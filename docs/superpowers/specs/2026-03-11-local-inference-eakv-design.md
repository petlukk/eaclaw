# Design: Local Inference Runtime with eakv KV Persistence

## Problem

eaclaw currently calls the Anthropic API for every LLM turn. Each tool-call round-trip
re-sends the full conversation context, costing O(N^2) in attention compute on the server side,
plus network latency and API costs.

## Goal

Replace the Anthropic backend with an embedded local inference runtime using:
- **llama.cpp** as the inference engine (linked as a C library)
- **eakv** as the KV cache backend (Q4 compressed, fused AVX-512 attention)
- **Qwen2.5-3B-Instruct Q4_K_M** as the initial model (~1.8GB weights)

The key innovation: **O(1) KV cache checkpointing** around tool calls. Since the KV cache
is append-only, a checkpoint is just a `seq_len` integer. Restore = reset `seq_len`. No memcpy.
This makes tool-call loops 10-50x faster than re-prefilling.

## Constraints

- ~4GB available RAM
- CPU-only, x86-64 with AVX-512
- Model must reliably produce structured JSON tool calls
- Minimal changes to agent loop — the `LlmProvider` trait interface stays the same,
  but `LocalLlmProvider` manages stateful KV sessions internally behind the stateless
  `chat()`/`chat_stream()` interface (see "Stateless-to-Stateful Bridge" below)

## Architecture

```
eaclaw (single process)
│
├── agent loop (existing, unchanged)
│   ├── tool registry (existing, unchanged)
│   ├── SIMD kernels (existing, unchanged)
│   └── uses LlmProvider trait (existing interface)
│
├── LocalLlmProvider (NEW — implements LlmProvider)
│   ├── holds llama_model + llama_context
│   ├── holds eakv_cache (KV backend)
│   ├── KV session: checkpoint(seq_len) / restore(seq_len)
│   ├── chat() = prefill + generate + parse tool calls
│   └── chat_stream() = same with token callback
│
├── llama.cpp (linked as C library)
│   └── patched: KV cache ops → eakv
│
└── eakv (linked as C library)
    └── Q4 KV storage + fused attention kernels
```

## Agent Loop with KV Checkpointing

```
user message arrives
    ↓
tokenize + append to context
    ↓
prefill new tokens only (incremental)
    ↓
checkpoint = current seq_len           ← O(1)
    ↓
generate tokens (sampling loop)
    ↓
tool call detected in output?
    ├─ yes:
    │   return LlmResponse with StopReason::ToolUse
    │   (agent loop executes tool, calls chat() again
    │    with updated message list — the Stateless-to-Stateful
    │    Bridge handles incremental prefill automatically:
    │    common prefix matches existing KV, only the new
    │    assistant + tool_result tokens get prefilled)
    │
    └─ no:
        return LlmResponse with StopReason::EndTurn
```

### Stateless-to-Stateful Bridge

The `LlmProvider` trait is stateless: each `chat()` / `chat_stream()` call receives the
full `&[Message]` history. But `LocalLlmProvider` maintains a persistent KV session
internally. The bridge works as follows:

1. **Provider tracks prefilled state:** `LocalLlmProvider` stores the token sequence
   it has already prefilled in an internal `Vec<Token>` called `prefilled_tokens`.

2. **On each `chat()` call:** The provider tokenizes the incoming `&[Message]` list
   into a new token sequence. It then computes the longest common prefix between
   `prefilled_tokens` and the new sequence.

3. **Delta prefill:** If the common prefix matches the current KV state, the provider
   only prefills the suffix (new tokens since last call). If there is a divergence
   (e.g., the conversation was edited), the provider restores to the divergence point
   and re-prefills from there.

4. **Typical case is O(suffix):** In the normal agent loop flow, each call adds
   an assistant message + tool results. The common prefix is the entire previous
   conversation, so only the new tokens get prefilled.

This keeps the `LlmProvider` trait unchanged while enabling KV persistence internally.

### Cost comparison

Note: these assume KV checkpoint mode (delta prefill only). Approach A adds ~10-30ms
of state export overhead on the initial prefill; Approach B (future) eliminates this.

| Context  | Traditional (re-prefill) | With KV checkpoint |
|----------|-------------------------|--------------------|
| 2K tokens  | ~40 ms   | ~5 ms (tool result tokens only) |
| 8K tokens  | ~160 ms  | ~6 ms |
| 32K tokens | ~600 ms  | ~7 ms |

## Component Details

### 1. eakv: Checkpoint API

New functions in `eakv.h`:

```c
/* Checkpoint = current seq_len. O(1). */
int eakv_checkpoint(eakv_cache_t *cache);

/* Restore to a previous checkpoint. O(1) — just resets seq_len.
 * Data beyond the checkpoint is logically discarded (overwritten on next append). */
void eakv_restore(eakv_cache_t *cache, int seq_len);
```

Implementation is trivial:
- `eakv_checkpoint()` returns `cache->seq_len`
- `eakv_restore()` validates `0 <= seq_len <= cache->seq_len` (current position, not max —
  restoring beyond what has been written would expose uninitialized data), then sets
  `cache->seq_len = seq_len`

This works because the KV cache is append-only and pre-allocated to `max_seq_len`.
Data beyond the checkpoint remains in memory but is ignored/overwritten.

### 2. eakv: llama.cpp KV Backend Integration

The integration point is replacing llama.cpp's internal KV cache with eakv.
Two approaches:

**Approach A: Intercept at state export/import (simpler)**
- After each prefill, export KV state via `llama_state_seq_get_data()`
- Feed into `eakv_from_llama_state()` (existing bridge)
- For attention, use eakv's fused kernels
- Pro: no llama.cpp source changes. Con: export/import overhead per prefill.

**Approach B: Patch llama.cpp KV allocation (deeper, better)**
- Replace `llama_kv_cache` internals to store Q4 via eakv
- Fused attention kernels called directly during inference
- Pro: zero-copy, maximum performance. Con: requires maintaining a llama.cpp fork.

**Recommendation: Start with Approach A**, migrate to B once the integration is validated.
The export/import overhead (~10-30ms for 4K context) only happens during prefill,
not during cached generation. For Approach A, the existing `eakv_from_llama_state()`
creates a new cache each time — an incremental variant `eakv_from_llama_state_append()`
is needed that appends new KV data to an existing cache rather than creating a fresh one.

### 3. eaclaw: llama.cpp FFI Layer

New Rust module: `crates/eaclaw-core/src/llm/llama_ffi.rs`

Minimal FFI bindings (~200-300 lines):

```rust
extern "C" {
    fn llama_model_load_from_file(path: *const c_char, params: llama_model_params) -> *mut llama_model;
    fn llama_new_context_with_model(model: *mut llama_model, params: llama_context_params) -> *mut llama_context;
    fn llama_free(ctx: *mut llama_context);
    fn llama_free_model(model: *mut llama_model);
    fn llama_tokenize(model: *const llama_model, text: *const c_char, ...) -> i32;
    fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;
    fn llama_token_to_piece(model: *const llama_model, token: i32, ...) -> i32;
    fn llama_state_seq_get_data(ctx: *mut llama_context, ...) -> usize;
    // sampling functions
}
```

Wrapped in a safe Rust struct `LlamaEngine` that manages lifetimes.

### 4. eaclaw: LocalLlmProvider

New file: `crates/eaclaw-core/src/llm/local.rs`

Implements `LlmProvider` trait:

```rust
pub struct LocalLlmProvider {
    // Interior mutability needed: LlmProvider trait takes &self,
    // but inference mutates the llama context and KV cache.
    inner: Mutex<LocalLlmInner>,
}

struct LocalLlmInner {
    engine: LlamaEngine,        // llama.cpp context (mutated by llama_decode)
    kv_cache: eakv::Cache,      // eakv KV backend (mutated by append)
    prefilled_tokens: Vec<i32>, // tokens already in KV cache
    tool_id_counter: u64,       // monotonic counter for tool-use IDs
}

impl LlmProvider for LocalLlmProvider {
    async fn chat(&self, messages, tools, system) -> Result<LlmResponse> { ... }
    async fn chat_stream(&self, messages, tools, system, on_text) -> Result<LlmResponse> { ... }
}
```

**Async/blocking bridge:** `llama_decode` is a blocking CPU-bound call. Both `chat()`
and `chat_stream()` use `tokio::task::spawn_blocking` to run the inference loop on
a blocking thread. For `chat_stream()`, the token callback is invoked from within the
blocking task via a `tokio::sync::mpsc` channel — the blocking task sends tokens,
an async task receives and calls `on_text`.

**Concurrency:** `LocalLlmProvider` serializes all inference calls via the `Mutex`.
A second `chat()` call blocks until the first completes. This is intentional — a single
model on CPU cannot usefully parallelize generation. Unlike `AnthropicProvider`, this
provider does not support concurrent requests.

Key responsibilities:
- Convert `Message` list → token sequence (using Qwen2.5 `<|im_start|>` / `<|im_end|>` chat template)
- Diff against `prefilled_tokens` to find the common prefix
- Incremental prefill (only new tokens since last call)
- Sampling loop with token callback for streaming
- Tool-call detection: when `<tool_call>` is detected, parse JSON, set `StopReason::ToolUse`
- Generate unique tool-use IDs: `format!("local_{}", self.tool_id_counter)` (monotonic counter)
- KV checkpoint/restore around tool calls

**StopReason mapping:**
- `<tool_call>` block detected in output → `StopReason::ToolUse`
- EOS token or max tokens → `StopReason::EndTurn` / `StopReason::MaxTokens`

### 5. Tool-Call Detection (Token-Level, Zero-String)

Tool calls use the `<tool_call>` / `</tool_call>` XML format. Detection happens at the
**token level** — no string construction or regex needed during the hot path.

#### Token pattern matching

At provider init, tokenize the tag prefixes once:

```rust
let open_pattern: Vec<i32> = tokenize("<tool_call>");   // e.g. [<, tool, _call, >]
let close_pattern: Vec<i32> = tokenize("</tool_call>"); // e.g. [</, tool, _call, >]
```

During generation, maintain a small ring buffer of recent token IDs. After each
sampled token, check if the tail of the buffer matches `open_pattern`:

```rust
if ring_buffer.ends_with(&open_pattern) {
    // Enter tool-call capture mode
    state = ToolCallCapture;
}
```

**Fast-path optimization:** Before the full `memcmp`, check only the last token
against the pattern's final token. This is a single integer comparison that
short-circuits ~99% of tokens:

```rust
if last_token == open_pattern.last()
    && ring_buffer.ends_with(&open_pattern)
{
    state = ToolCallCapture;
}
```

Full `memcmp` of ~3-5 integers only runs when the sentinel matches — making
detection effectively free per token.

#### Three states during generation

```
Normal → generating text, streaming tokens via on_text callback
         match open_pattern → transition to ToolCallCapture

ToolCallCapture → buffering token IDs (NOT streaming to on_text)
                  match close_pattern → transition to ToolCallParse

ToolCallParse → detokenize buffered IDs → serde_json::from_str
                extract "name" + "arguments"
                map to ContentBlock::ToolUse
                set StopReason::ToolUse
```

The JSON body is only detokenized and parsed once, when the closing tag is found.
During capture, tokens are accumulated as raw `i32` IDs — no string allocation.

#### Existing kernel reuse

For the JSON parse step, eaclaw's `extract_json_structural` kernel could be used
to find structural positions (`{ } : , "`) in the detokenized JSON string. However,
tool-call JSON is typically 50-200 bytes — too small for SIMD to outperform
`serde_json::from_str()`. The kernel is available as a future optimization for
batch/multi-stream scenarios where many tool calls are parsed concurrently.

#### Why token-level detection is future-safe

- Works identically in single-stream and multi-stream scenarios
- No intermediate string state to manage per stream — just a ring buffer of ints
- Pattern matching cost is constant regardless of context length
- Naturally extends to detecting other patterns (e.g., `<think>`, `<code>`)
  by adding more patterns to the match set

#### Edge cases

**Nested tags:** If the model produces `<tool_call>` inside an already-open capture,
reject it — do not stack-parse. Treat the inner tag as literal text in the JSON body.
Nesting is not a valid tool-call format.

**Runaway capture:** If `ToolCallCapture` accumulates more than 512 tokens without
seeing `close_pattern`, abort capture. Detokenize what was buffered, treat it as
regular text output (flush to `on_text`), and return to `Normal` state. This prevents
a malformed model output from silently swallowing the rest of generation.

**Token sequence stability:** The exact tokenization of `<tool_call>` depends on the
model's vocabulary. Tokenizing at init via `llama_tokenize` guarantees the pattern
matches the model's actual token sequence, regardless of how the tokenizer splits it
(`["<tool", "_call", ">"]` vs `["<", "tool", "_call", ">"]` etc.).

This works with Qwen, Llama, Mistral — all produce similar token sequences for
XML-delimited tags. The system prompt instructs the model to use this exact format.

### 6. System Prompt for Tool Use

Prepended to the system prompt:

```
You have access to the following tools. To call a tool, output:

<tool_call>
{"name": "tool_name", "arguments": {"key": "value"}}
</tool_call>

Available tools:
[generated from ToolRegistry::tool_defs()]
```

This replaces Anthropic's native tool-use API with prompt-based tool use,
which small models handle well with clear instructions.

### 7. Qwen2.5 Chat Template

The model uses `<|im_start|>` / `<|im_end|>` delimiters:

```
<|im_start|>system
{system prompt with tool definitions}<|im_end|>
<|im_start|>user
{user message}<|im_end|>
<|im_start|>assistant
{assistant response}<|im_end|>
```

The `LocalLlmProvider` formats `&[Message]` into this template before tokenization.
llama.cpp's tokenizer handles the special tokens natively for Qwen2.5 GGUF files.

### 8. Configuration

New environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `EACLAW_MODEL_PATH` | `~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf` | GGUF model file |
| `EACLAW_BACKEND` | `anthropic` | `anthropic` or `local` |
| `EACLAW_CTX_SIZE` | `4096` | Max context length |
| `EACLAW_THREADS` | num_cpus | Inference thread count |

When `EACLAW_BACKEND=local`, the CLI instantiates `LocalLlmProvider` instead of
`AnthropicProvider`. `Config::from_env()` must be updated to make `ANTHROPIC_API_KEY`
optional when backend is `local` (currently it errors if the key is missing).

## Build Changes

```bash
# New build dependencies
# - llama.cpp (git submodule or vendored)
# - eakv (sibling directory, linked)

# Cargo.toml additions:
# [build-dependencies]
# cc = "1"        # compile llama.cpp

# build.rs:
# - compile llama.cpp via cc crate
# - link libeakv.a
```

Feature-gated: `cargo build --features local-llm` to avoid requiring llama.cpp
for users who only want the Anthropic backend.

## File Plan

| File | Type | Purpose |
|------|------|---------|
| `crates/eaclaw-core/src/llm/llama_ffi.rs` | New | Raw FFI bindings to llama.cpp |
| `crates/eaclaw-core/src/llm/local.rs` | New | `LocalLlmProvider` implementation |
| `crates/eaclaw-core/src/llm/tool_parse.rs` | New | Token-level tool-call detector + JSON extractor |
| `crates/eaclaw-core/src/llm/mod.rs` | Edit | Export new modules, add `Backend` enum |
| `crates/eaclaw-core/src/config.rs` | Edit | Add local inference config vars |
| `eaclaw-cli/src/main.rs` | Edit | Backend selection logic |
| `build.rs` | New | Compile llama.cpp + link eakv |
| `Cargo.toml` | Edit | Add `cc` build dep, `local-llm` feature |

### eakv changes (in /root/dev/eakv/)

| File | Type | Purpose |
|------|------|---------|
| `include/eakv.h` | Edit | Add `eakv_checkpoint()`, `eakv_restore()` |
| `include/eakv_llama.h` | Edit | Add `eakv_from_llama_state_append()` |
| `src/cache.c` | Edit | Implement checkpoint/restore (2 functions, ~10 lines) |
| `src/llama_bridge.c` | Edit | Add incremental append variant (~40 lines) |

## Memory Budget (Qwen2.5-3B, 4K context)

| Component | Size |
|-----------|------|
| Model weights (Q4_K_M) | ~1.8 GB |
| KV cache (eakv Q4, 4K seq) | ~5 MB |
| KV checkpoint overhead | ~0 (just an integer) |
| llama.cpp scratch buffers | ~100 MB |
| eaclaw + SIMD kernels | ~10 MB |
| **Total** | **~1.9 GB** |

Fits comfortably in 4GB with room for OS and other processes.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Qwen 3B produces invalid JSON tool calls | Medium | System prompt engineering + fallback: retry once, then return text-only |
| llama.cpp API changes break FFI | Low | Pin to specific llama.cpp release tag |
| eakv Q4 quantization degrades model quality | Low | Already validated with TinyLlama — SNR 19.1 dB |
| Build complexity from C dependencies | Medium | Feature-gate behind `local-llm`, provide pre-built binaries |
| Approach A (state export) too slow | Low | ~10-30ms overhead per prefill, acceptable. Migrate to Approach B later. |
| Token diff false mismatch | Low | Hash-based comparison of tokenized messages; on mismatch, full re-prefill (correct but slow) |
| Model file missing at startup | Low | Clear error message with download instructions; do not crash silently |

## Future Work (Not In Scope)

- Approach B: patching llama.cpp KV internals for zero-copy eakv integration
- Multi-model support (load different GGUFs)
- Speculative decoding with a smaller draft model
- KV cache persistence to disk (session resume across restarts — eakv already has save/load)
- GPU offloading for specific layers
