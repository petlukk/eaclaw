# eaclaw — Cache-Resident SIMD Agent Framework

## Overview

eaclaw is an AI agent framework that combines SIMD-accelerated security scanning with streaming LLM integration. Every user message passes through cache-resident Eä kernels for injection detection and secret leak prevention before reaching the LLM. The entire security pipeline runs in single-digit microseconds — six orders of magnitude faster than the LLM call it protects.

**11,000+ lines** of Rust + Eä + Go across 66 source files. **296 tests**. Zero regex. Zero aho-corasick. All pattern matching compiled to native SIMD instructions via the Eä compiler. Single binary — all kernels embedded and auto-extracted at runtime. WhatsApp integration via Go bridge.

---

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │                   eaclaw agent                   │
                        │                                                  │
  User input ──────────►│  ┌─────────────┐   ┌──────────────┐             │
                        │  │ SIMD command │   │  Fused SIMD  │             │
                        │  │   router     │──►│ safety scan  │             │
                        │  │  (1,338 B)   │   │  (2,024 B)   │             │
                        │  └─────────────┘   └──────┬───────┘             │
                        │        │                   │                     │
                        │   /help /quit          injection?                │
                        │   /tools /clear        leak?                     │
                        │   /model /profile        │                       │
                        │   /recall /tasks     ┌───▼───────────────┐      │
                        │        │             │ Anthropic Claude  │      │
                        │        │             │  SSE streaming    │      │
                        │        │             │  (tool loop)      │      │
                        │        ▼             └───┬───────────────┘      │
                        │    [respond]             │                      │
                        │                    ┌─────▼──────────┐          │
                        │                    │  Tool executor  │          │
                        │                    │  19 built-in│          │
                        │                    │  tools          │          │
                        │                    └─────┬──────────┘          │
                        │                          │                      │
                        │                    SIMD leak scan               │
                        │                    on tool output               │
                        │                          │                      │
                        │                    ┌─────▼──────────┐          │
                        │                    │  SIMD recall    │          │
                        │                    │  index update   │          │
                        │                    └────────────────┘          │
                        └──────────────────────────┼──────────────────────┘
                                                   ▼
                                             Streamed text
                                             to terminal / WhatsApp
```

### WhatsApp Integration

```
WhatsApp servers ←→ whatsmeow bridge (Go, 229 lines)
                         │
                    JSON lines stdin/stdout
                         │
                    WhatsAppChannel (Rust)
                         │
                    Gateway (per-group routing)
                         ├── trigger filter (~20 ns, SIMD)
                         ├── safety scan (~2 µs, fused kernel)
                         ├── VectorStore recall (per group)
                         ├── HistoryLog persistence (JSONL)
                         └── LLM call → response → send
```

The bridge links as a companion device (like WhatsApp Web). Each chat gets isolated agent state — vector recall, conversation history, message count. History persists to `~/.eaclaw/groups/<jid>/history.jsonl` and is replayed into the VectorStore on startup.

### Core Principle: SIMD Filter + Scalar Verify

Eä kernels process input at memory bandwidth (~GB/s), rejecting ~97% of bytes. Only candidate positions are passed to cheap scalar verification in Rust. This two-phase design means:

- SIMD kernels never allocate — all memory is caller-provided
- Output is bitmasks, not position arrays — one i32 per 16-byte block
- Rust processes masks with `trailing_zeros()` + `mask &= mask - 1`
- The entire hot path (Rust + kernel) fits in L1 instruction cache (5,410 bytes)

### Single Binary Packaging

All 8 Eä kernel `.so` files are embedded in the binary via `include_bytes!` at compile time. On first run, they are extracted to `~/.eaclaw/lib/v{VERSION}/` and loaded via `libloading`. The search kernel ships in two variants (AVX and AVX-512); runtime CPUID detection selects the appropriate one. No `LD_LIBRARY_PATH` required.

> **Note:** Eä kernels target x86_64 SIMD. The compiler supports aarch64 cross-compilation but this has not been tested for eaclaw.

---

## Project Layout

```
eaclaw/
├── kernels/                          # Eä SIMD source files
│   ├── byte_classifier.ea            #   Byte flag classification (2,478 B .text)
│   ├── command_router.ea             #   Hash-based slash command matching (1,338 B)
│   ├── sanitizer.ea                  #   Injection pattern prefix filter (1,622 B)
│   ├── leak_scanner.ea               #   Secret leak prefix filter (1,395 B)
│   ├── fused_safety.ea               #   Combined inject+leak single pass (2,024 B)
│   ├── json_scanner.ea               #   JSON structural char finding (2,890 B)
│   ├── search.ea                     #   Vector similarity operations, f32x8 AVX fallback
│   └── search_avx512.ea              #   Vector similarity operations, f32x16 AVX-512 (15,740 B)
│
├── crates/eaclaw-core/               # Core Rust library
│   └── src/
│       ├── lib.rs                    #   Module root
│       ├── config.rs                 #   Env-var + identity + allowlist config
│       ├── error.rs                  #   Error types
│       ├── recall.rs                 #   SIMD conversation recall (byte-histogram)
│       ├── agent/
│       │   ├── mod.rs                #   Agent loop, streaming, timing, /recall
│       │   ├── router.rs             #   Command enum + parse
│       │   └── tool_dispatch.rs      #   Direct tool execution, pipelines
│       ├── channel/
│       │   ├── mod.rs                #   Channel trait (recv/send/send_chunk/flush)
│       │   ├── repl.rs               #   Rustyline REPL (You>/eaclaw> prompts)
│       │   ├── types.rs              #   InboundMessage, GroupChannel trait, trigger matching
│       │   ├── whatsapp.rs           #   WhatsApp bridge subprocess channel
│       │   ├── gateway.rs            #   Per-group message routing + safety + recall
│       │   └── wa_loop.rs            #   WhatsApp agent loop (gateway → LLM → respond)
│       ├── llm/
│       │   ├── mod.rs                #   Message, ContentBlock, LlmProvider trait
│       │   ├── anthropic.rs          #   Anthropic API + SSE streaming parser
│       │   ├── local.rs              #   LocalLlmProvider (llama.cpp + eakv bridge)
│       │   ├── llama_ffi.rs          #   llama.cpp FFI bindings + LlamaEngine wrapper
│       │   ├── eakv_ffi.rs           #   eakv FFI bindings + EakvCache wrapper
│       │   └── tool_parse.rs         #   Token-level tool-call detection
│       ├── safety/
│       │   ├── mod.rs                #   SafetyLayer (fused SIMD + verify)
│       │   ├── sanitizer.rs          #   Injection pattern verification (24 patterns)
│       │   ├── leak_detector.rs      #   Secret leak verification (20+ patterns)
│       │   ├── shell_guard.rs         #   Shell command risk classifier
│       │   └── validator.rs          #   Input/output length validation
│       ├── tools/
│       │   ├── mod.rs                #   Tool trait + ToolRegistry
│       │   ├── time.rs               #   UTC timestamp
│       │   ├── calc.rs               #   Math expression evaluator (i128 + f64)
│       │   ├── shell.rs              #   Shell command execution (streaming)
│       │   ├── http.rs               #   HTTP GET (with endpoint allowlisting)
│       │   ├── memory.rs             #   In-memory key-value store
│       │   ├── read_file.rs          #   File read
│       │   ├── write_file.rs         #   File write
│       │   ├── ls.rs                 #   Directory listing
│       │   ├── json_tool.rs          #   JSON operations (keys, get, pretty)
│       │   ├── cpu.rs                #   System resource info
│       │   ├── tokens.rs             #   Token count estimator
│       │   ├── bench_tool.rs         #   Subsystem benchmarking
│       │   ├── weather.rs            #   Weather via wttr.in
│       │   ├── define.rs             #   Word definition via dictionaryapi.dev
│       │   ├── translate.rs          #   Translation via LLM
│       │   ├── summarize.rs          #   URL summarization via LLM
│       │   ├── grep.rs               #   Regex file search (grep -rn)
│       │   ├── git.rs                #   Read-only git commands
│       │   ├── remind.rs             #   Timer-based reminders
│       │   └── echo.rs               #   Echo (test tool)
│       └── kernels/
│           ├── mod.rs                #   Module exports + init()
│           ├── ffi.rs                #   Runtime kernel loading (libloading + embed)
│           ├── command_router.rs     #   Safe wrapper + CMD_* constants (24 commands)
│           ├── sanitizer_kernel.rs   #   Injection prefix scan wrapper
│           ├── leak_scanner.rs       #   Leak prefix scan wrapper
│           ├── fused_safety.rs       #   Fused scan wrapper (reusable buffers)
│           ├── json_scanner.rs       #   JSON scanner wrapper
│           ├── byte_classifier.rs    #   Byte classifier wrapper
│           ├── arg_tokenizer.rs      #   SIMD argument tokenizer
│           └── search.rs             #   Vector search wrapper (6 functions)
│
├── crates/eaclaw-core/tests/         # Integration tests
│   ├── tool_integration.rs           #   Full tool + router tests
│   ├── recall_integration.rs         #   Recall system tests
│   ├── recall_bench.rs               #   Recall latency benchmarks
│   └── edge_cases.rs                 #   Safety, allowlist, identity tests
│
│       └── persist.rs                 #   JSONL history log + VectorStore replay
│
├── eaclaw-cli/                       # Binary entry point
│   └── src/main.rs                   #   REPL mode + --whatsapp mode
│
├── bridge/                           # WhatsApp bridge (Go)
│   ├── main.go                       #   whatsmeow client, JSON lines protocol, QR rendering
│   ├── go.mod                        #   Go module
│   └── go.sum                        #   Go dependencies
│
├── benches/
│   └── safety_bench.rs               #   Criterion benchmarks (SIMD vs aho-corasick)
│
├── build.sh                          #   Compile .ea → .so + FFI bindings + Go bridge
├── Cargo.toml                        #   Workspace (LTO, codegen-units=1)
└── README.md                         #   User documentation
```

---

## Eä Kernels

Eight kernels compiled to shared libraries by the Eä compiler. Safety and routing kernels use `u8x16` SIMD vectors (SSE2). The search kernel uses `f32x16` AVX-512 on supported CPUs, falling back to `f32x8` AVX. Compiled and tested on x86_64.

### command_router (1,338 bytes)

Hash-based slash command matching. Reads 2–4 bytes after `/`, computes hash, compares against 27 known hashes. Two-stage verification in Rust prevents hash collisions. Measured at **9 ns/call** (release build, 13-command benchmark).

```
Meta:  /help /quit /tools /clear /model /profile /tasks /recall
Tools: /time /calc /http /shell /memory /read /write /ls /json /cpu /tokens /bench
       /weather /translate /define /summarize /grep /git /remind
```

### fused_safety (2,024 bytes)

Combined injection + leak detection in a single memory pass. Loads `text[i..i+16]` and `text[i+1..i+17]` once, checks all two-byte prefix pairs for both pattern sets. Outputs separate bitmask arrays.

Two-byte pairs checked per 16-byte block:
- **Injection (15 pairs):** ig, di, fo, yo, ac, pr, sy, as, us, ne, up, <|, |>, [I, [/
- **Leak (10 pairs):** sk, AK, gh, gi, xo, SG, se, --, AI, Be

Trades doubled compute for halved memory traffic. Fits L1 instruction cache.

### sanitizer (1,622 bytes)

Standalone injection prefix filter. Same 15 two-byte pairs as fused_safety but as a separate kernel. Used in benchmarks for comparison.

### leak_scanner (1,395 bytes)

Standalone leak prefix filter. Same 10 two-byte pairs. Matches: `sk-*`, `AKIA*`, `ghp_*`, `gho_*`, `xoxb-*`, `SG.*`, `sess_*`, `-----BEGIN`, `AIza*`, `Bearer`.

### byte_classifier (2,478 bytes)

Classifies each byte into flag categories via SIMD range comparisons:
- `FLAG_WS (1)` — whitespace
- `FLAG_LETTER (2)` — a-z, A-Z
- `FLAG_DIGIT (4)` — 0-9
- `FLAG_PUNCT (8)` — punctuation
- `FLAG_NONASCII (16)` — bytes > 127

### json_scanner (2,890 bytes)

Finds JSON structural characters (`{}[]:",\`) via SIMD equality checks. Two exports: `count_json_structural` (total count) and `extract_json_structural` (positions + types arrays).

### search (15,740 bytes AVX-512 / 23,196 bytes AVX fallback)

Vector similarity operations for conversation recall. Six exports: `batch_dot`, `batch_cosine`, `batch_l2`, `normalize_vectors`, `threshold_filter`, `top_k`. AVX-512 variant uses `f32x16` on x86_64 (selected at runtime via CPUID), falling back to `f32x8` AVX. aarch64 uses `f32x4`.

---

## Safety Pipeline

Every user message and every tool output passes through SIMD safety scanning:

```
User input
    │
    ▼
fused_safety SIMD kernel (single pass, ~2 µs at 1KB)
    │
    ├──► injection bitmasks ──► scalar verify against 24 patterns
    │                           "ignore previous", "system:", "<|", "[INST]",
    │                           "disregard", "forget", "you are now",
    │                           "act as", "new instructions", "override",
    │                           "<script", "[/INST]", ...
    │
    └──► leak bitmasks ──► scalar verify against 20+ patterns
                           sk-ant-api*, sk-proj-*, AKIA*, ghp_*, gho_*,
                           xoxb-*, xoxp-*, SG.*, sk_live_*, sk_test_*,
                           sess_*, -----BEGIN*, AIza*, Bearer*, github_pat_*

Tool output
    │
    ▼
fused SIMD scan (injection + leak detection)
    │
    ├──► blocks tool results containing injection attempts
    └──► blocks tool results containing secrets

Shell command (before execution)
    │
    ▼
ShellGuard classifier (pure Rust, ~ns)
    │
    ├──► ALLOW: read-only (ls, cat, grep, git log, ...)
    ├──► WRITE: recoverable (cp, mv, mkdir, git push, ...)
    └──► DESTRUCTIVE: irreversible (rm -rf, mkfs, dd, shutdown, ...)
         Blocked in safe/strict modes.
```

### Shell Guard

The `ShellGuard` classifies shell commands by risk before execution. Three policy modes: `open` (no restrictions), `safe` (block destructive), `strict` (read-only). Handles compound commands (`;`, `&&`, `||`, `|`), strips prefixes (`sudo`, env vars), and recognizes subcommand semantics (`git push` = write, `git log` = read, `cargo test` = read, `cargo build` = write).

Configured via `EACLAW_SHELL_POLICY=safe` or `~/.eaclaw/shell_policy`. Default is `safe`.

### Two-Phase Detection

1. **SIMD filter** — The fused kernel scans all bytes at memory bandwidth. For each 16-byte block, it checks if any adjacent byte pair matches a known prefix. Output: one bitmask per block. On typical English text, this rejects ~97% of byte positions (measured across 8 representative prompts).

2. **Scalar verify** — For each set bit in the mask, Rust checks the full pattern (case-insensitive for injection, case-sensitive + format validation for leaks). This runs on candidates only — typically 0–5 positions per message.

---

## Conversation Recall

`/recall` uses byte-histogram embeddings — a 256-dimensional f32 vector where each dimension counts the frequency of that byte value. The SIMD `search` kernel handles normalization, cosine similarity, and top-k extraction.

Pipeline: `embed_bytes → normalize_vectors (SIMD) → batch_cosine (SIMD FMA) → top_k (SIMD)`

All conversation turns (user + assistant) are indexed. Zero API calls. Measured latency: **1.6 µs** at 20 entries (typical conversation, AVX-512), **22 µs** at 500 entries.

---

## Local Inference (llama.cpp + eakv)

When built with `--features local-llm`, eaclaw can run fully offline using llama.cpp for inference and eakv for Q4 KV cache compression. The `LocalLlmProvider` wraps the stateless `LlmProvider` trait into a stateful engine with incremental KV cache reuse.

```
Qwen2.5-3B Q4_K_M (.gguf, 1.95 GiB)
    │
    ▼
LlamaEngine (FFI)              EakvCache (FFI)
├── tokenize                    ├── checkpoint/restore
├── decode (batch chunked)      ├── import_llama_state
├── sample                      └── seq_len tracking
├── kv_cache_truncate
└── export_kv_state ──────────► (best-effort sync)
    │
    ▼
LocalLlmProvider
├── format_chat_template (Qwen2.5 <|im_start|> format)
├── common_prefix_len → incremental prefill
├── ToolCallDetector (token-level <tool_call> tags)
└── fallback: raw JSON → ToolUse parsing
```

**Key design decisions:**
- Stateless-to-stateful bridge: `prefilled_tokens` vec tracks what's in the KV cache; new turns only decode the diff
- Batch chunking: llama.cpp's `n_batch=512` limit requires splitting large prefills
- eakv sync is best-effort (non-fatal) — llama.cpp's internal KV cache handles inference alone
- Tool detection: token-level sentinel matching + fallback raw JSON parser (Qwen2.5-3B often skips `<tool_call>` tags)
- Single-pass streaming: pre-built vocab lookup table (token ID → string) allows text conversion, streaming, and tool detection inside the C++ generation callback — no replay pass needed
- C++ generation loop: decode+sample runs in a tight C++ loop, eliminating per-token Rust↔C FFI roundtrips

**Performance vs standalone llama.cpp (Qwen2.5-3B Q4_K_M, 4 threads, ctx=4096):**

| Metric | eaclaw | llama.cpp standalone |
|--------|--------|---------------------|
| Model load | **2.5s** | 3.7s |
| Decode (pure generation) | **9.5 tok/s** | 10.1 tok/s |
| End-to-end (incl. prefill) | 4.2 tok/s | — |
| Decode (KV reuse) | **9.0 tok/s** | N/A |
| Tool-call round-trip | 5.4s | N/A |

eaclaw reaches **94% of standalone llama.cpp** decode speed. The remaining ~6% gap is the per-token callback trampoline (C++ → Rust). Multi-turn conversations get free KV cache reuse via eakv checkpointing.

---

## Streaming SSE

The Anthropic Messages API streams tokens via Server-Sent Events when `"stream": true` is set. Our SSE parser:

1. Receives raw byte chunks via `reqwest::Response::bytes_stream()`
2. Buffers and splits on newlines to extract `data: {...}` lines
3. Dispatches on event type:
   - `content_block_start` — begin text or tool_use accumulation
   - `content_block_delta` — `text_delta` printed to stdout immediately; `input_json_delta` accumulated for tool input
   - `content_block_stop` — finalize content block
   - `message_delta` — extract stop_reason
   - `message_stop` — break stream loop immediately
4. Returns complete `LlmResponse` with all content blocks for the agent loop

The `on_text` callback prints each delta with `print!` + `flush()`, giving character-by-character streaming output.

---

## Agent Loop

```rust
loop {
    let input = channel.recv().await;          // readline (You> prompt)

    // Pipeline detection: "cmd1 | /cmd2"
    if input.contains(" | /") { execute_pipeline(); continue; }

    let cmd = SIMD_command_router(input);       // ~9 ns
    if cmd == RECALL { recall_search(); continue; }
    if cmd == TASKS  { show_tasks(); continue; }
    if cmd.is_meta() { handle_meta(); continue; }
    if cmd.is_tool() { execute_tool(); continue; }  // direct, bypasses LLM

    let scan = fused_safety_scan(input);        // ~2 µs at 1KB
    if scan.injection_found { warn; continue; }
    if scan.leaks_found { block; continue; }

    messages.push(user_message);
    recall_store.insert(input);

    loop {  // tool loop (max 10 turns)
        let response = llm.chat_stream(         // ~2s (tokens stream live)
            messages, tools, system,
            |chunk| print_chunk(chunk),          // eaclaw> prefix
        );

        if response.has_tool_use() {
            for tool_use in response.tool_uses {
                let output = tool.execute(input);
                let scan = safety.scan_output(output);  // injection + leak check
                messages.push(tool_result);
            }
            continue;  // next LLM turn
        }

        recall_store.insert(response_text);
        flush();
        break;
    }

    store_timing();  // for /profile
}
```

---

## Tools

| Tool | Command | Description |
|------|---------|-------------|
| **time** | `/time` | Current UTC date and time (formatted) |
| **calc** | `/calc <expr>` | Math evaluator (i128 integers + f64 floats) |
| **shell** | `/shell <cmd>` | Shell execution with streaming output |
| **http** | `/http <url>` | HTTP GET with endpoint allowlisting |
| **memory** | `/memory <action> [key] [value]` | In-memory key-value store |
| **read** | `/read <path>` | Read file contents |
| **write** | `/write <path> <content>` | Write file |
| **ls** | `/ls [path]` | Directory listing |
| **json** | `/json <action> <input> [path]` | JSON operations (keys, get, pretty) |
| **cpu** | `/cpu` | System resource info (CPU, memory, uptime) |
| **tokens** | `/tokens <text>` | Token count estimator |
| **bench** | `/bench <target>` | Benchmark subsystem (safety, router) |
| **weather** | `/weather <city>` | Current weather (via wttr.in) |
| **translate** | `/translate <lang> <text>` | Translate text (via LLM) |
| **define** | `/define <word>` | Word definition (via dictionaryapi.dev) |
| **summarize** | `/summarize <url>` | Fetch and summarize URL (via LLM) |
| **grep** | `/grep <pattern> [path]` | Regex file search (`file:line:match` output) |
| **git** | `/git <subcommand> [args]` | Read-only git (`status`, `log`, `diff`, `branch`, `show`, `blame`, `stash`) |
| **remind** | `/remind <time> <message>` | Timer reminder (use with `&` for background) |
| **echo** | *(LLM only)* | Returns input unchanged (test tool) |

Background execution: append `&` (e.g., `/shell sleep 5 &`, `/remind 30m check deploy &`). Completed background tasks are shown automatically at the next prompt. Pipelines: `/shell ls | /tokens`.

---

## Commands

| Command | SIMD hash ID | Description |
|---------|-------------|-------------|
| `/help` | CMD_HELP (0) | Show available commands |
| `/quit` | CMD_QUIT (1) | Exit cleanly |
| `/tools` | CMD_TOOLS (2) | List available tools |
| `/clear` | CMD_CLEAR (3) | Clear conversation + recall history |
| `/model` | CMD_MODEL (4) | Show current model |
| `/profile` | CMD_PROFILE (5) | Show last turn timing breakdown |
| `/tasks` | CMD_TASKS (18) | List background tasks |
| `/recall` | CMD_RECALL (19) | Search conversation history (SIMD) |

All 27 commands (8 meta + 19 tools) matched by the SIMD command router with two-stage verification (hash + full name check). Measured at **9 ns/call**.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | **required (cloud)** | API authentication |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Model to use (cloud mode) |
| `EACLAW_BACKEND` | `anthropic` | `anthropic` (cloud) or `local` (llama.cpp) |
| `EACLAW_MODEL_PATH` | `~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf` | GGUF model file (local mode) |
| `EACLAW_CTX_SIZE` | `4096` | Context window size (local mode) |
| `EACLAW_THREADS` | CPU count | Inference threads (local mode) |
| `AGENT_NAME` | `eaclaw` | Prompt prefix |
| `MAX_TURNS` | `10` | Max tool loop iterations |
| `COMMAND_PREFIX` | `/` | Slash command marker |
| `EACLAW_SHELL_POLICY` | `safe` | Shell guard: `open`, `safe`, `strict` |

### WhatsApp

| Variable | Default | Description |
|----------|---------|-------------|
| `EACLAW_BRIDGE_PATH` | auto-detect | Path to `eaclaw-bridge` binary |
| `EACLAW_WA_SESSION_DIR` | `~/.eaclaw/whatsapp` | Session data directory |

### Identity

`~/.eaclaw/identity.md` or `EACLAW_IDENTITY` env var — contents prepended to system prompt.

### Endpoint Allowlisting

`~/.eaclaw/allowed_hosts.txt` or `EACLAW_ALLOWED_HOSTS` (comma-separated) — restricts `/http` tool. Subdomain matching supported.

---

## Performance Results

### Kernel Code Sizes (.text sections)

| Kernel | .text size | L1i budget (of 64KB) |
|--------|----------:|--------------------|
| command_router | 1,338 B | 2.1% |
| leak_scanner | 1,395 B | 2.2% |
| sanitizer | 1,622 B | 2.5% |
| fused_safety | 2,024 B | 3.2% |
| byte_classifier | 2,478 B | 3.9% |
| json_scanner | 2,890 B | 4.5% |
| search (AVX-512) | 15,740 B | 24.6% |
| search (AVX fallback) | 23,196 B | 36.2% |
| **Hot path (Rust scan_input + fused kernel)** | **5,410 B** | **8.5%** |
| **All kernels (AVX-512)** | **27,587 B** | **43.1%** |
| **All kernels (AVX fallback)** | **35,043 B** | **54.8%** |

On AVX-512 machines, the full hot path (all kernels) fits in L1 icache (27.6 KB < 32 KB). On AVX-only machines, the search kernel spills to L2.

### Throughput (Criterion)

| Operation | Input | Latency | Throughput |
|-----------|-------|---------|------------|
| Fused SIMD kernel | 1 KB | **930 ns** | 1.1 GB/s |
| Fused SIMD kernel | 10 KB | 9 µs | 1.1 GB/s |
| Fused SIMD kernel | 100 KB | 98 µs | 1.0 GB/s |
| Full safety layer (SIMD + verify) | 1 KB | **2.0 µs** | 0.5 GB/s |
| Full safety layer (SIMD + verify) | 10 KB | 17 µs | 0.6 GB/s |
| Full safety layer (SIMD + verify) | 100 KB | 222 µs | 0.4 GB/s |

### Cache Behavior (perf stat)

**200-byte input (typical prompt), 2M iterations:**

| Counter | Value | Assessment |
|---------|-------|------------|
| Instructions | 17.4 B | |
| IPC | **3.71** | Near 4-wide superscalar ceiling |
| L1-icache misses | 111,585 | 0.00064% — negligible |
| L1-dcache misses | 2.36M | Expected (output arrays) |
| Branch mispredictions | **0** | Perfect prediction |
| Per call | **721 ns** | |

**1KB input, 2M iterations:**

| Counter | Value | Assessment |
|---------|-------|------------|
| Instructions | 46.4 B | |
| IPC | **3.67** | Near ceiling |
| L1-icache misses | 342,955 | 0.00074% — negligible |
| Branch mispredictions | **0** | Perfect prediction |
| Per call | **1,997 ns** | |
| Cycles/byte | **6.2** | ~6 cycles/byte (VM-dependent) |

**10KB input, 200K iterations:**

| Counter | Value | Assessment |
|---------|-------|------------|
| Instructions | 37.8 B | |
| IPC | **3.39** | Good (memory-bound at this size) |
| L1-icache misses | 317,539 | 0.00084% — negligible |
| Branch mispredictions | **0** | Perfect prediction |
| Per call | **18 µs** | |
| Cycles/byte | **5.4** | Below target |

Zero branch mispredictions and <0.001% L1-icache miss rate across all sizes confirms the kernels run entirely from L1 instruction cache after warmup.

### Live Timing (/profile)

```
Last turn timing:                    Last turn timing:
  Safety scan:    2 µs                 Safety scan:    3 µs
  LLM call:       1963 ms              LLM call:       3188 ms
  Total:          1964 ms              Tool: time      0 ms
                                       Total:          3189 ms
```

### Time Budget

```
                    ┌─────────────────────────────────────────────────┐
 Safety scan (3µs)  │▏                                                │  0.0001%
 Tool exec   (0ms)  │                                                 │  0.0000%
 LLM call (3188ms)  │█████████████████████████████████████████████████│ 99.9999%
                    └─────────────────────────────────────────────────┘
```

Safety scanning adds **2–3 microseconds** per turn. Six orders of magnitude faster than the LLM call. Zero measurable overhead.

---

## Line Counts

| Component | Files | Lines |
|-----------|------:|------:|
| Rust — eaclaw-core (src) | 56 | 9,366 |
| Rust — eaclaw-cli | 1 | 98 |
| Rust — integration tests | 6 | 1,368 |
| Eä kernels | 7 | 1,325 |
| Go — WhatsApp bridge | 1 | 229 |
| Benchmarks | 1 | 283 |
| **Total** | **72** | **12,386** |

---

## Test Results

296 tests across unit tests, integration tests, and edge case tests:

```
test result: ok. 220 passed  (unit tests — eaclaw-core lib, incl. shell guard)
test result: ok. 33 passed   (edge cases — safety, allowlist, identity, calc)
test result: ok. 10 passed   (recall — conversation, unicode, large store)
test result: ok. 2 passed    (recall bench — latency benchmarks)
test result: ok. 31 passed   (tool integration — all 19 tools + router)
─────────────────────────────
         296 passed, 0 failed
```

---

## Build & Run

```bash
./build.sh                          # Compile .ea → .so + build WhatsApp bridge
cargo build --release               # Build single binary (LTO, embeds kernels)
cargo test                          # Run all 296 tests (no LD_LIBRARY_PATH needed)
cargo bench                         # Criterion benchmarks
cargo run --release                 # Start REPL (requires ANTHROPIC_API_KEY)
cargo run --release -- --whatsapp   # Start WhatsApp mode
```

### Local Inference

```bash
# Build with llama.cpp + eakv support
cargo build --release --features local-llm

# Download Qwen2.5-3B-Instruct Q4_K_M (~1.8 GB)
mkdir -p ~/.eaclaw/models
wget -O ~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf

# Run (no API key needed)
EACLAW_BACKEND=local cargo run --release --features local-llm
EACLAW_BACKEND=local cargo run --release --features local-llm -- --whatsapp

# Run benchmarks
cargo test --features local-llm -p eaclaw-core --test local_llm_bench -- --nocapture
```

---

## Design Decisions

1. **Two-byte prefixes, not single-byte** — Single-byte SIMD matching produces too many false candidates (e.g., matching `s` alone). Two-byte pairs (e.g., `sk`, `ig`) reduce candidates ~100x with negligible extra compute.

2. **Bitmasks, not position arrays** — SIMD kernels output one i32 per 16-byte block. Set bits indicate candidate positions. This avoids dynamic allocation and branch-heavy scatter stores.

3. **Fused kernel** — The fused_safety kernel loads each byte pair once and checks all 25 prefix pairs. This halves memory traffic vs calling sanitizer + leak_scanner separately.

4. **No regex, no aho-corasick** — Eä kernels replace these entirely. Benchmarks show comparable or better throughput with zero runtime dependencies.

5. **`u8x16` not `u8x32` for safety kernels** — Wider vectors cause movemask sign-bit issues when extracting match positions. 16-byte vectors give clean 16-bit masks. The search kernel uses `f32x16` (AVX-512) where available, since it operates on f32 vectors and doesn't need movemask.

6. **Caller-provided memory** — Eä kernels never allocate. The Rust wrapper pre-allocates output arrays based on input length. Kernel code stays pure computation.

7. **Embedded kernels with runtime dispatch** — All `.so` files are embedded in the binary via `include_bytes!` and extracted to `~/.eaclaw/lib/v{VERSION}/` on first run. Version-stamped for clean upgrades. The search kernel ships in AVX and AVX-512 variants; `is_x86_feature_detected!("avx512f")` selects the right one at startup.

8. **Byte-histogram recall** — 256-dim embeddings (one per byte value) for conversation search. Zero external API calls, microsecond latency, pure SIMD cosine similarity.

9. **SSE break on `message_stop`** — The streaming parser breaks immediately on `message_stop` instead of waiting for TCP connection close. Prevents hangs from HTTP keep-alive.

10. **Terminal restoration via libc** — `std::process::exit()` skips destructors, leaving rustyline's raw mode active. Explicit `tcsetattr` restores `ECHO | ICANON | ISIG` before exit.

---

## Dependencies

### Production
- **tokio** — async runtime
- **reqwest** — HTTP client (json + stream features)
- **futures** — stream utilities for SSE parsing
- **serde / serde_json** — serialization
- **async-trait** — async trait methods
- **rustyline** — line editing
- **chrono** — UTC timestamps
- **libc** — terminal state restoration
- **thiserror** — error derive
- **tracing** — structured logging
- **home** — home directory resolution
- **libloading** — runtime shared library loading

### Local inference (feature = "local-llm")
- **llama.cpp** — vendored, statically linked (libllama, libggml, libggml-base, libggml-cpu)
- **eakv** — Q4 KV cache compression with fused AVX-512 attention kernels

### Dev only
- **criterion** — benchmarking
- **aho-corasick** — benchmark comparison baseline
- **rand** — random input generation for benchmarks
