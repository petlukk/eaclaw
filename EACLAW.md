# eaclaw — Cache-Resident SIMD Agent Framework

## Overview

eaclaw is an AI agent framework that combines SIMD-accelerated security scanning with streaming LLM integration. Every user message passes through cache-resident Ea kernels for injection detection and secret leak prevention before reaching the LLM. The entire security pipeline runs in single-digit microseconds — six orders of magnitude faster than the LLM call it protects.

**4,714 lines** of Rust + Ea across 40 source files. **69 tests**. Zero regex. Zero aho-corasick. All pattern matching compiled to native SIMD instructions via the Ea compiler.

---

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │                   eaclaw agent                   │
                        │                                                  │
  User input ──────────►│  ┌─────────────┐   ┌──────────────┐             │
                        │  │ SIMD command │   │  Fused SIMD  │             │
                        │  │   router     │──►│ safety scan  │             │
                        │  │  (331 B)     │   │  (961 B)     │             │
                        │  └─────────────┘   └──────┬───────┘             │
                        │        │                   │                     │
                        │   /help /quit          injection?                │
                        │   /tools /clear        leak?                     │
                        │   /model /profile        │                       │
                        │        │              ┌───▼───────────────┐      │
                        │        │              │ Anthropic Claude  │      │
                        │        │              │  SSE streaming    │      │
                        │        │              │  (tool loop)      │      │
                        │        ▼              └───┬───────────────┘      │
                        │    [respond]              │                      │
                        │                     ┌─────▼──────────┐          │
                        │                     │  Tool executor  │          │
                        │                     │  shell, http,   │          │
                        │                     │  time, memory   │          │
                        │                     └─────┬──────────┘          │
                        │                           │                      │
                        │                     SIMD leak scan               │
                        │                     on tool output               │
                        │                           │                      │
                        └───────────────────────────┼──────────────────────┘
                                                    ▼
                                              Streamed text
                                              to terminal
```

### Core Principle: SIMD Filter + Scalar Verify

Ea kernels process input at memory bandwidth (~GB/s), rejecting >99% of bytes. Only candidate positions are passed to cheap scalar verification in Rust. This two-phase design means:

- SIMD kernels never allocate — all memory is caller-provided
- Output is bitmasks, not position arrays — one i32 per 16-byte block
- Rust processes masks with `trailing_zeros()` + `mask &= mask - 1`
- The entire hot path fits in L1 instruction cache (1,292 bytes)

---

## Project Layout

```
eaclaw/
├── kernels/                          # Ea SIMD source files
│   ├── byte_classifier.ea            #   Byte flag classification (1,688 B .text)
│   ├── command_router.ea             #   Hash-based slash command matching (331 B)
│   ├── sanitizer.ea                  #   Injection pattern prefix filter (721 B)
│   ├── leak_scanner.ea               #   Secret leak prefix filter (584 B)
│   ├── fused_safety.ea               #   Combined inject+leak single pass (961 B)
│   ├── json_scanner.ea               #   JSON structural char finding (2,153 B)
│   └── search.ea                     #   Vector similarity operations (22,282 B)
│
├── crates/eaclaw-core/               # Core Rust library
│   └── src/
│       ├── lib.rs                    #   Module root
│       ├── config.rs                 #   Env-var configuration
│       ├── error.rs                  #   Error types
│       ├── agent/
│       │   ├── mod.rs                #   Agent loop, streaming, timing, /profile
│       │   └── router.rs             #   Command enum + parse
│       ├── channel/
│       │   ├── mod.rs                #   Channel trait (recv/send/send_chunk/flush)
│       │   └── repl.rs               #   Rustyline REPL with terminal restore
│       ├── llm/
│       │   ├── mod.rs                #   Message, ContentBlock, LlmProvider trait
│       │   └── anthropic.rs          #   Anthropic API + SSE streaming parser
│       ├── safety/
│       │   ├── mod.rs                #   SafetyLayer (fused SIMD + verify)
│       │   ├── sanitizer.rs          #   Injection pattern verification (24 patterns)
│       │   ├── leak_detector.rs      #   Secret leak verification (20+ patterns)
│       │   └── validator.rs          #   Input/output length validation
│       ├── tools/
│       │   ├── mod.rs                #   Tool trait + ToolRegistry
│       │   ├── echo.rs               #   Echo (test tool)
│       │   ├── time.rs               #   UTC timestamp
│       │   ├── shell.rs              #   Shell command execution
│       │   ├── http.rs               #   HTTP GET requests
│       │   └── memory.rs             #   In-memory key-value store
│       └── kernels/
│           ├── mod.rs                #   Module exports
│           ├── ffi.rs                #   Generated extern "C" declarations
│           ├── command_router.rs     #   Safe wrapper + CMD_* constants
│           ├── sanitizer_kernel.rs   #   Injection prefix scan wrapper
│           ├── leak_scanner.rs       #   Leak prefix scan wrapper
│           ├── fused_safety.rs       #   Fused scan wrapper
│           ├── json_scanner.rs       #   JSON scanner wrapper
│           ├── byte_classifier.rs    #   Byte classifier wrapper
│           └── search.rs             #   Vector search wrapper
│
├── eaclaw-cli/                       # Binary entry point
│   └── src/main.rs                   #   Config → Provider → Agent → run()
│
├── benches/
│   └── safety_bench.rs               #   Criterion benchmarks (SIMD vs aho-corasick)
│
├── build.sh                          #   Compile .ea → .so + FFI bindings
├── Cargo.toml                        #   Workspace (LTO, codegen-units=1)
└── CLAUDE.md                         #   Development conventions
```

---

## Ea Kernels

Seven kernels compiled to shared libraries by the Ea v1.6.0 compiler. All use `u8x16` SIMD vectors (not `u8x32`, which has movemask sign-bit issues). All provide x86_64 SIMD and aarch64 scalar implementations.

### command_router (331 bytes)

Hash-based slash command matching. Reads 4 bytes after `/`, computes `b1 + b2*256 + b3*65536 + b4*16777216`, compares against 6 known hashes. Runs entirely in registers.

```
/help → 0    /quit → 1    /tools → 2
/clear → 3   /model → 4   /profile → 5
```

### fused_safety (961 bytes)

Combined injection + leak detection in a single memory pass. Loads `text[i..i+16]` and `text[i+1..i+17]` once, checks all two-byte prefix pairs for both pattern sets. Outputs separate bitmask arrays.

Two-byte pairs checked per 16-byte block:
- **Injection (15 pairs):** ig, di, fo, yo, ac, pr, sy, as, us, ne, up, <|, |>, [I, [/
- **Leak (10 pairs):** sk, AK, gh, gi, xo, SG, se, --, AI, Be

Trades doubled compute for halved memory traffic. ~90 vector ops, fits L1.

### sanitizer (721 bytes)

Standalone injection prefix filter. Same 15 two-byte pairs as fused_safety but as a separate kernel. Used in benchmarks for comparison.

### leak_scanner (584 bytes)

Standalone leak prefix filter. Same 10 two-byte pairs. Matches: `sk-*`, `AKIA*`, `ghp_*`, `gho_*`, `xoxb-*`, `SG.*`, `sess_*`, `-----BEGIN`, `AIza*`, `Bearer`.

### byte_classifier (1,688 bytes)

Classifies each byte into flag categories via SIMD range comparisons:
- `FLAG_WS (1)` — whitespace
- `FLAG_LETTER (2)` — a-z, A-Z
- `FLAG_DIGIT (4)` — 0-9
- `FLAG_PUNCT (8)` — punctuation
- `FLAG_NONASCII (16)` — bytes > 127

### json_scanner (2,153 bytes)

Finds JSON structural characters (`{}[]:",\`) via SIMD equality checks. Two exports: `count_json_structural` (total count) and `extract_json_structural` (positions + types arrays).

### search (22,282 bytes)

Vector similarity operations for embedding search. Six exports: `batch_dot`, `batch_cosine`, `batch_l2`, `normalize_vectors`, `threshold_filter`, `top_k`. Uses `f32x8` on x86_64, `f32x4` on aarch64.

---

## Safety Pipeline

Every user message and every tool output passes through SIMD safety scanning:

```
User input
    │
    ▼
fused_safety SIMD kernel (single pass, ~4µs)
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
leak scan only (no injection check on outputs)
    │
    └──► blocks tool results containing secrets
```

### Two-Phase Detection

1. **SIMD filter** — The fused kernel scans all bytes at memory bandwidth. For each 16-byte block, it checks if any adjacent byte pair matches a known prefix. Output: one bitmask per block. This rejects >99% of bytes.

2. **Scalar verify** — For each set bit in the mask, Rust checks the full pattern (case-insensitive for injection, case-sensitive + format validation for leaks). This runs on candidates only — typically 0–5 positions per message.

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

The `on_text` callback prints each delta with `print!` + `flush()`, giving character-by-character streaming output with a leading `\n` before the first chunk to separate from the prompt.

---

## Agent Loop

```rust
loop {
    let input = channel.recv().await;          // readline

    let cmd = SIMD_command_router(input);       // 8 ns
    if cmd != NONE { handle_command(); continue; }

    let scan = fused_safety_scan(input);        // 3-9 µs
    if scan.injection_found { warn; continue; }
    if scan.leaks_found { block; continue; }

    messages.push(user_message);

    loop {  // tool loop (max 10 turns)
        let response = llm.chat_stream(         // ~2s (tokens stream live)
            messages, tools, system,
            |chunk| print_chunk(chunk),
        );

        if response.has_tool_use() {
            for tool_use in response.tool_uses {
                let output = tool.execute(input);
                let scan = safety.scan_output(output);  // leak check
                messages.push(tool_result);
            }
            continue;  // next LLM turn
        }

        flush();  // trailing newlines
        break;
    }

    store_timing();  // for /profile
}
```

---

## Tools

| Tool | Description |
|------|-------------|
| **echo** | Returns input unchanged (test tool) |
| **time** | Current UTC timestamp (RFC 3339) |
| **shell** | Executes shell commands, returns stdout + stderr |
| **http** | HTTP GET, returns body (truncated at 32KB) |
| **memory** | In-memory key-value store (write/read/list) |

---

## Commands

| Command | SIMD hash | Description |
|---------|-----------|-------------|
| `/help` | CMD_HELP (0) | Show available commands |
| `/quit` | CMD_QUIT (1) | Exit cleanly |
| `/tools` | CMD_TOOLS (2) | List available tools |
| `/clear` | CMD_CLEAR (3) | Clear conversation history |
| `/model` | CMD_MODEL (4) | Show current model |
| `/profile` | CMD_PROFILE (5) | Show last turn timing breakdown |

All commands are matched by the SIMD command router in 8 ns.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | **required** | API authentication |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Model to use |
| `AGENT_NAME` | `eaclaw` | Prompt prefix |
| `MAX_TURNS` | `10` | Max tool loop iterations |
| `COMMAND_PREFIX` | `/` | Slash command marker |

---

## Performance Results

### Kernel Code Sizes (.text sections)

| Kernel | .text size | L1i budget (of 64KB) |
|--------|----------:|--------------------|
| command_router | 331 B | 0.5% |
| leak_scanner | 584 B | 0.9% |
| sanitizer | 721 B | 1.1% |
| fused_safety | 961 B | 1.5% |
| byte_classifier | 1,688 B | 2.6% |
| json_scanner | 2,153 B | 3.4% |
| search | 22,282 B | 34.8% |
| **Hot path (router + fused)** | **1,292 B** | **2.0%** |
| **All kernels** | **29,400 B** | **45.9%** |

The hot path uses 2% of L1 instruction cache. All kernels combined use under half.

### Throughput

| Operation | Input | Latency | Throughput |
|-----------|-------|---------|------------|
| command_router | 5–48 B | **8.1 ns/call** | 123M calls/s |
| fused_safety | 44 B | **67.5 ns/call** | 0.65 GB/s |
| fused_safety | 4,500 B | **12.8 µs/call** | 0.35 GB/s |

### Cache Behavior (perf stat over 2.3 billion instructions)

| Counter | Value | Assessment |
|---------|-------|------------|
| Instructions per cycle | **1.84 IPC** | Near superscalar peak |
| L1-icache misses | **29,375** | 0.0013% — negligible |
| L1-dcache misses | **2.99M** | Expected (output arrays) |
| Branch mispredictions | **0.27%** | Near-perfect prediction |

29K L1-icache misses across 2.3B instructions confirms the kernels run entirely from L1 instruction cache after warmup.

### Live Timing (/profile)

```
Last turn timing:                    Last turn timing:
  Safety scan:    3 µs                 Safety scan:    9 µs
  LLM call:       1963 ms              LLM call:       3188 ms
  Total:          1964 ms              Tool: time      0 ms
                                       Total:          3189 ms
```

### Time Budget

```
                    ┌─────────────────────────────────────────────────┐
 Safety scan (9µs)  │▏                                                │  0.0003%
 Tool exec   (0ms)  │                                                 │  0.0000%
 LLM call (3188ms)  │█████████████████████████████████████████████████│ 99.9997%
                    └─────────────────────────────────────────────────┘
```

Safety scanning adds **3–9 microseconds** per turn. Six orders of magnitude faster than the LLM call. Zero measurable overhead.

---

## Line Counts

| Component | Files | Lines |
|-----------|------:|------:|
| Rust — eaclaw-core | 28 | 3,154 |
| Rust — eaclaw-cli | 1 | 48 |
| Ea kernels | 7 | 1,201 |
| Benchmarks | 1 | 267 |
| Config / build | 4 | 44 |
| **Total** | **41** | **4,714** |

---

## Test Results

69 tests across 10 modules, all passing:

| Module | Tests | What's tested |
|--------|------:|---------------|
| kernels/byte_classifier | 8 | Classification flags, edge sizes, non-ASCII, 1KB random |
| kernels/command_router | 9 | All 6 commands, no-match, too-short, no-slash, empty |
| kernels/sanitizer_kernel | 6 | Empty, no-match, prefix detection, case, special tokens, false-positive rate |
| kernels/leak_scanner | 6 | Empty, no-match, sk-prefix, AWS key, PEM, false-positive rate |
| kernels/fused_safety | 5 | Empty, inject-matches-separate, leak-matches-separate, both, clean |
| kernels/json_scanner | 5 | Empty, simple JSON, no structural, 16-byte boundary, 17-byte |
| kernels/search | 7 | dot, cosine, L2, normalize, threshold, top_k, top_k edge |
| safety/sanitizer | 7 | Clean, injection patterns, case-insensitive, special tokens, false-positive rate |
| safety/leak_detector | 7 | Clean, OpenAI, AWS, GitHub, PEM, short-sk, empty |
| safety/validator | 4 | Valid, empty, too-long, null-byte |
| agent/router | 4 | Parse commands, not-a-command, unknown |

```
test result: ok. 69 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

---

## Build & Run

```bash
./build.sh                          # Compile .ea → .so + generate FFI bindings
cargo build --release               # Build optimized binary (LTO enabled)
cargo test                          # Run all 69 tests
cargo bench                         # Criterion benchmarks (SIMD vs aho-corasick)
LD_LIBRARY_PATH=target/kernels \
  cargo run --release               # Start REPL (requires ANTHROPIC_API_KEY)
```

---

## Design Decisions

1. **Two-byte prefixes, not single-byte** — Single-byte SIMD matching produces too many false candidates (e.g., matching `s` alone). Two-byte pairs (e.g., `sk`, `ig`) reduce candidates ~100x with negligible extra compute.

2. **Bitmasks, not position arrays** — SIMD kernels output one i32 per 16-byte block. Set bits indicate candidate positions. This avoids dynamic allocation and branch-heavy scatter stores.

3. **Fused kernel** — The fused_safety kernel loads each byte pair once and checks all 25 prefix pairs. This halves memory traffic vs calling sanitizer + leak_scanner separately.

4. **No regex, no aho-corasick** — Ea kernels replace these entirely. Benchmarks show comparable or better throughput with zero runtime dependencies.

5. **`u8x16` not `u8x32`** — Wider vectors cause movemask sign-bit issues when extracting match positions. 16-byte vectors give clean 16-bit masks.

6. **Caller-provided memory** — Ea kernels never allocate. The Rust wrapper pre-allocates output arrays based on input length. Kernel code stays pure computation.

7. **SSE break on `message_stop`** — The streaming parser breaks immediately on `message_stop` instead of waiting for TCP connection close. Prevents hangs from HTTP keep-alive.

8. **Terminal restoration via libc** — `std::process::exit()` skips destructors, leaving rustyline's raw mode active. Explicit `tcsetattr` restores `ECHO | ICANON | ISIG` before exit.

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

### Dev only
- **criterion** — benchmarking
- **aho-corasick** — benchmark comparison baseline
- **rand** — random input generation for benchmarks
