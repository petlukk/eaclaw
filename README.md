# eaclaw

A high-performance AI assistant powered by SIMD kernels written in [Eä](https://github.com/petlukk/eacompute) and Rust. Uses the Anthropic Claude API or local LLM inference (llama.cpp + eakv) for conversation, with embedded SIMD acceleration for safety scanning, command routing, and conversation recall.

**Every kernel fits in L1 cache.** The entire hot path — safety scanning, command routing, conversation recall — runs at memory bandwidth with zero allocations on the fast path. Local inference uses llama.cpp with [eakv](https://github.com/petlukk/eakv) KV cache compression and O(1) checkpointing, so tool-call loops resume generation without re-prefilling the full context.

**No sandbox. No container. Still safe.** eaclaw uses a deterministic shell classifier and policy layer to prevent destructive commands (`rm -rf`, `mkfs`, `dd`, `shutdown`, fork bombs) without Docker, VMs, or seccomp overhead. Every `/shell` invocation is classified as read-only, write, or destructive — and the policy decides what runs. Combined with SIMD prompt injection scanning and endpoint allowlisting, the agent can operate with real system access while keeping guardrails that work at nanosecond speed.

> **Note:** The shell guard and safety scanner are defense-in-depth layers, not a security boundary. eaclaw runs with the full permissions of the user. Review your policy (`EACLAW_SHELL_POLICY`, default `safe`) and endpoint allowlist (`~/.eaclaw/allowed_hosts.txt`) before deploying.

## Install

Download the latest binary for your platform from [Releases](https://github.com/petlukk/eaclaw/releases):

```bash
curl -LO https://github.com/petlukk/eaclaw/releases/latest/download/eaclaw-v0.1.1-linux-x86_64.tar.gz
tar xzf eaclaw-v0.1.1-linux-x86_64.tar.gz
chmod +x eaclaw-cli eaclaw-bridge
```

The archive contains two binaries:

| Binary | Description |
|--------|-------------|
| `eaclaw-cli` | Main agent (REPL + WhatsApp mode) |
| `eaclaw-bridge` | WhatsApp bridge (only needed for `--whatsapp` mode) |

## Quick Start

### Cloud Mode (Anthropic API)

```bash
git clone https://github.com/petlukk/eaclaw
cd eaclaw
cargo build
ANTHROPIC_API_KEY=sk-ant-... cargo run
```

### Local Mode (no API key needed)

```bash
git clone --recursive https://github.com/petlukk/eaclaw
cd eaclaw
cargo build --features local-llm

# Download a model (~1.8 GB, one-time)
mkdir -p ~/.eaclaw/models
wget -O ~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf

EACLAW_BACKEND=local cargo run --features local-llm
```

No Eä compiler needed. No manual cmake. Submodules auto-init if you forget `--recursive`.

### WhatsApp Mode

```bash
ANTHROPIC_API_KEY=sk-ant-... cargo run -- --whatsapp  # or EACLAW_BACKEND=local
```

On first run, scan the QR code with WhatsApp ("Link a device"). Then mention `@eaclaw` in any chat to trigger it:

```
[eaclaw] Connected to WhatsApp!
[eaclaw] Listening for messages mentioning @eaclaw or !eaclaw

  ⚡ Triggered by Peter — calling LLM...
  ↳ Tool: shell (local)
  → [12036342...] eaclaw: I'm in /root/dev/eaclaw
```

The binary is self-contained — SIMD kernels are embedded and auto-extracted on first run to `~/.eaclaw/lib/`.

## Building from Source

Requires only a Rust toolchain. Pre-built SIMD kernels are checked into the repo.

| Mode | Clone | Build |
|------|-------|-------|
| Cloud | `git clone` | `cargo build` |
| Local | `git clone --recursive` | `cargo build --features local-llm` |

The `local-llm` feature builds llama.cpp and eakv from source automatically via cmake. First build takes ~5 minutes; subsequent builds are cached.

### Rebuilding SIMD Kernels (optional)

Only needed if modifying `.ea` kernel sources. Requires the [Eä compiler](https://github.com/petlukk/eacompute). Go is required for the WhatsApp bridge.

```bash
./build.sh      # Compile .ea kernels → .so + build WhatsApp bridge
```

## Modes

| Mode | Command | Description |
|------|---------|-------------|
| **REPL** | `./eaclaw-cli` | Interactive terminal session with `You>` / `eaclaw>` prompts |
| **WhatsApp** | `./eaclaw-cli --whatsapp` | WhatsApp bridge — respond to `@eaclaw` mentions in any chat |

## Commands

### Meta Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quit` | Exit the session |
| `/tools` | List available tools |
| `/clear` | Clear conversation history and recall index |
| `/model` | Show current LLM model |
| `/profile` | Show timing data for the last turn |
| `/tasks` | List background tasks |
| `/recall <query>` | Search conversation history using SIMD similarity |

### Tool Commands

Tools can be invoked directly (bypassing the LLM) or used by the LLM during conversation.

| Command | Description |
|---------|-------------|
| `/time` | Current UTC date and time |
| `/calc <expr>` | Evaluate a math expression (supports `+`, `-`, `*`, `/`, `%`, parentheses) |
| `/http <url>` | Fetch a URL |
| `/shell <cmd>` | Run a shell command (streams output) |
| `/read <path>` | Read a file |
| `/write <path> <content>` | Write content to a file |
| `/ls [path]` | List directory contents |
| `/memory <action> [key] [value]` | Key-value memory store (`read`, `write`, `list`) |
| `/json <action> <input> [path]` | JSON operations (`keys`, `get`, `pretty`) |
| `/cpu` | System resource info (CPU, memory, uptime) |
| `/tokens <text>` | Estimate token count for text |
| `/bench <target>` | Benchmark a subsystem (`safety`, `router`) |
| `/weather <city>` | Get current weather for a city |
| `/translate <lang> <text>` | Translate text to another language |
| `/define <word>` | Look up a word definition |
| `/summarize <url>` | Fetch a URL and summarize its content |
| `/grep <pattern> [path]` | Search files for a regex pattern (`file:line:match` output) |
| `/git <subcommand> [args]` | Read-only git commands (`status`, `log`, `diff`, `branch`, `show`, `blame`, `stash`) |
| `/remind <time> <message>` | Set a reminder (e.g. `/remind 30m check deploy &`) |

### Background Execution

Append `&` to run any tool command in the background:

```
/shell sleep 10 &
/tasks              # check status
```

### Pipelines

Chain tool commands with `|`:

```
/shell ls -la | /tokens
```

## WhatsApp Integration

eaclaw can run as a WhatsApp agent via a Go bridge binary (whatsmeow).

### How It Works

```
WhatsApp ←→ whatsmeow bridge (Go) ←JSON lines→ eaclaw (Rust)
                                                  ├── trigger filter (~20 ns)
                                                  ├── safety scan (~2 µs)
                                                  ├── recall context
                                                  └── LLM → response
```

1. The bridge links as a companion device to your WhatsApp account
2. All messages are received; only those mentioning `@eaclaw` or `!eaclaw` are processed
3. Each chat gets its own agent with isolated memory (VectorStore) and history (JSONL)
4. History persists to `~/.eaclaw/groups/` and is replayed into recall on startup

### WhatsApp Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EACLAW_BRIDGE_PATH` | auto-detect | Path to the `eaclaw-bridge` binary |
| `EACLAW_WA_SESSION_DIR` | `~/.eaclaw/whatsapp` | WhatsApp session data directory |

### Trigger Patterns

Messages are processed when they contain:
- `@eaclaw` — mention anywhere in the message
- `!eaclaw` — bang prefix
- `eaclaw ...` — message starts with the agent name

The trigger name matches `AGENT_NAME` (case-insensitive).

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required for cloud)* | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Model to use (cloud mode) |
| `EACLAW_BACKEND` | `anthropic` | `anthropic` (cloud) or `local` (llama.cpp) |
| `EACLAW_MODEL_PATH` | `~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf` | GGUF model file (local mode) |
| `EACLAW_CTX_SIZE` | `2048` | Context window size in tokens (local mode) |
| `EACLAW_BATCH_SIZE` | `512` | Prefill batch size (local mode) |
| `EACLAW_THREADS` | CPU count | Inference threads (local mode) |
| `EACLAW_MLOCK` | `0` | Set to `1` to pin model in RAM (needs `ulimit -l unlimited`) |
| `AGENT_NAME` | `eaclaw` | Agent display name and trigger word |
| `MAX_TURNS` | `10` | Max tool-use turns per conversation message |
| `COMMAND_PREFIX` | `/` | Prefix for slash commands |
| `EACLAW_SHELL_POLICY` | `safe` | Shell command policy: `open`, `safe`, or `strict` |

### Shell Policy

Controls what `/shell` commands the agent can execute:

| Mode | Read-only | Write | Destructive |
|------|-----------|-------|-------------|
| `open` | allow | allow | allow |
| `safe` (default) | allow | allow | **block** |
| `strict` | allow | **block** | **block** |

- **Read-only**: `ls`, `cat`, `grep`, `git log`, `ps`, `df`, etc.
- **Write**: `cp`, `mv`, `mkdir`, `chmod`, `git push`, `pip install`, etc.
- **Destructive**: `rm -rf`, `mkfs`, `dd`, `shutdown`, fork bombs, etc.

Set via `EACLAW_SHELL_POLICY=safe` or `~/.eaclaw/shell_policy` file.

### Identity

Customize the agent's personality by creating `~/.eaclaw/identity.md` or setting `EACLAW_IDENTITY` to a file path. The contents are prepended to the system prompt.

### Endpoint Allowlisting

Restrict which hosts the `/http` tool can access:

- **File:** `~/.eaclaw/allowed_hosts.txt` (one host per line, `#` comments supported)
- **Env:** `EACLAW_ALLOWED_HOSTS=api.example.com,docs.example.com`

Subdomain matching is supported — allowing `example.com` also allows `api.example.com`. When no allowlist is configured, all hosts are permitted.

## Performance

The entire hot path — safety scanning, command routing, conversation recall — fits in L1 icache (27.6 KB total, under the typical 32 KB budget). This means no kernel evicts another during a message turn: the CPU never stalls on instruction fetch across the full scan → route → recall sequence.

| Operation | Input Size | Time | Throughput |
|-----------|-----------|------|------------|
| Safety scan (injection + leak) | 1 KB | **930 ns** | 1.1 GB/s |
| Safety scan (injection + leak) | 10 KB | 9 µs | 1.1 GB/s |
| Safety scan (injection + leak) | 100 KB | 98 µs | 1.0 GB/s |
| Full safety layer (SIMD + verify) | 1 KB | **2.4 µs** | 0.4 GB/s |
| Full safety layer (SIMD + verify) | 100 KB | 262 µs | 0.4 GB/s |
| Command routing (SIMD hash + verify) | per command | **9 ns** | — |
| Conversation recall (20 entries) | top-5 query | **1.6 µs** | — |

Safety and routing kernels use `u8x16` SIMD (SSE2). The search kernel uses `f32x16` AVX-512 on supported CPUs (EPYC, Sapphire Rapids), falling back to `f32x8` AVX on others — runtime CPU detection via CPUID, no recompilation needed.

| Kernel | `.text` size | Fits in |
|--------|-------------|---------|
| `command_router` | 1.3 KB | L1 icache |
| `leak_scanner` | 1.4 KB | L1 icache |
| `sanitizer` | 1.6 KB | L1 icache |
| `fused_safety` | 2.0 KB | L1 icache |
| `byte_classifier` | 2.5 KB | L1 icache |
| `json_scanner` | 2.9 KB | L1 icache |
| `search` | 15.4 KB | L1 icache |
| **Total hot path** | **27.6 KB** | **L1 icache (32 KB)** |

The safety scan adds **~2 µs** of latency to every message — invisible next to the LLM round-trip.

### Local Inference (Qwen2.5-3B Q4_K_M, 2-core VPS)

| Metric | Value | Notes |
|--------|-------|-------|
| Model load | **1.9s** | mmap'd, fast on warm page cache |
| Decode throughput | **9.7 tok/s** | Memory-bandwidth-bound (single-token) |
| Prefill throughput | **31 tok/s** | Batched, scales with core count |
| End-to-end first turn | 3.9 tok/s | Prefill-dominated on low core count |
| End-to-end second turn | 4.5 tok/s | KV prefix reuse saves ~35% prefill |
| Tool-call round-trip | 5.3s | Including 100-token prefill |
| eakv KV sync | 2-8 ms | Negligible overhead |

The generation loop runs in C++ with a pre-built vocab lookup table for single-pass streaming and tool detection — no replay pass or per-token FFI overhead. Multi-turn conversations get free KV cache prefix reuse via eakv checkpointing.

**Scaling note:** These numbers are from a 2-core VPS where prefill is memory-bandwidth-bound. Prefill throughput scales strongly with core count (more cores overlap compute with dequantization loads). On 4+ core machines, expect significantly lower first-response latency. Decode throughput (~9 tok/s) is bandwidth-bound in a different way and won't scale as dramatically — for latency-sensitive workloads, use the cloud backend.

## Architecture

eaclaw uses a **SIMD filter + scalar verify** architecture. Eä kernels process input at cache-line speed, rejecting ~97% of byte positions. Rust scalar code verifies only at candidate positions.

### SIMD Kernels

All kernels are compiled from Eä (`.ea`) to shared libraries, embedded in the binary at compile time.

| Kernel | Purpose |
|--------|---------|
| `fused_safety` | Combined injection + secret leak detection |
| `command_router` | Slash command hash matching |
| `search` | Vector similarity (cosine, dot, L2), normalization, top-k |
| `byte_classifier` | Byte → flag classification |
| `json_scanner` | JSON structural character detection |
| `leak_scanner` | Secret pattern prefix scanning |
| `sanitizer` | Injection pattern prefix scanning |

### Safety Pipeline

Every user message passes through the SIMD safety scanner before reaching the LLM:

```
Input → fused_safety kernel → injection verify → leak verify → LLM
```

Tool outputs are also scanned for injection attempts and secret leaks before being fed back to the LLM.

### KV Cache Compression (eakv)

Local inference uses [eakv](https://github.com/petlukk/eakv), a Q4 KV cache quantization library with fused attention kernels written in Eä.

**Why not just use llama.cpp's KV cache?** Vanilla llama.cpp stores the KV cache in F16 — fine for large machines, but on a 2 GB VPS running a 3B model, every megabyte counts. eakv quantizes the KV cache to Q4_1 (4-bit with per-group scale and bias) and runs attention directly on packed nibbles — no decompression step.

| | llama.cpp F16 | eakv Q4 | Benefit |
|--|---------------|---------|---------|
| KV cache size (3B, 2K ctx) | ~8 MB | ~2.5 MB | **3.2x smaller** |
| Attention speed | baseline | 5x faster | Fused dot-product on packed Q4 |
| Tool-call loop | re-prefill ~40 ms | delta prefill ~5 ms | **8x faster** |

**O(1) checkpointing** is the key feature for agent workloads. The KV cache is append-only — a checkpoint is just a `seq_len` integer. Restoring a checkpoint resets the counter with no data copy. This means:

- Multi-turn conversations reuse the KV prefix (system prompt + earlier turns are never re-prefilled)
- Tool-call loops only prefill the new tokens (tool result + assistant prefix), not the entire context
- Checkpoint/restore cost: **< 1 µs**

The fused attention kernels (`fused_k_score`, `fused_v_sum`) operate directly on Q4 nibbles using AVX-512/AVX2 with 2-position loop unrolling and grouped-query attention (GQA) support. Pre-compiled kernel `.o` files are checked into the repo — no Eä compiler needed for builds.

### Conversation Recall

`/recall` uses byte-histogram embeddings (256-dim, one per byte value) with SIMD cosine similarity for microsecond-latency conversation search. No external APIs or models — pure SIMD.

Ring buffer (1024 entries) with recency boost ensures recent conversation context is preferred.

## Acknowledgements

eaclaw draws inspiration from:

- **[ironclaw](https://github.com/nearai/ironclaw)** — Rust agent framework with tool pipelines and safety scanning
- **[nanoclaw](https://github.com/qwibitai/nanoclaw)** — WhatsApp agent integration via whatsmeow bridge

## License

MIT
