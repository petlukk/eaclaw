# eaclaw

A high-performance AI agent powered by SIMD kernels ([Eä](https://github.com/petlukk/eacompute)) and Rust. The entire hot path — safety scanning, command routing, conversation recall — fits in L1 cache (27.6 KB). Cloud or fully local — no API key needed.

| vs. | eaclaw advantage |
|-----|-----------------|
| Claude Code / Cursor | Runs fully local, no API key needed, WhatsApp interface |
| Ollama + Open WebUI | Built-in tool use, shell safety, KV checkpointing |
| LangChain agents | Single binary, no Python, SIMD-speed routing, deterministic safety |
| Shell GPT | Multi-turn agent loop, background tasks, structured tool dispatch |

## Quick Start

### Cloud (Anthropic API)

```bash
git clone https://github.com/petlukk/eaclaw && cd eaclaw
ANTHROPIC_API_KEY=sk-ant-... cargo run
```

### Local (no API key)

```bash
git clone --recursive https://github.com/petlukk/eaclaw && cd eaclaw
cargo build --features local-llm

# Download a model (~1.8 GB, one-time)
mkdir -p ~/.eaclaw/models
wget -O ~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf

EACLAW_BACKEND=local cargo run --features local-llm
```

### WhatsApp

```bash
ANTHROPIC_API_KEY=sk-ant-... cargo run -- --whatsapp  # or EACLAW_BACKEND=local
```

Scan the QR code on first run. Mention `@eaclaw` in any chat to trigger it.

No Eä compiler needed. No manual cmake. Submodules auto-init if you forget `--recursive`.

## Tools

19 built-in tools — invokable directly via slash commands or by the LLM during conversation.

| Command | Description |
|---------|-------------|
| `/shell <cmd>` | Run a shell command (streams output) |
| `/read <path>` | Read a file |
| `/write <path> <content>` | Write content to a file |
| `/ls [path]` | List directory contents |
| `/grep <pattern> [path]` | Search files for a regex pattern |
| `/git <subcmd> [args]` | Read-only git (`status`, `log`, `diff`, `branch`, `show`, `blame`, `stash`) |
| `/calc <expr>` | Math expression (`+`, `-`, `*`, `/`, `%`, parentheses) |
| `/json <action> <input> [path]` | JSON operations (`keys`, `get`, `pretty`) |
| `/http <url>` | Fetch a URL |
| `/weather <city>` | Current weather |
| `/define <word>` | Word definition |
| `/translate <lang> <text>` | Translate text |
| `/summarize <url>` | Fetch and summarize a URL |
| `/memory <action> [key] [value]` | Key-value memory store |
| `/cpu` | System resource info |
| `/tokens <text>` | Estimate token count |
| `/time` | Current UTC time |
| `/bench <target>` | Benchmark a subsystem (`safety`, `router`) |
| `/remind <time> <message>` | Set a reminder (e.g. `/remind 30m check deploy &`) |

**Pipelines** — chain commands with `|`:

```
/shell ls -la | /tokens
```

**Background tasks** — append `&` to run in the background:

```
/shell sleep 10 &
/tasks              # check status
```

**Meta commands:** `/help`, `/quit`, `/tools`, `/clear`, `/model`, `/profile`, `/tasks`, `/recall <query>`

## Safety

**No sandbox. No container. Still safe.**

eaclaw uses a deterministic shell classifier and three-tier policy layer to prevent destructive commands without Docker, VMs, or seccomp overhead. Every `/shell` invocation is classified and checked before execution.

| Policy | Read-only | Write | Destructive |
|--------|-----------|-------|-------------|
| `open` | allow | allow | allow |
| **`safe`** (default) | allow | allow | **block** |
| `strict` | allow | **block** | **block** |

- **Read-only**: `ls`, `cat`, `grep`, `git log`, `ps`, `df`, ...
- **Write**: `cp`, `mv`, `mkdir`, `chmod`, `git push`, `pip install`, ...
- **Destructive**: `rm -rf`, `mkfs`, `dd`, `shutdown`, fork bombs, ...

The classifier handles compound commands (`;`, `&&`, `||`, `|`), `sudo` prefixes, and subcommand semantics (`git push` = write, `git log` = read).

Set via `EACLAW_SHELL_POLICY=safe` or `~/.eaclaw/shell_policy` file.

**Endpoint allowlisting** restricts which hosts `/http`, `/weather`, `/define`, and `/summarize` can reach. Configure via `~/.eaclaw/allowed_hosts.txt` or `EACLAW_ALLOWED_HOSTS`. Subdomain matching is supported.

**SIMD safety scanning** checks every message and tool output for prompt injection and secret leaks at 1.1 GB/s — adds ~2 µs per message.

> **Note:** These are defense-in-depth layers, not a security boundary. eaclaw runs with the full permissions of the user. Review your policy and allowlist before deploying.

## WhatsApp Integration

```
WhatsApp ←→ whatsmeow bridge (Go) ←JSON lines→ eaclaw (Rust)
                                                  ├── trigger filter (~20 ns)
                                                  ├── safety scan (~2 µs)
                                                  ├── recall context
                                                  └── LLM → response
```

1. The bridge links as a companion device to your WhatsApp account
2. Only messages mentioning `@eaclaw`, `!eaclaw`, or starting with `eaclaw` are processed
3. Each chat gets its own agent with isolated memory and history
4. History persists to `~/.eaclaw/groups/` and replays on startup

Trigger name matches `AGENT_NAME` (case-insensitive).

## Performance

The entire hot path fits in L1 icache (27.6 KB total, under the 32 KB budget). No kernel evicts another during a message turn.

| Operation | Input | Time | Throughput |
|-----------|-------|------|------------|
| Safety scan (injection + leak) | 1 KB | **930 ns** | 1.1 GB/s |
| Safety scan (injection + leak) | 100 KB | 98 µs | 1.0 GB/s |
| Full safety layer (SIMD + verify) | 1 KB | **2.4 µs** | 0.4 GB/s |
| Command routing (SIMD hash + verify) | per cmd | **9 ns** | — |
| Conversation recall (20 entries) | top-5 | **1.6 µs** | — |

| Kernel | `.text` size |
|--------|-------------|
| `command_router` | 1.3 KB |
| `leak_scanner` | 1.4 KB |
| `sanitizer` | 1.6 KB |
| `fused_safety` | 2.0 KB |
| `byte_classifier` | 2.5 KB |
| `json_scanner` | 2.9 KB |
| `search` | 15.4 KB |
| **Total** | **27.6 KB** |

SSE2 for safety/routing kernels, AVX-512/AVX2 for search — runtime CPU detection via CPUID.

### Local Inference (Qwen2.5-3B Q4_K_M, 2-core VPS)

| Metric | Value | Notes |
|--------|-------|-------|
| Model load | **1.9s** | mmap'd, fast on warm cache |
| Decode | **9.7 tok/s** | Memory-bandwidth-bound |
| Prefill | **31 tok/s** | Scales with core count |
| Tool-call round-trip | 5.3s | Including 100-token prefill |
| eakv KV sync | 2-8 ms | Negligible overhead |

Multi-turn conversations get free KV cache prefix reuse. On 4+ cores, expect significantly lower first-response latency.

## Architecture

**SIMD filter + scalar verify.** Eä kernels scan at cache-line speed, rejecting ~97% of byte positions. Rust verifies only at candidate positions.

### SIMD Kernels

All kernels compile from Eä (`.ea`) to shared libraries, embedded in the binary at compile time.

| Kernel | Purpose |
|--------|---------|
| `fused_safety` | Combined injection + secret leak detection |
| `command_router` | Slash command hash matching |
| `search` | Vector similarity (cosine, dot, L2), normalization, top-k |
| `byte_classifier` | Byte → flag classification |
| `json_scanner` | JSON structural character detection |
| `leak_scanner` | Secret pattern prefix scanning |
| `sanitizer` | Injection pattern prefix scanning |

### KV Cache Compression (eakv)

Local inference uses [eakv](https://github.com/petlukk/eakv) — Q4 KV cache quantization with fused attention kernels written in Eä.

| | llama.cpp F16 | eakv Q4 | Benefit |
|--|---------------|---------|---------|
| KV cache size (3B, 2K ctx) | ~8 MB | ~2.5 MB | **3.2x smaller** |
| Attention speed | baseline | 5x faster | Fused dot-product on packed Q4 |
| Tool-call loop | re-prefill ~40 ms | delta prefill ~5 ms | **8x faster** |

**O(1) checkpointing:** The KV cache is append-only — a checkpoint is just a `seq_len` integer. Restoring resets the counter with no data copy. Cost: **< 1 µs**.

- Multi-turn conversations reuse the KV prefix (system prompt + earlier turns are never re-prefilled)
- Tool-call loops only prefill new tokens, not the entire context

Fused attention kernels operate directly on Q4 nibbles using AVX-512/AVX2 with GQA support.

### Conversation Recall

`/recall` uses byte-histogram embeddings (256-dim) with SIMD cosine similarity for microsecond-latency search. Ring buffer (1024 entries) with recency boost. No external APIs — pure SIMD.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required for cloud)* | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Model (cloud mode) |
| `EACLAW_BACKEND` | `anthropic` | `anthropic` or `local` |
| `EACLAW_MODEL_PATH` | `~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf` | GGUF model file (local) |
| `EACLAW_CTX_SIZE` | `2048` | Context window tokens (local) |
| `EACLAW_BATCH_SIZE` | `512` | Prefill batch size (local) |
| `EACLAW_THREADS` | CPU count | Inference threads (local) |
| `EACLAW_MLOCK` | `0` | `1` to pin model in RAM |
| `EACLAW_SHELL_POLICY` | `safe` | `open`, `safe`, or `strict` |
| `AGENT_NAME` | `eaclaw` | Display name and trigger word |
| `MAX_TURNS` | `10` | Max tool-use turns per message |
| `COMMAND_PREFIX` | `/` | Slash command prefix |

**Identity:** Create `~/.eaclaw/identity.md` or set `EACLAW_IDENTITY` to customize the system prompt.

**WhatsApp:** `EACLAW_BRIDGE_PATH` (auto-detect) and `EACLAW_WA_SESSION_DIR` (`~/.eaclaw/whatsapp`).

## Building from Source

| Mode | Clone | Build |
|------|-------|-------|
| Cloud | `git clone` | `cargo build` |
| Local | `git clone --recursive` | `cargo build --features local-llm` |

The `local-llm` feature builds llama.cpp and eakv via cmake. First build ~5 minutes, then cached.

**Rebuilding SIMD kernels** (optional — only for modifying `.ea` sources):

```bash
./build.sh      # Requires Eä compiler + Go (for WhatsApp bridge)
```

The binary is self-contained — SIMD kernels are embedded and auto-extracted on first run to `~/.eaclaw/lib/`.

## Acknowledgements

- **[ironclaw](https://github.com/nearai/ironclaw)** — Rust agent framework with tool pipelines and safety scanning
- **[nanoclaw](https://github.com/qwibitai/nanoclaw)** — WhatsApp agent integration via whatsmeow bridge

## License

MIT
