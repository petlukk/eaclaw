# eaclaw

A high-performance AI assistant powered by SIMD kernels written in [Eä](https://github.com/petlukk/eacompute) and Rust. Uses the Anthropic Claude API for conversation, with embedded SIMD acceleration for safety scanning, command routing, and conversation recall.

**Every kernel fits in L1 cache.** The entire hot path — safety scanning, command routing, conversation recall — runs at memory bandwidth with zero allocations on the fast path.

## Quick Start

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run
./eaclaw
```

The binary is self-contained — SIMD kernels are embedded and auto-extracted on first run to `~/.eaclaw/lib/`.

## Building from Source

Requires the [Eä compiler](https://github.com/petlukk/eacompute) and a Rust toolchain.

```bash
./build.sh      # Compile .ea kernels → .so
cargo build     # Build the binary (embeds kernels)
cargo test      # Run tests (209 tests, no LD_LIBRARY_PATH needed)
cargo bench     # Run benchmarks
```

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
| `/time` | Current UTC timestamp |
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

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Model to use |
| `AGENT_NAME` | `eaclaw` | Agent display name |
| `MAX_TURNS` | `10` | Max tool-use turns per conversation message |
| `COMMAND_PREFIX` | `/` | Prefix for slash commands |

### Identity

Customize the agent's personality by creating `~/.eaclaw/identity.md` or setting `EACLAW_IDENTITY` to a file path. The contents are prepended to the system prompt.

### Endpoint Allowlisting

Restrict which hosts the `/http` tool can access:

- **File:** `~/.eaclaw/allowed_hosts.txt` (one host per line, `#` comments supported)
- **Env:** `EACLAW_ALLOWED_HOSTS=api.example.com,docs.example.com`

Subdomain matching is supported — allowing `example.com` also allows `api.example.com`. When no allowlist is configured, all hosts are permitted.

## Performance

The hot path is designed to stay in L1 cache. Every SIMD kernel fits comfortably — the largest safety kernel is 2 KB of instructions, the command router is 1.2 KB.

| Operation | Input Size | Time | Throughput |
|-----------|-----------|------|------------|
| Safety scan (injection + leak) | 1 KB | **930 ns** | 1.1 GB/s |
| Safety scan (injection + leak) | 10 KB | 9 µs | 1.1 GB/s |
| Safety scan (injection + leak) | 100 KB | 98 µs | 1.0 GB/s |
| Full safety layer (SIMD + verify) | 1 KB | **2.1 µs** | 0.5 GB/s |
| Full safety layer (SIMD + verify) | 100 KB | 179 µs | 0.6 GB/s |
| Command routing | per command | **960 ns** | — |

All kernels use `u8x16` SIMD (SSE2/NEON), keeping instruction footprint small:

| Kernel | `.text` size | Fits in |
|--------|-------------|---------|
| `command_router` | 1.2 KB | L1 icache |
| `leak_scanner` | 1.4 KB | L1 icache |
| `sanitizer` | 1.6 KB | L1 icache |
| `fused_safety` | 2.0 KB | L1 icache |
| `byte_classifier` | 2.5 KB | L1 icache |
| `json_scanner` | 2.9 KB | L1 icache |
| `search` | 23 KB | L2 (multi-function) |

The safety scan adds **~2 µs** of latency to every message — invisible next to the LLM round-trip.

## Architecture

eaclaw uses a **SIMD filter + scalar verify** architecture. Eä kernels process input at cache-line speed, rejecting >99% of bytes. Rust scalar code verifies only at candidate positions.

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

Tool outputs are also scanned for secret leaks before display.

### Conversation Recall

`/recall` uses byte-histogram embeddings (256-dim, one per byte value) with SIMD cosine similarity for microsecond-latency conversation search. No external APIs or models — pure SIMD.

## License

MIT
