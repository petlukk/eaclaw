# eaclaw — Cache-Resident SIMD Agent Framework

## Architecture

**SIMD filter + scalar verify.** Eä kernels rapidly identify candidate positions
(rejecting >99% of bytes). Cheap Rust scalar code verifies matches at candidates only.

## Project Layout

- `kernels/` — Eä source files (`.ea`), compiled to `.so` by `build.sh`
- `crates/eaclaw-core/` — Core library (FFI wrappers, safety, agent loop, LLM, tools)
- `eaclaw-cli/` — Binary entry point
- `benches/` — Criterion benchmarks

## Build

```bash
./build.sh              # Compile .ea → .so + generate FFI bindings
cargo build             # Build Rust crates (links pre-built .so files)
cargo test              # Run all tests
cargo bench             # Run benchmarks
cargo run               # Start REPL (requires ANTHROPIC_API_KEY)
```

## Environment Variables

- `ANTHROPIC_API_KEY` — Required for LLM
- `ANTHROPIC_MODEL` — Default: `claude-sonnet-4-20250514`
- `AGENT_NAME` — Default: `eaclaw`
- `MAX_TURNS` — Default: `10`
- `COMMAND_PREFIX` — Default: `/`

## Conventions

- No file exceeds 500 lines
- Kernels use `u8x16` (not `u8x32`) to avoid movemask sign-bit issues
- Bitmask output from SIMD kernels, not position arrays
- Rust processes masks with `trailing_zeros()` + `mask &= mask - 1` bit loop
- No regex/aho-corasick in production deps — Eä replaces these
- All memory is caller-provided in Eä kernels
- Each kernel fits in L1 cache (~0.5–1.5 KB instructions)

## Kernel Reference

| Kernel | Export | Purpose |
|--------|--------|---------|
| byte_classifier | `classify_bytes` | Byte → flag classification |
| json_scanner | `count_json_structural`, `extract_json_structural` | JSON structure finding |
| leak_scanner | `scan_leak_prefixes` | Secret pattern prefix filter |
| sanitizer | `scan_injection_prefixes` | Injection pattern prefix filter |
| command_router | `match_command` | Slash command hash matching |

## Safety Pipeline

```
Input → classify_bytes → scan_injection_prefixes → verify → scan_leak_prefixes → verify → ScanResult
```
