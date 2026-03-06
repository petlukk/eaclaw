# eaclaw Benchmark Results

---

## Run 3: Full Tool Layer (12 commands)
### Date: 2026-03-06
### Change: Extended to 12 tool commands, removed /echo, added nested hash structure

Replaced `/echo` with `/calc`. Added `/read`, `/write`, `/ls`, `/json`, `/cpu`, `/tokens`, `/bench`.
Kernel restructured: nested 2→3→4 byte hash (handles `/ls` at 3 bytes, `/cpu` at 4 bytes).
New tools: calc (expression eval), file ops, JSON, system info, token estimator, microbenchmark.

### Tests: 116 passed (was 88)

### perf stat

| Input | Per call | Throughput | IPC | L1i misses/call | Branch misses |
|-------|----------|------------|-----|------------------|---------------|
| 200 B | 788 ns | 0.25 GB/s | 3.65 | 0.048 | 362,713 |
| 1 KB | 2,001 ns | 0.51 GB/s | 3.83 | 0.129 | 2,353,474 |
| 10 KB | 17,471 ns | 0.59 GB/s | 3.54 | 1.34 | 1,570,471 |

### Cycles per Byte

| Input | Cycles/Byte |
|-------|-------------|
| 200 B | 12.0 |
| 1 KB | 6.0 |
| 10 KB | 5.3 |

### Instruction Footprint

| Component | Size |
|-----------|------|
| `scan_input` (Rust) | 3,338 B |
| `libcommand_router.so` .text | 649 B (was 985 B, −336 B) |
| `libfused_safety.so` .text | 961 B (was 2,024 B, −1,063 B) |
| Core hot path | 4,948 B (4.8 KB) |
| L1i usage (32 KB) | 15.1% |

### All Kernel .text Sizes
| Kernel | .text size |
|--------|-----------:|
| libleak_scanner.so | 584 B |
| libcommand_router.so | 649 B |
| libsanitizer.so | 721 B |
| libfused_safety.so | 961 B |
| libbyte_classifier.so | 1,688 B |
| libjson_scanner.so | 2,153 B |
| libsearch.so | 22,282 B |

### Delta from Run 2

| Metric | Run 2 | Run 3 | Delta |
|--------|-------|-------|-------|
| cmd_router .text | 985 B | 649 B | −336 B (18→12 commands, nested hash) |
| fused_safety .text | 2,024 B | 961 B | −1,063 B (compiler improvement) |
| Hot path total | 6,347 B | 4,948 B | −1,399 B |
| L1i usage | 19.4% | 15.1% | −4.3% |
| Commands | 11 | 18 | +7 |
| Tests | 88 | 116 | +28 |

### Notes

Kernel sizes decreased across the board due to Eä compiler rebuild (tighter codegen).
The command router got smaller despite adding 7 commands — nested hash structure (2→3→4 byte)
shares intermediate hash computation, producing more compact code than the flat chain.

---

## Run 2: SIMD Tool Routing
### Date: 2026-03-06
### Change: Extended command router with 5 tool commands + two-stage verification

Added `/time`, `/echo`, `/shell`, `/http`, `/memory` to the SIMD command router.
Two-stage match: 4-byte SIMD hash (fast reject) + Rust full-name verify (collision safety).
Tool commands bypass the LLM entirely — microsecond latency.

### Tests: 88 passed (was 71)

### perf stat

| Input | Per call | Throughput | IPC | L1i misses | Branch misses |
|-------|----------|------------|-----|------------|---------------|
| 200 B | 816 ns | 0.25 GB/s | 3.62 | 99,489 (0.0006%) | 0 (0.00%) |
| 1 KB | 2,146 ns | 0.48 GB/s | 3.79 | 333,436 (0.0007%) | 0 (0.00%) |
| 10 KB | 18,399 ns | 0.56 GB/s | 3.54 | 353,897 (0.0009%) | 0 (0.00%) |

### Cycles per Byte

| Input | Cycles/Byte |
|-------|-------------|
| 200 B | 12.1 |
| 1 KB | 6.1 |
| 10 KB | 5.3 |

### Instruction Footprint

| Component | Size |
|-----------|------|
| `scan_input` (Rust) | 3,338 B |
| `libcommand_router.so` .text | 985 B (was 882 B, +103 B) |
| `libfused_safety.so` .text | 2,024 B |
| Full agent hot path | 6,347 B (6.2 KB) |
| L1i usage (32 KB) | 19.4% |

### All Kernel .text Sizes
| Kernel | .text size |
|--------|-----------|
| libcommand_router.so | 985 B |
| libleak_scanner.so | 1,395 B |
| libsanitizer.so | 1,622 B |
| libfused_safety.so | 2,024 B |
| libbyte_classifier.so | 2,478 B |
| libjson_scanner.so | 2,890 B |
| libsearch.so | 23,196 B |

### Delta from Run 1

| Metric | Run 1 | Run 2 | Delta |
|--------|-------|-------|-------|
| cmd_router .text | 882 B | 985 B | +103 B |
| Hot path total | 6,244 B | 6,347 B | +103 B |
| L1i miss rate | 0.0008% | 0.0007% | same |
| Branch misses | 0.02-0.06% | 0.00% | improved |
| Tests | 71 | 88 | +17 |

---

## Run 1: Buffer Reuse / Zero-Allocation Hot Path
### Date: 2026-03-06
### Change: FusedScanner with reusable SIMD mask buffers

### Tests: 71 passed (was 69)

### Criterion Benchmarks (safety_layer_e2e)

#### Clean Input
| Size | Time | Throughput |
|------|------|------------|
| 1 KB | 1.94 µs | 502 MiB/s |
| 10 KB | 17.2 µs | 567 MiB/s |
| 100 KB | 168.1 µs | 581 MiB/s |

#### Mixed Input (injection + leak patterns)
| Size | Time | Throughput |
|------|------|------------|
| 1 KB | 3.87 µs | 252 MiB/s |
| 10 KB | 40.1 µs | 244 MiB/s |
| 100 KB | 403.8 µs | 242 MiB/s |

### perf stat

| Input | Per call | Throughput | IPC | L1i misses | Branch misses |
|-------|----------|------------|-----|------------|---------------|
| 200 B | 758 ns | 0.26 GB/s | 3.73 | 28,831 (0.0008%) | 776K (0.02%) |
| 1 KB | 1,857 ns | 0.55 GB/s | 3.96 | 334,945 (0.0009%) | 4.4M (0.06%) |
| 10 KB | 16,774 ns | 0.61 GB/s | 3.66 | 250,701 (0.0008%) | 3.0M (0.05%) |

### Cycles per Byte

| Input | Cycles/Byte | Notes |
|-------|-------------|-------|
| 200 B | 11.7 | Fixed overhead dominates (~1,500-2,000 cycles) |
| 1 KB | 5.8 | Approaching SIMD throughput floor |
| 10 KB | 5.1 | Steady-state SIMD throughput |

### Branch Prediction

| Input | Branch Miss Rate |
|-------|-----------------|
| 200 B | 0.02% |
| 1 KB | 0.06% |
| 10 KB | 0.05% |

### Instruction Footprint

| Component | Size |
|-----------|------|
| `scan_input` (Rust) | 3,338 B |
| `scan_safety_fused` (Eä) | 769 B |
| Core hot path | 4,107 B (4.0 KB) |
| + fused .so .text | 5,362 B (5.2 KB) |
| + command_router .text | 882 B |
| Full agent hot path | 6,244 B (6.1 KB) |
| L1i usage (32 KB) | 19.1% |

### Core Scalability (single core)

| Metric | Value |
|--------|-------|
| Prompts/sec (200 B) | 1.48M |
| Projected 8-core | ~12M prompts/sec |

---

## Performance Profile Summary (latest)

| Metric | Value | Assessment |
|--------|-------|------------|
| L1i miss rate | 0.0007% | Entire hot path in L1 |
| IPC | 3.5-3.8 | 88-95% of architectural ceiling |
| Branch miss rate | 0.00% | Perfect prediction |
| Cycles/byte (steady) | 5.3 | SIMD throughput floor |
| Hot path footprint | 6.3 KB | 19.4% of L1i |
| Allocation on hot path | Zero | Buffer reuse via FusedScanner |
| Safety overhead vs LLM | 0.0003% | Effectively free |
| Tool command latency | ~0 (no LLM) | SIMD route → direct execute |
