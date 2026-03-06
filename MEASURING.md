# eaclaw Measurement Methodology

How and what we measure after code changes.

---

## Prerequisites

```bash
cd /root/dev/eaclaw
```

---

## 1. Tests (correctness gate)

Run all tests before and after changes. Nothing ships with failures.

```bash
cargo test
```

Expected: 230 tests pass.

---

## 2. Criterion Benchmarks (throughput + latency)

Measures the full safety_layer_e2e pipeline at 1KB, 10KB, 100KB with clean and mixed input.

```bash
cargo bench -p eaclaw-core -- safety_layer_e2e
```

**What to look for:**
- Time per call (ns/µs)
- Throughput (MiB/s)
- Regression vs baseline (criterion reports % change automatically)

---

## 3. perf stat: Cache + Branch + IPC

Verifies the hot path stays in L1 instruction cache and branches predict well.

```bash
# Build the perf example
cargo build --release --example perf_hotpath

# Typical prompt size (200 B)
perf stat -e instructions,cycles,L1-icache-load-misses,L1-dcache-load-misses,branches,branch-misses \
  target/release/examples/perf_hotpath 2000000 200

# Medium input (1 KB)
perf stat -e instructions,cycles,L1-icache-load-misses,L1-dcache-load-misses,branches,branch-misses \
  target/release/examples/perf_hotpath 2000000 1024

# Large input (10 KB)
perf stat -e instructions,cycles,L1-icache-load-misses,L1-dcache-load-misses,branches,branch-misses \
  target/release/examples/perf_hotpath 200000 10240
```

**What to look for:**
- L1-icache miss rate < 0.01% (hot path fits in L1)
- IPC > 3.5 (near architectural ceiling of ~4)
- Branch miss rate < 0.1%

**How to calculate:**
- L1i miss rate = L1-icache-load-misses / instructions * 100
- Branch miss rate = branch-misses / branches * 100
- Cycles/byte = cycles / (iterations * input_size)

---

## 4. Cycles per Byte (efficiency)

Derived from perf stat output.

```
cycles_per_byte = total_cycles / (iterations * input_size)
```

**Targets:**
- ~6 cycles/byte at 1KB+ (SIMD throughput floor, VM-dependent)
- 200B will show higher due to fixed per-call overhead (~1,500-2,000 cycles)

---

## 5. Instruction Footprint (code size)

Verifies code stays small enough for L1i residency.

```bash
# Rust function sizes
nm -S --size-sort target/release/examples/perf_hotpath | grep -E 'scan_input|verify_from_masks'

# Kernel .text sizes
for f in target/kernels/lib*.so; do
    name=$(basename $f)
    text=$(size $f | tail -1 | awk '{print $1}')
    echo "$name: $text bytes"
done
```

**Target:** Full hot path < 8 KB (< 25% of 32 KB L1i).

---

## 6. perf record + report (profiling)

Identifies where time is spent. Run if regressions appear.

```bash
perf record -g -o perf.data -- target/release/examples/perf_hotpath 2000000 1024
perf report --stdio --no-children -g none -i perf.data
```

**What to look for:**
- >95% of time in scan_input + scan_safety_fused
- No unexpected functions in the profile

---

## 7. Multi-core Scalability

Tests linear scaling (each thread has independent SafetyLayer).

```bash
cargo build --release --example perf_multicore

# Single core baseline
target/release/examples/perf_multicore 1 2000000 200

# Scale up (if cores available)
target/release/examples/perf_multicore 2 1000000 200
target/release/examples/perf_multicore 4 500000 200
target/release/examples/perf_multicore 8 250000 200
```

**Target:** Linear scaling (N cores ≈ N× prompts/sec).

---

## Measurement Checklist

After every code change to the hot path:

1. [ ] `cargo test` — all pass
2. [ ] `cargo bench` — no regressions
3. [ ] `perf stat` — L1i miss rate < 0.01%, IPC > 3.5, branch miss < 0.1%
4. [ ] Cycles/byte — < 6 at 1KB+
5. [ ] Code size — hot path < 8 KB
6. [ ] Update BENCHMARKS.md with new results
