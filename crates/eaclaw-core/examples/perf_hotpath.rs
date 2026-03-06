//! Tight-loop benchmark for perf stat cache analysis.
//!
//! Usage:
//!   cargo build --release --example perf_hotpath
//!   perf stat -e instructions,cycles,L1-icache-load-misses,L1-dcache-load-misses,branch-misses \
//!     target/release/examples/perf_hotpath [iterations] [input_size]

use eaclaw_core::safety::SafetyLayer;

fn generate_input(size: usize) -> String {
    let base = b"The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = &base[..remaining.min(base.len())];
        data.extend_from_slice(chunk);
    }
    String::from_utf8(data).unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let input_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);

    let input = generate_input(input_size);
    let mut safety = SafetyLayer::with_capacity(input_size);

    // Warmup
    for _ in 0..1000 {
        let _ = safety.scan_input(&input);
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let result = safety.scan_input(std::hint::black_box(&input));
        std::hint::black_box(&result);
    }
    let elapsed = start.elapsed();

    let total_bytes = iterations as f64 * input_size as f64;
    let throughput_gbs = total_bytes / elapsed.as_secs_f64() / 1e9;
    let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;

    eprintln!("--- Hot Path Performance ---");
    eprintln!("Input size:    {} bytes", input_size);
    eprintln!("Iterations:    {}", iterations);
    eprintln!("Total time:    {:.3} s", elapsed.as_secs_f64());
    eprintln!("Per call:      {:.1} ns", ns_per_call);
    eprintln!("Throughput:    {:.2} GB/s", throughput_gbs);
}
