//! Multi-core scalability benchmark.
//!
//! Usage:
//!   cargo build --release --example perf_multicore
//!   target/release/examples/perf_multicore [threads] [iterations_per_thread] [input_size]

use eaclaw_core::safety::SafetyLayer;
use std::sync::Arc;
use std::thread;

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
    let n_threads: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let iterations: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500_000);
    let input_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);

    let input = Arc::new(generate_input(input_size));

    // Warmup on main thread
    let mut safety = SafetyLayer::with_capacity(input_size);
    for _ in 0..1000 {
        let _ = safety.scan_input(&input);
    }
    drop(safety);

    let barrier = Arc::new(std::sync::Barrier::new(n_threads));
    let start = std::time::Instant::now();

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let input = Arc::clone(&input);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut safety = SafetyLayer::with_capacity(input_size);
                // Warmup per thread
                for _ in 0..100 {
                    let _ = safety.scan_input(&input);
                }
                barrier.wait();
                for _ in 0..iterations {
                    let result = safety.scan_input(std::hint::black_box(&*input));
                    std::hint::black_box(&result);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    let elapsed = start.elapsed();

    let total_calls = n_threads as u64 * iterations;
    let total_bytes = total_calls as f64 * input_size as f64;
    let calls_per_sec = total_calls as f64 / elapsed.as_secs_f64();
    let throughput_gbs = total_bytes / elapsed.as_secs_f64() / 1e9;

    eprintln!("--- Multi-Core Scalability ---");
    eprintln!("Threads:       {}", n_threads);
    eprintln!("Input size:    {} bytes", input_size);
    eprintln!("Iter/thread:   {}", iterations);
    eprintln!("Total calls:   {}", total_calls);
    eprintln!("Total time:    {:.3} s", elapsed.as_secs_f64());
    eprintln!("Calls/sec:     {:.0}", calls_per_sec);
    eprintln!("Prompts/sec:   {:.2}M", calls_per_sec / 1e6);
    eprintln!("Throughput:    {:.2} GB/s", throughput_gbs);
}
