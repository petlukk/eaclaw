//! Recall latency benchmark — measures the exact /recall workload from README.
//! Run: cargo test -p eaclaw-core --test recall_bench -- --nocapture

use eaclaw_core::recall::VectorStore;
use std::time::Instant;

#[test]
fn bench_recall_20_entries() {
    eaclaw_core::kernels::init().unwrap();

    let mut store = VectorStore::with_capacity(1024);

    // Insert 20 conversation-like entries (realistic content)
    let entries = [
        "hello, how are you doing today?",
        "can you help me with my rust code?",
        "what is the capital of france?",
        "explain how async works in rust",
        "write a function to sort a list",
        "the weather today is sunny and warm",
        "i need to fix a bug in my application",
        "how do i use cargo build features?",
        "what are the best practices for error handling?",
        "tell me about the history of computing",
        "can you review my pull request?",
        "how to set up a docker container",
        "what is the difference between tcp and udp?",
        "help me understand ownership in rust",
        "write a test for my database module",
        "how to optimize memory usage in my app",
        "explain the concept of zero-cost abstractions",
        "what tools do you recommend for profiling?",
        "can you help debug this segfault?",
        "how does the linux scheduler work?",
    ];
    for entry in &entries {
        store.insert(entry);
    }
    assert_eq!(store.len(), 20);

    // Warm up
    for _ in 0..100 {
        let _ = store.recall("rust async programming", 5);
    }

    // Benchmark: 10000 iterations
    let n = 10_000;
    let t0 = Instant::now();
    for _ in 0..n {
        let results = store.recall("rust async programming", 5);
        std::hint::black_box(&results);
    }
    let elapsed = t0.elapsed();
    let per_call_ns = elapsed.as_nanos() / n as u128;
    let per_call_us = per_call_ns as f64 / 1000.0;

    println!("\n=== Recall Benchmark (20 entries, top-5, dim=256) ===");
    println!("Iterations:  {n}");
    println!("Total:       {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!("Per call:    {per_call_us:.2} µs ({per_call_ns} ns)");
    println!("=====================================================\n");
}

#[test]
fn bench_recall_100_entries() {
    eaclaw_core::kernels::init().unwrap();

    let mut store = VectorStore::with_capacity(1024);
    for i in 0..100 {
        store.insert(&format!("conversation entry number {i} with realistic length text content"));
    }

    // Warm up
    for _ in 0..100 {
        let _ = store.recall("conversation about code", 5);
    }

    let n = 10_000;
    let t0 = Instant::now();
    for _ in 0..n {
        let results = store.recall("conversation about code", 5);
        std::hint::black_box(&results);
    }
    let elapsed = t0.elapsed();
    let per_call_us = elapsed.as_nanos() as f64 / n as f64 / 1000.0;

    println!("\n=== Recall Benchmark (100 entries, top-5, dim=256) ===");
    println!("Per call:    {per_call_us:.2} µs");
    println!("======================================================\n");
}
