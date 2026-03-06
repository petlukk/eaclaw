use eaclaw_core::recall::VectorStore;
use std::time::Instant;

fn main() {
    // Small store (typical conversation)
    let mut store = VectorStore::new();
    for i in 0..20 {
        store.insert(&format!("message {i}: the quick brown fox jumps over the lazy dog"));
    }
    let _ = store.recall("fox", 5); // warm up
    let iters = 100_000u32;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = store.recall("fox", 5);
    }
    let per_call = start.elapsed() / iters;
    println!("Recall 20 entries, top-5:  {:?}/call", per_call);

    // Medium store
    let mut store = VectorStore::with_capacity(500);
    for i in 0..500 {
        store.insert(&format!("entry {i}: rust programming language"));
    }
    let _ = store.recall("rust", 5);
    let iters = 10_000u32;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = store.recall("rust", 5);
    }
    let per_call = start.elapsed() / iters;
    println!("Recall 500 entries, top-5: {:?}/call", per_call);
}
