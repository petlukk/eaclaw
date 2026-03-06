//! Integration tests for the SIMD recall system.

use eaclaw_core::recall::VectorStore;

#[test]
fn test_recall_conversation_flow() {
    let mut store = VectorStore::new();

    // Simulate a conversation
    store.insert("tell me about cargo build");
    store.insert("The cargo build system compiles Rust projects using Cargo.toml");
    store.insert("how does the safety scanner work");
    store.insert("The safety scanner uses SIMD kernels to detect injection patterns");
    store.insert("write me a function to sort numbers");
    store.insert("Here is a Rust sort function: fn sort(v: &mut Vec<i32>) { v.sort(); }");

    // "rust" should match entries containing rust-related text
    let results = store.recall("rust", 3);
    assert!(!results.is_empty(), "should find rust-related entries");
    // At least one result should contain "Rust" or "rust"
    let has_rust = results.iter().any(|r| r.text.to_lowercase().contains("rust"));
    assert!(has_rust, "expected a rust-related result, got: {:?}", results.iter().map(|r| &r.text).collect::<Vec<_>>());

    // "safety" should match the safety-related entries
    let results = store.recall("safety", 2);
    assert!(!results.is_empty());
    let has_safety = results.iter().any(|r| r.text.contains("safety"));
    assert!(has_safety, "expected safety result");

    // "SIMD" should match the scanner description
    let results = store.recall("SIMD", 2);
    assert!(!results.is_empty());
    let has_simd = results.iter().any(|r| r.text.contains("SIMD"));
    assert!(has_simd, "expected SIMD result");
}

#[test]
fn test_recall_numbers() {
    let mut store = VectorStore::new();
    store.insert("calculate 123 * 456");
    store.insert("the time is 2026-03-06");
    store.insert("system has 16384 MB memory");

    let results = store.recall("123", 1);
    assert!(!results.is_empty());
    assert!(results[0].text.contains("123"), "expected 123 match, got: {}", results[0].text);
}

#[test]
fn test_recall_self_similarity() {
    let mut store = VectorStore::new();
    let text = "the quick brown fox jumps over the lazy dog";
    store.insert(text);
    store.insert("completely different content with numbers 12345");

    let results = store.recall(text, 1);
    assert_eq!(results.len(), 1);
    assert!(results[0].score > 0.95, "self-similarity should be ~1.0, got {}", results[0].score);
    assert_eq!(results[0].text, text);
}

#[test]
fn test_recall_formatted_output() {
    let mut store = VectorStore::new();
    store.insert("hello world test");
    store.insert("goodbye cruel world");

    let output = store.recall_formatted("world", 5);
    assert!(output.contains("Recall"), "expected Recall header, got: {output}");
    assert!(output.contains("world"), "expected world in results");
}

#[test]
fn test_recall_formatted_empty_query() {
    let mut store = VectorStore::new();
    store.insert("test");
    let output = store.recall_formatted("", 5);
    assert!(output.contains("usage"), "expected usage hint, got: {output}");
    assert!(output.contains("1 entries"), "expected entry count");
}

#[test]
fn test_recall_large_store() {
    let mut store = VectorStore::with_capacity(500);
    // Insert 500 entries with different content
    for i in 0..500 {
        store.insert(&format!("entry {i}: {}", if i % 3 == 0 { "rust programming" } else if i % 3 == 1 { "python scripting" } else { "javascript frontend" }));
    }
    assert_eq!(store.len(), 500);

    let results = store.recall("rust", 5);
    assert_eq!(results.len(), 5);
    // All top results should be rust entries
    for r in &results {
        assert!(r.text.contains("rust"), "expected rust entry, got: {}", r.text);
    }
}

#[test]
fn test_recall_unicode() {
    let mut store = VectorStore::new();
    store.insert("こんにちは世界");
    store.insert("hello world");
    store.insert("مرحبا بالعالم");

    // Should find the Japanese entry by matching its byte pattern
    let results = store.recall("こんにちは", 1);
    assert!(!results.is_empty());
    assert!(results[0].text.contains("こんにちは"));
}

#[test]
fn test_recall_empty_and_whitespace() {
    let mut store = VectorStore::new();
    store.insert("   spaces   ");
    store.insert("normal text");

    let results = store.recall("   ", 1);
    assert!(!results.is_empty());
    // The spaces entry should match because whitespace bytes (0x20) dominate
    assert!(results[0].text.contains("spaces"), "expected spaces entry, got: {}", results[0].text);
}

// --- Tool + Recall interaction ---

#[tokio::test]
async fn test_tools_still_work_after_recall_changes() {
    use eaclaw_core::tools::ToolRegistry;
    let reg = ToolRegistry::with_defaults_open();

    // Calc
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "2+2"})).await.unwrap();
    assert_eq!(result, "4");

    // Time
    let tool = reg.get("time").unwrap();
    let result = tool.execute(serde_json::json!({})).await.unwrap();
    assert!(result.contains("T"));

    // Shell
    let tool = reg.get("shell").unwrap();
    let result = tool.execute(serde_json::json!({"command": "echo ok"})).await.unwrap();
    assert_eq!(result.trim(), "ok");

    // Memory
    let tool = reg.get("memory").unwrap();
    tool.execute(serde_json::json!({"action": "write", "key": "rtest", "value": "rval"})).await.unwrap();
    let result = tool.execute(serde_json::json!({"action": "read", "key": "rtest"})).await.unwrap();
    assert_eq!(result, "rval");

    // Ls
    let tool = reg.get("ls").unwrap();
    let result = tool.execute(serde_json::json!({"path": "/tmp"})).await.unwrap();
    assert!(result.starts_with("/tmp:"));

    // CPU
    let tool = reg.get("cpu").unwrap();
    let result = tool.execute(serde_json::json!({})).await.unwrap();
    assert!(result.contains("CPU") || result.contains("Memory"));

    // Tokens
    let tool = reg.get("tokens").unwrap();
    let result = tool.execute(serde_json::json!({"text": "hello world"})).await.unwrap();
    assert!(result.contains("Words:"));

    // JSON
    let tool = reg.get("json").unwrap();
    let result = tool.execute(serde_json::json!({"action": "keys", "input": r#"{"a":1}"#})).await.unwrap();
    assert!(result.contains("a"));

    // Bench
    let tool = reg.get("bench").unwrap();
    let result = tool.execute(serde_json::json!({"target": "router"})).await.unwrap();
    assert!(result.contains("Per call:"));
}

// --- Router with /recall ---

#[test]
fn test_router_recall_command() {
    use eaclaw_core::kernels::command_router::*;

    let (id, arg) = match_command_verified(b"/recall test query");
    assert_eq!(id, CMD_RECALL);
    assert_eq!(arg, b"test query");

    let (id, _) = match_command_verified(b"/recall");
    assert_eq!(id, CMD_RECALL);

    // Near-miss should not match
    let (id, _) = match_command_verified(b"/recalling");
    assert_eq!(id, CMD_NONE);
}
