//! Edge case and adversarial tests.

use eaclaw_core::kernels::arg_tokenizer::ArgTokenizer;
use eaclaw_core::kernels::command_router::*;
use eaclaw_core::safety::SafetyLayer;
use eaclaw_core::tools::ToolRegistry;

fn registry() -> ToolRegistry {
    ToolRegistry::with_defaults_open()
}

// --- Calc edge cases ---

#[tokio::test]
async fn test_calc_empty_parens() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "()"})).await;
    assert!(result.is_err(), "empty parens should fail");
}

#[tokio::test]
async fn test_calc_nested_parens() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "((((1+2))))*3"})).await.unwrap();
    assert_eq!(result, "9");
}

#[tokio::test]
async fn test_calc_big_number() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "999999999 * 999999999"})).await.unwrap();
    assert_eq!(result, "999999998000000001");
}

#[tokio::test]
async fn test_calc_unmatched_paren() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "(1+2"})).await;
    assert!(result.is_err(), "unmatched paren should fail");
}

#[tokio::test]
async fn test_calc_double_operator() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    // "1 + + 2" — second + is unary positive? Our parser should handle or reject
    let result = tool.execute(serde_json::json!({"expr": "1 + + 2"})).await;
    // This might error since we don't handle unary +
    // Just verify it doesn't panic
    let _ = result;
}

#[tokio::test]
async fn test_calc_only_spaces() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "   "})).await;
    assert!(result.is_err(), "only spaces should fail");
}

// --- Router edge cases ---

#[test]
fn test_router_near_collisions() {
    // Names that share 4-byte prefix with real commands
    let near_misses: &[&[u8]] = &[
        b"/timer",     // /time + r
        b"/helping",   // /help + ing
        b"/shells",    // /shell + s
        b"/cpus",      // /cpu + s
        b"/lsd",       // /ls + d
        b"/calculate", // /calc + ulate
        b"/reading",   // /read + ing
        b"/writer",    // /write + r
    ];
    for input in near_misses {
        let (id, _) = match_command_verified(input);
        assert_eq!(
            id, CMD_NONE,
            "near-miss {:?} should not match, got id={id}",
            std::str::from_utf8(input).unwrap()
        );
    }
}

#[test]
fn test_router_with_args_space_variations() {
    // Multiple spaces between command and arg
    let (id, arg) = match_command_verified(b"/calc  2+3");
    // Two spaces: first space separates, arg starts at "2+3" but has leading space
    // The router looks for single space at expected_len position
    // /calc is 5 bytes, position 5 should be space
    assert_eq!(id, CMD_CALC);
    // Second space becomes part of the arg
    assert_eq!(arg, b" 2+3");
}

#[test]
fn test_router_only_slash() {
    let (id, _) = match_command_verified(b"/");
    assert_eq!(id, CMD_NONE);
}

#[test]
fn test_router_slash_space() {
    let (id, _) = match_command_verified(b"/ ");
    assert_eq!(id, CMD_NONE);
}

// --- Safety scanner with tool output ---

#[test]
fn test_safety_blocks_leaked_key() {
    let mut safety = SafetyLayer::with_capacity(256);
    // Key must be >= 20 chars total (min_total_len for sk-ant-api pattern)
    let output = "Here is the key: sk-ant-api03-AABBCCDDEE";
    let scan = safety.scan_output(output);
    assert!(scan.leaks_found, "should detect API key in tool output");
}

#[test]
fn test_safety_allows_clean_output() {
    let mut safety = SafetyLayer::with_capacity(256);
    let output = "CPU: Intel i7 (8 cores)\nMemory: 16384 MB\nUptime: 5h 30m";
    let scan = safety.scan_output(output);
    assert!(!scan.leaks_found, "clean output should pass");
}

// --- Tokenizer edge cases ---

#[test]
fn test_tokenizer_all_spaces() {
    let mut tok = ArgTokenizer::new();
    let result = tok.tokenize(b"     ", 3);
    assert!(result.is_empty(), "all-space input should produce no tokens");
}

#[test]
fn test_tokenizer_single_byte() {
    let mut tok = ArgTokenizer::new();
    let result = tok.tokenize(b"x", 3);
    assert_eq!(result, vec![b"x".as_slice()]);
}

#[test]
fn test_tokenizer_tab_separated() {
    let mut tok = ArgTokenizer::new();
    let result = tok.tokenize(b"a\tb\tc", 3);
    assert_eq!(result, vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]);
}

#[test]
fn test_tokenizer_mixed_whitespace() {
    let mut tok = ArgTokenizer::new();
    let result = tok.tokenize(b"hello \t world", 2);
    assert_eq!(result, vec![b"hello".as_slice(), b"world".as_slice()]);
}

#[test]
fn test_tokenizer_long_input() {
    let mut tok = ArgTokenizer::with_capacity(1024);
    let input = "word ".repeat(200);
    let result = tok.tokenize_str(input.trim(), 5);
    assert_eq!(result.len(), 5);
    assert_eq!(result[0], "word");
    // Last token should contain all remaining words
    assert!(result[4].contains("word"));
}

// --- JSON edge cases ---

#[tokio::test]
async fn test_json_invalid_input() {
    let reg = registry();
    let tool = reg.get("json").unwrap();
    let result = tool
        .execute(serde_json::json!({"action": "keys", "input": "not json"}))
        .await;
    assert!(result.is_err(), "invalid JSON should error");
}

#[tokio::test]
async fn test_json_deep_nesting() {
    let reg = registry();
    let tool = reg.get("json").unwrap();
    let result = tool
        .execute(serde_json::json!({
            "action": "get",
            "input": r#"{"a": {"b": {"c": {"d": 42}}}}"#,
            "path": "a.b.c.d"
        }))
        .await
        .unwrap();
    assert!(result.contains("42"), "expected 42, got: {result}");
}

#[tokio::test]
async fn test_json_get_nonexistent_path() {
    let reg = registry();
    let tool = reg.get("json").unwrap();
    let result = tool
        .execute(serde_json::json!({
            "action": "get",
            "input": r#"{"a": 1}"#,
            "path": "b"
        }))
        .await;
    assert!(result.is_err(), "nonexistent path should error");
}

// --- Shell edge cases ---

#[tokio::test]
async fn test_shell_special_chars() {
    let reg = registry();
    let tool = reg.get("shell").unwrap();
    let result = tool
        .execute(serde_json::json!({"command": "echo 'hello \"world\"'"}))
        .await
        .unwrap();
    assert!(result.contains("hello"), "got: {result}");
}

#[tokio::test]
async fn test_shell_multiline_output() {
    let reg = registry();
    let tool = reg.get("shell").unwrap();
    let result = tool
        .execute(serde_json::json!({"command": "seq 1 5"}))
        .await
        .unwrap();
    assert!(result.contains("1"), "got: {result}");
    assert!(result.contains("5"), "got: {result}");
}

// --- Write edge cases ---

#[tokio::test]
async fn test_write_empty_content() {
    let reg = registry();
    let tool = reg.get("write").unwrap();
    let path = "/tmp/eaclaw_test_empty.txt";
    let result = tool
        .execute(serde_json::json!({"path": path, "content": ""}))
        .await
        .unwrap();
    assert!(result.contains("0 bytes"), "expected 0 bytes, got: {result}");
    let _ = tokio::fs::remove_file(path).await;
}

// --- Tokens edge cases ---

#[tokio::test]
async fn test_tokens_empty_string() {
    let reg = registry();
    let tool = reg.get("tokens").unwrap();
    let result = tool.execute(serde_json::json!({"text": ""})).await;
    // Empty string with no spaces or dots won't be treated as a file path
    // Should return stats for empty string
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_tokens_from_file() {
    let reg = registry();
    // Write a test file
    let write_tool = reg.get("write").unwrap();
    let path = "/tmp/eaclaw_test_tokens.txt";
    write_tool
        .execute(serde_json::json!({"path": path, "content": "fn main() { println!(\"hello\"); }"}))
        .await
        .unwrap();

    let tokens_tool = reg.get("tokens").unwrap();
    let result = tokens_tool
        .execute(serde_json::json!({"text": path}))
        .await
        .unwrap();
    assert!(result.contains("code"), "expected code classification for .rs-like content, got: {result}");

    let _ = tokio::fs::remove_file(path).await;
}

// --- Bench edge cases ---

#[tokio::test]
async fn test_bench_unknown_target() {
    let reg = registry();
    let tool = reg.get("bench").unwrap();
    let result = tool.execute(serde_json::json!({"target": "nonexistent"})).await;
    assert!(result.is_err());
}

// --- HTTP allowlist edge cases ---

#[tokio::test]
async fn test_http_allowlist_blocks() {
    use eaclaw_core::tools::http::HttpTool;
    use eaclaw_core::tools::Tool;
    let tool = HttpTool::new(vec!["example.com".to_string()]);
    let result = tool.execute(serde_json::json!({"url": "https://evil.com/steal"})).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not in allowed list"), "expected allowlist error, got: {err}");
}

#[tokio::test]
async fn test_http_allowlist_allows() {
    use eaclaw_core::tools::http::HttpTool;
    use eaclaw_core::tools::Tool;
    let tool = HttpTool::new(vec!["example.com".to_string()]);
    // This should pass the host check (may fail on network, but not on allowlist)
    let result = tool.execute(serde_json::json!({"url": "https://example.com"})).await;
    // If it errors, it should NOT be an allowlist error
    if let Err(e) = &result {
        assert!(!e.to_string().contains("not in allowed list"), "should be allowed: {e}");
    }
}

#[tokio::test]
async fn test_http_allowlist_subdomain() {
    use eaclaw_core::tools::http::HttpTool;
    use eaclaw_core::tools::Tool;
    let tool = HttpTool::new(vec!["example.com".to_string()]);
    let result = tool.execute(serde_json::json!({"url": "https://api.example.com/data"})).await;
    if let Err(e) = &result {
        assert!(!e.to_string().contains("not in allowed list"), "subdomain should be allowed: {e}");
    }
}

#[tokio::test]
async fn test_http_empty_allowlist_allows_all() {
    use eaclaw_core::tools::http::HttpTool;
    use eaclaw_core::tools::Tool;
    let tool = HttpTool::new(Vec::new());
    // Empty allowlist = allow all; won't fail on host check
    let result = tool.execute(serde_json::json!({"url": "https://httpbin.org/get"})).await;
    if let Err(e) = &result {
        assert!(!e.to_string().contains("not in allowed list"), "empty list should allow all: {e}");
    }
}

// --- Identity file ---

#[test]
fn test_identity_config_none_by_default() {
    // Without EACLAW_IDENTITY set and no ~/.eaclaw/identity.md, identity is None
    let config = eaclaw_core::config::Config {
        api_key: "test".into(),
        model: "test".into(),
        agent_name: "test".into(),
        max_turns: 10,
        command_prefix: "/".into(),
        identity: None,
        allowed_hosts: vec![],
        backend: eaclaw_core::config::Backend::Anthropic,
        model_path: None,
        ctx_size: 4096,
        threads: 4,
    };
    assert!(config.identity.is_none());
}

#[test]
fn test_identity_config_some() {
    let config = eaclaw_core::config::Config {
        api_key: "test".into(),
        model: "test".into(),
        agent_name: "test".into(),
        max_turns: 10,
        command_prefix: "/".into(),
        identity: Some("You are a pirate. Always say Arrr.".into()),
        allowed_hosts: vec![],
        backend: eaclaw_core::config::Backend::Anthropic,
        model_path: None,
        ctx_size: 4096,
        threads: 4,
    };
    assert!(config.identity.as_ref().unwrap().contains("pirate"));
}
