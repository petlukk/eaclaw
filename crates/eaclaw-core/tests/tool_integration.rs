//! Integration tests exercising every tool command end-to-end.

use eaclaw_core::tools::ToolRegistry;

fn registry() -> ToolRegistry {
    ToolRegistry::with_defaults_open()
}

// --- /time ---

#[tokio::test]
async fn test_time_tool() {
    let reg = registry();
    let tool = reg.get("time").unwrap();
    let result = tool.execute(serde_json::json!({})).await.unwrap();
    // Should be an RFC3339 timestamp
    assert!(result.contains("T"), "expected timestamp, got: {result}");
    assert!(result.contains("20"), "expected year in timestamp");
}

// --- /calc ---

#[tokio::test]
async fn test_calc_basic() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "2 + 3"})).await.unwrap();
    assert_eq!(result, "5");
}

#[tokio::test]
async fn test_calc_complex() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "(10 + 5) * 3 - 2"})).await.unwrap();
    assert_eq!(result, "43");
}

#[tokio::test]
async fn test_calc_float() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "7 / 2"})).await.unwrap();
    assert_eq!(result, "3.5");
}

#[tokio::test]
async fn test_calc_negative() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "-3 * -4"})).await.unwrap();
    assert_eq!(result, "12");
}

#[tokio::test]
async fn test_calc_div_zero() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "1/0"})).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_calc_modulo() {
    let reg = registry();
    let tool = reg.get("calc").unwrap();
    let result = tool.execute(serde_json::json!({"expr": "17 % 5"})).await.unwrap();
    assert_eq!(result, "2");
}

// --- /shell ---

#[tokio::test]
async fn test_shell_echo() {
    let reg = registry();
    let tool = reg.get("shell").unwrap();
    let result = tool.execute(serde_json::json!({"command": "echo hello"})).await.unwrap();
    assert_eq!(result.trim(), "hello");
}

#[tokio::test]
async fn test_shell_exit_code() {
    let reg = registry();
    let tool = reg.get("shell").unwrap();
    let result = tool.execute(serde_json::json!({"command": "false"})).await.unwrap();
    assert!(result.contains("exit code"), "expected exit code, got: {result}");
}

#[tokio::test]
async fn test_shell_pipe() {
    let reg = registry();
    let tool = reg.get("shell").unwrap();
    let result = tool
        .execute(serde_json::json!({"command": "echo 'hello world' | wc -w"}))
        .await
        .unwrap();
    assert_eq!(result.trim(), "2");
}

#[tokio::test]
async fn test_shell_stderr() {
    let reg = registry();
    let tool = reg.get("shell").unwrap();
    let result = tool
        .execute(serde_json::json!({"command": "echo err >&2"}))
        .await
        .unwrap();
    assert!(result.contains("[stderr]"), "expected stderr marker, got: {result}");
}

#[tokio::test]
async fn test_shell_streaming() {
    let reg = registry();
    let tool = reg.get("shell").unwrap();
    assert!(tool.supports_streaming());
    let mut chunks = Vec::new();
    tool.execute_stream(
        serde_json::json!({"command": "echo line1; echo line2; echo line3"}),
        &mut |chunk: &str| chunks.push(chunk.to_string()),
    )
    .await
    .unwrap();
    let output: String = chunks.concat();
    assert!(output.contains("line1"), "missing line1 in: {output}");
    assert!(output.contains("line2"), "missing line2 in: {output}");
    assert!(output.contains("line3"), "missing line3 in: {output}");
}

// --- /read ---

#[tokio::test]
async fn test_read_file() {
    let reg = registry();
    let tool = reg.get("read").unwrap();
    let result = tool.execute(serde_json::json!({"path": "/etc/hostname"})).await.unwrap();
    assert!(!result.is_empty(), "hostname should not be empty");
}

#[tokio::test]
async fn test_read_nonexistent() {
    let reg = registry();
    let tool = reg.get("read").unwrap();
    let result = tool.execute(serde_json::json!({"path": "/nonexistent/file.txt"})).await;
    assert!(result.is_err());
}

// --- /write ---

#[tokio::test]
async fn test_write_and_read() {
    let reg = registry();
    let write_tool = reg.get("write").unwrap();
    let read_tool = reg.get("read").unwrap();

    let path = "/tmp/eaclaw_test_write.txt";
    let content = "hello from eaclaw integration test";

    write_tool
        .execute(serde_json::json!({"path": path, "content": content}))
        .await
        .unwrap();

    let result = read_tool.execute(serde_json::json!({"path": path})).await.unwrap();
    assert_eq!(result, content);

    // Cleanup
    let _ = tokio::fs::remove_file(path).await;
}

// --- /ls ---

#[tokio::test]
async fn test_ls_current_dir() {
    let reg = registry();
    let tool = reg.get("ls").unwrap();
    let result = tool.execute(serde_json::json!({"path": "."})).await.unwrap();
    assert!(result.contains("Cargo.toml") || result.contains("src"), "expected project files, got: {result}");
}

#[tokio::test]
async fn test_ls_tmp() {
    let reg = registry();
    let tool = reg.get("ls").unwrap();
    let result = tool.execute(serde_json::json!({"path": "/tmp"})).await.unwrap();
    assert!(result.starts_with("/tmp:"), "expected /tmp: header, got: {result}");
}

#[tokio::test]
async fn test_ls_nonexistent() {
    let reg = registry();
    let tool = reg.get("ls").unwrap();
    let result = tool.execute(serde_json::json!({"path": "/nonexistent"})).await;
    assert!(result.is_err());
}

// --- /cpu ---

#[tokio::test]
async fn test_cpu_info() {
    let reg = registry();
    let tool = reg.get("cpu").unwrap();
    let result = tool.execute(serde_json::json!({})).await.unwrap();
    // Should have at least CPU or Memory info on Linux
    assert!(
        result.contains("CPU") || result.contains("Memory") || result.contains("Uptime"),
        "expected system info, got: {result}"
    );
}

// --- /json ---

#[tokio::test]
async fn test_json_keys() {
    let reg = registry();
    let tool = reg.get("json").unwrap();
    let result = tool
        .execute(serde_json::json!({
            "action": "keys",
            "input": r#"{"name": "eaclaw", "version": 1}"#
        }))
        .await
        .unwrap();
    assert!(result.contains("name"), "expected 'name' key, got: {result}");
    assert!(result.contains("version"), "expected 'version' key, got: {result}");
}

#[tokio::test]
async fn test_json_get() {
    let reg = registry();
    let tool = reg.get("json").unwrap();
    let result = tool
        .execute(serde_json::json!({
            "action": "get",
            "input": r#"{"users": [{"name": "alice"}, {"name": "bob"}]}"#,
            "path": "users.1.name"
        }))
        .await
        .unwrap();
    assert!(result.contains("bob"), "expected 'bob', got: {result}");
}

#[tokio::test]
async fn test_json_pretty() {
    let reg = registry();
    let tool = reg.get("json").unwrap();
    let result = tool
        .execute(serde_json::json!({
            "action": "pretty",
            "input": r#"{"a":1,"b":2}"#
        }))
        .await
        .unwrap();
    assert!(result.contains('\n'), "expected formatted output with newlines");
    assert!(result.contains("\"a\""), "expected key 'a'");
}

#[tokio::test]
async fn test_json_array() {
    let reg = registry();
    let tool = reg.get("json").unwrap();
    let result = tool
        .execute(serde_json::json!({
            "action": "keys",
            "input": "[1,2,3]"
        }))
        .await
        .unwrap();
    assert!(result.contains("3 elements"), "expected array info, got: {result}");
}

#[tokio::test]
async fn test_json_from_file() {
    let reg = registry();
    // Write a JSON file first
    let write_tool = reg.get("write").unwrap();
    let path = "/tmp/eaclaw_test.json";
    write_tool
        .execute(serde_json::json!({"path": path, "content": r#"{"test": true}"#}))
        .await
        .unwrap();

    let json_tool = reg.get("json").unwrap();
    let result = json_tool
        .execute(serde_json::json!({"action": "keys", "input": path}))
        .await
        .unwrap();
    assert!(result.contains("test"), "expected 'test' key from file, got: {result}");

    let _ = tokio::fs::remove_file(path).await;
}

// --- /tokens ---

#[tokio::test]
async fn test_tokens_text() {
    let reg = registry();
    let tool = reg.get("tokens").unwrap();
    let result = tool
        .execute(serde_json::json!({"text": "Hello world, this is a test."}))
        .await
        .unwrap();
    assert!(result.contains("Words:"), "expected word count, got: {result}");
    assert!(result.contains("Estimated tokens:"), "expected token estimate, got: {result}");
    assert!(result.contains("text"), "expected 'text' classification, got: {result}");
}

#[tokio::test]
async fn test_tokens_code() {
    let reg = registry();
    let tool = reg.get("tokens").unwrap();
    let result = tool
        .execute(serde_json::json!({"text": "fn main() { let x = 5; println!(\"{}\", x); }"}))
        .await
        .unwrap();
    assert!(result.contains("code"), "expected 'code' classification, got: {result}");
}

// --- /bench ---

#[tokio::test]
async fn test_bench_safety() {
    let reg = registry();
    let tool = reg.get("bench").unwrap();
    let result = tool.execute(serde_json::json!({"target": "safety"})).await.unwrap();
    assert!(result.contains("Per call:"), "expected timing, got: {result}");
    assert!(result.contains("GB/s"), "expected throughput, got: {result}");
}

#[tokio::test]
async fn test_bench_router() {
    let reg = registry();
    let tool = reg.get("bench").unwrap();
    let result = tool.execute(serde_json::json!({"target": "router"})).await.unwrap();
    assert!(result.contains("Per call:"), "expected timing, got: {result}");
    assert!(result.contains("Total calls:"), "expected call count, got: {result}");
}

// --- /memory ---

#[tokio::test]
async fn test_memory_write_read_list() {
    let reg = registry();
    let tool = reg.get("memory").unwrap();

    // Write
    let result = tool
        .execute(serde_json::json!({"action": "write", "key": "test_key", "value": "test_value"}))
        .await
        .unwrap();
    assert!(result.contains("Stored"), "expected stored, got: {result}");

    // Read
    let result = tool
        .execute(serde_json::json!({"action": "read", "key": "test_key"}))
        .await
        .unwrap();
    assert_eq!(result, "test_value");

    // List
    let result = tool
        .execute(serde_json::json!({"action": "list"}))
        .await
        .unwrap();
    assert!(result.contains("test_key"), "expected key in list, got: {result}");

    // Read nonexistent
    let result = tool
        .execute(serde_json::json!({"action": "read", "key": "nope"}))
        .await
        .unwrap();
    assert!(result.contains("not found"), "expected not found, got: {result}");
}

// --- Command Router ---

#[test]
fn test_all_commands_route() {
    use eaclaw_core::kernels::command_router::*;
    let commands: &[(&[u8], i32)] = &[
        (b"/help", CMD_HELP),
        (b"/quit", CMD_QUIT),
        (b"/tools", CMD_TOOLS),
        (b"/clear", CMD_CLEAR),
        (b"/model", CMD_MODEL),
        (b"/profile", CMD_PROFILE),
        (b"/time", CMD_TIME),
        (b"/calc 2+2", CMD_CALC),
        (b"/http url", CMD_HTTP),
        (b"/shell ls", CMD_SHELL),
        (b"/memory list", CMD_MEMORY),
        (b"/read file", CMD_READ),
        (b"/write f c", CMD_WRITE),
        (b"/ls", CMD_LS),
        (b"/json keys {}", CMD_JSON),
        (b"/cpu", CMD_CPU),
        (b"/tokens hi", CMD_TOKENS),
        (b"/bench safety", CMD_BENCH),
        (b"/tasks", CMD_TASKS),
        (b"/recall test", CMD_RECALL),
    ];
    for (input, expected) in commands {
        let (id, _) = match_command_verified(input);
        assert_eq!(
            id, *expected,
            "routing failed for {:?}: got {id}, expected {expected}",
            std::str::from_utf8(input).unwrap()
        );
    }
}

// --- SIMD Arg Tokenizer ---

#[test]
fn test_tokenizer_matches_splitn_comprehensive() {
    use eaclaw_core::kernels::arg_tokenizer::ArgTokenizer;
    let mut tok = ArgTokenizer::new();
    let cases: &[(&str, usize)] = &[
        ("list", 3),
        ("read mykey", 3),
        ("write key some value with spaces", 3),
        ("keys {\"a\": 1}", 3),
        ("get input users.0.name", 3),
        ("single", 1),
        ("a b c d e", 2),
        ("", 3),
    ];
    for &(input, n) in cases {
        let simd: Vec<&str> = tok.tokenize_str(input, n);
        let scalar: Vec<&str> = if input.is_empty() {
            vec![]
        } else {
            input.splitn(n, ' ').collect()
        };
        assert_eq!(simd, scalar, "mismatch: input={input:?}, n={n}");
    }
}
