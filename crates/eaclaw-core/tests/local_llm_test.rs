//! Integration test for LocalLlmProvider.
//! Only runs when local-llm feature is enabled AND a model file exists.
//!
//! Run: cargo test --features local-llm -- local_llm --nocapture

#[cfg(feature = "local-llm")]
mod local_llm_tests {
    use eaclaw_core::llm::{LocalLlmProvider, LlmProvider, Message, Role, ContentBlock, ToolDef};

    fn model_path() -> Option<String> {
        std::env::var("EACLAW_MODEL_PATH").ok().or_else(|| {
            let home = std::env::var("HOME").ok()?;
            let path = format!("{home}/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf");
            if std::path::Path::new(&path).exists() { Some(path) } else { None }
        })
    }

    #[tokio::test]
    async fn test_basic_generation() {
        let path = match model_path() {
            Some(p) => p,
            None => { eprintln!("Skipping: no model file"); return; }
        };

        let provider = LocalLlmProvider::new(&path, 2048, 4, 36, 2, 128).unwrap();
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "Say hello in one word.".into() }],
        }];

        let response = provider.chat(&messages, &[], "You are helpful.").await.unwrap();
        assert!(!response.content.is_empty());

        if let ContentBlock::Text { text } = &response.content[0] {
            assert!(!text.is_empty(), "Expected non-empty response");
            println!("Response: {text}");
        }
    }

    #[tokio::test]
    async fn test_tool_call_detection() {
        let path = match model_path() {
            Some(p) => p,
            None => { eprintln!("Skipping: no model file"); return; }
        };

        let provider = LocalLlmProvider::new(&path, 2048, 4, 36, 2, 128).unwrap();
        let tools = vec![ToolDef {
            name: "calc".into(),
            description: "Evaluate a math expression".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "expr": { "type": "string" } },
                "required": ["expr"]
            }),
        }];

        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "What is 42 * 17? Use the calc tool.".into(),
            }],
        }];

        let response = provider.chat(&messages, &tools, "You are helpful.").await.unwrap();
        println!("Response: {response:?}");

        let has_tool_use = response.content.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. }));
        if has_tool_use {
            assert_eq!(response.stop_reason, eaclaw_core::llm::StopReason::ToolUse);
            println!("Tool call detected successfully");
        } else {
            println!("Model did not produce tool call (may need prompt tuning)");
        }
    }

    #[tokio::test]
    async fn test_incremental_prefill() {
        let path = match model_path() {
            Some(p) => p,
            None => { eprintln!("Skipping: no model file"); return; }
        };

        let provider = LocalLlmProvider::new(&path, 2048, 4, 36, 2, 128).unwrap();

        let messages1 = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "Hi".into() }],
        }];
        let r1 = provider.chat(&messages1, &[], "You are helpful.").await.unwrap();

        let mut messages2 = messages1.clone();
        messages2.push(Message {
            role: Role::Assistant,
            content: r1.content.clone(),
        });
        messages2.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "How are you?".into() }],
        });

        let r2 = provider.chat(&messages2, &[], "You are helpful.").await.unwrap();
        assert!(!r2.content.is_empty());
        println!("Multi-turn response: {r2:?}");
    }
}
