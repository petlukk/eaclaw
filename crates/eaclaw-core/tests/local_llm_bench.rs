//! Timing benchmark for local inference.
//! Run: EACLAW_MODEL_PATH=~/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf \
//!   cargo test --features local-llm -p eaclaw-core --test local_llm_bench -- --nocapture

#[cfg(feature = "local-llm")]
mod bench {
    use eaclaw_core::llm::{ContentBlock, LlmProvider, LocalLlmProvider, Message, Role, ToolDef};
    use std::time::Instant;

    fn model_path() -> Option<String> {
        std::env::var("EACLAW_MODEL_PATH").ok().or_else(|| {
            let home = std::env::var("HOME").ok()?;
            let path = format!("{home}/.eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf");
            if std::path::Path::new(&path).exists() {
                Some(path)
            } else {
                None
            }
        })
    }

    #[tokio::test]
    async fn bench_timing() {
        let path = match model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping: no model file");
                return;
            }
        };

        // 1. Model load time
        let t0 = Instant::now();
        let n_ctx: u32 = std::env::var("EACLAW_CTX_SIZE")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
        let provider = LocalLlmProvider::new(&path, n_ctx, 512, 4, false, 36, 2, 128).unwrap();
        let load_ms = t0.elapsed().as_millis();
        println!("\n=== Local Inference Benchmark ===");
        println!("Model load:        {load_ms} ms");

        // 2. First message (cold prefill — system prompt + user msg)
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("Say hello in exactly 5 words.")],
        }];

        let t1 = Instant::now();
        let mut token_count = 0usize;
        let r1 = provider
            .chat_stream(
                &messages,
                &[],
                "You are a helpful assistant.",
                &mut |chunk: &str| {
                    token_count += 1;
                    let _ = chunk; // just count
                },
            )
            .await
            .unwrap();
        let first_ms = t1.elapsed().as_millis();
        let first_text = match &r1.content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => String::new(),
        };
        println!("First response:    {first_ms} ms  ({token_count} tokens)  \"{first_text}\"");
        if first_ms > 0 {
            let tps = token_count as f64 / (first_ms as f64 / 1000.0);
            println!("  tokens/sec:      {tps:.1}");
        }

        // 3. Second message (incremental prefill — reuses KV prefix)
        let mut messages2 = messages.clone();
        messages2.push(Message {
            role: Role::Assistant,
            content: r1.content.clone(),
        });
        messages2.push(Message {
            role: Role::User,
            content: vec![ContentBlock::text("Now say goodbye in 5 words.")],
        });

        let t2 = Instant::now();
        let mut token_count2 = 0usize;
        let r2 = provider
            .chat_stream(
                &messages2,
                &[],
                "You are a helpful assistant.",
                &mut |_chunk: &str| {
                    token_count2 += 1;
                },
            )
            .await
            .unwrap();
        let second_ms = t2.elapsed().as_millis();
        let second_text = match &r2.content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => String::new(),
        };
        println!("Second response:   {second_ms} ms  ({token_count2} tokens)  \"{second_text}\"");
        if second_ms > 0 {
            let tps = token_count2 as f64 / (second_ms as f64 / 1000.0);
            println!("  tokens/sec:      {tps:.1}");
        }

        // 4. Tool-use prompt
        let tools = vec![ToolDef {
            name: "calc".into(),
            description: "Evaluate a math expression and return the result".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "expr": { "type": "string" } },
                "required": ["expr"]
            }),
        }];
        let tool_msgs = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("What is 123 * 456? Use the calc tool.")],
        }];

        let t3 = Instant::now();
        let r3 = provider
            .chat(&tool_msgs, &tools, "You are a helpful assistant.")
            .await
            .unwrap();
        let tool_ms = t3.elapsed().as_millis();
        println!("Tool-use response: {tool_ms} ms  stop_reason={:?}", r3.stop_reason);
        for block in &r3.content {
            match block {
                ContentBlock::Text { text } => println!("  text: \"{text}\""),
                ContentBlock::ToolUse { name, input, .. } => {
                    println!("  tool_use: {name}({input})")
                }
                _ => {}
            }
        }

        println!("=================================\n");
    }
}
