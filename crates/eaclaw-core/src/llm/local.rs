use crate::llm::{ContentBlock, Message, Role, ToolDef};

#[cfg(feature = "local-llm")]
use std::sync::Mutex;
#[cfg(feature = "local-llm")]
use super::llama_ffi::LlamaEngine;
#[cfg(feature = "local-llm")]
use super::eakv_ffi::EakvCache;
#[cfg(feature = "local-llm")]
use super::tool_parse::{ToolCallDetector, DetectResult};
#[cfg(feature = "local-llm")]
use super::{LlmProvider, LlmResponse, StopReason, OnTextFn};

/// Format a conversation into the Qwen2.5 `<|im_start|>` / `<|im_end|>` chat template.
///
/// When `tools` is non-empty, tool instructions and definitions are appended to the
/// system message. The returned string ends with `<|im_start|>assistant\n` to prompt
/// the model to generate its next turn.
pub fn format_chat_template(system: &str, messages: &[Message], tools: &[ToolDef]) -> String {
    let mut out = String::new();

    // --- system block ---
    out.push_str("<|im_start|>system\n");
    out.push_str(system);

    if !tools.is_empty() {
        out.push_str("\n\nYou have access to the following tools. To call a tool, output:\n\n");
        out.push_str("<tool_call>\n");
        out.push_str("{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}\n");
        out.push_str("</tool_call>\n");
        out.push_str("\nAvailable tools:\n");
        for tool in tools {
            out.push_str(&format!("- **{}**: {}\n", tool.name, tool.description));
            out.push_str(&format!(
                "  Parameters: {}\n",
                tool.input_schema.to_string()
            ));
        }
    }

    out.push_str("\n<|im_end|>\n");

    // --- conversation turns ---
    for msg in messages {
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!("<|im_start|>{}\n", role_str));

        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => {
                    out.push_str(text);
                }
                ContentBlock::ToolUse { id: _, name, input } => {
                    out.push_str("<tool_call>\n");
                    let obj = serde_json::json!({ "name": name, "arguments": input });
                    out.push_str(&obj.to_string());
                    out.push('\n');
                    out.push_str("</tool_call>");
                }
                ContentBlock::ToolResult {
                    tool_use_id: _,
                    content,
                    ..
                } => {
                    out.push_str("<tool_result>\n");
                    out.push_str(content);
                    out.push('\n');
                    out.push_str("</tool_result>");
                }
            }
        }

        out.push_str("\n<|im_end|>\n");
    }

    // --- prompt the model ---
    out.push_str("<|im_start|>assistant\n");

    out
}

/// Compute the length of the common prefix between two token sequences.
///
/// Used for incremental prefill: find how many tokens from the previous context
/// can be reused before the first point of divergence.
pub fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(feature = "local-llm")]
pub struct LocalLlmProvider {
    inner: Mutex<LocalLlmInner>,
}

#[cfg(feature = "local-llm")]
struct LocalLlmInner {
    engine: LlamaEngine,
    kv_cache: EakvCache,
    prefilled_tokens: Vec<i32>,
    tool_id_counter: u64,
    open_pattern: Vec<i32>,
    close_pattern: Vec<i32>,
    eakv_seq_len: i32,
}

#[cfg(feature = "local-llm")]
impl LocalLlmProvider {
    pub fn new(model_path: &str, n_ctx: u32, n_threads: u32,
               n_layers: i32, n_kv_heads: i32, head_dim: i32) -> crate::error::Result<Self> {
        let engine = LlamaEngine::new(model_path, n_ctx, n_threads)
            .map_err(|e| crate::error::Error::Llm(e))?;

        let kv_cache = EakvCache::new(n_layers, n_kv_heads, head_dim, n_ctx as i32)
            .ok_or_else(|| crate::error::Error::Llm("failed to create eakv cache".into()))?;

        let open_pattern = engine.tokenize("<tool_call>", false);
        let close_pattern = engine.tokenize("</tool_call>", false);

        Ok(Self {
            inner: Mutex::new(LocalLlmInner {
                engine,
                kv_cache,
                prefilled_tokens: Vec::new(),
                tool_id_counter: 0,
                open_pattern,
                close_pattern,
                eakv_seq_len: 0,
            }),
        })
    }

    fn generate_tool_id(inner: &mut LocalLlmInner) -> String {
        inner.tool_id_counter += 1;
        format!("local_{}", inner.tool_id_counter)
    }
}

#[cfg(feature = "local-llm")]
#[async_trait::async_trait]
impl LlmProvider for LocalLlmProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
    ) -> crate::error::Result<LlmResponse> {
        let noop: OnTextFn<'_> = &mut |_: &str| {};
        self.chat_stream(messages, tools, system, noop).await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
        on_text: OnTextFn<'_>,
    ) -> crate::error::Result<LlmResponse> {
        let formatted = format_chat_template(system, messages, tools);

        let mut inner = self.inner.lock()
            .map_err(|e| crate::error::Error::Llm(format!("lock poisoned: {e}")))?;

        let new_tokens = inner.engine.tokenize(&formatted, true);
        let prefix_len = common_prefix_len(&inner.prefilled_tokens, &new_tokens);

        // If prefix diverges, truncate llama.cpp KV cache (and eakv if synced)
        if prefix_len < inner.prefilled_tokens.len() {
            inner.engine.kv_cache_truncate(prefix_len as i32);
            // eakv restore is best-effort — the bridge format may not be active yet
            let _ = inner.kv_cache.restore(prefix_len as i32);
            inner.eakv_seq_len = prefix_len as i32;
        }

        // Prefill only the new suffix
        let suffix = &new_tokens[prefix_len..];
        if !suffix.is_empty() {
            inner.engine.decode(suffix, prefix_len as i32)
                .map_err(|e| crate::error::Error::Llm(e))?;

            // Export KV state from llama.cpp -> eakv (Approach A, best-effort)
            let kv_state = inner.engine.export_kv_state();
            let seq_len = inner.eakv_seq_len;
            if let Err(e) = inner.kv_cache.import_llama_state(&kv_state, seq_len) {
                tracing::debug!("eakv KV import skipped: {e}");
            } else {
                inner.eakv_seq_len = inner.kv_cache.seq_len();
            }
        }

        inner.prefilled_tokens = new_tokens.clone();

        // Checkpoint eakv before generation (best-effort)
        let _checkpoint = inner.kv_cache.checkpoint();

        // Generation loop — runs in C++ for minimal per-token overhead.
        // C++ handles decode+sample in a tight loop; Rust processes tokens after.
        let gen_pos = inner.prefilled_tokens.len() as i32;
        let max_gen = 2048i32;

        // Tool-call detection state shared with the callback via raw pointer.
        // The callback only touches detector state — engine is borrowed by C++.
        let mut detector = ToolCallDetector::new(
            inner.open_pattern.clone(),
            inner.close_pattern.clone(),
            512,
        );
        let mut tool_detected = false;

        let generated = inner.engine.generate_stream(gen_pos, max_gen, |token| {
            let result = detector.feed(token);
            if matches!(result, DetectResult::ToolCall(_)) {
                tool_detected = true;
                return true; // stop generation
            }
            false // continue
        });

        // Update prefilled_tokens with generated tokens
        inner.prefilled_tokens.extend_from_slice(&generated);

        // Now process all generated tokens through the detector for streaming
        // and content block construction. Reset detector and replay.
        let mut detector = ToolCallDetector::new(
            inner.open_pattern.clone(),
            inner.close_pattern.clone(),
            512,
        );
        let mut content_blocks = Vec::new();
        let mut text_buf = String::new();
        let mut stop_reason = StopReason::EndTurn;

        for &token in &generated {
            match detector.feed(token) {
                DetectResult::Text(t) => {
                    let piece = inner.engine.token_to_str(t);
                    text_buf.push_str(&piece);
                    on_text(&piece);
                }
                DetectResult::TagOpen | DetectResult::Captured => {
                    // Suppressed from output
                }
                DetectResult::ToolCall(body_tokens) => {
                    if !text_buf.is_empty() {
                        content_blocks.push(ContentBlock::Text {
                            text: std::mem::take(&mut text_buf),
                        });
                    }

                    let json_str = inner.engine.detokenize(&body_tokens);
                    match serde_json::from_str::<serde_json::Value>(&json_str) {
                        Ok(val) => {
                            let name = val["name"].as_str().unwrap_or("").to_string();
                            let arguments = val["arguments"].clone();
                            let id = Self::generate_tool_id(&mut inner);
                            content_blocks.push(ContentBlock::ToolUse {
                                id,
                                name,
                                input: arguments,
                            });
                            stop_reason = StopReason::ToolUse;
                        }
                        Err(_e) => {
                            text_buf.push_str(&format!("<tool_call>{json_str}</tool_call>"));
                            on_text(&format!("<tool_call>{json_str}</tool_call>"));
                        }
                    }
                    break;
                }
                DetectResult::Aborted(tokens) => {
                    let text = inner.engine.detokenize(&tokens);
                    text_buf.push_str(&text);
                    on_text(&text);
                }
            }
        }

        if !text_buf.is_empty() {
            content_blocks.push(ContentBlock::Text { text: text_buf });
        }

        // Fallback: if tools were provided but model output raw JSON without
        // <tool_call> tags, try to parse accumulated text as a tool call.
        if !tools.is_empty() && stop_reason != StopReason::ToolUse {
            if let Some(idx) = content_blocks.iter().position(|b| {
                matches!(b, ContentBlock::Text { text } if text.trim_start().starts_with('{'))
            }) {
                if let ContentBlock::Text { text } = &content_blocks[idx] {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text.trim()) {
                        if val.get("name").and_then(|n| n.as_str()).is_some() {
                            let name = val["name"].as_str().unwrap().to_string();
                            let arguments = val.get("arguments")
                                .cloned()
                                .unwrap_or(serde_json::Value::Object(Default::default()));
                            let id = Self::generate_tool_id(&mut inner);
                            content_blocks[idx] = ContentBlock::ToolUse {
                                id, name, input: arguments,
                            };
                            stop_reason = StopReason::ToolUse;
                        }
                    }
                }
            }
        }

        if content_blocks.is_empty() {
            content_blocks.push(ContentBlock::Text { text: String::new() });
        }

        Ok(LlmResponse {
            content: content_blocks,
            stop_reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ContentBlock, Message, Role, ToolDef};
    use serde_json::json;

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
        }
    }

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::text(text)],
        }
    }

    #[test]
    fn format_messages_basic() {
        let messages = vec![user_msg("Hello!"), assistant_msg("Hi there.")];
        let result = format_chat_template("You are helpful.", &messages, &[]);

        assert!(result.starts_with("<|im_start|>system\nYou are helpful."));
        assert!(result.contains("<|im_end|>\n<|im_start|>user\nHello!"));
        assert!(result.contains("<|im_end|>\n<|im_start|>assistant\nHi there."));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_messages_with_tools() {
        let tools = vec![ToolDef {
            name: "calc".to_string(),
            description: "Calculate math".to_string(),
            input_schema: json!({"type":"object","properties":{"expr":{"type":"string"}}}),
        }];
        let messages = vec![user_msg("What is 2+2?")];
        let result = format_chat_template("You are helpful.", &messages, &tools);

        assert!(result.contains("You have access to the following tools"));
        assert!(result.contains("**calc**: Calculate math"));
        assert!(result.contains("Parameters:"));
        assert!(result.contains("expr"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn common_prefix_length_identical() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        assert_eq!(common_prefix_len(&a, &b), 5);
    }

    #[test]
    fn common_prefix_length_partial() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 6, 7, 8];
        assert_eq!(common_prefix_len(&a, &b), 3);
    }

    #[test]
    fn common_prefix_length_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        assert_eq!(common_prefix_len(&a, &b), 0);
    }

    #[test]
    fn common_prefix_length_diverge_at_start() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert_eq!(common_prefix_len(&a, &b), 0);
    }
}
