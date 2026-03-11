use crate::llm::{ContentBlock, Message, Role, ToolDef};

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
