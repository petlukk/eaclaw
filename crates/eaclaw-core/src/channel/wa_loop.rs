//! WhatsApp agent loop: gateway + LLM integration.
//!
//! Receives messages from WhatsAppChannel, routes through Gateway,
//! calls LLM for forwarded messages, and sends responses back.

use crate::channel::gateway::{Action, Gateway};
use crate::channel::types::GroupChannel;
use crate::channel::whatsapp::WhatsAppChannel;
use crate::config::Config;
use crate::error::Result;
use crate::llm::{ContentBlock, LlmProvider, Message, Role, StopReason, ToolDef};
use crate::safety::SafetyLayer;
use crate::tools::ToolRegistry;
use std::sync::Arc;

const WA_SYSTEM_PROMPT: &str = "\
You are eaclaw, an AI assistant in a WhatsApp group. \
Be concise — WhatsApp messages should be short and readable. \
You have access to tools. Use them when they help answer the question. \
Reply in the same language the user writes in.";

/// Run the WhatsApp agent loop.
pub async fn run(
    bridge_path: &str,
    session_dir: &str,
    config: &Config,
    llm: Arc<dyn LlmProvider>,
    tools: &ToolRegistry,
) -> Result<()> {
    let channel = WhatsAppChannel::start(bridge_path, session_dir).await?;
    let mut gateway = Gateway::new(config);
    let mut safety = SafetyLayer::new();
    let tool_defs: Vec<ToolDef> = tools.tool_defs();

    let system_prompt = match &config.identity {
        Some(identity) => format!("{WA_SYSTEM_PROMPT}\n\n{identity}"),
        None => WA_SYSTEM_PROMPT.to_string(),
    };

    tracing::info!("WhatsApp agent loop started, waiting for messages...");

    // Wait for connection
    while !channel.is_connected() {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    tracing::info!("WhatsApp bridge connected, listening for messages");

    loop {
        let msg = match channel.recv().await {
            Some(m) => m,
            None => {
                tracing::info!("WhatsApp channel closed");
                break;
            }
        };

        let processed = match gateway.process_inbound(&msg) {
            Some(p) => p,
            None => continue,
        };

        match processed.action {
            Action::Blocked(reason) => {
                tracing::warn!(jid = %processed.jid, %reason, "message blocked");
                channel
                    .send(&processed.jid, &format!("Message blocked: {reason}"))
                    .await;
            }
            Action::Forward {
                text,
                sender_name,
                context,
            } => {
                // Build conversation with context
                let mut messages = Vec::new();

                // Add recall context as a system-like user message
                if !context.is_empty() {
                    let context_text = context.join("\n");
                    messages.push(Message {
                        role: Role::User,
                        content: vec![ContentBlock::text(format!(
                            "[Recent conversation context]\n{context_text}"
                        ))],
                    });
                    messages.push(Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::text(
                            "I have the conversation context. What can I help with?",
                        )],
                    });
                }

                // Add the actual user message
                messages.push(Message {
                    role: Role::User,
                    content: vec![ContentBlock::text(format!("{sender_name}: {text}"))],
                });

                // LLM call with tool loop
                let response_text = match run_llm_turn(
                    &llm,
                    &mut messages,
                    &tool_defs,
                    &system_prompt,
                    tools,
                    &mut safety,
                    config.max_turns,
                )
                .await
                {
                    Ok(text) => text,
                    Err(e) => {
                        tracing::error!(error = %e, "LLM error");
                        format!("Sorry, I encountered an error: {e}")
                    }
                };

                // Send response and record it
                channel.send(&processed.jid, &response_text).await;
                gateway.record_response(&processed.jid, &response_text);

                tracing::info!(
                    jid = %processed.jid,
                    sender = %sender_name,
                    response_len = response_text.len(),
                    "responded"
                );
            }
        }
    }

    Ok(())
}

/// Run an LLM turn with tool loop, returning the final text response.
async fn run_llm_turn(
    llm: &Arc<dyn LlmProvider>,
    messages: &mut Vec<Message>,
    tool_defs: &[ToolDef],
    system: &str,
    tools: &ToolRegistry,
    safety: &mut SafetyLayer,
    max_turns: usize,
) -> Result<String> {
    let mut turns = 0;

    loop {
        if turns >= max_turns {
            return Ok("I've reached my tool use limit for this message.".into());
        }
        turns += 1;

        let response = llm.chat(messages, tool_defs, system).await?;

        let mut tool_uses = Vec::new();
        let mut text_parts = Vec::new();
        let mut assistant_blocks = Vec::new();

        for block in &response.content {
            match block {
                ContentBlock::Text { text } => {
                    text_parts.push(text.clone());
                    assistant_blocks.push(block.clone());
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_uses.push((id.clone(), name.clone(), input.clone()));
                    assistant_blocks.push(block.clone());
                }
                _ => {
                    assistant_blocks.push(block.clone());
                }
            }
        }

        messages.push(Message {
            role: Role::Assistant,
            content: assistant_blocks,
        });

        if tool_uses.is_empty() || response.stop_reason != StopReason::ToolUse {
            return Ok(text_parts.join(""));
        }

        // Execute tools
        let mut result_blocks = Vec::new();
        for (id, name, input) in &tool_uses {
            let result = match tools.get(name) {
                Some(tool) => match tool.execute(input.clone()).await {
                    Ok(output) => {
                        let scan = safety.scan_output(&output);
                        if scan.leaks_found {
                            ContentBlock::tool_error(
                                id,
                                "Tool output blocked: contains potential secrets",
                            )
                        } else {
                            ContentBlock::tool_result(id, &output)
                        }
                    }
                    Err(e) => ContentBlock::tool_error(id, e.to_string()),
                },
                None => ContentBlock::tool_error(id, format!("Unknown tool: {name}")),
            };
            result_blocks.push(result);
        }

        messages.push(Message {
            role: Role::User,
            content: result_blocks,
        });
    }
}
