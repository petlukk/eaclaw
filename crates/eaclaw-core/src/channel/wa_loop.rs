//! WhatsApp agent loop: gateway + LLM integration.
//!
//! Receives messages from WhatsAppChannel, routes through Gateway,
//! calls LLM for forwarded messages, and sends responses back.

use crate::agent::ainur;
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

/// Print a status line to stderr (visible in the terminal).
fn status(msg: &str) {
    eprintln!("[eaclaw] {msg}");
}

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

    status("Waiting for WhatsApp connection...");

    // Wait for connection
    while !channel.is_connected() {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    status("Connected to WhatsApp!");
    status(&format!(
        "Listening for messages mentioning @{} or !{}",
        config.agent_name, config.agent_name
    ));
    status("Press Ctrl+C to stop.");
    eprintln!();

    loop {
        let msg = match channel.recv().await {
            Some(m) => m,
            None => {
                status("WhatsApp channel closed.");
                break;
            }
        };

        let processed = match gateway.process_inbound(&msg) {
            Some(p) => p,
            None => continue,
        };

        match processed.action {
            Action::Blocked(reason) => {
                eprintln!("  ⚠ Blocked: {reason}");
                channel
                    .send(&processed.jid, &format!("Message blocked: {reason}"))
                    .await;
            }
            Action::Forward {
                text,
                sender_name,
                context,
            } => {
                // Strip trigger prefix to get the actual command/message
                let stripped = strip_trigger(&text, &config.agent_name);

                // Check for /ainur command
                if let Some((count, task)) = ainur::parse_ainur(stripped) {
                    eprintln!(
                        "  ⚡ Triggered by {sender_name} — /ainur {count}..."
                    );

                    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(64);
                    let llm2 = llm.clone();
                    let tools2 = tools.clone();
                    let td2 = tool_defs.clone();
                    let sys2 = system_prompt.clone();
                    let mt = config.max_turns;
                    let task_str = task.to_string();

                    let ainur_handle = tokio::spawn(async move {
                        let mut s = SafetyLayer::new();
                        ainur::execute(
                            count, &task_str, &llm2, &tools2, &mut s,
                            &td2, &sys2, mt, event_tx,
                        ).await
                    });

                    // Drain events — send to both terminal and WhatsApp
                    let jid = processed.jid.clone();
                    while let Some(event) = event_rx.recv().await {
                        let chat_msg = match &event {
                            ainur::AinurEvent::Planning => {
                                eprintln!("  /ainur {count} — decomposing task...");
                                Some(format!("🎵 /ainur {count} — decomposing task..."))
                            }
                            ainur::AinurEvent::Planned { subtasks } => {
                                let lines: Vec<String> = subtasks.iter().enumerate()
                                    .map(|(i, s)| format!("♪ Ainur {}/{count}: {s}", i + 1))
                                    .collect();
                                let msg = lines.join("\n");
                                eprintln!("{}", lines.iter().map(|l| format!("  {l}")).collect::<Vec<_>>().join("\n"));
                                Some(msg)
                            }
                            ainur::AinurEvent::WorkerProgress { index, total, status } => {
                                eprintln!("  ♪ Ainur {}/{total} {status}", index + 1);
                                Some(format!("♪ Ainur {}/{total} {status}", index + 1))
                            }
                            ainur::AinurEvent::WorkerDone { index, total } => {
                                eprintln!("  ♪ Ainur {}/{total} ✓", index + 1);
                                Some(format!("♪ Ainur {}/{total} ✓", index + 1))
                            }
                            ainur::AinurEvent::Merging => {
                                eprintln!("  ♪ All voices complete — merging...");
                                Some("🎵 All voices complete — merging...".into())
                            }
                            ainur::AinurEvent::RateLimitWait { seconds } => {
                                eprintln!("  ⏳ Rate limit — waiting {seconds}s...");
                                Some(format!("⏳ Rate limit — waiting {seconds}s..."))
                            }
                        };
                        if let Some(msg) = chat_msg {
                            channel.send(&jid, &msg).await;
                        }
                    }

                    let response_text = match ainur_handle.await {
                        Ok(Ok(text)) => text,
                        Ok(Err(e)) => {
                            eprintln!("  ✗ Ainur error: {e}");
                            format!("Ainur error: {e}")
                        }
                        Err(e) => format!("Ainur task error: {e}"),
                    };

                    channel.send(&processed.jid, &response_text).await;
                    gateway.record_response(&processed.jid, &response_text);
                    eprintln!(
                        "  → [{jid}] eaclaw: {resp}",
                        jid = short_jid(&processed.jid),
                        resp = truncate(&response_text, 120),
                    );
                    continue;
                }

                eprintln!(
                    "  ⚡ Triggered by {sender_name} — calling LLM..."
                );

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
                        eprintln!("  ✗ LLM error: {e}");
                        format!("Sorry, I encountered an error: {e}")
                    }
                };

                // Send response and record it
                channel.send(&processed.jid, &response_text).await;
                gateway.record_response(&processed.jid, &response_text);

                eprintln!(
                    "  → [{jid}] eaclaw: {text}",
                    jid = short_jid(&processed.jid),
                    text = truncate(&response_text, 120),
                );
            }
        }
    }

    Ok(())
}

/// Strip trigger prefix (@eaclaw, !eaclaw, or "eaclaw ") from message text.
fn strip_trigger<'a>(text: &'a str, trigger: &str) -> &'a str {
    let lower = text.to_lowercase();
    let trig = trigger.to_lowercase();

    // Try @trigger or !trigger
    for prefix in [format!("@{trig}"), format!("!{trig}")] {
        if let Some(pos) = lower.find(&prefix) {
            let after = pos + prefix.len();
            return text[after..].trim_start();
        }
    }
    // Try "trigger " at start
    if lower.starts_with(&trig) {
        return text[trig.len()..].trim_start();
    }
    text
}

/// Shorten a JID for display: "1234567890@s.whatsapp.net" → "1234567890"
fn short_jid(jid: &str) -> &str {
    jid.split('@').next().unwrap_or(jid)
}

/// Truncate text for terminal display.
fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max {
        s
    } else {
        format!("{}...", &s[..max])
    }
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
            eprintln!("  🔧 Tool: {name}");
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
