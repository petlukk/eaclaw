pub mod router;

use crate::channel::Channel;
use crate::config::Config;
use crate::error::Result;
use crate::kernels::command_router as cmd_router;
use crate::llm::{
    ContentBlock, LlmProvider, Message, Role, StopReason, ToolDef,
};
use crate::safety::SafetyLayer;
use crate::tools::ToolRegistry;
use std::sync::Arc;
use std::time::Instant;

const SYSTEM_PROMPT: &str = "\
You are eaclaw, a high-performance AI assistant. \
You have access to tools that you can use to help the user. \
Be concise and helpful. Use tools when they would help answer the user's question.";

/// Timing data for a single agent turn.
pub struct TurnTiming {
    pub safety_scan_us: u64,
    pub llm_call_ms: u64,
    pub tool_execs: Vec<(String, u64)>,
}

impl TurnTiming {
    fn total_ms(&self) -> u64 {
        let tool_ms: u64 = self.tool_execs.iter().map(|(_, ms)| ms).sum();
        // safety_scan_us converted to ms (rounded up)
        let safety_ms = (self.safety_scan_us + 999) / 1000;
        safety_ms + self.llm_call_ms + tool_ms
    }

    fn format(&self) -> String {
        let mut lines = vec![
            "Last turn timing:".to_string(),
            format!("  Safety scan:    {} µs", self.safety_scan_us),
            format!("  LLM call:       {} ms", self.llm_call_ms),
        ];
        for (name, ms) in &self.tool_execs {
            lines.push(format!("  Tool: {:<10}{} ms", name, ms));
        }
        lines.push(format!("  Total:          {} ms", self.total_ms()));
        lines.join("\n")
    }
}

pub struct Agent {
    config: Config,
    llm: Arc<dyn LlmProvider>,
    tools: ToolRegistry,
    safety: SafetyLayer,
    messages: Vec<Message>,
    last_timing: Option<TurnTiming>,
}

impl Agent {
    pub fn new(
        config: Config,
        llm: Arc<dyn LlmProvider>,
        tools: ToolRegistry,
        safety: SafetyLayer,
    ) -> Self {
        Self {
            config,
            llm,
            tools,
            safety,
            messages: Vec::new(),
            last_timing: None,
        }
    }

    /// Run the agent loop on the given channel.
    pub async fn run(&mut self, channel: &dyn Channel) -> Result<()> {
        channel
            .send(&format!(
                "Welcome to {}! Type /help for commands, /quit to exit.",
                self.config.agent_name
            ))
            .await;

        let tool_defs: Vec<ToolDef> = self.tools.tool_defs();

        loop {
            let msg = match channel.recv().await {
                Some(m) => m,
                None => break,
            };

            // Two-stage SIMD command routing (hash + verify)
            let (cmd_id, cmd_arg) = cmd_router::match_command_verified(msg.as_bytes());

            // Handle meta commands
            if cmd_id >= cmd_router::CMD_HELP && cmd_id <= cmd_router::CMD_PROFILE {
                match cmd_id {
                    cmd_router::CMD_QUIT => {
                        channel.send("Goodbye!").await;
                        break;
                    }
                    cmd_router::CMD_HELP => {
                        channel.send(&self.help_text()).await;
                        continue;
                    }
                    cmd_router::CMD_TOOLS => {
                        let names = self.tools.list_names();
                        channel
                            .send(&format!("Available tools: {}", names.join(", ")))
                            .await;
                        continue;
                    }
                    cmd_router::CMD_CLEAR => {
                        self.messages.clear();
                        channel.send("Context cleared.").await;
                        continue;
                    }
                    cmd_router::CMD_MODEL => {
                        channel
                            .send(&format!("Model: {}", self.config.model))
                            .await;
                        continue;
                    }
                    cmd_router::CMD_PROFILE => {
                        match &self.last_timing {
                            Some(timing) => channel.send(&timing.format()).await,
                            None => channel.send("No timing data yet. Send a message first.").await,
                        }
                        continue;
                    }
                    _ => {}
                }
            }

            // Handle direct tool commands — bypass the LLM
            if cmd_id >= cmd_router::CMD_TOOL_FIRST && cmd_id <= cmd_router::CMD_TOOL_LAST {
                let tool_start = Instant::now();
                let arg_str = String::from_utf8_lossy(cmd_arg);
                let result = self.execute_direct_tool(cmd_id, &arg_str).await;
                let tool_ms = tool_start.elapsed().as_millis() as u64;
                let tool_name = cmd_router::command_name(cmd_id).unwrap_or("unknown");

                match result {
                    Ok(output) => {
                        // Leak scan on tool output
                        let scan = self.safety.scan_output(&output);
                        if scan.leaks_found {
                            channel
                                .send("Tool output blocked: contains potential secrets.")
                                .await;
                        } else {
                            channel.send(&output).await;
                        }
                    }
                    Err(e) => {
                        channel.send(&format!("Tool error: {e}")).await;
                    }
                }

                self.last_timing = Some(TurnTiming {
                    safety_scan_us: 0,
                    llm_call_ms: 0,
                    tool_execs: vec![(tool_name.to_string(), tool_ms)],
                });
                continue;
            }

            // Check for unknown slash commands
            if msg.starts_with(&self.config.command_prefix) && cmd_id == cmd_router::CMD_NONE {
                channel
                    .send(&format!("Unknown command: {msg}. Type /help for available commands."))
                    .await;
                continue;
            }

            // Safety scan on input (timed, reuses SIMD buffers)
            let safety_start = Instant::now();
            let scan = self.safety.scan_input(&msg);
            let safety_scan_us = safety_start.elapsed().as_micros() as u64;

            if scan.injection_found {
                let details: Vec<String> = scan
                    .details
                    .iter()
                    .map(|w| format!("  - {} at position {}", w.pattern, w.position))
                    .collect();
                channel
                    .send(&format!(
                        "Warning: potential injection detected:\n{}",
                        details.join("\n")
                    ))
                    .await;
                continue;
            }
            if scan.leaks_found {
                channel
                    .send("Warning: your message appears to contain secrets. Message not sent.")
                    .await;
                continue;
            }

            // Add user message
            self.messages.push(Message {
                role: Role::User,
                content: vec![ContentBlock::text(&msg)],
            });

            // Agentic tool loop
            let mut turns = 0;
            let mut total_llm_ms: u64 = 0;
            let mut tool_execs: Vec<(String, u64)> = Vec::new();

            loop {
                if turns >= self.config.max_turns {
                    channel
                        .send("Max tool turns reached. Stopping.")
                        .await;
                    break;
                }
                turns += 1;

                // Stream LLM response (timed)
                let llm_start = Instant::now();
                let mut streamed_any_text = false;

                let response = {
                    let streamed_flag = &mut streamed_any_text;
                    let mut on_text = |chunk: &str| {
                        if !chunk.is_empty() {
                            if !*streamed_flag {
                                // Leading newline to separate from prompt,
                                // matching the \n prefix in channel.send().
                                print!("\n");
                            }
                            *streamed_flag = true;
                            print!("{chunk}");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                        }
                    };

                    match self
                        .llm
                        .chat_stream(&self.messages, &tool_defs, SYSTEM_PROMPT, &mut on_text)
                        .await
                    {
                        Ok(r) => r,
                        Err(e) => {
                            if streamed_any_text {
                                channel.flush().await;
                            }
                            channel.send(&format!("LLM error: {e}")).await;
                            break;
                        }
                    }
                };
                total_llm_ms += llm_start.elapsed().as_millis() as u64;

                // Collect tool uses and text
                let mut tool_uses = Vec::new();
                let mut text_parts = Vec::new();
                let mut assistant_blocks = Vec::new();

                for block in &response.content {
                    match block {
                        ContentBlock::Text { text } => {
                            text_parts.push(text.as_str());
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

                // Push assistant message
                self.messages.push(Message {
                    role: Role::Assistant,
                    content: assistant_blocks,
                });

                if tool_uses.is_empty() || response.stop_reason != StopReason::ToolUse {
                    // Text was already streamed, just flush
                    if streamed_any_text {
                        channel.flush().await;
                    } else if !text_parts.is_empty() {
                        // Fallback: non-streaming provider
                        channel.send(&text_parts.join("")).await;
                    }
                    break;
                }

                // Flush any leading streamed text before tool execution
                if streamed_any_text {
                    channel.flush().await;
                }

                // Execute tools (timed)
                let mut result_blocks = Vec::new();
                for (id, name, input) in &tool_uses {
                    let tool_start = Instant::now();
                    let result = match self.tools.get(name) {
                        Some(tool) => match tool.execute(input.clone()).await {
                            Ok(output) => {
                                // Safety scan on tool output
                                let scan = self.safety.scan_output(&output);
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
                        None => {
                            ContentBlock::tool_error(id, format!("Unknown tool: {name}"))
                        }
                    };
                    let tool_ms = tool_start.elapsed().as_millis() as u64;
                    tool_execs.push((name.clone(), tool_ms));
                    result_blocks.push(result);
                }

                self.messages.push(Message {
                    role: Role::User,
                    content: result_blocks,
                });
            }

            // Store timing for /profile
            self.last_timing = Some(TurnTiming {
                safety_scan_us,
                llm_call_ms: total_llm_ms,
                tool_execs,
            });
        }

        Ok(())
    }

    /// Execute a tool directly from a slash command, bypassing the LLM.
    async fn execute_direct_tool(&self, cmd_id: i32, arg: &str) -> Result<String> {
        match cmd_id {
            cmd_router::CMD_TIME => {
                self.run_tool("time", serde_json::json!({})).await
            }
            cmd_router::CMD_CALC => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /calc <expression>".into()));
                }
                self.run_tool("calc", serde_json::json!({"expr": arg})).await
            }
            cmd_router::CMD_HTTP => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /http <url>".into()));
                }
                self.run_tool("http", serde_json::json!({"url": arg})).await
            }
            cmd_router::CMD_SHELL => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /shell <command>".into()));
                }
                self.run_tool("shell", serde_json::json!({"command": arg})).await
            }
            cmd_router::CMD_MEMORY => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /memory <action> [key] [value]".into()));
                }
                let parts: Vec<&str> = arg.splitn(3, ' ').collect();
                let action = parts[0];
                let key = parts.get(1).unwrap_or(&"");
                let value = parts.get(2).unwrap_or(&"");
                self.run_tool("memory", serde_json::json!({
                    "action": action, "key": key, "value": value,
                })).await
            }
            cmd_router::CMD_READ => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /read <path>".into()));
                }
                self.run_tool("read", serde_json::json!({"path": arg})).await
            }
            cmd_router::CMD_WRITE => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /write <path> <content>".into()));
                }
                let parts: Vec<&str> = arg.splitn(2, ' ').collect();
                let path = parts[0];
                let content = parts.get(1).unwrap_or(&"");
                self.run_tool("write", serde_json::json!({
                    "path": path, "content": content,
                })).await
            }
            cmd_router::CMD_LS => {
                let path = if arg.is_empty() { "." } else { arg };
                self.run_tool("ls", serde_json::json!({"path": path})).await
            }
            cmd_router::CMD_JSON => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /json <keys|get|pretty> <input> [path]".into()));
                }
                let parts: Vec<&str> = arg.splitn(3, ' ').collect();
                let action = parts[0];
                let input = parts.get(1).unwrap_or(&"");
                let path = parts.get(2).unwrap_or(&"");
                let mut params = serde_json::json!({"action": action, "input": input});
                if !path.is_empty() {
                    params["path"] = serde_json::json!(path);
                }
                self.run_tool("json", params).await
            }
            cmd_router::CMD_CPU => {
                self.run_tool("cpu", serde_json::json!({})).await
            }
            cmd_router::CMD_TOKENS => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /tokens <text or file>".into()));
                }
                self.run_tool("tokens", serde_json::json!({"text": arg})).await
            }
            cmd_router::CMD_BENCH => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /bench <safety|router>".into()));
                }
                self.run_tool("bench", serde_json::json!({"target": arg})).await
            }
            _ => Err(crate::error::Error::Tool(format!("unknown tool command: {cmd_id}"))),
        }
    }

    async fn run_tool(&self, name: &str, params: serde_json::Value) -> Result<String> {
        match self.tools.get(name) {
            Some(tool) => tool.execute(params).await,
            None => Err(crate::error::Error::Tool(format!("{name} tool not registered"))),
        }
    }

    fn help_text(&self) -> String {
        format!(
            "Commands:\n  \
             /help              — Show this help\n  \
             /quit              — Exit\n  \
             /tools             — List available tools\n  \
             /clear             — Clear conversation history\n  \
             /model             — Show current model\n  \
             /profile           — Show last turn timing\n\
             \nDirect tool commands:\n  \
             /time              — Current UTC time\n  \
             /calc <expr>       — Evaluate math expression\n  \
             /http <url>        — HTTP GET\n  \
             /shell <command>   — Run shell command\n  \
             /memory <action>   — Key-value store (list/read key/write key value)\n  \
             /read <path>       — Read file contents\n  \
             /write <path> <content> — Write to file\n  \
             /ls [path]         — List directory\n  \
             /json <action> <input> — JSON operations (keys/get/pretty)\n  \
             /cpu               — System info (CPU, memory, uptime)\n  \
             /tokens <text>     — Estimate token count\n  \
             /bench <target>    — Microbenchmark (safety/router)\n\
             \nAgent: {} | Model: {}",
            self.config.agent_name, self.config.model
        )
    }
}
