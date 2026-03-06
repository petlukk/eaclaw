pub mod background;
pub mod router;

use crate::channel::Channel;
use crate::config::Config;
use crate::error::Result;
use crate::kernels::arg_tokenizer::ArgTokenizer;
use crate::kernels::command_router as cmd_router;
use crate::llm::{
    ContentBlock, LlmProvider, Message, Role, StopReason, ToolDef,
};
use crate::safety::SafetyLayer;
use crate::tools::ToolRegistry;
use std::sync::Arc;
use std::time::Instant;

/// Escape a string for safe use in shell commands (single-quote wrapping).
fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

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
    bg_tasks: background::TaskTable,
    tokenizer: ArgTokenizer,
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
            bg_tasks: background::TaskTable::new(),
            tokenizer: ArgTokenizer::with_capacity(256),
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

            // Pipeline detection: split on " | /" before routing
            if msg.starts_with(&self.config.command_prefix) && msg.contains(" | /") {
                match self.execute_pipeline(&msg, channel).await {
                    Ok(()) => {}
                    Err(e) => channel.send(&format!("Pipeline error: {e}")).await,
                }
                continue;
            }

            // Two-stage SIMD command routing (hash + verify)
            let (cmd_id, cmd_arg) = cmd_router::match_command_verified(msg.as_bytes());

            // Handle /tasks meta command (ID 18, outside 0-5 range)
            if cmd_id == cmd_router::CMD_TASKS {
                channel.send(&self.bg_tasks.format_list()).await;
                continue;
            }

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
                let arg_str = String::from_utf8_lossy(cmd_arg).into_owned();
                let tool_name = cmd_router::command_name(cmd_id).unwrap_or("unknown");

                // Detect background execution: trailing " &"
                let (arg_str, is_background) = if arg_str.ends_with(" &") {
                    (arg_str[..arg_str.len() - 2].to_string(), true)
                } else if arg_str == "&" {
                    (String::new(), true)
                } else {
                    (arg_str, false)
                };

                if is_background {
                    // Spawn as background task
                    match self.build_tool_params(cmd_id, &arg_str) {
                        Ok((name, params)) => {
                            let tool = self.tools.get(name).cloned();
                            if let Some(tool) = tool {
                                let task_id = self.bg_tasks.register(
                                    tool_name,
                                    &format!("/{tool_name} {arg_str}"),
                                );
                                let bg_tasks = self.bg_tasks.clone();
                                tokio::spawn(async move {
                                    match tool.execute(params).await {
                                        Ok(output) => bg_tasks.complete(task_id, output),
                                        Err(e) => bg_tasks.fail(task_id, e.to_string()),
                                    }
                                });
                                channel
                                    .send(&format!("[{task_id}] Started in background: /{tool_name} {arg_str}"))
                                    .await;
                            } else {
                                channel
                                    .send(&format!("Tool error: {tool_name} not registered"))
                                    .await;
                            }
                        }
                        Err(e) => {
                            channel.send(&format!("Tool error: {e}")).await;
                        }
                    }
                    continue;
                }

                // Foreground execution
                let tool_start = Instant::now();

                // Check if tool supports streaming
                let tool_streams = self.tools.get(tool_name)
                    .map_or(false, |t| t.supports_streaming());

                if tool_streams {
                    let result = self.execute_direct_tool_stream(cmd_id, &arg_str, channel).await;
                    if let Err(e) = result {
                        channel.send(&format!("Tool error: {e}")).await;
                    }
                } else {
                    let result = self.execute_direct_tool(cmd_id, &arg_str).await;
                    match result {
                        Ok(output) => {
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
                }

                let tool_ms = tool_start.elapsed().as_millis() as u64;
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
                    let prefix = channel.response_prefix();
                    let mut on_text = |chunk: &str| {
                        if !chunk.is_empty() {
                            if !*streamed_flag {
                                print!("\r\x1b[2K{prefix} ");
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

    /// Execute a pipeline of tool commands separated by " | ".
    /// Output of each stage becomes the argument to the next.
    async fn execute_pipeline(&mut self, input: &str, channel: &dyn Channel) -> Result<()> {
        let stages: Vec<&str> = input.split(" | ").collect();
        if stages.len() < 2 {
            return Err(crate::error::Error::Tool("not a pipeline".into()));
        }

        let mut pipe_data: Option<String> = None;
        let mut tool_execs: Vec<(String, u64)> = Vec::new();
        let _pipeline_start = Instant::now();

        for (i, stage) in stages.iter().enumerate() {
            let stage = stage.trim();
            let (cmd_id, cmd_arg) = cmd_router::match_command_verified(stage.as_bytes());

            if cmd_id < cmd_router::CMD_TOOL_FIRST || cmd_id > cmd_router::CMD_TOOL_LAST {
                return Err(crate::error::Error::Tool(format!(
                    "stage {}: not a tool command: {stage}",
                    i + 1
                )));
            }

            let tool_name = cmd_router::command_name(cmd_id).unwrap_or("unknown");
            let arg_str = String::from_utf8_lossy(cmd_arg).into_owned();

            // Build params — for piped stages, use piped data as default arg
            let effective_arg = if pipe_data.is_some() && arg_str.is_empty() {
                // Use a placeholder so build_tool_params doesn't reject empty args
                "__piped__".to_string()
            } else {
                arg_str.clone()
            };
            let (name, mut params) = self.build_tool_params(cmd_id, &effective_arg)?;
            if let Some(ref piped) = pipe_data {
                self.inject_pipe_data(cmd_id, &mut params, piped, &arg_str);
            }

            let tool_start = Instant::now();
            let output = self.run_tool(name, params).await?;
            let tool_ms = tool_start.elapsed().as_millis() as u64;
            tool_execs.push((tool_name.to_string(), tool_ms));

            pipe_data = Some(output);
        }

        // Safety scan on final output
        if let Some(ref output) = pipe_data {
            let scan = self.safety.scan_output(output);
            if scan.leaks_found {
                channel
                    .send("Pipeline output blocked: contains potential secrets.")
                    .await;
            } else {
                channel.send(output).await;
            }
        }

        self.last_timing = Some(TurnTiming {
            safety_scan_us: 0,
            llm_call_ms: 0,
            tool_execs,
        });

        Ok(())
    }

    /// Inject piped data into tool parameters.
    /// Each tool has a primary input field that receives piped data.
    fn inject_pipe_data(
        &self,
        cmd_id: i32,
        params: &mut serde_json::Value,
        piped: &str,
        _arg: &str,
    ) {
        match cmd_id {
            cmd_router::CMD_CALC => {
                params["expr"] = serde_json::json!(piped);
            }
            cmd_router::CMD_SHELL => {
                // Pipe as stdin: wrap command with echo piped | command
                if let Some(cmd) = params["command"].as_str() {
                    let wrapped = format!("echo {} | {cmd}", shell_escape(piped));
                    params["command"] = serde_json::json!(wrapped);
                }
            }
            cmd_router::CMD_HTTP => {
                // Piped data becomes the URL
                params["url"] = serde_json::json!(piped.trim());
            }
            cmd_router::CMD_TOKENS => {
                params["text"] = serde_json::json!(piped);
            }
            cmd_router::CMD_JSON => {
                // Piped JSON goes to input field; action/path kept from args
                params["input"] = serde_json::json!(piped);
            }
            cmd_router::CMD_READ => {
                // Piped data as file path
                params["path"] = serde_json::json!(piped.trim());
            }
            cmd_router::CMD_WRITE => {
                // Piped data as content; path must come from args
                params["content"] = serde_json::json!(piped);
            }
            cmd_router::CMD_MEMORY => {
                // Piped data as value for write, or key for read
                if params["action"].as_str() == Some("write") {
                    params["value"] = serde_json::json!(piped);
                } else if params["action"].as_str() == Some("read") {
                    params["key"] = serde_json::json!(piped.trim());
                }
            }
            _ => {
                // For tools without a clear pipe target, ignore piped data
            }
        }
    }

    /// Execute a tool directly from a slash command, bypassing the LLM.
    async fn execute_direct_tool(&mut self, cmd_id: i32, arg: &str) -> Result<String> {
        let (name, params) = self.build_tool_params(cmd_id, arg)?;
        self.run_tool(name, params).await
    }

    /// Execute a streaming tool command, sending chunks directly to the channel.
    async fn execute_direct_tool_stream(
        &mut self,
        cmd_id: i32,
        arg: &str,
        channel: &dyn Channel,
    ) -> Result<()> {
        let (tool_name, params) = self.build_tool_params(cmd_id, arg)?;

        let tool = self.tools.get(tool_name)
            .ok_or_else(|| crate::error::Error::Tool(format!("{tool_name} tool not registered")))?;

        // Collect chunks, then leak-scan full output before sending to channel.
        let mut chunks: Vec<String> = Vec::new();
        let mut on_chunk = |chunk: &str| {
            chunks.push(chunk.to_string());
        };

        tool.execute_stream(params, &mut on_chunk).await?;

        // Leak scan the full output
        let full_output: String = chunks.iter().map(|s| s.as_str()).collect();
        let scan = self.safety.scan_output(&full_output);
        if scan.leaks_found {
            channel
                .send("Tool output blocked: contains potential secrets.")
                .await;
            return Ok(());
        }

        // Stream chunks to channel
        if !chunks.is_empty() {
            print!("\r\x1b[2K{} ", channel.response_prefix());
            for chunk in &chunks {
                channel.send_chunk(chunk).await;
            }
            channel.flush().await;
        }

        Ok(())
    }

    /// Build tool name and params from command ID and argument string.
    /// Uses SIMD arg tokenizer for multi-arg commands.
    fn build_tool_params(&mut self, cmd_id: i32, arg: &str) -> Result<(&'static str, serde_json::Value)> {
        match cmd_id {
            cmd_router::CMD_TIME => Ok(("time", serde_json::json!({}))),
            cmd_router::CMD_CALC => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /calc <expression>".into()));
                }
                Ok(("calc", serde_json::json!({"expr": arg})))
            }
            cmd_router::CMD_HTTP => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /http <url>".into()));
                }
                Ok(("http", serde_json::json!({"url": arg})))
            }
            cmd_router::CMD_SHELL => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /shell <command>".into()));
                }
                Ok(("shell", serde_json::json!({"command": arg})))
            }
            cmd_router::CMD_MEMORY => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /memory <action> [key] [value]".into()));
                }
                // SIMD tokenize: action key value (3 tokens max)
                let parts = self.tokenizer.tokenize_str(arg, 3);
                let action = parts.first().unwrap_or(&"");
                let key = parts.get(1).unwrap_or(&"");
                let value = parts.get(2).unwrap_or(&"");
                Ok(("memory", serde_json::json!({"action": action, "key": key, "value": value})))
            }
            cmd_router::CMD_READ => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /read <path>".into()));
                }
                Ok(("read", serde_json::json!({"path": arg})))
            }
            cmd_router::CMD_WRITE => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /write <path> <content>".into()));
                }
                // SIMD tokenize: path content (2 tokens max)
                let parts = self.tokenizer.tokenize_str(arg, 2);
                let path = parts.first().unwrap_or(&"");
                let content = parts.get(1).unwrap_or(&"");
                Ok(("write", serde_json::json!({"path": path, "content": content})))
            }
            cmd_router::CMD_LS => {
                let path = if arg.is_empty() { "." } else { arg };
                Ok(("ls", serde_json::json!({"path": path})))
            }
            cmd_router::CMD_JSON => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /json <keys|get|pretty> <input> [path]".into()));
                }
                // SIMD tokenize: action input path (3 tokens max)
                let parts = self.tokenizer.tokenize_str(arg, 3);
                let action = parts.first().unwrap_or(&"");
                let input = parts.get(1).unwrap_or(&"");
                let path = parts.get(2).unwrap_or(&"");
                let mut params = serde_json::json!({"action": action, "input": input});
                if !path.is_empty() {
                    params["path"] = serde_json::json!(path);
                }
                Ok(("json", params))
            }
            cmd_router::CMD_CPU => Ok(("cpu", serde_json::json!({}))),
            cmd_router::CMD_TOKENS => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /tokens <text or file>".into()));
                }
                Ok(("tokens", serde_json::json!({"text": arg})))
            }
            cmd_router::CMD_BENCH => {
                if arg.is_empty() {
                    return Err(crate::error::Error::Tool("usage: /bench <safety|router>".into()));
                }
                Ok(("bench", serde_json::json!({"target": arg})))
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
             /profile           — Show last turn timing\n  \
             /tasks             — List background tasks\n\
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
             \nAppend & to run any tool in background (e.g. /shell sleep 5 &)\n\
             Pipe tools with | (e.g. /shell ls | /tokens)\n\
             \nAgent: {} | Model: {}",
            self.config.agent_name, self.config.model
        )
    }
}
