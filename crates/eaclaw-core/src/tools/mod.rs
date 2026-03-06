pub mod bench_tool;
pub mod calc;
pub mod cpu;
pub mod http;
pub mod json_tool;
pub mod ls;
pub mod memory;
pub mod read_file;
pub mod shell;
pub mod time;
pub mod tokens;
pub mod write_file;

use crate::config::Config;
use crate::error::Result;
use crate::llm::ToolDef;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// Tool trait — each tool provides a JSON schema and an execute method.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    async fn execute(&self, params: serde_json::Value) -> Result<String>;

    /// Whether this tool supports streaming output.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Execute with streaming output via callback.
    /// Default: calls execute() and sends result as one chunk.
    async fn execute_stream(
        &self,
        params: serde_json::Value,
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> Result<()> {
        let result = self.execute(params).await?;
        on_chunk(&result);
        Ok(())
    }

    fn as_tool_def(&self) -> ToolDef {
        ToolDef {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.parameters_schema(),
        }
    }
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Create a registry with all default tools (no host restrictions).
    pub fn with_defaults_open() -> Self {
        let mut reg = Self::new();
        reg.register(Arc::new(time::TimeTool));
        reg.register(Arc::new(calc::CalcTool));
        reg.register(Arc::new(http::HttpTool::new(Vec::new())));
        reg.register(Arc::new(shell::ShellTool));
        reg.register(Arc::new(memory::MemoryTool::new()));
        reg.register(Arc::new(read_file::ReadFileTool));
        reg.register(Arc::new(write_file::WriteFileTool));
        reg.register(Arc::new(ls::LsTool));
        reg.register(Arc::new(json_tool::JsonTool));
        reg.register(Arc::new(cpu::CpuTool));
        reg.register(Arc::new(tokens::TokensTool));
        reg.register(Arc::new(bench_tool::BenchTool));
        reg
    }

    /// Create a registry with all default tools using config.
    pub fn with_defaults(config: &Config) -> Self {
        let mut reg = Self::new();
        reg.register(Arc::new(time::TimeTool));
        reg.register(Arc::new(calc::CalcTool));
        reg.register(Arc::new(http::HttpTool::new(config.allowed_hosts.clone())));
        reg.register(Arc::new(shell::ShellTool));
        reg.register(Arc::new(memory::MemoryTool::new()));
        reg.register(Arc::new(read_file::ReadFileTool));
        reg.register(Arc::new(write_file::WriteFileTool));
        reg.register(Arc::new(ls::LsTool));
        reg.register(Arc::new(json_tool::JsonTool));
        reg.register(Arc::new(cpu::CpuTool));
        reg.register(Arc::new(tokens::TokensTool));
        reg.register(Arc::new(bench_tool::BenchTool));
        reg
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    pub fn tool_defs(&self) -> Vec<ToolDef> {
        self.tools.values().map(|t| t.as_tool_def()).collect()
    }

    pub fn list_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
