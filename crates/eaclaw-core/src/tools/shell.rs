use super::Tool;
use async_trait::async_trait;
use tokio::process::Command;

pub struct ShellTool;

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return stdout/stderr."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let cmd = params["command"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'command' parameter".into()))?;

        let output = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
            .await
            .map_err(|e| crate::error::Error::Tool(format!("failed to execute: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("[stderr] ");
            result.push_str(&stderr);
        }

        if result.is_empty() {
            result.push_str(&format!("(exit code: {})", output.status.code().unwrap_or(-1)));
        }

        Ok(result)
    }
}
