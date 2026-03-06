use super::Tool;
use async_trait::async_trait;

pub struct HttpTool;

#[async_trait]
impl Tool for HttpTool {
    fn name(&self) -> &str {
        "http"
    }

    fn description(&self) -> &str {
        "Make an HTTP GET request and return the response body."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let url = params["url"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'url' parameter".into()))?;

        let response = reqwest::get(url).await?;
        let status = response.status();
        let body = response.text().await?;

        // Truncate large responses
        let max_len = 32 * 1024;
        let body = if body.len() > max_len {
            format!("{}... (truncated, {} bytes total)", &body[..max_len], body.len())
        } else {
            body
        };

        Ok(format!("HTTP {status}\n{body}"))
    }
}
