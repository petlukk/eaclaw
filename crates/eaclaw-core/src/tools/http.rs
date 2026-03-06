use super::Tool;
use async_trait::async_trait;
use futures::StreamExt;

pub struct HttpTool {
    /// Allowed hosts. Empty = allow all.
    allowed_hosts: Vec<String>,
}

impl HttpTool {
    pub fn new(allowed_hosts: Vec<String>) -> Self {
        Self { allowed_hosts }
    }

    fn check_host(&self, url: &str) -> crate::error::Result<()> {
        if self.allowed_hosts.is_empty() {
            return Ok(());
        }
        let host = url
            .split("://")
            .nth(1)
            .unwrap_or(url)
            .split('/')
            .next()
            .unwrap_or("")
            .split(':')
            .next()
            .unwrap_or("")
            .to_lowercase();
        if self.allowed_hosts.iter().any(|h| host == *h || host.ends_with(&format!(".{h}"))) {
            Ok(())
        } else {
            Err(crate::error::Error::Tool(format!(
                "host '{host}' not in allowed list. Set EACLAW_ALLOWED_HOSTS or ~/.eaclaw/allowed_hosts.txt"
            )))
        }
    }
}

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

        self.check_host(url)?;
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

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn execute_stream(
        &self,
        params: serde_json::Value,
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> crate::error::Result<()> {
        let url = params["url"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'url' parameter".into()))?;

        self.check_host(url)?;
        let response = reqwest::get(url).await?;
        let status = response.status();
        on_chunk(&format!("HTTP {status}\n"));

        let mut stream = response.bytes_stream();
        let mut total = 0usize;
        let max_len = 32 * 1024;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| crate::error::Error::Tool(format!("stream error: {e}")))?;
            if total >= max_len {
                break;
            }
            let remaining = max_len - total;
            let slice = if bytes.len() > remaining {
                &bytes[..remaining]
            } else {
                &bytes[..]
            };
            let text = String::from_utf8_lossy(slice);
            on_chunk(&text);
            total += slice.len();
        }

        if total >= max_len {
            on_chunk("... (truncated)");
        }

        Ok(())
    }
}
