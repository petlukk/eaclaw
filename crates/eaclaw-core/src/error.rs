use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("config error: {0}")]
    Config(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("tool error: {0}")]
    Tool(String),

    #[error("safety violation: {0}")]
    Safety(String),

    #[error("channel error: {0}")]
    Channel(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
