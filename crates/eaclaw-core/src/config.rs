use crate::error::{Error, Result};
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub api_key: String,
    pub model: String,
    pub agent_name: String,
    pub max_turns: usize,
    pub command_prefix: String,
}

impl Config {
    /// Load configuration from environment variables.
    pub fn from_env() -> Result<Self> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .map_err(|_| Error::Config("ANTHROPIC_API_KEY not set".into()))?;

        let model = env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());

        let agent_name = env::var("AGENT_NAME")
            .unwrap_or_else(|_| "eaclaw".into());

        let max_turns = env::var("MAX_TURNS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        let command_prefix = env::var("COMMAND_PREFIX")
            .unwrap_or_else(|_| "/".into());

        Ok(Self {
            api_key,
            model,
            agent_name,
            max_turns,
            command_prefix,
        })
    }
}
