use eaclaw_core::agent::Agent;
use eaclaw_core::channel::repl::ReplChannel;
use eaclaw_core::config::Config;
use eaclaw_core::llm::anthropic::AnthropicProvider;
use eaclaw_core::safety::SafetyLayer;
use eaclaw_core::tools::ToolRegistry;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Configuration error: {e}");
            eprintln!("Set ANTHROPIC_API_KEY to get started.");
            std::process::exit(1);
        }
    };

    let llm: Arc<dyn eaclaw_core::llm::LlmProvider> =
        Arc::new(AnthropicProvider::new(&config));

    let tools = ToolRegistry::with_defaults(&config);
    let safety = SafetyLayer::new();
    let channel = ReplChannel::new(&config.agent_name);

    let mut agent = Agent::new(config, llm, tools, safety);

    let result = agent.run(&channel).await;

    // Restore terminal state before exiting — the readline thread may
    // still be blocked on input with raw mode enabled.
    channel.shutdown();

    if let Err(e) = result {
        eprintln!("Agent error: {e}");
        std::process::exit(1);
    }

    std::process::exit(0);
}
