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

    if let Err(e) = eaclaw_core::kernels::init() {
        eprintln!("Failed to initialize SIMD kernels: {e}");
        std::process::exit(1);
    }

    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Configuration error: {e}");
            eprintln!("Set ANTHROPIC_API_KEY to get started.");
            std::process::exit(1);
        }
    };

    // Check for --whatsapp mode
    let args: Vec<String> = std::env::args().collect();
    let whatsapp_mode = args.iter().any(|a| a == "--whatsapp");

    if whatsapp_mode {
        run_whatsapp(&config).await;
    } else {
        run_repl(&config).await;
    }
}

async fn run_repl(config: &Config) {
    let llm: Arc<dyn eaclaw_core::llm::LlmProvider> =
        Arc::new(AnthropicProvider::new(config));

    let tools = ToolRegistry::with_defaults(config, llm.clone());
    let safety = SafetyLayer::new();
    let channel = ReplChannel::new(&config.agent_name);

    let mut agent = Agent::new(config.clone(), llm, tools, safety);

    let result = agent.run(&channel).await;

    // Restore terminal state before exiting
    channel.shutdown();

    if let Err(e) = result {
        eprintln!("Agent error: {e}");
        std::process::exit(1);
    }

    std::process::exit(0);
}

async fn run_whatsapp(config: &Config) {
    let llm: Arc<dyn eaclaw_core::llm::LlmProvider> =
        Arc::new(AnthropicProvider::new(config));

    let tools = ToolRegistry::with_defaults(config, llm.clone());

    // Bridge path: check env, then look next to the binary, then PATH
    let bridge_path = std::env::var("EACLAW_BRIDGE_PATH").unwrap_or_else(|_| {
        // Look next to our binary
        if let Ok(exe) = std::env::current_exe() {
            let sibling = exe.parent().unwrap().join("eaclaw-bridge");
            if sibling.exists() {
                return sibling.to_string_lossy().into_owned();
            }
        }
        "eaclaw-bridge".into()
    });

    let session_dir = std::env::var("EACLAW_WA_SESSION_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME not set");
        format!("{home}/.eaclaw/whatsapp")
    });

    eprintln!("Starting WhatsApp bridge: {bridge_path}");
    eprintln!("Session directory: {session_dir}");

    if let Err(e) =
        eaclaw_core::channel::wa_loop::run(&bridge_path, &session_dir, config, llm, &tools).await
    {
        eprintln!("WhatsApp error: {e}");
        std::process::exit(1);
    }
}
