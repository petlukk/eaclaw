#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/petlukk/eaclaw"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_DIR="$HOME/.eaclaw/models"
MODEL_FILE="$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"

echo "=== eaclaw setup ==="

# 1. Clone (with submodules for local mode)
if [ ! -d eaclaw ]; then
    git clone --recursive "$REPO"
else
    echo "eaclaw/ already exists, skipping clone"
fi
cd eaclaw

# 2. Build
if command -v cargo &>/dev/null; then
    cargo build --features local-llm
else
    echo "Error: Rust toolchain not found. Install from https://rustup.rs"
    exit 1
fi

# 3. Download model
if [ ! -f "$MODEL_FILE" ]; then
    mkdir -p "$MODEL_DIR"
    echo "Downloading model (~1.8 GB)..."
    wget -O "$MODEL_FILE" "$MODEL_URL"
else
    echo "Model already downloaded"
fi

echo ""
echo "=== Ready ==="
echo "  Cloud:  ANTHROPIC_API_KEY=sk-ant-... cargo run"
echo "  Local:  EACLAW_BACKEND=local cargo run --features local-llm"
