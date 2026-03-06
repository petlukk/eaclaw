#!/bin/bash
set -euo pipefail

EA_VERSION=v1.6.0
KERNEL_DIR=kernels
OUT_DIR=target/kernels
EA="${EA:-/root/dev/eacompute/target/release/ea}"

mkdir -p "$OUT_DIR"

echo "Building Eä $EA_VERSION kernels..."

for f in $KERNEL_DIR/*.ea; do
    stem=$(basename "$f" .ea)
    echo "  $stem.ea → lib${stem}.so"
    "$EA" "$f" --lib -o "$OUT_DIR/lib${stem}.so"
    "$EA" bind "$f" --rust
done

echo "Done. Kernels in $OUT_DIR/"
ls -la "$OUT_DIR/"

# Build WhatsApp bridge (requires Go)
if command -v go &>/dev/null; then
    echo ""
    echo "Building WhatsApp bridge..."
    (cd bridge && go build -o ../target/debug/eaclaw-bridge .)
    echo "  bridge/main.go → target/debug/eaclaw-bridge"
else
    echo ""
    echo "Skipping WhatsApp bridge (Go not installed)"
fi
