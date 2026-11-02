#!/usr/bin/env bash
# Download Hugging Face tokenizer files used for exact token counts (ARCHITECTURE.md §11).
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/tokenizers"
mkdir -p "$DEST"
fetch() {  # <family> <hf repo>
  echo "Downloading $1 tokenizer from $2"
  curl -fsSL "https://huggingface.co/$2/resolve/main/tokenizer.json" -o "$DEST/$1.json"
}
fetch qwen35 Qwen/Qwen3.5-4B
ls -l "$DEST"
