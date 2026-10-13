#!/usr/bin/env bash
# Local-model setup, for the "run against a local model first" path.
# Run this on YOUR machine (not needed in CI/cloud if you later switch
# to a hosted provider).
set -euo pipefail

if ! command -v ollama &> /dev/null; then
  echo "Ollama not found. Install it from https://ollama.com/download, then re-run this script."
  exit 1
fi

MODEL="${1:-llama3.1}"

echo "Pulling model: $MODEL (this can take a while the first time)..."
ollama pull "$MODEL"

echo ""
echo "Done. Start the server with:  ollama serve"
echo "Then run the demo with:       python scripts/run_demo.py"
echo "Or the full experiment with:  python -m src.evaluation.run_experiment --provider ollama --model $MODEL"
