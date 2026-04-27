#!/usr/bin/env bash
# Run the full pipeline: generate → audit → analyze → visualize.
# Usage: ./scripts/run_pipeline.sh configs/mvp.yaml
set -euo pipefail

CONFIG="${1:-configs/mvp.yaml}"

echo ">>> [1/4] Generating counterfactuals (config: $CONFIG)"
cap-generate --config "$CONFIG"

echo ">>> [2/4] Auditing"
cap-audit --config "$CONFIG"

echo ">>> [3/4] Analyzing"
cap-analyze --config "$CONFIG"

echo ">>> [4/4] Visualizing"
cap-visualize --config "$CONFIG"

echo ">>> Pipeline complete."
