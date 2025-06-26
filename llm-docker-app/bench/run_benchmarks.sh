#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "phi-1_5-8bit"
  "phi-1_5-pruned-20pct"
  "phi-8B-mps"
)

for M in "${MODELS[@]}"; do
  echo "=== Benchmarking $M ==="
  python3 bench/benchmark_inference.py \
    --model-path "/app/quantized/$M" \
    --runs 50 \
    --warmup 10
  echo
done

echo "=== All benchmarks complete ==="
