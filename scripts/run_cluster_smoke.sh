#!/usr/bin/env bash
set -euo pipefail

# Example cluster smoke test. Adjust module/venv commands for your environment.
export HF_HOME="${HF_HOME:-/scratch/$USER/huggingface}"
export HF_HUB_DISABLE_XET=1

diffgate generate \
  --prompt "A red car on a snowy road" \
  --record-mode rich \
  --scorer training_free \
  --threshold 25 \
  --seed 12345 \
  --device cuda \
  --torch-dtype float16 \
  --save-trajectory \
  --output-dir outputs/cluster_smoke
