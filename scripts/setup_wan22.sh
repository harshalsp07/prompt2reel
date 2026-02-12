#!/usr/bin/env bash
set -euo pipefail

# Colab-friendly WAN setup helper.
# Usage:
#   HF_TOKEN=hf_xxx WAN_MODEL_ID=your/repo bash scripts/setup_wan22.sh
# Optional:
#   WAN_REVISION=main WAN_LOCAL_DIR=models/wan22 WAN_ALLOW_PATTERNS='*.json,*.safetensors,*.bin,*.txt'

WAN_MODEL_ID="${WAN_MODEL_ID:-}"
HF_TOKEN="${HF_TOKEN:-}"
WAN_REVISION="${WAN_REVISION:-main}"
WAN_LOCAL_DIR="${WAN_LOCAL_DIR:-models/wan22}"
WAN_ALLOW_PATTERNS="${WAN_ALLOW_PATTERNS:-*.json,*.safetensors,*.bin,*.txt,*.model,*.py}"

if [[ -z "$WAN_MODEL_ID" ]]; then
  echo "❌ WAN_MODEL_ID is required (e.g. WAN_MODEL_ID=Wan-AI/Wan2.2-T2V)"
  exit 1
fi

python scripts/setup_wan22.py \
  --repo-id "$WAN_MODEL_ID" \
  --revision "$WAN_REVISION" \
  --local-dir "$WAN_LOCAL_DIR" \
  --allow-patterns "$WAN_ALLOW_PATTERNS" \
  ${HF_TOKEN:+--token "$HF_TOKEN"}

cat <<MSG

✅ WAN setup complete.

Use one of these before launching app:
  export WAN_MODEL_ID="$WAN_LOCAL_DIR"
  # or keep WAN_MODEL_ID as HF repo id if you prefer remote loading.

MSG
