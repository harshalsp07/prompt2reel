#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

# Optional ffmpeg binary is usually present in Colab, but keep fallback.
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update && apt-get install -y ffmpeg
fi

echo "✅ Colab environment setup complete"
