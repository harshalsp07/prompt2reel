#!/usr/bin/env python3
"""Download/cache WAN model files from Hugging Face for Colab/local usage."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and validate WAN model snapshot from Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. Wan-AI/Wan2.2-T2V")
    parser.add_argument("--revision", default="main", help="Repo revision/branch/tag/commit")
    parser.add_argument("--local-dir", default="models/wan22", help="Directory to store model files")
    parser.add_argument(
        "--allow-patterns",
        default="*.json,*.safetensors,*.bin,*.txt,*.model,*.py",
        help="Comma-separated allow patterns for snapshot download",
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN", ""), help="HF token for gated/private repos")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        print(f"❌ Missing dependency huggingface_hub: {exc}")
        print("Install with: pip install huggingface_hub")
        return 2

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]

    print(f"ℹ️ Checking access to repo: {args.repo_id}@{args.revision}")
    api = HfApi(token=args.token or None)
    try:
        _ = api.model_info(repo_id=args.repo_id, revision=args.revision)
    except Exception as exc:
        print("❌ Cannot access model repo.")
        print("Common causes: wrong repo id, gated model without token, token missing scope.")
        print(f"Details: {exc}")
        return 3

    print(f"ℹ️ Downloading snapshot to: {local_dir}")
    try:
        snapshot_path = snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            token=args.token or None,
            resume_download=True,
        )
    except Exception as exc:
        print("❌ Snapshot download failed.")
        print(f"Details: {exc}")
        return 4

    print(f"✅ Snapshot downloaded at: {snapshot_path}")

    config_file = local_dir / "model_index.json"
    if not config_file.exists():
        print("⚠️ model_index.json not found in local dir.")
        print("This may still work for some custom repos, but verify pipeline compatibility.")
    else:
        print("✅ Found model_index.json (Diffusers-compatible layout detected).")

    print("\nSet this before app launch:")
    print(f"export WAN_MODEL_ID={local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
