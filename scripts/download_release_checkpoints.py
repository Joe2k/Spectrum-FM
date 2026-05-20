#!/usr/bin/env python3
"""Re-download official release checkpoints from W&B using MANIFEST.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RELEASE = REPO / "checkpoints" / "release"
MANIFEST = RELEASE / "MANIFEST.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None, help="Download one model; default: all in MANIFEST")
    args = parser.parse_args()

    if not MANIFEST.is_file():
        print(f"Missing {MANIFEST}. Run scripts/setup_release_checkpoints.py first.", file=sys.stderr)
        return 1

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass

    try:
        import wandb
    except ImportError:
        print("wandb not installed", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST.read_text())
    api = wandb.Api()
    models = manifest.get("models", {})
    ids = [args.model_id] if args.model_id else list(models.keys())

    for model_id in ids:
        if model_id not in models:
            print(f"Unknown model_id: {model_id}", file=sys.stderr)
            return 1
        uri = models[model_id]["wandb_artifact"]
        out_dir = RELEASE / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        art = api.artifact(uri, type="model")
        art.download(root=str(out_dir))
        pts = sorted(out_dir.glob("*.pt"))
        if not pts:
            print(f"No .pt in {out_dir} after download", file=sys.stderr)
            return 1
        best = out_dir / "best.pt"
        if pts[0] != best and best.exists():
            best.unlink()
        if not best.exists():
            pts[0].rename(best)
        print(f"[wandb] {model_id} -> {best}")

    print("Done. For Git LFS commit, ensure best.pt are real files (not symlinks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
