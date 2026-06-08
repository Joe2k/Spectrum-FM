#!/usr/bin/env python3
"""
Populate checkpoints/release/ from W&B artifact downloads.

Copies best.pt into named model directories and writes MANIFEST.json
plus per-model config.json files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
WANDB_ARTIFACTS = REPO / "checkpoints" / "wandb_artifacts"
RELEASE = REPO / "checkpoints" / "release"

WANDB_ENTITY = "jjayaseelan-university-of-san-francisco"
WANDB_PROJECT = "redshifty"

# W&B artifact folder suffix -> release model_id
ARTIFACT_MAP = {
    "spectrum_tokenizer_v1": {
        "model_id": "spectrum_tokenizer_v1",
        "display_name": "Spectrum-FM Spectrum Tokenizer v1",
        "wandb_artifact": f"{WANDB_ENTITY}/{WANDB_PROJECT}/spectrum_tokenizer_v1:best",
        "role": "spectrum_tokenizer",
    },
    "approach_a_fm_v1_10k_a_ddp4_redmask50_v9": {
        "model_id": "approach_a_fm_v1_10k_a_ddp4_redmask50_v9",
        "display_name": "Spectrum-FM Transformer — Approach A (redshift conditioning dropout)",
        "wandb_artifact": f"{WANDB_ENTITY}/{WANDB_PROJECT}/approach_a_fm_v1_10k_a_ddp4_redmask50_v9:best",
        "role": "transformer",
        "approach": "a",
        "requires_tokenizer": "spectrum_tokenizer_v1",
    },
    "approach_a_fm_v1_10k_a_ddp4_rw10_v8": {
        "model_id": "transformer_approach_a_fm_v1_10k_ddp4_rw10_v8",
        "display_name": "Spectrum-FM Transformer — Approach A v8 (superseded)",
        "wandb_artifact": f"{WANDB_ENTITY}/{WANDB_PROJECT}/approach_a_fm_v1_10k_a_ddp4_rw10_v8:best",
        "role": "transformer",
        "approach": "a",
        "requires_tokenizer": "spectrum_tokenizer_v1",
    },
}


def _find_wandb_pt(artifact_name: str) -> Path:
    pattern = f"*__redshifty__{artifact_name}@best"
    matches = list(WANDB_ARTIFACTS.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No wandb download for {artifact_name!r} under {WANDB_ARTIFACTS}. "
            f"Expected folder matching {pattern}"
        )
    pt = matches[0] / "best.pt"
    if not pt.is_file():
        raise FileNotFoundError(f"Missing {pt}")
    return pt


def _infer_transformer_config(sd: dict) -> dict:
    enc_idx = {int(k.split(".")[1]) for k in sd if k.startswith("encoder_layers.")}
    dec_idx = {int(k.split(".")[1]) for k in sd if k.startswith("decoder_layers.")}
    d_model = int(sd["token_embedding.weight"].shape[1])
    n_heads = d_model // 64
    return {
        "d_model": d_model,
        "n_encoder_layers": len(enc_idx),
        "n_decoder_layers": len(dec_idx),
        "n_heads": n_heads,
        "dropout": 0.0,
    }


def _install_checkpoint(src_pt: Path, dst_pt: Path, copy: bool) -> None:
    if dst_pt.exists() or dst_pt.is_symlink():
        dst_pt.unlink()
    if copy:
        shutil.copy2(src_pt, dst_pt)
        print(f"copied {src_pt} -> {dst_pt}")
    else:
        dst_pt.symlink_to(src_pt.resolve())
        print(f"symlinked {dst_pt} -> {src_pt}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy .pt files into release/ (needs ~1.4 GB free). "
        "Default: symlink to wandb_artifacts/ to save disk.",
    )
    args = parser.parse_args()

    if not WANDB_ARTIFACTS.is_dir():
        print(f"ERROR: {WANDB_ARTIFACTS} not found", file=sys.stderr)
        return 1

    RELEASE.mkdir(parents=True, exist_ok=True)
    models_manifest = {}

    for artifact_name, meta in ARTIFACT_MAP.items():
        model_id = meta["model_id"]
        src_pt = _find_wandb_pt(artifact_name)
        out_dir = RELEASE / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dst_pt = out_dir / "best.pt"
        _install_checkpoint(src_pt, dst_pt, copy=args.copy)
        load_path = dst_pt.resolve()

        cfg = {
            "model_id": model_id,
            "display_name": meta["display_name"],
            "wandb_artifact": meta["wandb_artifact"],
            "checkpoint": f"{model_id}/best.pt",
            "role": meta["role"],
        }

        if meta["role"] == "transformer":
            ckpt = torch.load(load_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model", ckpt)
            cfg.update(_infer_transformer_config(sd))
            cfg["approach"] = ckpt.get("approach", meta.get("approach", "a"))
            cfg["encoder_mask_ratio"] = float(ckpt.get("encoder_mask_ratio", 0.0))
            cfg["val_loss"] = float(ckpt.get("val_loss", float("nan")))
            cfg["step"] = int(ckpt.get("step", 0))
            if ckpt.get("z_tokenizer") is not None:
                cfg["has_embedded_z_tokenizer"] = True
            if meta.get("requires_tokenizer"):
                cfg["requires_tokenizer"] = meta["requires_tokenizer"]
            models_manifest[model_id] = {
                "display_name": meta["display_name"],
                "wandb_artifact": meta["wandb_artifact"],
                "checkpoint": cfg["checkpoint"],
                "approach": cfg["approach"],
                "requires_tokenizer": meta.get("requires_tokenizer"),
            }
        else:
            models_manifest[model_id] = {
                "display_name": meta["display_name"],
                "wandb_artifact": meta["wandb_artifact"],
                "checkpoint": cfg["checkpoint"],
                "role": "spectrum_tokenizer",
            }

        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    manifest = {
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "models": models_manifest,
        "default_tokenizer": "spectrum_tokenizer_v1",
        "default_transformer": "approach_a_fm_v1_10k_a_ddp4_redmask50_v9",
    }
    # Preserve metrics if MANIFEST already synced from W&B; else write stub.
    existing = RELEASE / "MANIFEST.json"
    if existing.is_file():
        old = json.loads(existing.read_text())
        for key in ("approach_a_results", "approach_b_results", "reference_runs", "metrics_source"):
            if key in old:
                manifest[key] = old[key]
    (RELEASE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {RELEASE / 'MANIFEST.json'}")
    print("  Tip: run scripts/sync_wandb_metrics.py to refresh metrics from W&B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
