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
    # v2 ships final.pt (step ~80k, χ²≈1.00), NOT the best-val W&B artifact
    # (step 57.5k): best.pt was selected on val/total, which mixes in the
    # capped entropy reward, so the genuinely best reconstructor is the
    # fully-annealed end-of-run state. final.pt is not a W&B artifact, so it
    # is sourced from a local file staged under wandb_artifacts/ (see local_pt).
    "spectrum_tokenizer_v2": {
        "model_id": "spectrum_tokenizer_v2",
        "display_name": "Spectrum-FM Spectrum Tokenizer v2 (Huber ivar-NLL, wavelength-aware, balanced DR1)",
        "wandb_artifact": (
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/tokenizer_tokenizer_v2_3k_v3 "
            "(run waotf2n0, final.pt @ step ~80k; not the best-val artifact)"
        ),
        "role": "spectrum_tokenizer",
        "local_pt": "checkpoints/wandb_artifacts/spectrum_tokenizer_v2_final/best.pt",
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
    # V2/V3 transformers (v2 tokens). best.pt is the run's best-val checkpoint
    # stripped of optimizer/scaler (~400MB vs 1.2GB) and committed directly via
    # git-LFS — there is no W&B artifact, so these are sourced from a staged
    # local file (NERSC $SCRATCH/deepsrch/checkpoints/<run>/best.pt → strip → LFS).
    "transformer_v2_512hard": {
        "model_id": "transformer_v2_512hard",
        "display_name": "Spectrum-FM Transformer V2 — 512-bin hard redshift (v2 tokens, masked-targets X2)",
        "wandb_artifact": f"{WANDB_ENTITY}/{WANDB_PROJECT}/approach_a_v2cache_x2_512hard_ctrl_ddp4 (run cd1ikb99, best.pt stripped of optimizer)",
        "role": "transformer",
        "approach": "a",
        "requires_tokenizer": "spectrum_tokenizer_v2",
        "local_pt": "checkpoints/release/transformer_v2_512hard/best.pt",
    },
    "transformer_v3_4096soft": {
        "model_id": "transformer_v3_4096soft",
        "display_name": "Spectrum-FM Transformer V3 — 4096-bin soft-label redshift (v2 tokens, X2+X3)",
        "wandb_artifact": f"{WANDB_ENTITY}/{WANDB_PROJECT}/approach_a_v2cache_x2x3_ddp4 (run aqxmwgl1, best.pt stripped of optimizer)",
        "role": "transformer",
        "approach": "a",
        "requires_tokenizer": "spectrum_tokenizer_v2",
        "local_pt": "checkpoints/release/transformer_v3_4096soft/best.pt",
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
        try:
            if meta.get("local_pt"):
                # Local-file source (e.g. a checkpoint that was never pushed
                # to W&B, like tokenizer v2's final.pt). Stage it there first.
                src_pt = (REPO / meta["local_pt"]).resolve()
                if not src_pt.is_file():
                    raise FileNotFoundError(
                        f"local checkpoint {src_pt} not found — stage it there "
                        f"(e.g. copy final.pt off NERSC, renamed to best.pt) "
                        f"before running with --copy")
            else:
                src_pt = _find_wandb_pt(artifact_name)
        except FileNotFoundError as e:
            # Only some artifacts may be downloaded locally; skip the rest
            # instead of aborting (e.g. you pulled v9 but not the older v8).
            print(f"skip {model_id}: {e}", file=sys.stderr)
            continue
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
        for key in ("approach_a_results", "approach_a_v8_results",
                    "approach_b_results", "reference_runs", "metrics_source"):
            if key in old:
                manifest[key] = old[key]
    (RELEASE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {RELEASE / 'MANIFEST.json'}")
    print("  Tip: run scripts/sync_wandb_metrics.py to refresh metrics from W&B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
