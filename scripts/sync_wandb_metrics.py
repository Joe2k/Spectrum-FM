#!/usr/bin/env python3
"""
Refresh checkpoints/release/MANIFEST.json metrics from W&B.

Requires WANDB_API_KEY in .env or environment. Uses the same entity/project
and run names as the release artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RELEASE = REPO / "checkpoints" / "release"
MANIFEST_PATH = RELEASE / "MANIFEST.json"

ENTITY = "jjayaseelan-university-of-san-francisco"
PROJECT = "redshifty"

RUNS = {
    "approach_a_results": "fm_v1_10k_a_ddp4_rw10_v8",
    "approach_b_results": "phase10_mask50_b",
    "reference_runs.phase10_mask50_a_big": "phase10_mask50_a_big",
}


def _parse_summary(summary_metrics: str | dict) -> dict:
    if isinstance(summary_metrics, dict):
        return summary_metrics
    return json.loads(summary_metrics)


def _fetch_run(api, display_name: str) -> dict:
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": display_name})
    for r in runs:
        if r.display_name == display_name:
            return r
    raise RuntimeError(f"Run not found: {display_name}")


def _metrics_block(run, prefix: str = "") -> dict:
    s = _parse_summary(run.summary)
    p = prefix
    return {
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "wandb_run_state": run.state,
        "wandb_run_url": run.url,
        "steps": int(s.get(f"{p}_step", s.get("_step", 0))),
        "val_loss": float(s.get(f"{p}val/loss_total", s.get(f"{p}val/loss", 0))),
        "val_redshift_acc_teacher_forced": float(s.get(f"{p}val/redshift_acc", 0)),
        "val_redshift_acc_autoregressive": float(s.get(f"{p}val_ar/ar_redshift_acc", 0)),
        "val_spectrum_acc_teacher_forced": float(s.get(f"{p}val/spectrum_acc", 0)),
        "val_spectrum_acc_autoregressive": float(s.get(f"{p}val_ar/ar_spectrum_acc", 0)),
        "val_masked_spec_acc_teacher_forced": float(s.get(f"{p}val/masked_spec_acc", 0)),
        "val_ar_n_samples": int(s.get(f"{p}val_ar/n_samples", 0)),
    }


def main() -> int:
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

    api = wandb.Api()
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.is_file() else {}
    manifest.setdefault("wandb_entity", ENTITY)
    manifest.setdefault("wandb_project", PROJECT)

    a_run = _fetch_run(api, RUNS["approach_a_results"])
    b_run = _fetch_run(api, RUNS["approach_b_results"])
    ref_run = _fetch_run(api, RUNS["reference_runs.phase10_mask50_a_big"])

    manifest["metrics_source"] = "W&B API (sync_wandb_metrics.py)"
    manifest["approach_a_results"] = {
        "display_name": "Approach A — primary release model",
        "training_run": a_run.name,
        **_metrics_block(a_run),
        "encoder_mask_ratio": float(a_run.config.get("encoder_mask_ratio", 0.5)),
        "redshift_loss_weight": float(a_run.config.get("redshift_loss_weight", 1)),
        "batch_size": int(a_run.config.get("batch_size", 32)),
        "lr": float(a_run.config.get("lr", 0)),
        "manifest": Path(str(a_run.config.get("manifest", ""))).name,
        "note": "Synced from W&B; matches release artifact approach_a_fm_v1_10k_a_ddp4_rw10_v8:best",
    }
    manifest["approach_b_results"] = {
        "display_name": "Approach B (encoder never sees z) — metrics only",
        "training_run": b_run.name,
        **_metrics_block(b_run),
        "encoder_mask_ratio": float(b_run.config.get("encoder_mask_ratio", 0.5)),
        "redshift_loss_weight": float(b_run.config.get("redshift_loss_weight", 50)),
        "batch_size": int(b_run.config.get("batch_size", 8)),
        "lr": float(b_run.config.get("lr", 0)),
        "manifest": Path(str(b_run.config.get("manifest", ""))).name,
        "conclusion": "Encoder does not learn redshift from spectrum alone; contrast with Approach A release model.",
        "note": "No checkpoint shipped — metrics only.",
    }
    manifest["reference_runs"] = {
        "phase10_mask50_a_big": {
            "training_run": ref_run.name,
            **_metrics_block(ref_run),
            "encoder_mask_ratio": float(ref_run.config.get("encoder_mask_ratio", 0.5)),
            "redshift_loss_weight": float(ref_run.config.get("redshift_loss_weight", 50)),
            "batch_size": int(ref_run.config.get("batch_size", 32)),
            "note": "Phase-10 writeup run; not the shipped release checkpoint.",
        }
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Updated {MANIFEST_PATH}")
    print(f"  A: z_TF={manifest['approach_a_results']['val_redshift_acc_teacher_forced']:.3f} "
          f"z_AR={manifest['approach_a_results']['val_redshift_acc_autoregressive']:.3f}")
    print(f"  B: z_TF={manifest['approach_b_results']['val_redshift_acc_teacher_forced']:.3f} "
          f"z_AR={manifest['approach_b_results']['val_redshift_acc_autoregressive']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
