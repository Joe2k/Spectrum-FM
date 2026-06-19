#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained SpectrumTransformer checkpoint on the DESI val split and
report physical redshift metrics (σ_NMAD of Δz/(1+z), catastrophic outlier
fraction, bias) — the AION-comparable numbers, not just bin accuracy.

Path-agnostic: ``--checkpoint`` accepts any full-state checkpoint (NERSC scratch
best.pt / last.pt of a live run, or a local checkpoints/release/*.pt). The model
dims and the redshift tokenizer are restored straight from the checkpoint, so no
config or hyperparameters need to be supplied.

The redshift token is decoded two ways:
  - expected-value (probability-weighted, sub-bin precision) — the headline,
  - argmax (single most-likely bin) — for the sub-bin-gain delta.

Honest regime by default (--redshift-mask-ratio 1.0): the encoder's redshift
token is hidden, so the metrics reflect predict-z-from-spectrum at inference.

Usage (cached X1 tokens):

    python nersc/eval_redshift.py \\
        --checkpoint $SCRATCH/deepsrch/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt \\
        --tokenized-dir $SCRATCH/dr1_tokenized_v2 \\
        --manifest $SCRATCH/manifests/dr1_v2_full.jsonl \\
        --max-batches 50

Usage (on-the-fly tokenization):

    python nersc/eval_redshift.py --checkpoint <ckpt> \\
        --tokenizer-ckpt <spectrum tokenizer best.pt> --manifest <manifest>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src.eval.redshift_metrics import CATASTROPHIC_DZ  # noqa: E402
from src.eval.redshift_uncertainty import (  # noqa: E402
    DEFAULT_REJECTION_SCORE,
    apply_pit_recalibration,
    apply_temperature,
    coverage_quality_curve,
    fit_pit_recalibrator,
    fit_temperature,
    outlier_auroc,
    pit_calibration_error,
    pit_histogram,
    pit_values,
    uncertainty_scores,
)
from src.inference.release import _infer_transformer_dims, _restore_z_tokenizer  # noqa: E402
from src.models.transformer import SpectrumTransformer  # noqa: E402
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.training.data_split import split_records_by_healpix  # noqa: E402
from src.training.eval import evaluate  # noqa: E402

from dr1_dataset import (  # noqa: E402
    DR1IndexedDataset,
    collate_dr1_skip_none,
    load_manifest,
)
from dr1_tokenized_dataset import (  # noqa: E402
    DR1CachedTokenDataset,
    collate_cached_skip_none,
)
from train_transformer import _records_to_shards  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Redshift σ_NMAD eval for a checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Full-state checkpoint (.pt) — NERSC scratch or local release.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tokenized-dir", type=Path, default=None,
                   help="X1 token cache (cached path; spectrum tokenizer not loaded).")
    p.add_argument("--tokenizer-ckpt", type=Path, default=None,
                   help="SpectrumTokenizer .pt for on-the-fly tokenization "
                        "(unused with --tokenized-dir).")
    p.add_argument("--approach", default="a", choices=["a", "b"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--redshift-mask-ratio", type=float, default=1.0,
                   help="1.0 = honest predict-z-from-spectrum (default); "
                        "0.0 = z visible (copy-path upper bound).")
    p.add_argument("--encoder-mask-ratio", type=float, default=0.0,
                   help="Spectrum encoder masking at eval (default 0 = full spectrum).")
    p.add_argument("--healpix-holdout-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--uncertainty", action="store_true",
                   help="Also report the quality-vs-coverage curve, outlier-"
                        "detection AUROC per score, and PIT calibration.")
    p.add_argument("--temperature", type=float, default=None,
                   help="With --uncertainty: temperature for the calibrated "
                        "posterior. Default None = fit T on this val split "
                        "(X8b); pass a value to apply a known T instead.")
    p.add_argument("--dump-npz", type=Path, default=None,
                   help="With --uncertainty: write per-sample dz/scores/pit "
                        "to this .npz for notebook plotting.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model + z tokenizer from the checkpoint ----
    print(f"[eval] loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    dims = _infer_transformer_dims(sd)
    print(f"[eval] dims={dims}")
    model = SpectrumTransformer(**dims).to(device)
    model.load_state_dict(sd)
    model.eval()
    z_tok = _restore_z_tokenizer(ckpt)
    print(f"[eval] z_tok n_levels={z_tok.n_levels} "
          f"z range [{z_tok._min_z:.4f}, {z_tok._max_z:.4f}]")

    # ---- spectrum tokenizer (only for on-the-fly path) ----
    cached = args.tokenized_dir is not None
    if cached:
        spec_tok = None
        print(f"[eval] cached tokens from {args.tokenized_dir} (encoder not loaded)")
    else:
        if args.tokenizer_ckpt is None:
            print("ERROR: --tokenizer-ckpt required without --tokenized-dir", file=sys.stderr)
            sys.exit(1)
        spec_tok = SpectrumTokenizer().to(device)
        tck = torch.load(args.tokenizer_ckpt, map_location=device, weights_only=False)
        spec_tok.load_state_dict(tck.get("model", tck) if isinstance(tck, dict) else tck)
        spec_tok.eval()
        for pm in spec_tok.parameters():
            pm.requires_grad_(False)

    # ---- val loader (same healpix split as training) ----
    records = load_manifest(args.manifest)
    _, val_records = split_records_by_healpix(
        records, holdout_frac=args.healpix_holdout_frac, seed=args.seed)
    val_cap = None if args.max_spectra is None else max(50, args.max_spectra // 10)
    if cached:
        collate = collate_cached_skip_none
        val_ds = DR1CachedTokenDataset(
            _records_to_shards(val_records, args.tokenized_dir),
            require_good_zwarn=True, require_nonzero_flux=True, max_spectra=val_cap)
    else:
        collate = collate_dr1_skip_none
        val_ds = DR1IndexedDataset(
            val_records, require_good_zwarn=True, require_nonzero_flux=True,
            max_spectra=val_cap)
    print(f"[eval] val_ds={len(val_ds)} ({len(val_records)} healpix)")
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    v = evaluate(
        model, val_loader, spec_tok, z_tok, args.approach, device,
        amp=args.amp, redshift_weight=1.0,
        encoder_mask_ratio=args.encoder_mask_ratio,
        redshift_mask_ratio=args.redshift_mask_ratio,
        max_batches=args.max_batches, amp_dtype=amp_dtype,
        return_per_sample=args.uncertainty)

    regime = "honest (z hidden)" if args.redshift_mask_ratio >= 1.0 else \
        f"z-mask={args.redshift_mask_ratio}"
    print("\n" + "=" * 56)
    print(f"  Redshift eval — {args.checkpoint.name}  [{regime}]")
    print("=" * 56)
    print(f"  bin accuracy        : {v['redshift_acc']:.4f}")
    print(f"  bin within-2        : {v['redshift_acc_within2']:.4f}")
    print(f"  σ_NMAD (expected)   : {v['z_nmad']:.6f}")
    print(f"  σ_NMAD (argmax)     : {v['z_nmad_argmax']:.6f}")
    print(f"  catastrophic η>.0033: {v['z_outlier_frac']:.4f}")
    print(f"  outliers   η>.05    : {v['z_outlier_frac_05']:.4f}")
    print(f"  bias  median(Δz)    : {v['z_bias']:+.6f}")
    print(f"  spectrum_acc        : {v['spectrum_acc']:.4f}")
    print("=" * 56)

    if args.uncertainty:
        _report_uncertainty(v, args, z_tok)


def _coverage_table(z_pred, z_true, score):
    print(f"  {'retained':>9} {'σ_NMAD':>10} {'η>.0033':>9} {'η>.05':>8} {'n':>7}")
    for r in coverage_quality_curve(z_pred, z_true, score):
        print(f"  {r['retained_frac']*100:8.0f}% {r['sigma_nmad']:10.6f} "
              f"{r['outlier_frac']*100:8.2f}% {r['outlier_frac_05']*100:7.2f}% "
              f"{r['n']:7d}")


def _report_uncertainty(v, args, z_tok):
    """Coverage curve + per-score outlier AUROC + PIT calibration (X8) and the
    X8b temperature-calibration step (fit/apply T, before→after PIT)."""
    z_pred, z_true = v["z_pred"], v["z_true"]
    scores, pit, probs = v["scores"], v["pit"], v["probs"]
    dz = (z_pred - z_true) / (1.0 + z_true)
    is_outlier = dz.abs() > CATASTROPHIC_DZ

    print(f"\n  Reject most-uncertain by {DEFAULT_REJECTION_SCORE}:")
    _coverage_table(z_pred, z_true, scores[DEFAULT_REJECTION_SCORE])

    print(f"\n  Outlier-detection AUROC (η>{CATASTROPHIC_DZ}):")
    for name, s in scores.items():
        print(f"    {name:18s}: {outlier_auroc(s, is_outlier):.4f}")

    ks = pit_calibration_error(pit)
    hist = pit_histogram(pit, bins=10)
    bars = " ".join(f"{int(c)}" for c in hist)
    print(f"\n  PIT calibration KS : {ks:.4f}  (0 = uniform = calibrated)")
    print(f"  PIT histogram (10) : {bars}")

    # ---- X8b: temperature calibration -------------------------------------
    if args.temperature is not None:
        T = args.temperature
        how = "applied"
    else:
        T = fit_temperature(probs, z_true, z_tok)
        how = "fitted on this val split"
    probs_T = apply_temperature(probs, T)
    pit_T = pit_values(probs_T, z_true, z_tok)
    scores_T = uncertainty_scores(probs_T, z_tok)
    ks_T = pit_calibration_error(pit_T)
    bars_T = " ".join(f"{int(c)}" for c in pit_histogram(pit_T, bins=10))

    print("\n" + "-" * 56)
    print(f"  X8b temperature calibration  (T = {T:.3f}, {how})")
    print("-" * 56)
    print(f"  PIT KS  {ks:.4f}  ->  {ks_T:.4f}   ({'better' if ks_T < ks else 'worse'})")
    print(f"  PIT histogram (10) : {bars_T}")
    print(f"\n  Calibrated reject by {DEFAULT_REJECTION_SCORE}:")
    _coverage_table(z_pred, z_true, scores_T[DEFAULT_REJECTION_SCORE])
    print(f"\n  Calibrated outlier-detection AUROC (η>{CATASTROPHIC_DZ}):")
    for name, s in scores_T.items():
        print(f"    {name:18s}: {outlier_auroc(s, is_outlier):.4f}")
    # Recommendation keys off rejection (the metric we ship), not PIT: a soft,
    # over-dispersed posterior lets temperature nudge PIT while wrecking the
    # posterior_std_z ranking — only bake T if rejection is preserved. Use X8c
    # (below) for calibration instead.
    auroc_base = outlier_auroc(scores[DEFAULT_REJECTION_SCORE], is_outlier)
    auroc_T = outlier_auroc(scores_T[DEFAULT_REJECTION_SCORE], is_outlier)
    if auroc_T < auroc_base - 0.01 or ks_T >= ks:
        print(f"  -> temperature hurts rejection "
              f"({DEFAULT_REJECTION_SCORE} AUROC {auroc_base:.3f}->{auroc_T:.3f}) "
              f"or doesn't help PIT; keep redshift_temperature=1.0 "
              f"(use X8c recalibration for calibration)")
    else:
        print(f"  -> bake T={T:.3f} into eval/train via "
              f"`evaluate(..., redshift_temperature={T:.3f})`")

    # ---- X8c: isotonic PIT recalibration (monotone; rejection untouched) ----
    pit_flat = pit.flatten()
    nps = pit_flat.numel()
    fit_idx = torch.arange(0, nps, 2)   # honest held-out split: fit on evens,
    ev_idx = torch.arange(1, nps, 2)    # score KS on odds
    knots = fit_pit_recalibrator(pit_flat[fit_idx])
    ks_ev_raw = pit_calibration_error(pit_flat[ev_idx])
    ks_ev_rc = pit_calibration_error(apply_pit_recalibration(pit_flat[ev_idx], knots))
    pit_rc = apply_pit_recalibration(pit_flat, knots)
    bars_rc = " ".join(f"{int(c)}" for c in pit_histogram(pit_rc, bins=10))

    print("\n" + "-" * 56)
    print("  X8c isotonic PIT recalibration  (monotone; rejection unchanged)")
    print("-" * 56)
    print(f"  held-out PIT KS  {ks_ev_raw:.4f}  ->  {ks_ev_rc:.4f}   "
          f"({'better' if ks_ev_rc < ks_ev_raw else 'worse'})")
    print(f"  PIT histogram (10) : {bars_rc}")
    print("  (the posterior_std_z rejection table above is unchanged by design)")

    if args.dump_npz is not None:
        import numpy as np
        np.savez(
            args.dump_npz,
            dz=dz.numpy(), pit=pit.numpy(), pit_caltemp=pit_T.numpy(),
            pit_recal=pit_rc.numpy(), recal_knots=knots.numpy(),
            temperature=np.array(T), is_outlier=is_outlier.numpy(),
            **{f"score_{k}": s.numpy() for k, s in scores.items()},
            **{f"score_caltemp_{k}": s.numpy() for k, s in scores_T.items()},
        )
        print(f"\n  per-sample arrays -> {args.dump_npz}")


if __name__ == "__main__":
    main()
