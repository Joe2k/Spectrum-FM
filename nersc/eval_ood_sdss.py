#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X9 — out-of-distribution evaluation on SDSS (the generalization headline).

The model is trained on DESI DR1 only. This runs the *complete* eval stack on
SDSS spectra it has never seen — redshift σ_NMAD + `posterior_std_z` rejection +
calibration (X4/X8/X8c) AND flux-space reconstruction (X10) — to test the
out-of-sample claim against AION-1 / SpecPT. SDSS's log-λ grid is handled by the
tokenizer's wavelength-aware resampling, so nothing here is DESI-specific.

Needs the on-the-fly path: raw SDSS flux + the spectrum tokenizer decoder
(`--tokenizer-ckpt`). Reuses `evaluate()` (wavelength-aware) and
`evaluate_spectrum()` unchanged on an SDSS loader.

Usage:

    python nersc/eval_ood_sdss.py \\
        --checkpoint $SCRATCH/deepsrch/checkpoints/<run>/best.pt \\
        --tokenizer-ckpt $SCRATCH/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt \\
        --sdss-dir <dir of spec-*.fits>  [--max-spectra 5000 --amp --amp-dtype bf16]
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
    coverage_quality_curve,
    fit_pit_recalibrator,
    outlier_auroc,
    pit_calibration_error,
    pit_histogram,
)
from src.inference.release import _infer_transformer_dims, _restore_z_tokenizer  # noqa: E402
from src.models.transformer import SpectrumTransformer  # noqa: E402
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.training.eval import evaluate  # noqa: E402
from src.utils.sdss import SDSSSpectrumDataset, collate_sdss_skip_none  # noqa: E402

from eval_spectrum import (  # noqa: E402  (nersc/ sibling)
    evaluate_spectrum,
    evaluate_spectrum_blind,
)


def parse_args():
    p = argparse.ArgumentParser(description="X9 SDSS out-of-distribution eval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer-ckpt", type=Path, required=True,
                   help="SpectrumTokenizer v2 .pt (encode + decode for SDSS).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sdss-dir", type=Path, help="Directory of spec-*.fits.")
    src.add_argument("--sdss-manifest", type=Path,
                     help="Text file with one spec-*.fits path per line.")
    p.add_argument("--glob", default="spec-*.fits",
                   help="Glob for --sdss-dir. Use '**/spec-*.fits' to recurse the "
                        "SDSS lite tree (spectra are nested in per-plate dirs).")
    p.add_argument("--approach", default="a", choices=["a", "b"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=200)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--encoder-mask-ratio", type=float, default=0.5,
                   help="Spectrum masking for the X10 reconstruction pass.")
    p.add_argument("--blind", action="store_true",
                   help="Also run the fully-blind AR reconstruction (no teacher "
                        "forcing) — the honest OOD transfer number. Slow.")
    p.add_argument("--blind-max-batches", type=int, default=8)
    p.add_argument("--require-good-zwarn", action="store_true", default=True)
    p.add_argument("--keep-bad-zwarn", dest="require_good_zwarn",
                   action="store_false")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--dump-npz", type=Path, default=None)
    return p.parse_args()


def _coverage_table(z_pred, z_true, score):
    print(f"  {'retained':>9} {'σ_NMAD':>10} {'η>.0033':>9} {'η>.05':>8} {'n':>7}")
    for r in coverage_quality_curve(z_pred, z_true, score):
        print(f"  {r['retained_frac']*100:8.0f}% {r['sigma_nmad']:10.6f} "
              f"{r['outlier_frac']*100:8.2f}% {r['outlier_frac_05']*100:7.2f}% "
              f"{r['n']:7d}")


def _report_redshift(v):
    print("\n" + "=" * 64)
    print("  REDSHIFT (OOD, honest z-hidden)")
    print("=" * 64)
    print(f"  σ_NMAD (expected)   : {v['z_nmad']:.6f}")
    print(f"  σ_NMAD (argmax)     : {v['z_nmad_argmax']:.6f}")
    print(f"  catastrophic η>.0033: {v['z_outlier_frac']:.4f}")
    print(f"  outliers   η>.05    : {v['z_outlier_frac_05']:.4f}")
    print(f"  bias  median(Δz)    : {v['z_bias']:+.6f}")

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
    bars = " ".join(f"{int(c)}" for c in pit_histogram(pit, bins=10))
    print(f"\n  PIT calibration KS : {ks:.4f}  ({bars})")
    # X8c isotonic recalibration, honest held-out split (fit evens / score odds).
    n = pit.numel()
    knots = fit_pit_recalibrator(pit[0:n:2])
    ks_rc = pit_calibration_error(apply_pit_recalibration(pit[1:n:2], knots))
    print(f"  X8c recalibrated KS: {ks:.4f} -> {ks_rc:.4f} (held-out)")
    return is_outlier, dz


def _report_spectrum(vs, mask_ratio, vb=None):
    def _row(label, r):
        if not r:
            return f"  {label:30s}: (no data)"
        return (f"  {label:30s}: χ²/pixel {r['chi2_per_pixel']:7.3f}   "
                f"ivar-R² {r['ivar_r2']:7.4f}   n={r['n']}")

    print("\n" + "=" * 64)
    print(f"  SPECTRUM reconstruction (OOD, mask {mask_ratio:.0%}, z hidden)")
    print("=" * 64)
    print(f"  masked-token acc (top-1/1024): {vs['masked_token_acc']:.4f}  "
          f"(misleading; n={vs['n_masked_tokens']})")
    print(_row("codec ceiling (all px)", vs["recon_codec"]))
    print(_row("predicted (masked blocks)", vs["recon_masked"]))
    print(_row("predicted (all px)", vs["recon_all"]))
    if vb is not None:
        print(f"  -- fully-blind AR (no teacher forcing): acc {vb['masked_token_acc']:.4f}")
        print(_row("blind predicted (masked blk)", vb["recon_masked"]))
        print(_row("blind predicted (all px)", vb["recon_all"]))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[ood] loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    dims = _infer_transformer_dims(sd)
    print(f"[ood] dims={dims}")
    model = SpectrumTransformer(**dims).to(device)
    model.load_state_dict(sd)
    model.eval()
    z_tok = _restore_z_tokenizer(ckpt)

    spec_tok = SpectrumTokenizer().to(device)
    tck = torch.load(args.tokenizer_ckpt, map_location=device, weights_only=False)
    spec_tok.load_state_dict(tck.get("model", tck) if isinstance(tck, dict) else tck)
    spec_tok.eval()
    for pm in spec_tok.parameters():
        pm.requires_grad_(False)

    if args.sdss_manifest is not None:
        paths = [ln.strip() for ln in args.sdss_manifest.read_text().splitlines()
                 if ln.strip()]
        source = paths
    else:
        source = args.sdss_dir
    ds = SDSSSpectrumDataset(
        source, require_good_zwarn=args.require_good_zwarn,
        require_nonzero_flux=True, max_spectra=args.max_spectra, glob=args.glob)
    print(f"[ood] SDSS spectra loaded: {len(ds)}")
    if len(ds) == 0:
        print("[ood] no usable SDSS spectra — check --sdss-dir/--sdss-manifest "
              "and column names.", file=sys.stderr)
        sys.exit(1)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_sdss_skip_none,
        pin_memory=device.type == "cuda")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    # ---- redshift (full spectrum visible, z hidden) ----
    v = evaluate(
        model, loader, spec_tok, z_tok, args.approach, device,
        amp=args.amp, redshift_weight=1.0, encoder_mask_ratio=0.0,
        redshift_mask_ratio=1.0, max_batches=args.max_batches,
        amp_dtype=amp_dtype, return_per_sample=True, wavelength_aware=True)

    # ---- spectrum reconstruction (masked) ----
    vs = evaluate_spectrum(
        model, loader, spec_tok, z_tok, args.approach, device,
        amp=args.amp, amp_dtype=amp_dtype,
        encoder_mask_ratio=args.encoder_mask_ratio, max_batches=args.max_batches)
    vb = None
    if args.blind:
        vb = evaluate_spectrum_blind(
            model, loader, spec_tok, z_tok, args.approach, device,
            encoder_mask_ratio=args.encoder_mask_ratio,
            max_batches=args.blind_max_batches)

    print("\n" + "#" * 64)
    print(f"#  X9 OOD — {args.checkpoint.name} on SDSS (never trained on)")
    print("#" * 64)
    is_outlier, dz = _report_redshift(v)
    _report_spectrum(vs, args.encoder_mask_ratio, vb)
    print("\n" + "=" * 64)
    print("  σ_NMAD/η vs SpecPT (0.0006-0.0008 / 0.2-0.8%); χ²/pixel→1 = noise floor.")

    if args.dump_npz is not None:
        import numpy as np
        np.savez(
            args.dump_npz,
            z_pred=v["z_pred"].numpy(), z_true=v["z_true"].numpy(),
            dz=dz.numpy(), pit=v["pit"].numpy(), is_outlier=is_outlier.numpy(),
            **{f"score_{k}": s.numpy() for k, s in v["scores"].items()})
        print(f"\n  per-sample arrays -> {args.dump_npz}")


if __name__ == "__main__":
    main()
