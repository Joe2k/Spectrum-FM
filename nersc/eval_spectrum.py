#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux-space spectrum reconstruction eval (the X4-for-spectrum).

`spectrum_acc` (~0.27) is exact top-1 over the 1024-way codebook — and just as
misleading as redshift bin-accuracy was: tokenizer v2 is at the noise floor
(χ²/pixel ≈ 1.0), so a near-equivalent predicted code scores "wrong" yet decodes
to a spectrum within DESI's own noise. This script asks the honest question:
decode the transformer's predicted *masked* tokens back to flux and score the
reconstruction against the observation, inverse-variance weighted.

It reports, on the held-out DESI val split (honest regime: redshift hidden):
  - masked-token accuracy (the misleading 0.27, for reference);
  - the codec-only ceiling (decode the *true* tokens) — expect χ²/pixel ≈ 1.0;
  - predicted reconstruction on the **masked-token pixel blocks** (the blind
    number that mirrors `masked_spec_acc`) and over all valid pixels.

Requires the on-the-fly path (raw flux + ivar + the spectrum tokenizer decoder),
so pass --tokenizer-ckpt (the cached token path stores no flux / denorm).

Usage:

    python nersc/eval_spectrum.py \\
        --checkpoint $SCRATCH/deepsrch/checkpoints/<run>/best.pt \\
        --tokenizer-ckpt <spectrum tokenizer v2 best.pt> \\
        --manifest $SCRATCH/manifests/dr1_v2_full.jsonl \\
        --encoder-mask-ratio 0.5 --max-spectra 20000 --amp --amp-dtype bf16
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

from src.eval.spectrum_metrics import (  # noqa: E402
    add_sums,
    finalize_recon,
    recon_weighted_sums,
    token_mask_to_pixel_mask,
)
from src.inference.release import _infer_transformer_dims, _restore_z_tokenizer  # noqa: E402
from src.models.transformer import (  # noqa: E402
    REDSHIFT_TOKEN_OFFSET,
    SPECTRUM_TOKEN_OFFSET,
    SpectrumTransformer,
)
from src.tokenizers.spectrum import N_TOKENS, SpectrumTokenizer  # noqa: E402
from src.training.data_split import split_records_by_healpix  # noqa: E402
from src.training.sequences import tokenize_and_build  # noqa: E402

from dr1_dataset import (  # noqa: E402
    DR1IndexedDataset,
    collate_dr1_skip_none,
    load_manifest,
)

N_SPEC_CODES = REDSHIFT_TOKEN_OFFSET - SPECTRUM_TOKEN_OFFSET  # 1024


def parse_args():
    p = argparse.ArgumentParser(description="Flux-space spectrum reconstruction eval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer-ckpt", type=Path, required=True,
                   help="SpectrumTokenizer v2 .pt (decoder needed to map tokens→flux).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--approach", default="a", choices=["a", "b"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--encoder-mask-ratio", type=float, default=0.5,
                   help="Spectrum positions hidden from the encoder (match training).")
    p.add_argument("--healpix-holdout-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    return p.parse_args()


@torch.no_grad()
def evaluate_spectrum(model, loader, spec_tok, z_tok, approach, device, amp,
                      amp_dtype, encoder_mask_ratio, max_batches):
    model.eval()
    acc = {"masked": None, "all": None, "codec": None}
    tok_correct = tok_total = 0

    def _accum(key, new):
        new = {k: v.detach().cpu() for k, v in new.items()}
        acc[key] = new if acc[key] is None else add_sums(acc[key], new)

    for i, raw in enumerate(loader):
        if raw is None:
            continue
        if i >= max_batches:
            break
        flux = raw["flux"].to(device, non_blocking=True)
        ivar = raw["ivar"].to(device, non_blocking=True)
        wave = raw["wavelength"].to(device, non_blocking=True)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)  # (B, 2, L) native grid

        # Frozen tokenizer (full precision): true codes + denorm + gridded obs.
        indices, denorm = spec_tok.encode(x, wavelength=wave)  # (B,272), (B,)
        if indices.dim() == 3:
            indices = indices.squeeze(1)
        indices = indices.long()
        x_grid = spec_tok._to_grid(x, wave)                    # (B,2,8704)
        flux_obs, istd_grid = x_grid[:, 0], x_grid[:, 1]

        # Transformer masked prediction — honest regime (redshift hidden).
        enc, dec, tgt, mask_pos = tokenize_and_build(
            raw, spec_tok, z_tok, approach, device,
            encoder_mask_ratio=encoder_mask_ratio, redshift_mask_ratio=1.0,
            wavelength_aware=True, mask_targets_only=False)
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
            logits, _ = model(enc, dec, targets=tgt, redshift_weight=1.0)
        spec_logits = logits[:, 1:1 + N_TOKENS,
                             SPECTRUM_TOKEN_OFFSET:SPECTRUM_TOKEN_OFFSET + N_SPEC_CODES].float()
        pred_codes = spec_logits.argmax(dim=-1)            # (B, 272)
        true_codes = indices                               # (B, 272)
        m = mask_pos.bool()
        tok_correct += int(((pred_codes == true_codes) & m).sum().item())
        tok_total += int(m.sum().item())

        # Decode predicted (true tokens, masked ones replaced) and codec ceiling.
        # The LFQ (en|de)coder uses 2-D (B, n_tokens) index tensors.
        pred_idx = true_codes.clone()
        pred_idx[m] = pred_codes[m]
        flux_pred = spec_tok.decode(pred_idx, denorm)[:, 0]
        flux_codec = spec_tok.decode(indices, denorm)[:, 0]

        pix_mask = token_mask_to_pixel_mask(m)             # (B, 8704)
        _accum("masked", recon_weighted_sums(flux_pred, flux_obs, istd_grid, pix_mask))
        _accum("all", recon_weighted_sums(flux_pred, flux_obs, istd_grid))
        _accum("codec", recon_weighted_sums(flux_codec, flux_obs, istd_grid))

    return {
        "masked_token_acc": (tok_correct / tok_total) if tok_total else float("nan"),
        "n_masked_tokens": tok_total,
        "recon_masked": finalize_recon(acc["masked"]) if acc["masked"] else {},
        "recon_all": finalize_recon(acc["all"]) if acc["all"] else {},
        "recon_codec": finalize_recon(acc["codec"]) if acc["codec"] else {},
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[eval] loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    dims = _infer_transformer_dims(sd)
    print(f"[eval] dims={dims}")
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

    records = load_manifest(args.manifest)
    _, val_records = split_records_by_healpix(
        records, holdout_frac=args.healpix_holdout_frac, seed=args.seed)
    val_cap = None if args.max_spectra is None else max(50, args.max_spectra // 10)
    val_ds = DR1IndexedDataset(
        val_records, require_good_zwarn=True, require_nonzero_flux=True,
        max_spectra=val_cap)
    print(f"[eval] val_ds={len(val_ds)} ({len(val_records)} healpix)")
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_dr1_skip_none,
        pin_memory=device.type == "cuda")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    v = evaluate_spectrum(
        model, val_loader, spec_tok, z_tok, args.approach, device,
        amp=args.amp, amp_dtype=amp_dtype,
        encoder_mask_ratio=args.encoder_mask_ratio, max_batches=args.max_batches)

    def _row(label, r):
        if not r:
            return f"  {label:28s}: (no data)"
        return (f"  {label:28s}: χ²/pixel {r['chi2_per_pixel']:7.3f}   "
                f"ivar-R² {r['ivar_r2']:7.4f}   n={r['n']}")

    print("\n" + "=" * 64)
    print(f"  Spectrum reconstruction — {args.checkpoint.name}  "
          f"[mask {args.encoder_mask_ratio:.0%}, z hidden]")
    print("=" * 64)
    print(f"  masked-token acc (top-1/1024): {v['masked_token_acc']:.4f}   "
          f"(the misleading number; n={v['n_masked_tokens']})")
    print(_row("codec ceiling (all px)", v["recon_codec"]))
    print(_row("predicted (masked blocks)", v["recon_masked"]))
    print(_row("predicted (all px)", v["recon_all"]))
    print("=" * 64)
    print("  χ²/pixel → 1.0 = reconstruction at DESI's noise floor; "
          "ivar-R² → 1 = perfect.")


if __name__ == "__main__":
    main()
