#!/usr/bin/env python3
"""
T3 token-stability audit for the spectrum tokenizer.
====================================================
Two diagnostics on the SAME held-out val set the tokenizer trained on (same
manifest, seed and split as nersc/pretrain_tokenizer.py), to learn the last
thing isolated metrics can tell us before committing the transformer campaign:

A. Noise-realization flip rate. Re-draw the per-pixel noise (std = 1/sqrt(ivar))
   K times, re-tokenize, and measure how often each token index changes vs the
   observed spectrum's tokens. Stratified by per-token local signal-to-noise
   ratio (SNR): at chi^2 = 1.0 the codec encodes detail down to the noise, so
   LOW-SNR tokens flipping is expected and benign; HIGH-SNR tokens must stay
   stable, or the transformer would be learning noise instead of spectra.

B. Per-(survey, program) equity. chi^2/pixel, pooled R^2 and codebook usage
   per group, to verify the balanced DR1 manifest didn't leave dark-program
   spectra second-class.

No training; ~minutes on one GPU. Run in an interactive NERSC session where the
checkpoint and FITS data live, e.g.:

    python nersc/audit_tokenizer_stability.py \
        --checkpoint $CFS/tokenizer_v2_3k_v3/final.pt \
        --manifest   $SCRATCH/manifests/dr1_v2_balanced_3k_scratch.jsonl \
        --max-spectra 2000 --n-noise 16 --out results/t3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Repo imports
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.tokenizers.spectrum import SpectrumTokenizer, flux_r2_terms  # noqa: E402

# Local imports
sys.path.insert(0, str(HERE))
from dr1_dataset import (  # noqa: E402
    DR1IndexedDataset,
    collate_dr1_skip_none,
    load_manifest,
)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in tests/test_audit_stability.py)
# --------------------------------------------------------------------------- #
def flip_fraction(a, b) -> float:
    """Fraction of positions where two equal-shaped integer token tensors differ."""
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return float("nan")
    return (a != b).float().mean().item()


def token_local_snr(snr_grid: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Median |SNR| per token region.

    snr_grid: (B, L) per-pixel SNR on the model grid. Returns (B, n_tokens).
    First-order chunking: token t <- grid pixels [t*stride : (t+1)*stride] with
    stride = L // n_tokens. This ignores the ConvNeXt receptive-field overlap;
    it is an approximation adequate for SNR stratification, not exact provenance.
    """
    if snr_grid.dim() != 2:
        raise ValueError("snr_grid must be (B, L)")
    B, L = snr_grid.shape
    stride = L // n_tokens
    if stride < 1:
        raise ValueError(f"n_tokens {n_tokens} exceeds grid length {L}")
    usable = snr_grid[:, : stride * n_tokens].reshape(B, n_tokens, stride)
    return usable.abs().median(dim=-1).values


def stratify_by_snr(snr, flip, edges):
    """Bin per-token flip rates by per-token SNR.

    snr, flip: 1D array-likes of equal length. edges: ascending bin edges.
    Returns (labels, mean_flip_per_bin, count_per_bin) over len(edges)+1 bins
    (the digitize convention: bin 0 = (-inf, edges[0]), last = [edges[-1], inf)).
    """
    snr = np.asarray(snr, dtype=float)
    flip = np.asarray(flip, dtype=float)
    if snr.shape != flip.shape:
        raise ValueError("snr and flip must be the same shape")
    idx = np.digitize(snr, edges)
    labels, means, counts = [], [], []
    for b in range(len(edges) + 1):
        m = idx == b
        counts.append(int(m.sum()))
        means.append(float(flip[m].mean()) if m.any() else float("nan"))
        lo = "-inf" if b == 0 else f"{edges[b - 1]:g}"
        hi = "inf" if b == len(edges) else f"{edges[b]:g}"
        labels.append(f"[{lo},{hi})")
    return labels, means, counts


def pooled_r2(ssr: float, sst: float) -> float:
    """Corpus-pooled R^2 = 1 - sum(ss_res)/sum(ss_tot)."""
    return 1.0 - ssr / max(sst, 1e-12)


def summarize_group(acc: dict, codebook_size: int) -> dict:
    """Build a per-(survey, program) metrics row from an accumulator dict.

    acc keys: n_spectra, nll_sum, nll_n, ssr, sst, ssr_hi, sst_hi, n_codes_seen.
    chi^2/pixel = 2 * mean(nll_flux) (the held-out reconstruction quality).
    """
    chi2 = 2.0 * acc["nll_sum"] / max(acc["nll_n"], 1)
    return {
        "n_spectra": int(acc["n_spectra"]),
        "chi2_per_pixel": chi2,
        "flux_r2_pooled": pooled_r2(acc["ssr"], acc["sst"]),
        "flux_r2_pooled_snr3": (
            pooled_r2(acc["ssr_hi"], acc["sst_hi"]) if acc["sst_hi"] > 0 else float("nan")
        ),
        "codebook_use": acc["n_codes_seen"] / float(codebook_size),
    }


def verdict(high_snr_flip, group_rows, *, flip_thresh, chi2_thresh, codebook_thresh):
    """PASS/FLAG against the soft T3 criteria. Returns (passed, reasons)."""
    reasons = []
    if not (high_snr_flip < flip_thresh):
        reasons.append(
            f"high-SNR (>3) token flip rate {high_snr_flip:.3f} >= {flip_thresh:.3f} "
            f"— codec may be putting noise into structurally important tokens")
    for (survey, program), row in group_rows.items():
        tag = f"{survey}/{program}"
        if row["chi2_per_pixel"] >= chi2_thresh:
            reasons.append(f"{tag} chi^2/pixel {row['chi2_per_pixel']:.3f} >= {chi2_thresh:.2f}")
        if row["codebook_use"] < codebook_thresh:
            reasons.append(
                f"{tag} codebook use {row['codebook_use']:.3f} < {codebook_thresh:.2f}")
    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------- #
# Val-split replication (identical to pretrain_tokenizer.py:292-300)
# --------------------------------------------------------------------------- #
def build_val_indices(n_full: int, seed: int, val_frac: float):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_full, generator=g).tolist()
    n_val = max(1, int(n_full * val_frac))
    return perm[:n_val]


# --------------------------------------------------------------------------- #
# Audit driver
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="T3 token-stability audit")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="full-state tokenizer checkpoint (final.pt / last.pt / best.pt)")
    p.add_argument("--manifest", type=Path, required=True,
                   help="the SAME JSONL manifest the tokenizer trained on")
    p.add_argument("--seed", type=int, default=42,
                   help="must match training --seed so the val split is identical")
    p.add_argument("--val-frac", type=float, default=0.02,
                   help="must match training --val-frac")
    p.add_argument("--surveys", nargs="*", default=None,
                   help="optional manifest survey filter (match training)")
    p.add_argument("--programs", nargs="*", default=None,
                   help="optional manifest program filter (match training)")
    p.add_argument("--max-spectra", type=int, default=2000,
                   help="cap audited val spectra (FITS reads are the bottleneck)")
    p.add_argument("--n-noise", type=int, default=16,
                   help="noise realizations per spectrum for the flip-rate test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (matches training)")
    p.add_argument("--require-good-zwarn", action="store_true", default=True)
    p.add_argument("--require-nonzero-flux", action="store_true", default=True)
    p.add_argument("--snr-edges", type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 3.0, 5.0, 10.0],
                   help="SNR bin edges for flip-rate stratification")
    # Soft acceptance thresholds (documented in RESEARCH_LOG T3 spec)
    p.add_argument("--flip-thresh", type=float, default=0.15,
                   help="max acceptable high-SNR (>3) token flip rate")
    p.add_argument("--chi2-thresh", type=float, default=1.2,
                   help="max acceptable per-group chi^2/pixel")
    p.add_argument("--codebook-thresh", type=float, default=0.30,
                   help="min acceptable per-group codebook utilization")
    p.add_argument("--out", type=Path, default=Path("results/t3"),
                   help="output dir for t3_stability.json / .md / .npz")
    return p.parse_args()


@torch.no_grad()
def run_audit(args, model, full, val_idx, device):
    codebook_size = model.codebook_size
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # Group the audited val indices by (survey, program) → one I/O pass per
    # spectrum serves BOTH the flip-rate test (A) and the equity table (B).
    groups = defaultdict(list)
    for i in val_idx[: args.max_spectra]:
        groups[full.meta_for_index(i)].append(i)

    # Global Part-A accumulators
    spectrum_flips = []          # per-spectrum mean flip rate
    position_flip_sum = None     # (T,) summed per-token flip rate
    position_n = 0
    snr_flat, flip_flat = [], []  # per-token (SNR, flip rate) for stratification

    group_rows = {}
    for (survey, program), idxs in sorted(groups.items()):
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full, idxs),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_dr1_skip_none,
            pin_memory=device.type == "cuda",
        )
        acc = dict(n_spectra=0, nll_sum=0.0, nll_n=0,
                   ssr=0.0, sst=0.0, ssr_hi=0.0, sst_hi=0.0)
        codes_seen = torch.zeros(codebook_size, dtype=torch.bool, device=device)

        for batch in loader:
            if batch is None:
                continue
            flux = batch["flux"].to(device, non_blocking=True)
            ivar = batch["ivar"].to(device, non_blocking=True)
            wave = batch["wavelength"].to(device, non_blocking=True)
            istd = torch.sqrt(ivar.clamp(min=1e-10))
            x = torch.stack([flux, istd], dim=1)
            valid = ivar > 0
            noise_std = torch.where(valid, 1.0 / torch.sqrt(ivar.clamp(min=1e-30)),
                                    torch.zeros_like(ivar))

            with torch.amp.autocast("cuda", enabled=args.amp):
                # ----- Part B: reconstruction quality + codebook on this group
                recon, loss, indices = model(x, wavelength=wave)
                # ----- Part A: baseline tokens of the observed spectrum
                baseline = model.encode(x, wavelength=wave)[0]  # (B, T) int

            B, T = baseline.shape
            acc["n_spectra"] += B
            acc["nll_sum"] += float(loss["nll_flux"].item()) * B
            acc["nll_n"] += B
            x_grid = model._to_grid(x, wave)
            ivar_g = x_grid[:, 1].square()
            ss_res, ss_tot = flux_r2_terms(x_grid[:, 0], recon[:, 0].float(), ivar=ivar_g)
            acc["ssr"] += ss_res.sum().item()
            acc["sst"] += ss_tot.sum().item()
            snr_spec = (x_grid[:, 0] * x_grid[:, 1]).median(dim=-1).values  # (B,)
            hi = snr_spec > 3.0
            if hi.any():
                acc["ssr_hi"] += ss_res[hi].sum().item()
                acc["sst_hi"] += ss_tot[hi].sum().item()
            codes_seen[indices.unique()] = True

            # ----- Part A: flip rate over n_noise re-draws of the pixel noise
            flip_count = torch.zeros(B, T, device=device)
            for _ in range(args.n_noise):
                noise = torch.randn(flux.shape, generator=gen, device=device) * noise_std
                x_k = torch.stack([flux + noise, istd], dim=1)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    tokens_k = model.encode(x_k, wavelength=wave)[0]
                flip_count += (tokens_k != baseline).float()
            per_token_flip = flip_count / args.n_noise               # (B, T)
            spectrum_flips.extend(per_token_flip.mean(dim=1).tolist())
            position_flip_sum = (per_token_flip.sum(dim=0) if position_flip_sum is None
                                 else position_flip_sum + per_token_flip.sum(dim=0))
            position_n += B
            tsnr = token_local_snr((x_grid[:, 0] * x_grid[:, 1]), T)  # (B, T)
            snr_flat.extend(tsnr.flatten().tolist())
            flip_flat.extend(per_token_flip.flatten().tolist())

        acc["n_codes_seen"] = int(codes_seen.sum().item())
        group_rows[(survey, program)] = summarize_group(acc, codebook_size)
        print(f"[group {survey}/{program}] n={acc['n_spectra']} "
              f"chi2={group_rows[(survey, program)]['chi2_per_pixel']:.4f} "
              f"codebook={group_rows[(survey, program)]['codebook_use']:.3f}")

    # ---- Stratify flip rate by SNR; high-SNR summary number ----
    labels, bin_means, bin_counts = stratify_by_snr(snr_flat, flip_flat, args.snr_edges)
    snr_arr = np.asarray(snr_flat)
    flip_arr = np.asarray(flip_flat)
    hi_mask = snr_arr > 3.0
    high_snr_flip = float(flip_arr[hi_mask].mean()) if hi_mask.any() else float("nan")
    lo_mask = snr_arr <= 1.0
    low_snr_flip = float(flip_arr[lo_mask].mean()) if lo_mask.any() else float("nan")

    passed, reasons = verdict(
        high_snr_flip, group_rows,
        flip_thresh=args.flip_thresh, chi2_thresh=args.chi2_thresh,
        codebook_thresh=args.codebook_thresh)

    spectrum_flips = np.asarray(spectrum_flips)
    results = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "n_spectra_audited": int(spectrum_flips.size),
        "n_noise": args.n_noise,
        "flip_rate": {
            "per_spectrum_median": float(np.median(spectrum_flips)) if spectrum_flips.size else float("nan"),
            "per_spectrum_p90": float(np.percentile(spectrum_flips, 90)) if spectrum_flips.size else float("nan"),
            "high_snr_gt3": high_snr_flip,
            "low_snr_le1": low_snr_flip,
            "by_snr_bin": [
                {"snr": lab, "mean_flip": m, "n_tokens": c}
                for lab, m, c in zip(labels, bin_means, bin_counts)
            ],
            "per_token_position": (position_flip_sum / max(position_n, 1)).tolist()
            if position_flip_sum is not None else [],
        },
        "equity": {f"{s}/{p}": row for (s, p), row in group_rows.items()},
        "verdict": {"passed": passed, "reasons": reasons,
                    "thresholds": {"flip": args.flip_thresh, "chi2": args.chi2_thresh,
                                   "codebook": args.codebook_thresh}},
    }
    return results, (snr_arr, flip_arr, spectrum_flips)


def write_markdown(results: dict, path: Path):
    fr = results["flip_rate"]
    lines = [
        "# T3 token-stability audit",
        "",
        f"- checkpoint: `{results['checkpoint']}`",
        f"- spectra audited: {results['n_spectra_audited']}  |  noise realizations: {results['n_noise']}",
        "",
        "## A. Noise-realization flip rate",
        "",
        f"- per-spectrum median flip rate: **{fr['per_spectrum_median']:.3f}**  (p90 {fr['per_spectrum_p90']:.3f})",
        f"- high-SNR (>3) token flip rate: **{fr['high_snr_gt3']:.3f}**  ← the one that matters",
        f"- low-SNR (<=1) token flip rate: {fr['low_snr_le1']:.3f}  (expected to be higher — benign)",
        "",
        "| SNR bin | mean flip | n tokens |",
        "|---|---|---|",
    ]
    for row in fr["by_snr_bin"]:
        m = "nan" if row["mean_flip"] != row["mean_flip"] else f"{row['mean_flip']:.3f}"
        lines.append(f"| {row['snr']} | {m} | {row['n_tokens']} |")
    lines += [
        "",
        "Healthy signature: flip rate falls as SNR rises; high-SNR tokens stable.",
        "",
        "## B. Per-(survey, program) equity",
        "",
        "| survey/program | n | chi^2/pixel | pooled R^2 (SNR>3) | codebook use |",
        "|---|---|---|---|---|",
    ]
    for tag, row in results["equity"].items():
        hi = row["flux_r2_pooled_snr3"]
        hi = "nan" if hi != hi else f"{hi:.4f}"
        lines.append(
            f"| {tag} | {row['n_spectra']} | {row['chi2_per_pixel']:.4f} | "
            f"{hi} | {row['codebook_use']:.3f} |")
    v = results["verdict"]
    lines += ["", f"## Verdict: {'PASS' if v['passed'] else 'FLAG'}"]
    if v["reasons"]:
        lines += [""] + [f"- {r}" for r in v["reasons"]]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} amp={args.amp}")

    # Model — same construction as pretrain_tokenizer.py (nll recon by default).
    model = SpectrumTokenizer().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[model] loaded {args.checkpoint} (step {ckpt.get('step', '?') if isinstance(ckpt, dict) else '?'})")

    records = load_manifest(args.manifest)
    if args.surveys:
        records = [r for r in records if r.get("survey") in set(args.surveys)]
    if args.programs:
        records = [r for r in records if r.get("program") in set(args.programs)]
    full = DR1IndexedDataset(
        records,
        require_good_zwarn=args.require_good_zwarn,
        require_nonzero_flux=args.require_nonzero_flux,
    )
    val_idx = build_val_indices(len(full), args.seed, args.val_frac)
    print(f"[data] {len(full)} spectra, {len(val_idx)} val, auditing {min(len(val_idx), args.max_spectra)}")

    results, arrays = run_audit(args, model, full, val_idx, device)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "t3_stability.json").write_text(json.dumps(results, indent=2) + "\n")
    write_markdown(results, args.out / "t3_stability.md")
    snr_arr, flip_arr, spectrum_flips = arrays
    np.savez(args.out / "t3_stability.npz",
             token_snr=snr_arr, token_flip=flip_arr, spectrum_flip=spectrum_flips)

    v = results["verdict"]
    print("\n" + "=" * 60)
    print(f"VERDICT: {'PASS' if v['passed'] else 'FLAG'}")
    for r in v["reasons"]:
        print(f"  - {r}")
    print(f"high-SNR(>3) flip {results['flip_rate']['high_snr_gt3']:.3f} | "
          f"median {results['flip_rate']['per_spectrum_median']:.3f}")
    print(f"wrote {args.out / 't3_stability.json'} (+ .md, .npz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
