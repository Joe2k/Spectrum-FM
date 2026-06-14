#!/usr/bin/env python3
"""
T3 token-stability audit for the spectrum tokenizer.
====================================================
Two diagnostics on the SAME held-out val set the tokenizer trained on (same
manifest, seed and split as nersc/pretrain_tokenizer.py), to learn the last
thing isolated metrics can tell us before committing the transformer campaign:

A. Noise-realization flip rate. Re-draw the per-pixel noise (std = 1/sqrt(ivar))
   K times, re-tokenize, and measure how often the encoding changes vs the
   observed spectrum's tokens. Measured at TWO granularities:
     * per-token (the 1024-code index): a token "flips" if ANY of its `dim`
       sign-bits change, so this overcounts — at the noise floor the low-order
       bits encode the noise itself and flip readily while reconstruction holds;
     * per-bit (the `dim` sign-bits): the honest granularity. A token index of
       73% flip with dim=10 corresponds to only ~12% per-bit flip.
   Stratified by per-token local SNR, and swept over perturbation scale
   (0.25 / 0.5 / 1.0 sigma) to expose the sign-decision margin. The number that
   matters: per-bit flip on HIGH-SNR (>10) tokens — the signal-bearing bits must
   be stable, or the transformer would learn noise.

B. Per-(survey, program) equity. chi^2/pixel, pooled R^2 and codebook usage
   per group, to verify the balanced DR1 manifest didn't leave dark-program
   spectra second-class. Groups below --min-n-equity are reported but treated
   as informational (small-sample pooling noise), not hard failures.

No training; ~minutes on one GPU. Run in an interactive NERSC session where the
checkpoint and FITS data live, e.g.:

    python nersc/audit_tokenizer_stability.py \
        --checkpoint $CFS/tokenizer_v2_3k_v3/final.pt \
        --manifest   $SCRATCH/manifests/dr1_v2_balanced_3k_scratch.jsonl \
        --amp --max-spectra 2000 --n-noise 16 --out results/t3
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
    """Fraction of positions where two equal-shaped integer tensors differ.

    Works for token indices and for {0,1} sign-bits alike.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return float("nan")
    return (a != b).float().mean().item()


def bits_from_indices(indices, dim: int) -> torch.Tensor:
    """Decompose integer LFQ token indices into their `dim` sign-bits {0,1}.

    The LFQ index is index = sum_i bit_i * 2^i, so bit_i = (index >> i) & 1.
    A token-index "flip" needs only ONE of `dim` bits to change, which
    overcounts instability relative to the bits that carry the reconstruction;
    decomposing lets us measure flip rate at the per-bit granularity. Returns
    an int64 tensor of shape (*indices.shape, dim).
    """
    indices = torch.as_tensor(indices).to(torch.int64)
    powers = torch.arange(dim, device=indices.device, dtype=torch.int64)
    return (indices.unsqueeze(-1) >> powers) & 1


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


def stratify_by_snr(snr, value, edges):
    """Bin per-token values by per-token SNR.

    snr, value: 1D array-likes of equal length. edges: ascending bin edges.
    Returns (labels, mean_per_bin, count_per_bin) over len(edges)+1 bins
    (the digitize convention: bin 0 = (-inf, edges[0]), last = [edges[-1], inf)).
    """
    snr = np.asarray(snr, dtype=float)
    value = np.asarray(value, dtype=float)
    if snr.shape != value.shape:
        raise ValueError("snr and value must be the same shape")
    idx = np.digitize(snr, edges)
    labels, means, counts = [], [], []
    for b in range(len(edges) + 1):
        m = idx == b
        counts.append(int(m.sum()))
        means.append(float(value[m].mean()) if m.any() else float("nan"))
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


def verdict(bit_high_snr, token_high_snr, group_rows, *,
            bit_thresh, token_hi_thresh, chi2_thresh, codebook_thresh, min_n):
    """PASS/FLAG against the recalibrated T3 criteria.

    Hard gate = per-bit flip on signal-bearing (SNR>10) tokens (the honest
    granularity) + per-group reconstruction on adequately-sampled groups.
    Token-index flip and small-n group blips are informational notes, not fails.
    Returns (passed, reasons, notes).
    """
    reasons, notes = [], []
    if not (bit_high_snr < bit_thresh):
        reasons.append(
            f"high-SNR(>10) per-bit flip {bit_high_snr:.3f} >= {bit_thresh:.3f} "
            f"— signal-bearing bits unstable, codec is encoding noise into them")
    if not (token_high_snr < token_hi_thresh):
        notes.append(
            f"high-SNR(>10) token-index flip {token_high_snr:.3f} >= {token_hi_thresh:.2f} "
            f"(informational — the index collapses `dim` bits, so it overcounts)")
    for (survey, program), row in sorted(group_rows.items()):
        tag = f"{survey}/{program}"
        low_n = row["n_spectra"] < min_n
        bucket = notes if low_n else reasons
        suffix = f" (low-n {row['n_spectra']}, informational)" if low_n else ""
        if row["chi2_per_pixel"] >= chi2_thresh:
            bucket.append(f"{tag} chi^2/pixel {row['chi2_per_pixel']:.3f} >= {chi2_thresh:.2f}{suffix}")
        if row["codebook_use"] < codebook_thresh:
            bucket.append(f"{tag} codebook use {row['codebook_use']:.3f} < {codebook_thresh:.2f}{suffix}")
    return (len(reasons) == 0), reasons, notes


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
                   help="noise realizations at the primary scale (1.0 sigma)")
    p.add_argument("--sweep-noise", type=int, default=4,
                   help="noise realizations per non-primary scale in the margin sweep")
    p.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                   help="perturbation scales (x sigma) for the margin sweep")
    p.add_argument("--primary-scale", type=float, default=1.0,
                   help="the scale used for SNR stratification and per-position stats")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (matches training)")
    p.add_argument("--require-good-zwarn", action="store_true", default=True)
    p.add_argument("--require-nonzero-flux", action="store_true", default=True)
    p.add_argument("--snr-edges", type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 3.0, 5.0, 10.0],
                   help="SNR bin edges for flip-rate stratification")
    # Recalibrated soft acceptance thresholds (see RESEARCH_LOG T3 entry)
    p.add_argument("--bit-thresh", type=float, default=0.10,
                   help="max acceptable high-SNR(>10) per-bit flip rate (the hard gate)")
    p.add_argument("--token-hi-thresh", type=float, default=0.50,
                   help="high-SNR(>10) token-index flip ceiling (informational)")
    p.add_argument("--chi2-thresh", type=float, default=1.2,
                   help="max acceptable per-group chi^2/pixel")
    p.add_argument("--codebook-thresh", type=float, default=0.30,
                   help="min acceptable per-group codebook utilization")
    p.add_argument("--min-n-equity", type=int, default=100,
                   help="groups below this n are informational, not hard fails")
    p.add_argument("--out", type=Path, default=Path("results/t3"),
                   help="output dir for t3_stability.json / .md / .npz")
    return p.parse_args()


def _flip_for_scale(model, flux, istd, noise_std, wave, baseline, baseline_bits,
                    scale, K, dim, gen, amp):
    """Per-token (token-index, mean-bit) flip rate over K re-draws at `scale`.

    Returns (per_token_token_flip (B,T), per_token_bit_flip (B,T)).
    """
    B, T = baseline.shape
    ft = torch.zeros(B, T, device=flux.device)
    fb = torch.zeros(B, T, dim, device=flux.device)
    for _ in range(K):
        noise = torch.randn(flux.shape, generator=gen, device=flux.device) * noise_std * scale
        x_k = torch.stack([flux + noise, istd], dim=1)
        with torch.amp.autocast("cuda", enabled=amp):
            tok_k = model.encode(x_k, wavelength=wave)[0]
        bits_k = bits_from_indices(tok_k, dim)
        ft += (tok_k != baseline).float()
        fb += (bits_k != baseline_bits).float()
    return ft / K, (fb / K).mean(dim=-1)


@torch.no_grad()
def run_audit(args, model, full, val_idx, device):
    codebook_size = model.codebook_size
    dim = model.quantizer.dim
    gen = torch.Generator(device=device).manual_seed(args.seed)
    scales = sorted(set(args.scales) | {args.primary_scale})
    primary = args.primary_scale

    # Group the audited val indices by (survey, program) → one I/O pass per
    # spectrum serves BOTH the flip-rate test (A) and the equity table (B).
    groups = defaultdict(list)
    for i in val_idx[: args.max_spectra]:
        groups[full.meta_for_index(i)].append(i)

    # Global Part-A accumulators (primary scale)
    spectrum_flips = []           # per-spectrum mean token-flip
    position_flip_sum = None      # (T,) summed per-token token-flip
    position_n = 0
    snr_flat, tokflip_flat, bitflip_flat = [], [], []  # per-token at primary scale
    # Margin-sweep accumulators (global mean per scale)
    sweep = {s: {"tok": 0.0, "bit": 0.0, "n": 0} for s in scales}

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
            baseline_bits = bits_from_indices(baseline, dim)    # (B, T, dim)

            B, T = baseline.shape
            acc["n_spectra"] += B
            acc["nll_sum"] += float(loss["nll_flux"].item()) * B
            acc["nll_n"] += B
            x_grid = model._to_grid(x, wave)
            snr_grid = x_grid[:, 0] * x_grid[:, 1]
            ivar_g = x_grid[:, 1].square()
            ss_res, ss_tot = flux_r2_terms(x_grid[:, 0], recon[:, 0].float(), ivar=ivar_g)
            acc["ssr"] += ss_res.sum().item()
            acc["sst"] += ss_tot.sum().item()
            snr_spec = snr_grid.median(dim=-1).values  # (B,)
            hi = snr_spec > 3.0
            if hi.any():
                acc["ssr_hi"] += ss_res[hi].sum().item()
                acc["sst_hi"] += ss_tot[hi].sum().item()
            codes_seen[indices.unique()] = True

            # ----- Part A: flip rate over the margin sweep
            tsnr = token_local_snr(snr_grid, T)  # (B, T)
            for s in scales:
                K = args.n_noise if s == primary else args.sweep_noise
                ptf, pbf = _flip_for_scale(model, flux, istd, noise_std, wave,
                                           baseline, baseline_bits, s, K, dim, gen, args.amp)
                sweep[s]["tok"] += ptf.sum().item()
                sweep[s]["bit"] += pbf.sum().item()
                sweep[s]["n"] += ptf.numel()
                if s == primary:
                    spectrum_flips.extend(ptf.mean(dim=1).tolist())
                    position_flip_sum = (ptf.sum(dim=0) if position_flip_sum is None
                                         else position_flip_sum + ptf.sum(dim=0))
                    position_n += B
                    snr_flat.extend(tsnr.flatten().tolist())
                    tokflip_flat.extend(ptf.flatten().tolist())
                    bitflip_flat.extend(pbf.flatten().tolist())

        acc["n_codes_seen"] = int(codes_seen.sum().item())
        group_rows[(survey, program)] = summarize_group(acc, codebook_size)
        r = group_rows[(survey, program)]
        print(f"[group {survey}/{program}] n={acc['n_spectra']} "
              f"chi2={r['chi2_per_pixel']:.4f} codebook={r['codebook_use']:.3f}")

    # ---- Stratify primary-scale flip by SNR; high/low-SNR summary numbers ----
    snr_arr = np.asarray(snr_flat)
    tok_arr = np.asarray(tokflip_flat)
    bit_arr = np.asarray(bitflip_flat)
    labels, tok_bin, counts = stratify_by_snr(snr_arr, tok_arr, args.snr_edges)
    _, bit_bin, _ = stratify_by_snr(snr_arr, bit_arr, args.snr_edges)

    def _slice(mask, arr):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    gt3, gt10, le1 = snr_arr > 3.0, snr_arr > 10.0, snr_arr <= 1.0
    flip = {
        "token_high_snr_gt3": _slice(gt3, tok_arr),
        "token_high_snr_gt10": _slice(gt10, tok_arr),
        "token_low_snr_le1": _slice(le1, tok_arr),
        "bit_high_snr_gt3": _slice(gt3, bit_arr),
        "bit_high_snr_gt10": _slice(gt10, bit_arr),
        "bit_low_snr_le1": _slice(le1, bit_arr),
    }
    spectrum_flips = np.asarray(spectrum_flips)

    passed, reasons, notes = verdict(
        flip["bit_high_snr_gt10"], flip["token_high_snr_gt10"], group_rows,
        bit_thresh=args.bit_thresh, token_hi_thresh=args.token_hi_thresh,
        chi2_thresh=args.chi2_thresh, codebook_thresh=args.codebook_thresh,
        min_n=args.min_n_equity)

    results = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "n_spectra_audited": int(spectrum_flips.size),
        "n_noise": args.n_noise,
        "sweep_noise": args.sweep_noise,
        "primary_scale": primary,
        "codebook_dim_bits": dim,
        "flip_rate": {
            "per_spectrum_token_median": float(np.median(spectrum_flips)) if spectrum_flips.size else float("nan"),
            "per_spectrum_token_p90": float(np.percentile(spectrum_flips, 90)) if spectrum_flips.size else float("nan"),
            **flip,
            "by_snr_bin": [
                {"snr": lab, "token_flip": t, "bit_flip": b, "n_tokens": c}
                for lab, t, b, c in zip(labels, tok_bin, bit_bin, counts)
            ],
            "per_token_position": (position_flip_sum / max(position_n, 1)).tolist()
            if position_flip_sum is not None else [],
        },
        "margin_sweep": [
            {"scale": s, "token_flip": sweep[s]["tok"] / max(sweep[s]["n"], 1),
             "bit_flip": sweep[s]["bit"] / max(sweep[s]["n"], 1)}
            for s in scales
        ],
        "equity": {f"{s}/{p}": row for (s, p), row in group_rows.items()},
        "verdict": {"passed": passed, "reasons": reasons, "notes": notes,
                    "thresholds": {"bit": args.bit_thresh, "token_hi": args.token_hi_thresh,
                                   "chi2": args.chi2_thresh, "codebook": args.codebook_thresh,
                                   "min_n_equity": args.min_n_equity}},
    }
    return results, (snr_arr, tok_arr, bit_arr, spectrum_flips)


def write_markdown(results: dict, path: Path):
    fr = results["flip_rate"]
    lines = [
        "# T3 token-stability audit",
        "",
        f"- checkpoint: `{results['checkpoint']}`",
        f"- spectra audited: {results['n_spectra_audited']}  |  "
        f"noise realizations: {results['n_noise']} (sweep {results['sweep_noise']})  |  "
        f"bits/token: {results['codebook_dim_bits']}",
        "",
        "## A. Noise-realization flip rate (primary scale "
        f"{results['primary_scale']:g}σ)",
        "",
        "Per-bit flip is the honest granularity; the token index collapses "
        f"{results['codebook_dim_bits']} bits so it overcounts.",
        "",
        f"- **per-bit flip, high-SNR (>10): {fr['bit_high_snr_gt10']:.3f}**  ← the gate",
        f"- per-bit flip, SNR>3: {fr['bit_high_snr_gt3']:.3f}  |  low-SNR (<=1): {fr['bit_low_snr_le1']:.3f}",
        f"- token-index flip, high-SNR (>10): {fr['token_high_snr_gt10']:.3f}  (informational)",
        f"- per-spectrum token-flip median: {fr['per_spectrum_token_median']:.3f}  (p90 {fr['per_spectrum_token_p90']:.3f})",
        "",
        "| SNR bin | bit flip | token flip | n tokens |",
        "|---|---|---|---|",
    ]
    for row in fr["by_snr_bin"]:
        b = "nan" if row["bit_flip"] != row["bit_flip"] else f"{row['bit_flip']:.3f}"
        t = "nan" if row["token_flip"] != row["token_flip"] else f"{row['token_flip']:.3f}"
        lines.append(f"| {row['snr']} | {b} | {t} | {row['n_tokens']} |")
    lines += [
        "",
        "Healthy signature: bit flip falls as SNR rises; high-SNR bits stable.",
        "",
        "## Margin sweep (flip vs perturbation scale)",
        "",
        "| scale (×σ) | bit flip | token flip |",
        "|---|---|---|",
    ]
    for row in results["margin_sweep"]:
        lines.append(f"| {row['scale']:g} | {row['bit_flip']:.3f} | {row['token_flip']:.3f} |")
    lines += [
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
        lines += ["", "**Reasons (hard):**"] + [f"- {r}" for r in v["reasons"]]
    if v["notes"]:
        lines += ["", "**Notes (informational):**"] + [f"- {n}" for n in v["notes"]]
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
    step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
    print(f"[model] loaded {args.checkpoint} (step {step})")

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
    snr_arr, tok_arr, bit_arr, spectrum_flips = arrays
    np.savez(args.out / "t3_stability.npz",
             token_snr=snr_arr, token_flip=tok_arr, bit_flip=bit_arr,
             spectrum_flip=spectrum_flips)

    v = results["verdict"]
    print("\n" + "=" * 60)
    print(f"VERDICT: {'PASS' if v['passed'] else 'FLAG'}")
    for r in v["reasons"]:
        print(f"  [hard] {r}")
    for n in v["notes"]:
        print(f"  [note] {n}")
    print(f"per-bit flip high-SNR(>10) {results['flip_rate']['bit_high_snr_gt10']:.3f} "
          f"| token-flip high-SNR(>10) {results['flip_rate']['token_high_snr_gt10']:.3f}")
    print(f"wrote {args.out / 't3_stability.json'} (+ .md, .npz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
