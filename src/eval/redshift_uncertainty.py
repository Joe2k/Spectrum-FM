"""
Redshift uncertainty, calibration & outlier rejection (X8).
==========================================================

X4 showed our σ_NMAD is already SpecPT-competitive (~7e-4) with ~zero bias, but
the catastrophic-outlier fraction (|Δz|/(1+z) > 0.0033) sits at ~9-11% vs
SpecPT's <1%. Those failures are line-confusion / multimodal-posterior cases,
not resolution. The redshift position's softmax IS a discretized z-posterior, so
we can:

  - score each prediction's uncertainty (entropy, peak height, bimodality, …),
  - reject the most-uncertain fraction (a Redrock-ZWARN-style quality cut) and
    watch the outlier fraction collapse toward SpecPT levels,
  - check the posterior is calibrated (PIT ~ Uniform) — a real differentiator vs
    point-estimate baselines.

All functions are torch-only and CPU-safe; import them into notebooks (incl. the
SDSS OOD check) the same way as `redshift_metrics`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from src.eval.redshift_metrics import (
    CATASTROPHIC_DZ,
    _z_bin_logits,
    redshift_bin_centers,
    redshift_metrics,
)

DEFAULT_COVERAGE = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)

# Rejection score chosen by held-out outlier-detection AUROC (2026-06-18 eval):
# posterior_std_z separates catastrophic outliers far better than entropy —
# AUROC 0.976 (4096-soft) / 0.932 (512-hard) vs entropy 0.846 / 0.590. With a
# 10% std cut the 4096-soft model reaches η>0.0033 = 0.69% @ σ_NMAD 5.3e-4.
DEFAULT_REJECTION_SCORE = "posterior_std_z"


def redshift_posterior(red_logits: torch.Tensor, z_tok) -> torch.Tensor:
    """Softmax over the redshift sub-vocab → posterior probs (..., n_bins)."""
    return F.softmax(_z_bin_logits(red_logits, z_tok), dim=-1)


def uncertainty_scores(probs: torch.Tensor, z_tok, k: int = 2) -> Dict[str, torch.Tensor]:
    """Per-sample uncertainty scores from a bin posterior (higher = less sure).

    Args:
        probs: (..., n_bins) posterior over z bins.
        z_tok: fitted RedshiftTokenizer (for z-space std).
        k: half-width (in bins) of the mode window for `mass_outside_k`.

    Returns dict of (...,) tensors:
        entropy, neg_max_prob, second_mode_ratio, posterior_std_z, mass_outside_k.
    """
    probs = probs.float()
    n = probs.shape[-1]
    eps = 1e-12

    entropy = -(probs * (probs + eps).log()).sum(dim=-1)
    max_prob, mode = probs.max(dim=-1)
    neg_max_prob = 1.0 - max_prob

    # Bimodality: probability of the strongest peak OUTSIDE a window around the
    # mode (a second emission-line solution shows up here). Mask ±1 bin of mode.
    idx = torch.arange(n, device=probs.device)
    near_mode = (idx.view(*([1] * (probs.dim() - 1)), n)
                 - mode.unsqueeze(-1)).abs() <= 1
    masked = probs.masked_fill(near_mode, 0.0)
    second_peak = masked.max(dim=-1).values
    second_mode_ratio = second_peak / max_prob.clamp(min=eps)

    centers = redshift_bin_centers(z_tok).to(probs.device).float()  # (n,)
    mean_z = (probs * centers).sum(dim=-1)
    var_z = (probs * (centers - mean_z.unsqueeze(-1)) ** 2).sum(dim=-1)
    posterior_std_z = var_z.clamp(min=0.0).sqrt()

    # Mass outside ±k bins of the mode (local concentration).
    within_k = (idx.view(*([1] * (probs.dim() - 1)), n)
                - mode.unsqueeze(-1)).abs() <= k
    mass_within_k = (probs * within_k).sum(dim=-1)
    mass_outside_k = 1.0 - mass_within_k

    return {
        "entropy": entropy,
        "neg_max_prob": neg_max_prob,
        "second_mode_ratio": second_mode_ratio,
        "posterior_std_z": posterior_std_z,
        "mass_outside_k": mass_outside_k,
    }


def pit_values(probs: torch.Tensor, z_true: torch.Tensor, z_tok) -> torch.Tensor:
    """Probability Integral Transform: posterior CDF evaluated at the true z.

    PIT_i = Σ_{bins ≤ true bin} p_k. A calibrated posterior ⇒ PIT ~ Uniform(0,1).
    Returns (...,) in [0, 1].
    """
    probs = probs.float()
    true_bin = z_tok.encode(z_true.flatten().cpu()).to(probs.device).long()
    true_bin = true_bin.clamp(0, probs.shape[-1] - 1).reshape(probs.shape[:-1])
    cdf = probs.cumsum(dim=-1)
    return cdf.gather(-1, true_bin.unsqueeze(-1)).squeeze(-1).clamp(0.0, 1.0)


def coverage_quality_curve(
    z_pred: torch.Tensor,
    z_true: torch.Tensor,
    score: torch.Tensor,
    fractions: Sequence[float] = DEFAULT_COVERAGE,
    catastrophic_dz: float = CATASTROPHIC_DZ,
) -> List[Dict[str, float]]:
    """Quality vs coverage: keep the most-confident `frac` of predictions
    (lowest `score`) and report σ_NMAD / outlier fraction on the kept subset.

    Returns a list of rows {retained_frac, sigma_nmad, outlier_frac,
    outlier_frac_05, n} ordered by descending coverage.
    """
    z_pred = z_pred.flatten().float()
    z_true = z_true.flatten().float()
    score = score.flatten().float()
    n = z_true.numel()
    order = torch.argsort(score)  # most confident (lowest score) first
    rows: List[Dict[str, float]] = []
    for f in fractions:
        keep = max(1, int(round(f * n)))
        sel = order[:keep]
        m = redshift_metrics(z_pred[sel], z_true[sel], catastrophic_dz)
        rows.append({
            "retained_frac": float(keep) / n,
            "sigma_nmad": m["sigma_nmad"],
            "outlier_frac": m["outlier_frac"],
            "outlier_frac_05": m["outlier_frac_05"],
            "n": m["n"],
        })
    return rows


def outlier_auroc(score: torch.Tensor, is_outlier: torch.Tensor) -> float:
    """AUROC of `score` as a detector of `is_outlier` (rank-based, no sklearn).

    Returns 0.5 if either class is empty.
    """
    score = score.flatten().float()
    y = is_outlier.flatten().bool()
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Mann–Whitney U via average ranks.
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=score.dtype)
    # Average tied ranks so ties score 0.5.
    uniq, inv, counts = torch.unique(score, return_inverse=True, return_counts=True)
    sum_ranks = torch.zeros_like(uniq).scatter_add_(0, inv, ranks)
    ranks = (sum_ranks / counts)[inv]
    rank_pos = ranks[y].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc.item())


def pit_calibration_error(pit: torch.Tensor) -> float:
    """KS distance between the PIT empirical CDF and Uniform(0,1). 0 = perfect."""
    pit = pit.flatten().float()
    n = pit.numel()
    if n == 0:
        return float("nan")
    s = torch.sort(pit).values
    ecdf = torch.arange(1, n + 1, dtype=s.dtype) / n
    return float((ecdf - s).abs().max().item())


def pit_histogram(pit: torch.Tensor, bins: int = 10) -> torch.Tensor:
    """Coarse PIT histogram (counts per uniform bin) for a text/plot readout."""
    pit = pit.flatten().float().clamp(0.0, 1.0 - 1e-7)
    idx = (pit * bins).long().clamp(0, bins - 1)
    return torch.bincount(idx, minlength=bins)
