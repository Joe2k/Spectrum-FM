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

import math
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


def redshift_posterior(
    red_logits: torch.Tensor, z_tok, temperature: float = 1.0
) -> torch.Tensor:
    """Softmax over the redshift sub-vocab → posterior probs (..., n_bins).

    ``temperature > 1`` softens an over-confident posterior; ``< 1`` sharpens it.
    Temperature scaling is monotone in the logits, so the argmax (and hence the
    point prediction) is unchanged — only the spread / calibration moves.
    """
    z = _z_bin_logits(red_logits, z_tok)
    if temperature != 1.0:
        z = z / temperature
    return F.softmax(z, dim=-1)


def apply_temperature(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """Re-temperature an existing posterior in probability space.

    Equivalent to ``softmax(logits / T)`` even though we only kept the ``T = 1``
    probabilities: ``softmax(log p / T)`` differs from the logits only by the
    per-row normalisation constant, which cancels under the softmax. Lets the
    offline analysis (and notebooks) recalibrate from stored posteriors without
    needing the raw logits.
    """
    probs = probs.float()
    if temperature == 1.0:
        return probs
    logp = probs.clamp_min(1e-20).log()
    return F.softmax(logp / temperature, dim=-1)


def fit_temperature(
    probs: torch.Tensor,
    z_true: torch.Tensor,
    z_tok,
    bounds: Sequence[float] = (0.25, 8.0),
    iters: int = 60,
) -> float:
    """Fit a single temperature ``T > 0`` minimising the NLL of the true z bin
    under the re-tempered posterior (Guo et al. 2017, applied to the z sub-vocab).

    Operates on the ``T = 1`` posterior ``probs`` (..., n_bins) — equivalent to
    scaling the logits (see :func:`apply_temperature`). ``T > 1`` softens an
    over-confident model (high PIT-KS); ``T ≈ 1`` means already calibrated.

    Derivative-free golden-section search on ``log T`` (the NLL is unimodal in
    ``T``); CPU-safe, no autograd, so it runs the same in notebooks.
    """
    probs = probs.float().reshape(-1, probs.shape[-1])
    true_bin = (z_tok.encode(z_true.flatten().cpu()).long()
                .clamp(0, probs.shape[-1] - 1))
    logp = probs.clamp_min(1e-20).log()  # (N, n_bins); shifted logits

    def nll(t: float) -> float:
        lp = F.log_softmax(logp / t, dim=-1)
        return float(-lp.gather(-1, true_bin[:, None]).squeeze(-1).mean().item())

    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = nll(math.exp(c)), nll(math.exp(d))
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = nll(math.exp(c))
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = nll(math.exp(d))
    return float(math.exp(0.5 * (lo + hi)))


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


def fit_pit_recalibrator(pit_cal: torch.Tensor) -> torch.Tensor:
    """Fit a monotone PIT recalibration map (Kuleshov et al. 2018) on a
    calibration set.

    X8b showed a single temperature is the wrong instrument for this
    soft-labelled posterior: the σ=24-bin soft labels make it *over-dispersed by
    design* (central-peaked PIT), so NLL-temperature only sharpens to the search
    floor — barely helping PIT while destroying the `posterior_std_z` rejection
    that is the whole result. The fix is a monotone remap of the PIT itself: its
    empirical CDF, which for this objective IS the isotonic-regression solution.

    Because the map acts on PIT (not on the rejection score), the point
    predictions, σ_NMAD and the outlier-rejection ranking are **unchanged** — it
    fixes absolute calibration without the accuracy/rejection trade-off.

    Returns the sorted calibration PITs (the ECDF knots); pass to
    :func:`apply_pit_recalibration`. Fit on a split disjoint from the one you
    score, or it is trivially perfect.
    """
    return torch.sort(pit_cal.flatten().float()).values


def apply_pit_recalibration(pit: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Map PIT values through a fitted calibration ECDF (monotone
    non-decreasing, [0,1] → [0,1]). A calibrated posterior ⇒ recalibrated
    PIT ~ Uniform(0,1)."""
    pit = pit.flatten().float().contiguous()
    n = int(knots.numel())
    if n == 0:
        return pit
    # ECDF: P(cal ≤ t) = (#knots ≤ t) / n.
    idx = torch.searchsorted(knots.contiguous(), pit, right=True)
    return (idx.float() / n).clamp(0.0, 1.0)
