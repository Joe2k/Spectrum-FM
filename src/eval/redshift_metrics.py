"""
Physical redshift metrics + continuous (sub-bin) decoding.
=========================================================

The transformer predicts redshift as a categorical distribution over the
quantized z-bins (the redshift sub-vocab of its logits). Bin *accuracy*
(``redshift_acc`` / ``within2``) is the metric we have logged so far, but it is
not what the redshift literature — or AION-1 — reports, and it throws away the
probability mass spread across neighbouring bins (which hurts soft / fine-bin
models the most). The field measures redshift quality in physical units:

    Δz = (z_pred - z_true) / (1 + z_true)
    σ_NMAD          = 1.4826 · median(|Δz - median(Δz)|)
    catastrophic η  = fraction with |Δz| > 0.0033   (DESI/Redrock convention)
    bias            = median(Δz)

This module provides those metrics plus two ways to turn the predicted bin
distribution back into a continuous redshift:

  - ``mode="argmax"``  : decode the single most-likely bin (legacy behaviour).
  - ``mode="expected"``: probability-weighted expected value — sub-bin
    precision at zero training cost. Because the bins are uniform in the
    Gaussian *latent* of the ``RedshiftTokenizer`` (CDF→Gaussian→FSQ), the
    expectation is taken in that latent space and mapped back through the
    tokenizer's inverse CDF, which is unbiased w.r.t. the skewed z density.

All functions are torch-only and CPU-safe; nothing here depends on the training
stack, so they can be imported directly into notebooks (e.g. for the SDSS
out-of-distribution check) alongside ``SpectrumTokenizer.resample_to_grid``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.models.transformer import REDSHIFT_TOKEN_OFFSET

# DESI/Redrock catastrophic-failure threshold on |Δz|/(1+z).
CATASTROPHIC_DZ = 0.0033


def redshift_bin_centers(z_tok) -> torch.Tensor:
    """Continuous redshift value at the centre of each quantization bin.

    Returns a ``(n_levels,)`` tensor (on the tokenizer's device, i.e. CPU).
    """
    idx = torch.arange(z_tok.n_levels)
    return z_tok.decode(idx)


def _z_bin_logits(red_logits: torch.Tensor, z_tok) -> torch.Tensor:
    """Slice the redshift sub-vocab out of full-vocab logits → (..., n_levels)."""
    off = REDSHIFT_TOKEN_OFFSET
    n = z_tok.n_levels
    return red_logits[..., off:off + n].float()


def decode_redshift(red_logits: torch.Tensor, z_tok, mode: str = "expected") -> torch.Tensor:
    """Decode full-vocab redshift logits to continuous redshift(s).

    Args:
        red_logits: (..., V) logits at the redshift position (position 0 of the
            decoder output). Only the redshift sub-vocab is used.
        z_tok: a fitted ``RedshiftTokenizer``.
        mode: ``"expected"`` (probability-weighted, sub-bin precision, default)
            or ``"argmax"`` (single most-likely bin).

    Returns:
        A CPU float tensor of redshifts with the leading shape of ``red_logits``.
    """
    z_logits = _z_bin_logits(red_logits, z_tok)
    n = z_tok.n_levels

    if mode == "argmax":
        idx = z_logits.argmax(dim=-1).cpu()
        return z_tok.decode(idx).float()

    if mode != "expected":
        raise ValueError(f"mode must be 'expected' or 'argmax', got {mode!r}")

    # Expected value in the Gaussian latent (bins uniform there), then map back
    # through the tokenizer's CDF→z so the skewed z density is handled correctly.
    probs = F.softmax(z_logits, dim=-1)
    R = z_tok.gaussian_range
    k = torch.arange(n, device=z_logits.device, dtype=probs.dtype)
    g = k / (n - 1) * (2.0 * R) - R  # (n,) bin-centre Gaussian values
    g_exp = (probs * g).sum(dim=-1)  # (...,)
    cdf = z_tok.gaussian_to_cdf(g_exp)
    z = z_tok._inverse_cdf(cdf.cpu())
    return z.float()


def redshift_metrics(
    z_pred: torch.Tensor,
    z_true: torch.Tensor,
    catastrophic_dz: float = CATASTROPHIC_DZ,
) -> Dict[str, float]:
    """AION-comparable redshift quality metrics from continuous predictions.

    Args:
        z_pred: predicted redshifts (any shape).
        z_true: ground-truth continuous redshifts (same number of elements).
        catastrophic_dz: |Δz| threshold for the catastrophic outlier fraction
            (default 0.0033, the DESI/Redrock convention).

    Returns:
        dict with ``sigma_nmad``, ``bias``, ``outlier_frac`` (at
        ``catastrophic_dz``), ``outlier_frac_05`` (at 0.05), ``rms``,
        ``mean_abs`` and ``n``. NaN-safe on empty input.
    """
    z_pred = z_pred.flatten().float()
    z_true = z_true.flatten().float()
    n = int(z_true.numel())
    if n == 0:
        nan = float("nan")
        return {"sigma_nmad": nan, "bias": nan, "outlier_frac": nan,
                "outlier_frac_05": nan, "rms": nan, "mean_abs": nan, "n": 0}

    dz = (z_pred - z_true) / (1.0 + z_true)
    med = dz.median()
    sigma_nmad = 1.4826 * (dz - med).abs().median()
    adz = dz.abs()
    return {
        "sigma_nmad": float(sigma_nmad.item()),
        "bias": float(med.item()),
        "outlier_frac": float((adz > catastrophic_dz).float().mean().item()),
        "outlier_frac_05": float((adz > 0.05).float().mean().item()),
        "rms": float(dz.pow(2).mean().sqrt().item()),
        "mean_abs": float(adz.mean().item()),
        "n": n,
    }
