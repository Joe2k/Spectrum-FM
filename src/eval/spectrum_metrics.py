"""
Flux-space spectrum reconstruction metrics (the X4-for-spectrum).
=================================================================

`spectrum_acc` (exact top-1 over the 1024-way codebook) is to spectra what bin
accuracy was to redshift: **misleading**. Tokenizer v2 sits at the
information-theoretic floor (χ²/pixel ≈ 1.0, code-recon R² 0.998), so a predicted
code that is *off by one* in a fine codebook scores as "wrong" yet decodes to a
spectrum within the noise. The honest question is the same one X4 asked for z:
when the transformer's predicted (masked) tokens are decoded back to flux, how
close is that flux to the *observation*, weighted by DESI's own inverse-variance?

This module scores reconstruction in flux space:

    χ²/pixel        = mean_valid[ ivar · (flux_pred − flux_obs)² ]   (1.0 = at the
                      noise floor, indistinguishable from the observation)
    ivar-weighted R² = 1 − Σ ivar·(flux_pred−flux_obs)² / Σ ivar·(flux_obs−ȳ_w)²

Both are reported on the **masked-token pixel blocks** (the blind-reconstruction
number that mirrors `masked_spec_acc`) and over all valid pixels, against the
codec-only ceiling (decode the *true* tokens) so the transformer's added error is
explicit. Sums are accumulated so batches combine exactly (single- or two-pass).

Torch-only and CPU-safe — importable into notebooks (incl. the SDSS OOD check)
the same way as `redshift_metrics`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from src.tokenizers.spectrum import LATENT_GRID_SIZE, N_TOKENS

# Each spectrum token governs this many flux-grid pixels (stride-32 encoder).
PIXELS_PER_TOKEN = LATENT_GRID_SIZE // N_TOKENS  # 8704 // 272 = 32


def token_mask_to_pixel_mask(
    token_mask: torch.Tensor, pixels_per_token: int = PIXELS_PER_TOKEN
) -> torch.Tensor:
    """Expand a per-token boolean mask (B, T) to per-pixel (B, T·pixels_per_token).

    Token ``i`` maps to flux-grid pixels ``[i·s : (i+1)·s)`` (s = stride 32), so
    the masked-token reconstruction can be scored on exactly the pixels those
    tokens generate.
    """
    return token_mask.bool().repeat_interleave(pixels_per_token, dim=-1)


def recon_weighted_sums(
    flux_pred: torch.Tensor,
    flux_obs: torch.Tensor,
    istd: torch.Tensor,
    pixel_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Inverse-variance-weighted reconstruction sufficient statistics.

    Args:
        flux_pred, flux_obs, istd: (..., n_pixels) predicted flux, observed flux,
            and observed inverse-std (= √ivar; 0 outside wavelength coverage).
        pixel_mask: optional boolean of the same shape; restrict to these pixels
            (e.g. the masked-token blocks). Pixels with ``istd == 0`` are always
            excluded so out-of-coverage grid points don't count.

    Returns scalar tensors {Sw, Swy, Swy2, Swr2, n} that combine additively
    across batches; pass the running totals to :func:`finalize_recon`.
    """
    valid = istd > 0
    if pixel_mask is not None:
        valid = valid & pixel_mask.bool()
    w = (istd * istd)[valid]            # ivar
    y = flux_obs[valid]
    r = (flux_pred - flux_obs)[valid]
    return {
        "Sw": w.sum(),
        "Swy": (w * y).sum(),
        "Swy2": (w * y * y).sum(),
        "Swr2": (w * r * r).sum(),       # = Σ ivar·resid² (drives both χ² and R²)
        "n": valid.sum().to(w.dtype),
    }


def finalize_recon(sums: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Turn accumulated :func:`recon_weighted_sums` totals into χ²/pixel and the
    ivar-weighted R². NaN-safe on empty / degenerate input."""
    n = float(sums["n"])
    Sw = float(sums["Sw"])
    if n == 0 or Sw == 0:
        nan = float("nan")
        return {"chi2_per_pixel": nan, "ivar_r2": nan, "n": 0}
    ss_res = float(sums["Swr2"])
    ss_tot = float(sums["Swy2"]) - float(sums["Swy"]) ** 2 / Sw
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"chi2_per_pixel": ss_res / n, "ivar_r2": r2, "n": int(n)}


def add_sums(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Add two sufficient-statistic dicts (running accumulation over batches)."""
    return {k: a[k] + b[k] for k in a}


def flux_reconstruction_metrics(
    flux_pred: torch.Tensor,
    flux_obs: torch.Tensor,
    istd: torch.Tensor,
    pixel_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Single-shot χ²/pixel + ivar-weighted R² (sums → finalize in one call)."""
    return finalize_recon(recon_weighted_sums(flux_pred, flux_obs, istd, pixel_mask))
