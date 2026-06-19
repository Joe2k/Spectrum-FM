"""Unit tests for flux-space spectrum reconstruction metrics (X10).

CPU-only, no model/tokenizer. Validates the ivar-weighted χ²/pixel + R², the
token→pixel mask expansion, and the additive accumulation across batches.
"""

import torch

from src.eval.spectrum_metrics import (
    PIXELS_PER_TOKEN,
    add_sums,
    finalize_recon,
    flux_reconstruction_metrics,
    recon_weighted_sums,
    token_mask_to_pixel_mask,
)


def test_perfect_reconstruction():
    g = torch.Generator().manual_seed(0)
    flux = torch.randn(4, 8704, generator=g)
    istd = torch.rand(4, 8704, generator=g) + 0.5
    m = flux_reconstruction_metrics(flux.clone(), flux, istd)
    assert m["chi2_per_pixel"] < 1e-10
    assert abs(m["ivar_r2"] - 1.0) < 1e-6


def test_chi2_is_one_at_noise_floor():
    # Residual ~ N(0, σ) with istd = 1/σ ⇒ E[ivar·resid²] = 1.
    g = torch.Generator().manual_seed(1)
    n = 200_000
    sigma = 0.7
    flux_obs = torch.randn(n, generator=g) * 3.0
    resid = torch.randn(n, generator=g) * sigma
    istd = torch.full((n,), 1.0 / sigma)
    m = flux_reconstruction_metrics(flux_obs + resid, flux_obs, istd)
    assert abs(m["chi2_per_pixel"] - 1.0) < 0.05


def test_zero_istd_pixels_excluded():
    flux_obs = torch.zeros(1, 10)
    flux_pred = torch.full((1, 10), 5.0)   # would be a huge residual...
    istd = torch.ones(1, 10)
    istd[0, :5] = 0.0                       # ...but half the pixels are masked out
    m = flux_reconstruction_metrics(flux_pred, flux_obs, istd)
    assert m["n"] == 5


def test_token_mask_to_pixel_mask():
    tm = torch.tensor([[True, False, True]])
    pm = token_mask_to_pixel_mask(tm, pixels_per_token=4)
    assert pm.shape == (1, 12)
    assert pm[0].tolist() == [True] * 4 + [False] * 4 + [True] * 4


def test_pixel_mask_restricts_metric():
    flux_obs = torch.zeros(1, 8)
    flux_pred = torch.zeros(1, 8)
    flux_pred[0, 4:] = 10.0                 # error only in the second half
    istd = torch.ones(1, 8)
    pmask = torch.tensor([[True] * 4 + [False] * 4])  # score only the clean half
    m = flux_reconstruction_metrics(flux_pred, flux_obs, istd, pmask)
    assert m["n"] == 4
    assert m["chi2_per_pixel"] < 1e-10


def test_sums_combine_additively():
    g = torch.Generator().manual_seed(2)
    fp = torch.randn(6, 100, generator=g)
    fo = torch.randn(6, 100, generator=g)
    istd = torch.rand(6, 100, generator=g) + 0.2
    full = finalize_recon(recon_weighted_sums(fp, fo, istd))
    a = recon_weighted_sums(fp[:2], fo[:2], istd[:2])
    b = recon_weighted_sums(fp[2:], fo[2:], istd[2:])
    split = finalize_recon(add_sums(a, b))
    assert abs(full["chi2_per_pixel"] - split["chi2_per_pixel"]) < 1e-6
    assert abs(full["ivar_r2"] - split["ivar_r2"]) < 1e-6
    assert full["n"] == split["n"] == 600


def test_empty_is_nan_safe():
    out = finalize_recon({"Sw": torch.tensor(0.0), "Swy": torch.tensor(0.0),
                          "Swy2": torch.tensor(0.0), "Swr2": torch.tensor(0.0),
                          "n": torch.tensor(0.0)})
    assert out["n"] == 0
    assert out["chi2_per_pixel"] != out["chi2_per_pixel"]  # NaN


def test_pixels_per_token_constant():
    assert PIXELS_PER_TOKEN == 32
