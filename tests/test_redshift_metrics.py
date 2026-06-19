"""Unit tests for physical redshift metrics + continuous decoding (X4).

CPU-only, no model/training deps. Validates:
  - redshift_metrics closed-form behaviour (perfect / offset / empty),
  - decode_redshift expected==argmax for one-hot logits,
  - sharper bin distributions yield lower σ_NMAD (the sub-bin gain).
"""

import math

import torch

from src.eval.redshift_metrics import (
    decode_redshift,
    redshift_bin_centers,
    redshift_metrics,
)
from src.models.transformer import REDSHIFT_TOKEN_OFFSET, vocab_size_for_z_bins
from src.tokenizers.redshift import RedshiftTokenizer


def _fitted_tok(n_levels=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Skewed, DESI-like: many near 0, tail to ~3.
    z = torch.cat([
        torch.rand(4000, generator=g) * 0.3,
        torch.rand(2000, generator=g) * 3.0,
    ])
    tok = RedshiftTokenizer(n_levels=n_levels)
    tok.fit(z)
    return tok


def _onehot_logits(bins, n_levels, big=30.0):
    """(B, V) logits with a single large value at REDSHIFT_TOKEN_OFFSET+bin."""
    V = vocab_size_for_z_bins(n_levels)
    logits = torch.zeros(len(bins), V)
    for i, b in enumerate(bins):
        logits[i, REDSHIFT_TOKEN_OFFSET + int(b)] = big
    return logits


# --------------------------------------------------------------------------- #
# redshift_metrics
# --------------------------------------------------------------------------- #
def test_perfect_prediction_zero_error():
    z = torch.tensor([0.01, 0.5, 1.0, 2.0, 0.3])
    m = redshift_metrics(z.clone(), z.clone())
    assert m["sigma_nmad"] == 0.0
    assert m["bias"] == 0.0
    assert m["outlier_frac"] == 0.0
    assert m["outlier_frac_05"] == 0.0
    assert m["n"] == 5


def test_constant_offset_matches_closed_form():
    z_true = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    z_pred = z_true + 0.02
    m = redshift_metrics(z_pred, z_true)
    dz = (z_pred - z_true) / (1.0 + z_true)
    assert math.isclose(m["bias"], float(dz.median()), rel_tol=1e-6, abs_tol=1e-9)
    # All |dz| at z<=... cross 0.0033 -> all are catastrophic outliers here.
    assert m["outlier_frac"] == 1.0
    # sigma_nmad is small but non-negative.
    assert m["sigma_nmad"] >= 0.0


def test_empty_is_nan_safe():
    m = redshift_metrics(torch.empty(0), torch.empty(0))
    assert m["n"] == 0
    assert math.isnan(m["sigma_nmad"])
    assert math.isnan(m["bias"])


# --------------------------------------------------------------------------- #
# decode_redshift
# --------------------------------------------------------------------------- #
def test_bin_centers_shape_and_monotonic():
    tok = _fitted_tok()
    c = redshift_bin_centers(tok)
    assert c.shape == (tok.n_levels,)
    # CDF->Gaussian->z mapping is monotone non-decreasing.
    assert torch.all(c[1:] - c[:-1] >= -1e-6)


def test_onehot_expected_equals_argmax():
    tok = _fitted_tok()
    bins = [0, 5, 20, 40, 63]
    logits = _onehot_logits(bins, tok.n_levels)
    z_exp = decode_redshift(logits, tok, mode="expected")
    z_arg = decode_redshift(logits, tok, mode="argmax")
    z_dir = tok.decode(torch.tensor(bins)).float()
    assert torch.allclose(z_exp, z_arg, atol=1e-5)
    assert torch.allclose(z_exp, z_dir, atol=1e-5)


def test_decode_output_shape_and_finite():
    tok = _fitted_tok()
    logits = torch.randn(7, vocab_size_for_z_bins(tok.n_levels))
    for mode in ("expected", "argmax"):
        z = decode_redshift(logits, tok, mode=mode)
        assert z.shape == (7,)
        assert torch.isfinite(z).all()


def test_expected_beats_argmax_subbin():
    """For sub-bin truths, expected-value decoding interpolates between bin
    centres and beats argmax (which snaps to a centre) on σ_NMAD — the whole
    reason we report the continuous metric."""
    tok = _fitted_tok(n_levels=128)
    n_levels = tok.n_levels
    R = tok.gaussian_range
    g = torch.Generator().manual_seed(7)
    base_bins = torch.randint(10, n_levels - 11, (256,), generator=g)
    # True z sits half a bin above each integer centre (genuinely sub-bin).
    frac = base_bins.float() + 0.5
    gauss = frac / (n_levels - 1) * (2.0 * R) - R
    z_true = tok._inverse_cdf(tok.gaussian_to_cdf(gauss)).float()

    # Sharp distribution centred on the fractional position.
    k = torch.arange(n_levels).float()
    V = vocab_size_for_z_bins(n_levels)
    logits = torch.full((256, V), -1e4)
    logits[:, REDSHIFT_TOKEN_OFFSET:REDSHIFT_TOKEN_OFFSET + n_levels] = (
        -1.0 * (k[None, :] - frac[:, None]) ** 2)

    nmad_exp = redshift_metrics(
        decode_redshift(logits, tok, mode="expected"), z_true)["sigma_nmad"]
    nmad_arg = redshift_metrics(
        decode_redshift(logits, tok, mode="argmax"), z_true)["sigma_nmad"]
    assert nmad_exp < nmad_arg
