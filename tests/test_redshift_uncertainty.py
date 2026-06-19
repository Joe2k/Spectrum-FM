"""Unit tests for redshift uncertainty / calibration / outlier rejection (X8).

CPU-only, no model. Validates the scores, PIT, coverage curve, AUROC and the
PIT calibration statistic that back the rejection + calibration analysis.
"""

import math

import torch

from src.eval.redshift_uncertainty import (
    DEFAULT_REJECTION_SCORE,
    apply_pit_recalibration,
    apply_temperature,
    coverage_quality_curve,
    fit_pit_recalibrator,
    fit_temperature,
    outlier_auroc,
    pit_calibration_error,
    pit_histogram,
    pit_values,
    uncertainty_scores,
)
from src.tokenizers.redshift import RedshiftTokenizer


def test_default_rejection_score_is_a_real_score():
    tok = _fitted_tok(n_levels=64)
    probs = torch.softmax(torch.randn(4, tok.n_levels), dim=-1)
    assert DEFAULT_REJECTION_SCORE in uncertainty_scores(probs, tok)


def _fitted_tok(n_levels=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.cat([torch.rand(4000, generator=g) * 0.3,
                   torch.rand(2000, generator=g) * 3.0])
    tok = RedshiftTokenizer(n_levels=n_levels)
    tok.fit(z)
    return tok


def _gauss_probs(centers_bins, sigma, n):
    k = torch.arange(n).float()
    g = torch.exp(-0.5 * ((k[None, :] - centers_bins[:, None].float()) / sigma) ** 2)
    return g / g.sum(-1, keepdim=True)


# --------------------------------------------------------------------------- #
# uncertainty_scores
# --------------------------------------------------------------------------- #
def test_entropy_uniform_and_onehot():
    tok = _fitted_tok(n_levels=64)
    n = tok.n_levels
    uniform = torch.full((1, n), 1.0 / n)
    onehot = torch.zeros(1, n)
    onehot[0, 10] = 1.0
    s_u = uncertainty_scores(uniform, tok)
    s_o = uncertainty_scores(onehot, tok)
    assert math.isclose(s_u["entropy"].item(), math.log(n), rel_tol=1e-5)
    assert s_o["entropy"].item() < 1e-4
    assert math.isclose(s_o["neg_max_prob"].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(s_u["neg_max_prob"].item(), 1.0 - 1.0 / n, rel_tol=1e-5)


def test_bimodal_has_high_second_mode_ratio():
    tok = _fitted_tok(n_levels=128)
    # Two equal far-apart peaks vs one peak.
    bi = _gauss_probs(torch.tensor([30.0]), 1.0, 128) + \
        _gauss_probs(torch.tensor([90.0]), 1.0, 128)
    bi = bi / bi.sum(-1, keepdim=True)
    uni = _gauss_probs(torch.tensor([60.0]), 0.5, 128)
    r_bi = uncertainty_scores(bi, tok)["second_mode_ratio"].item()
    r_uni = uncertainty_scores(uni, tok)["second_mode_ratio"].item()
    assert r_bi > 0.5
    assert r_uni < 0.05
    assert r_bi > r_uni


# --------------------------------------------------------------------------- #
# pit_values + calibration
# --------------------------------------------------------------------------- #
def test_pit_in_range_and_sharp_at_truth():
    tok = _fitted_tok()
    n = tok.n_levels
    # Interior bins (edge bins fail the encode(decode(b)) round-trip by one).
    bins = torch.tensor([40, 90, 150, 220])
    onehot = torch.zeros(len(bins), n)
    onehot[torch.arange(len(bins)), bins] = 1.0
    z_true = tok.decode(bins).float()
    pit = pit_values(onehot, z_true, tok)
    assert torch.all((pit >= 0) & (pit <= 1))
    assert torch.allclose(pit, torch.ones_like(pit), atol=1e-5)


def test_calibrated_posterior_beats_overconfident():
    tok = _fitted_tok(n_levels=256)
    n = tok.n_levels
    g = torch.Generator().manual_seed(3)
    centers = torch.randint(30, n - 30, (3000,), generator=g).float()
    cal = _gauss_probs(centers, sigma=4.0, n=n)
    true_bin = torch.multinomial(cal, 1, generator=g).squeeze(1)   # truth ~ posterior
    z_true = tok.decode(true_bin).float()
    ks_cal = pit_calibration_error(pit_values(cal, z_true, tok))
    over = _gauss_probs(centers, sigma=0.5, n=n)                   # too sharp
    ks_over = pit_calibration_error(pit_values(over, z_true, tok))
    assert ks_cal < ks_over


def test_pit_calibration_error_extremes():
    uniform = torch.rand(5000, generator=torch.Generator().manual_seed(1))
    assert pit_calibration_error(uniform) < 0.05
    degenerate = torch.full((1000,), 0.5)
    assert pit_calibration_error(degenerate) > 0.4


def test_pit_histogram_sums_to_n():
    pit = torch.rand(500)
    h = pit_histogram(pit, bins=10)
    assert int(h.sum()) == 500
    assert h.shape == (10,)


# --------------------------------------------------------------------------- #
# coverage_quality_curve
# --------------------------------------------------------------------------- #
def test_coverage_rejection_lowers_nmad():
    g = torch.Generator().manual_seed(5)
    z_true = torch.rand(2000, generator=g) * 2.0
    noise = torch.randn(2000, generator=g) * 0.002
    z_pred = z_true + noise
    score = noise.abs()  # perfectly correlated with the error magnitude
    rows = coverage_quality_curve(z_pred, z_true, score)
    nmads = [r["sigma_nmad"] for r in rows]
    # σ_NMAD is non-increasing as we keep fewer (more-confident) predictions.
    assert all(b <= a + 1e-12 for a, b in zip(nmads, nmads[1:]))
    by_frac = {round(r["retained_frac"], 2): r["sigma_nmad"] for r in rows}
    assert by_frac[0.7] < by_frac[1.0]


# --------------------------------------------------------------------------- #
# outlier_auroc
# --------------------------------------------------------------------------- #
def test_auroc_perfect_reversed_random():
    is_out = torch.zeros(1000, dtype=torch.bool)
    is_out[:100] = True
    perfect = torch.where(is_out, 1.0, 0.0)
    assert outlier_auroc(perfect, is_out) == 1.0
    assert outlier_auroc(-perfect, is_out) == 0.0
    rand = torch.rand(1000, generator=torch.Generator().manual_seed(2))
    assert 0.4 < outlier_auroc(rand, is_out) < 0.6


def test_auroc_empty_class_is_half():
    score = torch.rand(50)
    assert outlier_auroc(score, torch.zeros(50, dtype=torch.bool)) == 0.5


# --------------------------------------------------------------------------- #
# fit_temperature / apply_temperature (X8b)
# --------------------------------------------------------------------------- #
def _calibration_setup(gen_sigma, model_sigma, n_levels=256, n=4000, seed=7):
    """Truth drawn from a Gaussian posterior of width `gen_sigma`; the model
    reports width `model_sigma`. model_sigma < gen_sigma ⇒ over-confident."""
    tok = _fitted_tok(n_levels=n_levels)
    g = torch.Generator().manual_seed(seed)
    centers = torch.randint(40, n_levels - 40, (n,), generator=g).float()
    gen = _gauss_probs(centers, sigma=gen_sigma, n=n_levels)
    true_bin = torch.multinomial(gen, 1, generator=g).squeeze(1)
    z_true = tok.decode(true_bin).float()
    model_probs = _gauss_probs(centers, sigma=model_sigma, n=n_levels)
    return tok, model_probs, z_true


def test_apply_temperature_softens_and_is_identity_at_one():
    tok = _fitted_tok(n_levels=128)
    probs = torch.softmax(torch.randn(16, tok.n_levels), dim=-1)
    assert torch.allclose(apply_temperature(probs, 1.0), probs, atol=1e-6)
    hot = apply_temperature(probs, 4.0)
    ent = -(probs * (probs + 1e-12).log()).sum(-1)
    ent_hot = -(hot * (hot + 1e-12).log()).sum(-1)
    assert torch.all(ent_hot > ent)  # higher T ⇒ more entropy
    assert torch.allclose(hot.sum(-1), torch.ones(16), atol=1e-5)


def test_fit_temperature_softens_overconfident():
    # Model is too sharp (sigma 1) vs the generative spread (sigma 4).
    tok, probs, z_true = _calibration_setup(gen_sigma=4.0, model_sigma=1.0)
    T = fit_temperature(probs, z_true, tok)
    assert T > 1.5  # must soften
    ks_before = pit_calibration_error(pit_values(probs, z_true, tok))
    ks_after = pit_calibration_error(
        pit_values(apply_temperature(probs, T), z_true, tok))
    assert ks_after < ks_before


def test_fit_temperature_near_one_when_calibrated():
    # Model width matches the generative width ⇒ already calibrated ⇒ T ≈ 1.
    tok, probs, z_true = _calibration_setup(gen_sigma=3.0, model_sigma=3.0)
    T = fit_temperature(probs, z_true, tok)
    assert 0.7 < T < 1.4


# --------------------------------------------------------------------------- #
# fit_pit_recalibrator / apply_pit_recalibration (X8c)
# --------------------------------------------------------------------------- #
def test_pit_recalibration_makes_uniform_and_is_monotone():
    g = torch.Generator().manual_seed(11)
    # Over-dispersed PIT (mass concentrated mid-range), like the soft model.
    pit = torch.rand(4000, generator=g) * 0.4 + 0.3
    knots = fit_pit_recalibrator(pit)
    rc = apply_pit_recalibration(pit, knots)
    assert pit_calibration_error(rc) < pit_calibration_error(pit)
    assert pit_calibration_error(rc) < 0.05            # in-sample → ~uniform
    # The map is monotone non-decreasing in its input.
    xs = torch.linspace(0.0, 1.0, 200)
    ys = apply_pit_recalibration(xs, knots)
    assert torch.all(ys[1:] >= ys[:-1] - 1e-6)


def test_pit_recalibration_generalises_to_held_out():
    g = torch.Generator().manual_seed(12)
    cal = torch.rand(4000, generator=g) * 0.4 + 0.3   # two independent draws of
    test = torch.rand(4000, generator=g) * 0.4 + 0.3  # the same miscalibration
    knots = fit_pit_recalibrator(cal)
    ks_before = pit_calibration_error(test)
    ks_after = pit_calibration_error(apply_pit_recalibration(test, knots))
    assert ks_after < ks_before
    assert ks_after < 0.1


def test_pit_recalibration_keeps_uniform_uniform():
    g = torch.Generator().manual_seed(13)
    u = torch.rand(5000, generator=g)                 # already calibrated
    knots = fit_pit_recalibrator(u)
    assert pit_calibration_error(apply_pit_recalibration(u, knots)) < 0.05
