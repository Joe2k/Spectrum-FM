"""Tests for the T3 token-stability audit helpers (nersc/audit_tokenizer_stability.py)."""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

# nersc/ is a sibling of tests/ at the repo root; add it to the path.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "nersc"))

from audit_tokenizer_stability import (  # noqa: E402
    flip_fraction,
    bits_from_indices,
    token_local_snr,
    stratify_by_snr,
    pooled_r2,
    summarize_group,
    verdict,
    build_val_indices,
)
from dr1_dataset import DR1IndexedDataset  # noqa: E402


class TestFlipFraction:
    def test_identical_is_zero(self):
        a = torch.tensor([[1, 2, 3, 4]])
        assert flip_fraction(a, a.clone()) == 0.0

    def test_disjoint_is_one(self):
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[5, 6, 7, 8]])
        assert flip_fraction(a, b) == 1.0

    def test_partial_is_exact_fraction(self):
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[1, 2, 9, 9]])  # 2 of 4 differ
        assert flip_fraction(a, b) == pytest.approx(0.5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            flip_fraction(torch.zeros(2, 3), torch.zeros(2, 4))


class TestBitsFromIndices:
    def test_zero_is_all_zero_bits(self):
        out = bits_from_indices(torch.tensor([0]), dim=10)
        assert out.shape == (1, 10)
        assert out.sum().item() == 0

    def test_one_sets_only_lowest_bit(self):
        out = bits_from_indices(torch.tensor([1]), dim=10)
        assert out[0, 0].item() == 1 and out[0, 1:].sum().item() == 0

    def test_five_is_101(self):
        out = bits_from_indices(torch.tensor([5]), dim=4)
        assert out[0].tolist() == [1, 0, 1, 0]  # 5 = 1 + 4

    def test_all_ones(self):
        out = bits_from_indices(torch.tensor([1023]), dim=10)
        assert out.sum().item() == 10

    def test_preserves_leading_shape(self):
        idx = torch.randint(0, 1024, (3, 7))
        out = bits_from_indices(idx, dim=10)
        assert out.shape == (3, 7, 10)

    def test_token_flip_overcounts_bit_flip(self):
        # A single bit change makes the token index differ — flip_fraction at
        # token granularity reads 1.0, at bit granularity reads 1/dim.
        base = torch.tensor([[0, 0]])
        noisy = torch.tensor([[1, 2]])  # bit0 of t0, bit1 of t1 → 1 bit each
        assert flip_fraction(base, noisy) == 1.0
        bb = bits_from_indices(base, 10)
        nb = bits_from_indices(noisy, 10)
        assert flip_fraction(bb, nb) == pytest.approx(1.0 / 10)


class TestTokenLocalSNR:
    def test_chunks_and_takes_median(self):
        # 2 tokens over a length-6 grid → stride 3; per-token median of |snr|.
        snr = torch.tensor([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]])
        out = token_local_snr(snr, n_tokens=2)
        assert out.shape == (1, 2)
        assert out[0, 0].item() == pytest.approx(2.0)   # median(1,2,3)
        assert out[0, 1].item() == pytest.approx(20.0)  # median(10,20,30)

    def test_takes_absolute_value(self):
        snr = torch.tensor([[-3.0, -2.0, -1.0]])
        out = token_local_snr(snr, n_tokens=1)
        assert out[0, 0].item() == pytest.approx(2.0)

    def test_drops_remainder_pixels(self):
        # length 7, 2 tokens, stride 3 → last pixel ignored, no error
        snr = torch.arange(7, dtype=torch.float32).unsqueeze(0)
        out = token_local_snr(snr, n_tokens=2)
        assert out.shape == (1, 2)

    def test_too_many_tokens_raises(self):
        with pytest.raises(ValueError):
            token_local_snr(torch.zeros(1, 3), n_tokens=4)


class TestStratifyBySNR:
    def test_bins_and_means(self):
        snr = [0.5, 1.5, 1.6, 4.0]
        flip = [0.9, 0.2, 0.4, 0.05]
        edges = [1.0, 3.0]
        labels, means, counts = stratify_by_snr(snr, flip, edges)
        # bins: (-inf,1), [1,3), [3,inf)
        assert counts == [1, 2, 1]
        assert means[0] == pytest.approx(0.9)
        assert means[1] == pytest.approx(0.3)   # mean(0.2, 0.4)
        assert means[2] == pytest.approx(0.05)
        assert labels[0].startswith("[-inf")
        assert labels[-1].endswith("inf)")

    def test_empty_bin_is_nan(self):
        labels, means, counts = stratify_by_snr([5.0], [0.1], [1.0, 3.0])
        assert counts[0] == 0
        assert np.isnan(means[0])

    def test_healthy_monotonic_shape(self):
        # flip rate decreasing with SNR is the healthy signature.
        snr = [0.5, 2.0, 8.0]
        flip = [0.6, 0.3, 0.05]
        _, means, _ = stratify_by_snr(snr, flip, [1.0, 3.0])
        assert means[0] > means[1] > means[2]


class TestPooledR2:
    def test_perfect_reconstruction(self):
        assert pooled_r2(0.0, 10.0) == pytest.approx(1.0)

    def test_no_better_than_mean(self):
        assert pooled_r2(10.0, 10.0) == pytest.approx(0.0)


class TestSummarizeGroup:
    def test_chi2_is_twice_mean_nll(self):
        acc = dict(n_spectra=4, nll_sum=2.0, nll_n=4,
                   ssr=1.0, sst=10.0, ssr_hi=0.5, sst_hi=8.0, n_codes_seen=512)
        row = summarize_group(acc, codebook_size=1024)
        assert row["chi2_per_pixel"] == pytest.approx(2.0 * (2.0 / 4))  # 1.0
        assert row["flux_r2_pooled"] == pytest.approx(0.9)
        assert row["flux_r2_pooled_snr3"] == pytest.approx(1.0 - 0.5 / 8.0)
        assert row["codebook_use"] == pytest.approx(0.5)
        assert row["n_spectra"] == 4

    def test_no_high_snr_is_nan(self):
        acc = dict(n_spectra=1, nll_sum=0.5, nll_n=1,
                   ssr=1.0, sst=2.0, ssr_hi=0.0, sst_hi=0.0, n_codes_seen=100)
        row = summarize_group(acc, codebook_size=1024)
        assert np.isnan(row["flux_r2_pooled_snr3"])


class TestVerdict:
    def _group(self, chi2, use, n=150):
        return {"n_spectra": n, "chi2_per_pixel": chi2,
                "flux_r2_pooled": 0.5, "flux_r2_pooled_snr3": 0.8, "codebook_use": use}

    def _kw(self):
        return dict(bit_thresh=0.10, token_hi_thresh=0.50,
                    chi2_thresh=1.2, codebook_thresh=0.30, min_n=100)

    def test_pass_on_low_bit_flip(self):
        # token flip can be high (overcounts) — gate is on per-BIT flip.
        rows = {("main", "bright"): self._group(1.0, 0.7),
                ("main", "dark"): self._group(1.05, 0.65)}
        passed, reasons, notes = verdict(0.06, 0.73, rows, **self._kw())
        assert passed and reasons == []

    def test_high_token_flip_is_only_a_note(self):
        rows = {("main", "bright"): self._group(1.0, 0.7)}
        passed, reasons, notes = verdict(0.06, 0.80, rows, **self._kw())
        assert passed and reasons == []
        assert any("token-index flip" in n for n in notes)

    def test_flag_on_high_bit_flip(self):
        rows = {("main", "bright"): self._group(1.0, 0.7)}
        passed, reasons, notes = verdict(0.25, 0.73, rows, **self._kw())
        assert not passed and any("per-bit flip" in r for r in reasons)

    def test_flag_on_bad_well_sampled_group(self):
        rows = {("main", "dark"): self._group(1.5, 0.1, n=300)}
        passed, reasons, notes = verdict(0.05, 0.4, rows, **self._kw())
        assert not passed
        assert any("chi^2" in r for r in reasons)
        assert any("codebook" in r for r in reasons)

    def test_small_n_group_blip_is_a_note_not_a_fail(self):
        # sv1/dark χ²=1.29 on n=81 (the real run) → informational, still PASS.
        rows = {("sv1", "dark"): self._group(1.29, 0.5, n=81),
                ("main", "dark"): self._group(1.0, 0.6, n=300)}
        passed, reasons, notes = verdict(0.05, 0.4, rows, **self._kw())
        assert passed and reasons == []
        assert any("sv1/dark" in n and "informational" in n for n in notes)


class TestBuildValIndices:
    def test_reproducible_and_sized(self):
        a = build_val_indices(1000, seed=42, val_frac=0.02)
        b = build_val_indices(1000, seed=42, val_frac=0.02)
        assert a == b
        assert len(a) == 20

    def test_matches_training_split_recipe(self):
        # Same recipe as pretrain_tokenizer.py:292-296.
        g = torch.Generator().manual_seed(7)
        perm = torch.randperm(1000, generator=g).tolist()
        expected = perm[: max(1, int(1000 * 0.05))]
        assert build_val_indices(1000, seed=7, val_frac=0.05) == expected


class TestMetaForIndex:
    def test_returns_survey_program_from_record(self):
        manifest = [
            {"coadd": "/a.fits", "redrock": "/ar.fits", "n_rows": 2,
             "survey": "sv3", "program": "bright"},
            {"coadd": "/b.fits", "redrock": "/br.fits", "n_rows": 3,
             "survey": "main", "program": "dark"},
        ]
        ds = DR1IndexedDataset(manifest)  # n_rows present → no FITS read
        # flat index: rec0 rows 0,1 ; rec1 rows 0,1,2
        assert len(ds) == 5
        assert ds.meta_for_index(0) == ("sv3", "bright")
        assert ds.meta_for_index(1) == ("sv3", "bright")
        assert ds.meta_for_index(2) == ("main", "dark")
        assert ds.meta_for_index(4) == ("main", "dark")

    def test_missing_fields_default_unknown(self):
        ds = DR1IndexedDataset([{"coadd": "/a.fits", "redrock": "/ar.fits", "n_rows": 1}])
        assert ds.meta_for_index(0) == ("unknown", "unknown")
