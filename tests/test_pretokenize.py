"""Tests for the X1 pre-tokenization cache: reader, collate, and the
cached==on-the-fly equivalence that guarantees training is unchanged."""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "nersc"))

from dr1_tokenized_dataset import (  # noqa: E402
    DR1CachedTokenDataset,
    collate_cached_skip_none,
    collect_redshifts_from_cache,
)
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.tokenizers.redshift import RedshiftTokenizer  # noqa: E402
from src.training.sequences import tokenize_and_build  # noqa: E402


def _write_shard(path, indices, *, z=None, zwarn=None, fiberstatus=None,
                 nonzero_flux=None, survey="sv3", program="bright", healpix=1):
    """Write a shard in the schema pretokenize_corpus.write_shard produces."""
    n = indices.shape[0]
    z = np.full(n, 0.5, np.float32) if z is None else z.astype(np.float32)
    np.savez_compressed(
        path,
        indices=indices.astype(np.uint16),
        z=z,
        denorm=np.ones(n, np.float32),
        zwarn=(np.zeros(n, np.int16) if zwarn is None else zwarn.astype(np.int16)),
        fiberstatus=(np.zeros(n, np.int32) if fiberstatus is None else fiberstatus.astype(np.int32)),
        nonzero_flux=(np.ones(n, np.int8) if nonzero_flux is None else nonzero_flux.astype(np.int8)),
        spectype=np.array(["GALAXY"] * n, dtype="<U8"),
        targetid=np.arange(n, dtype=np.int64),
        row=np.arange(n, dtype=np.int32),
        survey=np.array(survey), program=np.array(program), healpix=np.array(healpix),
        coadd=np.array("x.fits"), redrock=np.array("xr.fits"),
    )


class TestCachedReader:
    def test_len_and_getitem(self, tmp_path):
        idx = np.random.randint(0, 1024, (5, 272))
        _write_shard(tmp_path / "sv3_bright_1.npz", idx)
        ds = DR1CachedTokenDataset(tmp_path)
        assert len(ds) == 5
        item = ds[0]
        assert item["spec_indices"].shape == (272,)
        assert item["spec_indices"].dtype == torch.int64
        assert np.array_equal(item["spec_indices"].numpy(), idx[0])
        assert item["z"].dtype == torch.float32

    def test_quality_filter_from_flags(self, tmp_path):
        idx = np.random.randint(0, 1024, (6, 272))
        zwarn = np.array([0, 4, 0, 0, 0, 0], np.int16)        # row1 bad zwarn
        fstat = np.array([0, 0, 1, 0, 0, 0], np.int32)        # row2 bad fiber
        nzf = np.array([1, 1, 1, 0, 1, 1], np.int8)           # row3 zero flux
        _write_shard(tmp_path / "sv3_bright_1.npz", idx,
                     zwarn=zwarn, fiberstatus=fstat, nonzero_flux=nzf)
        ds = DR1CachedTokenDataset(tmp_path, require_good_zwarn=True,
                                   require_nonzero_flux=True)
        assert len(ds) == 3  # rows 0, 4, 5 survive
        # first surviving row is original row 0
        assert np.array_equal(ds[0]["spec_indices"].numpy(), idx[0])

    def test_no_filter_keeps_all(self, tmp_path):
        idx = np.random.randint(0, 1024, (4, 272))
        _write_shard(tmp_path / "sv3_bright_1.npz", idx,
                     zwarn=np.array([0, 4, 0, 4], np.int16))
        ds = DR1CachedTokenDataset(tmp_path, require_good_zwarn=False,
                                   require_nonzero_flux=False)
        assert len(ds) == 4

    def test_meta_for_index(self, tmp_path):
        _write_shard(tmp_path / "sv3_bright_1.npz", np.zeros((2, 272), int),
                     survey="sv3", program="bright")
        _write_shard(tmp_path / "main_dark_2.npz", np.zeros((3, 272), int),
                     survey="main", program="dark", healpix=2)
        ds = DR1CachedTokenDataset(tmp_path)
        metas = {ds.meta_for_index(i) for i in range(len(ds))}
        assert metas == {("sv3", "bright"), ("main", "dark")}

    def test_max_spectra_cap(self, tmp_path):
        _write_shard(tmp_path / "sv3_bright_1.npz", np.zeros((10, 272), int))
        ds = DR1CachedTokenDataset(tmp_path, max_spectra=4)
        assert len(ds) == 4

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DR1CachedTokenDataset(tmp_path / "empty")


class TestCollate:
    def test_stacks_and_drops_none(self, tmp_path):
        idx = np.random.randint(0, 1024, (3, 272))
        _write_shard(tmp_path / "sv3_bright_1.npz", idx)
        ds = DR1CachedTokenDataset(tmp_path)
        batch = collate_cached_skip_none([ds[0], None, ds[1]])
        assert batch["spec_indices"].shape == (2, 272)
        assert batch["z"].shape == (2,)

    def test_all_none_returns_none(self):
        assert collate_cached_skip_none([None, None]) is None


class TestCollectRedshiftsFromCache:
    def test_good_zwarn_only(self, tmp_path):
        _write_shard(tmp_path / "sv3_bright_1.npz", np.zeros((3, 272), int),
                     z=np.array([0.1, 0.2, 0.3], np.float32),
                     zwarn=np.array([0, 4, 0], np.int16))
        zs = collect_redshifts_from_cache(tmp_path)
        assert sorted(round(float(z), 3) for z in zs) == [0.1, 0.3]


class TestRoundtripEquivalence:
    """The correctness proof: a cached batch produces byte-identical sequences
    to the on-the-fly tokenization path, with no FITS involved."""

    def _setup(self, tmp_path, bs=4, L=7781):
        torch.manual_seed(0)
        spec_tok = SpectrumTokenizer().eval()
        for p in spec_tok.parameters():
            p.requires_grad_(False)
        g = torch.Generator().manual_seed(1)
        flux = torch.rand(bs, L, generator=g) * 5 + 0.5
        ivar = torch.rand(bs, L, generator=g) * 4 + 0.1
        wave = (3600.0 + 0.8 * torch.arange(L)).unsqueeze(0).expand(bs, -1).contiguous()
        z = torch.rand(bs, generator=g) * 2.0
        # on-the-fly indices == what the writer would cache
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)
        with torch.no_grad():
            idx, _ = spec_tok.encode(x, wavelength=wave)
        _write_shard(tmp_path / "sv3_bright_1.npz", idx.numpy(), z=z.numpy())
        z_tok = RedshiftTokenizer(n_levels=256)
        z_tok.fit(torch.rand(5000, generator=torch.Generator().manual_seed(2)) * 3)
        flux_batch = {"flux": flux, "ivar": ivar, "wavelength": wave, "z": z}
        return spec_tok, z_tok, flux_batch, idx

    def test_cached_indices_match_encode(self, tmp_path):
        _, _, _, idx = self._setup(tmp_path)
        ds = DR1CachedTokenDataset(tmp_path)
        for i in range(len(ds)):
            assert torch.equal(ds[i]["spec_indices"], idx[i].long())

    def test_sequences_identical_no_mask(self, tmp_path):
        spec_tok, z_tok, flux_batch, _ = self._setup(tmp_path)
        ds = DR1CachedTokenDataset(tmp_path)
        cached_batch = collate_cached_skip_none([ds[i] for i in range(len(ds))])
        dev = torch.device("cpu")
        on_fly = tokenize_and_build(flux_batch, spec_tok, z_tok, "a", dev,
                                    wavelength_aware=True)
        cached = tokenize_and_build(cached_batch, None, z_tok, "a", dev)
        for a, b in zip(on_fly[:3], cached[:3]):  # enc, dec, tgt
            assert torch.equal(a, b)

    def test_sequences_identical_with_masking(self, tmp_path):
        spec_tok, z_tok, flux_batch, _ = self._setup(tmp_path)
        ds = DR1CachedTokenDataset(tmp_path)
        cached_batch = collate_cached_skip_none([ds[i] for i in range(len(ds))])
        dev = torch.device("cpu")
        kw = dict(encoder_mask_ratio=0.5, redshift_mask_ratio=0.5)
        on_fly = tokenize_and_build(flux_batch, spec_tok, z_tok, "a", dev,
                                    wavelength_aware=True,
                                    rng=torch.Generator().manual_seed(7), **kw)
        cached = tokenize_and_build(cached_batch, None, z_tok, "a", dev,
                                    rng=torch.Generator().manual_seed(7), **kw)
        # encoder (masked), decoder, target, and mask positions all identical
        assert torch.equal(on_fly[0], cached[0])
        assert torch.equal(on_fly[3], cached[3])

    def test_sequences_identical_masked_targets_only(self, tmp_path):
        # The X2 masked-targets-only objective must produce byte-identical
        # sequences (incl. the -100 target positions) from cache vs on-the-fly.
        spec_tok, z_tok, flux_batch, _ = self._setup(tmp_path)
        ds = DR1CachedTokenDataset(tmp_path)
        cached_batch = collate_cached_skip_none([ds[i] for i in range(len(ds))])
        dev = torch.device("cpu")
        kw = dict(encoder_mask_ratio=0.5, mask_targets_only=True)
        on_fly = tokenize_and_build(flux_batch, spec_tok, z_tok, "a", dev,
                                    wavelength_aware=True,
                                    rng=torch.Generator().manual_seed(11), **kw)
        cached = tokenize_and_build(cached_batch, None, z_tok, "a", dev,
                                    rng=torch.Generator().manual_seed(11), **kw)
        for a, b in zip(on_fly[:4], cached[:4]):  # enc, dec, tgt (with -100), mask
            assert torch.equal(a, b)
        # The objective actually dropped some targets to ignore_index.
        assert (on_fly[2] == -100).any()
