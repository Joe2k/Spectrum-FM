"""Tests for the SDSS OOD loader (X9).

Synthesizes standard SDSS ``spec-*.fits`` files with astropy and exercises the
reader, the Dataset filtering, and the padding collate (the survey-agnostic
bridge into `tokenize_and_build(..., wavelength_aware=True)`).
"""

import numpy as np
import pytest
import torch

pytest.importorskip("astropy")
from astropy.io import fits  # noqa: E402

from src.utils.sdss import (  # noqa: E402
    SDSSSpectrumDataset,
    collate_sdss_skip_none,
    read_sdss_spec_fits,
)


def _write_spec(path, n=200, z=0.5, zwarn=0, flux_val=1.0, descending=False,
                lam_lo=3800.0, lam_hi=9200.0):
    loglam = np.linspace(np.log10(lam_lo), np.log10(lam_hi), n).astype(np.float32)
    if descending:
        loglam = loglam[::-1].copy()
    flux = np.full(n, flux_val, dtype=np.float32)
    ivar = np.ones(n, dtype=np.float32)
    coadd = fits.BinTableHDU.from_columns([
        fits.Column(name="flux", format="E", array=flux),
        fits.Column(name="loglam", format="E", array=loglam),
        fits.Column(name="ivar", format="E", array=ivar)], name="COADD")
    specobj = fits.BinTableHDU.from_columns([
        fits.Column(name="Z", format="E", array=np.array([z], dtype=np.float32)),
        fits.Column(name="ZWARNING", format="J", array=np.array([zwarn]))],
        name="SPECOBJ")
    fits.HDUList([fits.PrimaryHDU(), coadd, specobj]).writeto(path, overwrite=True)
    return path


def test_read_basic(tmp_path):
    p = _write_spec(tmp_path / "spec-1.fits", n=150, z=1.23, zwarn=0)
    rec = read_sdss_spec_fits(p)
    assert set(rec) == {"flux", "ivar", "wavelength", "z", "zwarn"}
    assert rec["flux"].shape == (150,)
    assert abs(rec["z"] - 1.23) < 1e-4
    assert rec["zwarn"] == 0
    # wavelength = 10**loglam, ascending, within the requested span.
    assert rec["wavelength"][0] < rec["wavelength"][-1]
    assert 3790 < rec["wavelength"][0] < 3810
    assert 9180 < rec["wavelength"][-1] < 9220


def test_descending_wavelength_is_flipped(tmp_path):
    p = _write_spec(tmp_path / "spec-desc.fits", descending=True)
    rec = read_sdss_spec_fits(p)
    w = rec["wavelength"]
    assert np.all(np.diff(w) > 0)  # strictly ascending after flip


def test_read_bad_file_returns_none(tmp_path):
    bad = tmp_path / "spec-bad.fits"
    bad.write_bytes(b"not a fits file")
    assert read_sdss_spec_fits(bad) is None


def test_dataset_filters_zwarn_and_flux(tmp_path):
    _write_spec(tmp_path / "spec-good.fits", z=0.4, zwarn=0)
    _write_spec(tmp_path / "spec-warn.fits", z=0.4, zwarn=4)       # bad pipeline z
    _write_spec(tmp_path / "spec-zero.fits", z=0.4, flux_val=0.0)  # all-zero flux
    ds = SDSSSpectrumDataset(tmp_path, require_good_zwarn=True,
                             require_nonzero_flux=True)
    assert len(ds) == 1  # only spec-good survives

    ds_all = SDSSSpectrumDataset(tmp_path, require_good_zwarn=False,
                                 require_nonzero_flux=False)
    assert len(ds_all) == 3


def test_dataset_getitem_keys(tmp_path):
    _write_spec(tmp_path / "spec-1.fits")
    ds = SDSSSpectrumDataset(tmp_path)
    item = ds[0]
    assert set(item) == {"flux", "ivar", "wavelength", "z"}
    assert item["z"].dtype == torch.float32
    assert item["flux"].ndim == 1


def test_collate_pads_and_skips_none(tmp_path):
    _write_spec(tmp_path / "spec-a.fits", n=180)
    _write_spec(tmp_path / "spec-b.fits", n=220)
    ds = SDSSSpectrumDataset(tmp_path)
    batch = collate_sdss_skip_none([ds[0], None, ds[1]])
    assert batch["flux"].shape == (2, 220)
    assert batch["wavelength"].shape == (2, 220)
    assert batch["z"].shape == (2,)
    # padded wavelengths stay strictly ascending per row (searchsorted needs it).
    for row in batch["wavelength"]:
        assert torch.all(torch.diff(row) > 0)
    # padded flux/ivar tail is zero (no loss weight there).
    assert torch.all(batch["ivar"][0, 180:] == 0)


def test_collate_all_none_is_none():
    assert collate_sdss_skip_none([None, None]) is None
