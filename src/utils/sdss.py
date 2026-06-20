"""
SDSS spectrum loader for the out-of-distribution (X9) evaluation.
================================================================

The foundation model is trained on DESI DR1 only. The headline generalization
claim — beating AION-1 / SpecPT *out of sample* — is tested on SDSS, which the
model has never seen. SDSS spectra live on a **log-λ grid** (different spacing,
different coverage, different flux calibration), so they go through the
tokenizer's `resample_to_grid` (wavelength-aware) path; nothing here is
DESI-specific.

This module reads standard SDSS/BOSS/eBOSS ``spec-*.fits`` files into the same
survey-agnostic batch dict the rest of the stack consumes
(``{"flux", "ivar", "wavelength", "z"}``), with a Dataset + collate that mirror
``nersc/dr1_dataset.py`` so `tokenize_and_build(..., wavelength_aware=True)` and
`evaluate(...)` work unchanged.

Standard SDSS spec FITS layout (spec-PLATE-MJD-FIBER.fits / spec-lite):
  - HDU "COADD" (or 1): columns ``flux``, ``loglam`` (log10 λ/Å), ``ivar``.
  - HDU "SPECOBJ" (or 2): scalars ``Z`` (pipeline redshift), ``ZWARNING``.
Column/HDU names are overridable for survey variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def read_sdss_spec_fits(
    path: Union[str, Path],
    *,
    coadd_hdu: Union[str, int] = "COADD",
    specobj_hdu: Union[str, int] = "SPECOBJ",
    flux_col: str = "flux",
    loglam_col: str = "loglam",
    ivar_col: str = "ivar",
    z_col: str = "Z",
    zwarn_col: str = "ZWARNING",
) -> Optional[dict]:
    """Read one SDSS ``spec-*.fits`` into ``{flux, ivar, wavelength, z, zwarn}``.

    Wavelength is ``10**loglam`` (Angstrom, ascending). Returns ``None`` if the
    file can't be parsed (so a directory scan can skip bad files).
    """
    from astropy.io import fits

    path = Path(path)
    try:
        with fits.open(path, memmap=False) as hdul:
            coadd = hdul[coadd_hdu].data
            flux = np.asarray(coadd[flux_col], dtype=np.float32)
            loglam = np.asarray(coadd[loglam_col], dtype=np.float64)
            ivar = np.asarray(coadd[ivar_col], dtype=np.float32)
            wavelength = (10.0 ** loglam).astype(np.float32)

            z = float("nan")
            zwarn = 0
            try:
                sp = hdul[specobj_hdu].data
                z = float(np.asarray(sp[z_col]).ravel()[0])
                if zwarn_col in sp.names:
                    zwarn = int(np.asarray(sp[zwarn_col]).ravel()[0])
            except (KeyError, IndexError, TypeError):
                pass
    except Exception:  # noqa: BLE001 — malformed FITS, skip it
        return None

    # Ensure strictly ascending wavelength (searchsorted-based resampling needs it).
    if wavelength.size >= 2 and wavelength[0] > wavelength[-1]:
        flux, ivar, wavelength = flux[::-1], ivar[::-1], wavelength[::-1]
    return {
        "flux": flux.copy(),
        "ivar": ivar.copy(),
        "wavelength": wavelength.copy(),
        "z": z,
        "zwarn": zwarn,
    }


class SDSSSpectrumDataset(Dataset):
    """Dataset over SDSS ``spec-*.fits`` files (one spectrum each).

    Args:
        paths: explicit list of FITS paths, or a single directory (globbed for
            ``spec-*.fits``).
        require_good_zwarn: keep only ``ZWARNING == 0`` (good pipeline z).
        require_nonzero_flux: drop spectra with all-zero / non-finite flux.
        max_spectra: cap (after filtering).
        reader_kwargs: forwarded to :func:`read_sdss_spec_fits` (column overrides).
    """

    def __init__(
        self,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        require_good_zwarn: bool = True,
        require_nonzero_flux: bool = True,
        max_spectra: Optional[int] = None,
        glob: str = "spec-*.fits",
        **reader_kwargs,
    ):
        if isinstance(paths, (str, Path)) and Path(paths).is_dir():
            file_list: List[Path] = sorted(Path(paths).glob(glob))
        elif isinstance(paths, (str, Path)):
            file_list = [Path(paths)]
        else:
            file_list = [Path(p) for p in paths]

        self.records: List[dict] = []
        for p in file_list:
            rec = read_sdss_spec_fits(p, **reader_kwargs)
            if rec is None:
                continue
            if require_good_zwarn and rec["zwarn"] != 0:
                continue
            if not np.isfinite(rec["z"]):
                continue
            if require_nonzero_flux and not np.any(np.abs(rec["flux"]) > 0):
                continue
            self.records.append(rec)
            if max_spectra is not None and len(self.records) >= max_spectra:
                break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "flux": torch.from_numpy(r["flux"].copy()),
            "ivar": torch.from_numpy(r["ivar"].copy()),
            "wavelength": torch.from_numpy(r["wavelength"].copy()),
            "z": torch.tensor(r["z"], dtype=torch.float32),
        }


def collate_sdss_skip_none(batch):
    """Pad variable-length SDSS spectra into a batch dict; drop ``None`` items.

    Mirrors ``collate_dr1_skip_none``: flux/ivar zero-padded (ivar 0 ⇒ no loss
    weight), wavelength extended past the data so it stays strictly ascending.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    L = max(b["flux"].shape[0] for b in batch)

    def _pad(t, fill=0.0):
        if t.shape[0] == L:
            return t
        out = torch.full((L,), fill, dtype=t.dtype)
        out[: t.shape[0]] = t
        return out

    def _pad_wave(t):
        n = t.shape[0]
        if n == L:
            return t
        step = (t[-1] - t[-2]) if n >= 2 else torch.tensor(1.0, dtype=t.dtype)
        ext = t[-1] + step * torch.arange(1, L - n + 1, dtype=t.dtype)
        return torch.cat([t, ext])

    return {
        "flux": torch.stack([_pad(b["flux"]) for b in batch]),
        "ivar": torch.stack([_pad(b["ivar"], 0.0) for b in batch]),
        "wavelength": torch.stack([_pad_wave(b["wavelength"]) for b in batch]),
        "z": torch.stack([b["z"] for b in batch]),
    }
