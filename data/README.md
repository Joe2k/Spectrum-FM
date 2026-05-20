# Data for Spectrum-FM

## Training data (DESI DR1)

- **Survey**: DESI Data Release 1, SV3 "one-percent" (bright/dark programs)
- **Format**: Per-healpix directories with paired FITS files:
  - `coadd-<brick>.fits` — flux, inverse variance, wavelength
  - `redrock-<brick>.fits` — pipeline redshifts (`ZWARN == 0` for training)
- **Typical shape**: Stitched B+R+Z spectrum; length varies by object (~5k–15k pixels before tokenizer grid)

Build a training manifest on NERSC:

```bash
python nersc/build_dr1_index.py --root /global/cfs/cdirs/desi/public/dr1 ...
```

## Local smoke data (instructor / laptop)

Download a small healpix patch:

```bash
python data/download_desi.py
# or
python scripts/download_desi_batch.py
```

Place files under `data/desi_raw/` so you have at least one `coadd-*.fits` and matching `redrock-*.fits`.

## Instructor: custom spectra (no DESI FITS)

The submission notebook accepts **NumPy arrays**:

- `flux`: 1D `float32`, calibrated flux
- `ivar`: 1D `float32`, inverse variance (same length as `flux`)
- `z`: optional scalar or array for evaluation only

The model resamples internally via the frozen spectrum tokenizer. For best results, spectra should be DESI-like (rest-frame or consistent with training); OOD inputs are supported but accuracy is not guaranteed.

Large FITS files are gitignored (`data/*.fits`). Do not commit raw survey data.
