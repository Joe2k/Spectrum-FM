# Unimodal Foundation Model for DESI Spectra

A course project for PHYS303/CS686 — Deep Learning & Bayesian Learning (Spring 2026).

## Overview

This repository implements a **unimodal foundation model for astrophysical spectra**, focusing exclusively on DESI (Dark Energy Spectroscopic Instrument) spectra and redshift (`z`). The project directly addresses a key critique of the AION-1 multimodal foundation model: its redshift token was treated identically to spectral tokens and only masked occasionally, preventing redshift from becoming an organizing principle of the learned representation.

Our model inverts AION-1's breadth-for-depth trade-off: **one modality, treated seriously**.

## Key Contributions

- **Approach A**: Joint training with a lightweight redshift predictor MLP head attached to the encoder representation, trained simultaneously with masked spectral token reconstruction.
- **Approach B**: Always-mask the redshift token — the model must reconstruct `z` from spectral context on every training step.
- Continuous **visualization and testing** at every stage.

## Data

- **DESI Data Release 1** (Early Data Release / SV3 "one-percent" survey)
- ~1 million galaxies, stars, and quasars
- 7,081-pixel wavelength grid spanning ~3,600–9,800 Å
- Redshift provided by DESI pipeline

## PHYS303 final submission

Official trained weights live under [`checkpoints/release/`](checkpoints/release/) (from W&B project `redshifty`, entity `jjayaseelan-university-of-san-francisco`):

| `model_id` | Role |
|------------|------|
| `spectrum_tokenizer_v1` | Frozen spectrum codec |
| `transformer_approach_a_fm_v1_10k_ddp4_rw10_v8` | Primary transformer (Approach A) |

**Run the submission notebook:** [`notebooks/08_PHYS303_final_submission.ipynb`](notebooks/08_PHYS303_final_submission.ipynb) — overview, training summary, instructor NumPy/FITS inference.

### Final validation metrics

Held-out validation on the DR1 10k-healpix manifest (`dr1_10k_scratch.jsonl`, encoder mask ratio 0.5). Source: W&B best checkpoint at step 22k ([`checkpoints/release/MANIFEST.json`](checkpoints/release/MANIFEST.json), verified 2026-05-20).

| Model | Run | Val loss | z (TF) | z (AR) | Spectrum (TF) | Spectrum (AR) | Masked spec (TF) |
|-------|-----|----------|--------|--------|---------------|---------------|------------------|
| **Approach A** (release) | `fm_v1_10k_a_ddp4_rw10_v8` | 0.868 | **100.0%** | **100.0%** | 72.8% | 64.7% | 45.5% |
| Approach B (no weights) | `phase10_mask50_b` | 2.755 | 1.8% | 3.6% | 28.2% | 5.3% | 28.1% |

- **TF** = teacher-forced decode; **AR** = autoregressive generation (no teacher forcing).
- **Masked spec (TF)** = accuracy only at decoder positions whose encoder input was `[MASK]` (the honest reconstruction metric; unmasked copy inflates overall spectrum accuracy).
- Approach A: encoder sees the redshift token (`redshift_loss_weight` = 1.0). Approach B: encoder never sees `z` (`redshift_loss_weight` = 50.0) — redshift is not learned from spectrum alone.

Release training run: [W&B `8fglr5zl`](https://wandb.ai/jjayaseelan-university-of-san-francisco/redshifty/runs/8fglr5zl). Refresh metrics: `python scripts/sync_wandb_metrics.py`.

**Checkpoints (~1.4 GB):** tracked with Git LFS.

```bash
git lfs install
git lfs pull
# If release/ has symlinks only, materialize real files before commit:
python scripts/setup_release_checkpoints.py --copy
```

Re-download from W&B: `python scripts/download_release_checkpoints.py` (requires `WANDB_API_KEY` in `.env`).

## Quick Start

### Local Smoke Testing (Mac MPS)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Verify package imports
python -c "import src; print('OK')"
```

### NERSC A100 Training

See `scripts/` for SLURM job scripts. NERSC uses SLURM with specific constraints:
- Must specify `--constraint`, `--qos`, `--account`, `--gpus`
- GPU jobs require `--gpus` or `-G` flag for CUDA visibility

```bash
# Example submission
sbatch scripts/train_approach_a.sh
```

## Repository Structure

```
FoundationModel/
├── data/                  # Data download & caching scripts
├── src/                   # Source code
│   ├── tokenizers/        # Spectrum & redshift tokenizers
│   ├── models/            # Transformer encoder-decoder
│   ├── training/          # Training loops (Approach A & B)
│   ├── evaluation/        # Metrics & benchmarking
│   └── utils/             # Plotting, logging, config
├── tests/                 # pytest suite
├── notebooks/             # Visualization notebooks
├── scripts/               # SLURM/job scripts for NERSC
└── RESEARCH_LOG.md        # Living document of findings
```

## Evaluation

The model will be tested on a held-out benchmark including:
1. **Redshift prediction** accuracy vs. DESI pipeline values
2. **Spectrum reconstruction** of masked spectral regions
3. **Out-of-distribution generalization** to non-DESI spectra

## References

- AION-1 Paper: Parker et al. (2025), *AION-1: Omnimodal Foundation Model for Astronomical Sciences*
- DESI Collaboration et al. (2016, 2024)

## License

MIT
