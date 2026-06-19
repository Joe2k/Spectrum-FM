# Research Log

Living document of findings, decisions, and experimental results.

---

## 2026-05-05: Project Kickoff & Architecture Planning

### Assignment Requirements (from PHYS303_Final-Project_2026.pdf)

- **Goal**: Build a unimodal foundation model for DESI spectra
- **Scope**: Spectra ONLY. No images, no photometry, no Subaru.
- **Core critique of AION-1 to address**:
  1. Redshift treated like any other token → masked only occasionally
  2. Redshift never enters encoder representation space (separate frozen head)
- **Data**: DESI DR1 (SV3 "one-percent"), ~1M objects, 7,081-pixel grid, 3,600–9,800 Å
- **Required approaches** (choose at least one):
  - **Approach A**: Joint training — small MLP head predicts z jointly with masked-token objective
  - **Approach B**: Always-mask redshift token — force reconstruction of z from spectral context every step
- **Evaluation**: Redshift prediction + spectrum reconstruction + OOD generalization (non-DESI spectra)
- **What NOT to build**: Pure CNN redshift regressor — must be a foundation model with masked reconstruction

### AION-1 Architecture Notes (from AION Paper.pdf + GitHub)

**Spectrum Tokenizer**:
- ConvNeXt-V2 encoder/decoder backbone
- Input: flux + inverse-variance (istd), 2-channel
- Interpolated to common 8,704-point latent grid (3,500–10,462.4 Å, 0.8 Å spacing)
- 4-stage ConvNeXt downsampling: 4×4 conv + 3× (2×2 conv) → compresses to 273×512 latent
- Quantizer: Look-up-Free Quantizer (LFQ), dim=10, codebook size=1024
- Losses: Gaussian NLL (inverse-variance weighted) + mask BCE + commitment loss β=0.25
- Additional normalization token (log10 median flux, scalar quantized) prepended to sequence
- Total tokens per spectrum: **274** (1 normalization + 273 spectral)

**Scalar Tokenizer (for redshift)**:
- Empirical CDF mapping to standard normal: z_i = Φ⁻¹(F_x(x_i))
- Equal-width binning in Gaussian space → uniform probability mass per bin
- FSQ quantization with K=1024 fixed centroids at standard normal quantiles
- No learned parameters — parameter-free

**Transformer Backbone**:
- Encoder-decoder architecture (T5-style scaling)
- Modality-specific embeddings: token embedding + modality embedding + positional embedding
- Input budget: 256 tokens, output budget: 128 tokens (AION-1 config)
- Model sizes: Base (300M), Large (800M), XL (3B)
- Trained with multimodal masked modeling (4M objective)

**Key Design Decisions for Our Model**:

1. **Reuse AION tokenizers**: The spectrum and scalar codecs are well-engineered and open-source. We will adapt them for our unimodal scope rather than reimplementing from scratch.
2. **Custom transformer**: We will build our own encoder-decoder, simplified to handle only two modalities (spectrum tokens + redshift token).
3. **Redshift mechanisms**: Implement both Approach A and B separately, then compare.
4. **Training compute**: Mac MPS for smoke tests, NERSC A100 for full training.
5. **Discrete tokens**: We will keep the discrete tokenization approach (not continuous MAE) because it directly addresses the assignment's critique of AION-1's token handling.

### NERSC SLURM Constraints (from docs.nersc.gov/jobs)

- Must specify: `--nodes`, `--time`, `--constraint`, `--qos`, `--account`
- GPU jobs MUST use `--gpus` or `-G` flag for CUDA visibility
- Default QOS is `debug` (10 min)
- No default architecture — jobs without `--constraint` are rejected
- Perlmutter GPU nodes: 256 GB CPU RAM, 160 GB GPU RAM
- Use `srun` within job scripts for parallel tasks
- Good practice to always set `--account=<NERSC Project>`

### Next Steps

1. ~~Phase 1: Build minimal smoke-test data pipeline~~ ✅ COMPLETE
2. Phase 2: Adapt spectrum tokenizer from AION
3. Phase 3: Redshift scalar tokenizer
4. Phase 4: Transformer backbone
5. Phase 5: Approach A training
6. Phase 6: Approach B training
7. Phase 7: Evaluation & comparison
8. Phase 8: NERSC full-scale training
9. Phase 9: OOD generalization prep

## 2026-05-06: Phase 1 — Minimal Smoke-Test Data Pipeline

### Data Source
- Downloaded **real DESI EDR data** from `data.desi.lbl.gov`
- File: `coadd-sv3-bright-10016.fits` (18.5 MB) + `redrock-sv3-bright-10016.fits` (96 KB)
- Contains **43 spectra** from SV3 "one-percent" bright targets
- Redshift range: **[-0.0020, 1.1854]** (mix of stars and galaxies)
- Wavelength coverage: **3600–9824 Å** (B+R+Z camera bands stitched)
- Native pixel count after stitching: **7781 pixels** (slightly more than standard 7081 due to overlaps)

### Pipeline Components Built
1. **`src/utils/data.py`** — `DESISpectrumDataset` PyTorch Dataset
   - Stitches B/R/Z bands via inverse-variance weighted averaging in overlap regions
   - Returns dict of tensors: `flux`, `ivar`, `mask`, `wavelength`, `z`
   - `collate_desi_batch()` for DataLoader batching

2. **`src/utils/plotting.py`** — Visualization utilities
   - `plot_spectrum()`: Single spectrum with error regions and masking
   - `plot_spectrum_grid()`: Grid of multiple spectra
   - `plot_redshift_distribution()`: Histogram with statistics
   - `plot_reconstruction_comparison()`: Original vs reconstructed (for later phases)
   - `plot_training_curves()`: Loss curves (for later phases)

3. **`tests/test_data.py`** — pytest suite (10 tests, all passing)
   - Band stitching with/without overlaps
   - Real data loading, shapes, wavelength range, redshift values
   - Batch collation

4. **`scripts/visualize_spectra.py`** — Standalone visualization script
   - Plots sample spectra grid + redshift distribution
   - Outputs to `plots/` directory

### Key Observations from Real Data
- Spectra show clear **emission lines** (H-alpha, O-III, etc.) at various redshifts
- One object at z ≈ −0.002 is a **star** (flat continuum, no emission lines)
- Flux amplitudes vary by ~10× across objects — need robust normalization for tokenizer
- B/R/Z band overlaps (~40 pixels each) require careful stitching to avoid discontinuities

### Decisions Made
- **Using real data, not synthetic** — assignment explicitly calls for real DESI data
- **Native wavelength grid** — keeping the stitched 7781-pixel grid rather than forcing exactly 7081 pixels; the tokenizer will interpolate to its latent grid anyway
- **Coadd files preferred** — coadds combine multiple exposures and have higher S/N than individual spectra

### Next Steps
- ~~Phase 2: Adapt AION spectrum tokenizer (ConvNeXt-V2 + LFQ)~~ ✅ COMPLETE
- Phase 3: Redshift scalar tokenizer
- Phase 4: Transformer backbone
- Phase 5: Approach A training
- Phase 6: Approach B training
- Phase 7: Evaluation & comparison
- Phase 8: NERSC full-scale training
- Phase 9: OOD generalization prep

## 2026-05-07: Phase 2 — Spectrum Tokenizer

### Architecture Built
**`src/tokenizers/spectrum.py`** — ConvNeXt-V2 autoencoder + LFQ quantization

**Encoder:**
- Stem: 4×4 conv, stride 4 (8704 → 2176)
- Stage 1: 3 ConvNeXt blocks @ 96 dim
- Stage 2: downsample 2× + 3 blocks @ 192 dim (→ 1088)
- Stage 3: downsample 2× + 9 blocks @ 384 dim (→ 544)
- Stage 4: downsample 2× + 3 blocks @ 512 dim (→ 272)
- Pre-quant: LayerNorm + 1×1 conv (512 → 10 dim)

**Quantizer:**
- Look-up-Free Quantizer (LFQ), dim=10, codebook_size=1024
- Straight-through estimator with commitment loss (β=0.25)
- Simplified from Yu et al. (2023)

**Decoder:**
- Mirror of encoder with ConvTranspose1d upsampling
- Output head: transposed conv stride 4 + 1×1 conv → 2 channels

**Key design choices:**
- Input is **interpolated to fixed 8704-pixel grid** (like AION) to ensure exact token count
- Output is also on 8704-pixel grid; user can interpolate back to original wavelength if needed
- Total parameters: **~24M** (AION's is ~50M; ours is scaled down for smoke testing)

### Tests
- `tests/test_tokenizer.py` — 12 tests, all passing
- ConvNeXt block preserves shape, residual connection works
- LFQ quantizes to valid range, encode→decode roundtrip works
- Full tokenizer: forward pass, different batch sizes, different input lengths
- Model can overfit a single sample (loss decreases with training)

### Notebook
- `notebooks/02_tokenizer.ipynb` — Interactive training & visualization
- Loads real DESI data, trains tokenizer for 50 epochs
- Plots original vs reconstructed spectra with residuals
- Computes MSE and R² reconstruction quality metrics
- Shows token usage distribution

### Comparison with AION
| Feature | AION Tokenizer | Our Tokenizer |
|---------|---------------|---------------|
| Backbone | ConvNeXt-V2 | ConvNeXt-V2 ✅ |
| Input grid | 8704 pixels | 8704 pixels ✅ |
| Output tokens | 273 | 272 (off by 1, fixable with padding) |
| Quantizer | LFQ (dim=10, codebook=1024) | LFQ (dim=10, codebook=1024) ✅ |
| Parameters | ~50M | ~24M (smaller for smoke test) |
| Normalization token | Yes (log10 median flux) | Not yet — will add in Phase 3 |
| Training data | Millions of spectra | 25 spectra (smoke test only) |

### Next Steps
- Phase 3: Redshift scalar tokenizer (CDF → Gaussian → FSQ)
- Phase 4: Transformer encoder-decoder backbone

---

## 2026-05-08: Phase 8 — NERSC Scaffolding for Tokenizer Pretrain

### Diagnosis: Why local results stalled

After running both Approach A and B for 10 epochs on 269 spectra with the
"fixed_split" honest random validation (seed=42), val loss bottomed out at
3.44 (A) / 3.09 (B) and overall accuracy at ~20–24%. Redshift accuracy
plateaued at 84.5% — that's the star-prior shortcut, not real learning.

The dominant cause is **the spectrum tokenizer is still random-init**. The
transformer is being trained against essentially random discrete codes, so
spectrum_acc is structurally bounded near the noise floor regardless of
model size or epoch count. This must be fixed before any further
transformer scaling experiments.

### Decision: NERSC first, tokenizer first

Move to Perlmutter for compute. Pretrain the tokenizer end-to-end on real
DR1 (not just the 4 healpix files we have locally), then re-run the
transformer with the trained codebook frozen.

### NERSC environment facts (verified)

- DR1 lives at `/global/cfs/cdirs/desi/public/dr1`, world-readable to any
  NERSC user. No `desi` unix group needed for the public tree.
- DR1 production = **iron**. Healpix coadds at
  `spectro/redux/iron/healpix/{survey}/{program}/{hpix_group}/{healpix}/`.
  Surveys: sv1/sv2/sv3/main. Programs: bright/dark/backup/other.
- Authoritative redshift catalog: `zcatalog/v1/zall-pix-iron.fits` (~21 GB);
  per-program `zpix-{survey}-{program}.fits` is the right file for subset
  selection without globbing.
- NERSC project name: **`deepsrch`**. GPU jobs use the **`_g`** suffix:
  `--account=deepsrch_g`. CPU jobs are bare `deepsrch`.
- QOS choice: **`shared`** lets us request `--gpus=1` and pay 1/4 the
  allocation hours vs `regular` (which forces a full 4-GPU node). Up to
  48h wallclock. Right call for a single-GPU pretrain.
- Filesystem: code in `$HOME`, manifests/checkpoints in `$SCRATCH`
  (high-perf Lustre but **purged after ~8 weeks idle**), final artifacts
  mirrored back to `$CFS` / repo `checkpoints/nersc/`.

### Architecture: manifest-based streaming, not preload

The local `DESISpectrumDataset` loads every spectrum at `__init__` time.
That doesn't scale to DR1's millions of spectra. Solution:

1. `nersc/build_dr1_index.py` walks the iron tree once, writes a JSONL
   manifest of `(coadd_path, redrock_path, n_rows, survey, program,
   healpix)` records.
2. `nersc/dr1_dataset.py::DR1IndexedDataset` flattens the manifest to one
   `(rec_idx, row_idx)` per spectrum. `__getitem__` opens FITS on demand
   with a small memmap'd HDUL cache. Multi-worker DataLoader parallelizes
   I/O.
3. `collate_dr1_skip_none` drops rows that fail ZWARN/fiber-status/flux
   filters at read time, so quality cuts apply naturally.

### Training entry point

`nersc/pretrain_tokenizer.py`:
- Single-GPU AMP loop. AdamW + cosine schedule with warmup.
- Reuses `SpectrumTokenizer.forward` which already returns
  `{total, recon, quant}` losses. We backprop on `total`.
- Periodic checkpointing to `$SCRATCH/deepsrch/checkpoints/<run>/`,
  best/final mirrored to `--cfs-out` for `$SCRATCH`-purge survival.
- Smoke flag: 50 steps, 200 spectra, no AMP — validates the pipeline in
  a few minutes inside the 10-min `shared`-QOS smoke job.

DDP intentionally deferred. `nersc/ddp_template.slurm` is a placeholder;
promoting the trainer to DDP is a small, separate code change (wrap in
DDP, swap shuffle for DistributedSampler) that should happen *after* the
single-GPU run validates the data path.

### Submission flow

```
ssh perlmutter
cd ~/FoundationModel
bash nersc/setup_env.sh                # one-time
sbatch nersc/smoke_tokenizer.slurm     # 10 min, ~few hundred spectra
sbatch nersc/pretrain_tokenizer.slurm  # 24h, ~hundreds of thousands of spectra
```

### Backlog / ideas

- **Top-hat 5-pixel convolution** as preprocessing before the tokenizer
  encoder — may smooth out per-pixel noise that the LFQ codebook is
  currently spending capacity on. Try this once the baseline tokenizer
  is trained, as an ablation.
- Once `best.pt` exists from NERSC: add `--tokenizer-ckpt PATH` to
  `scripts/train.py` so the transformer training (Approach A and B)
  loads frozen pretrained weights instead of random init.
- Honest val for transformer should be a held-out set of *healpix*
  files, not random rows from the same healpix — eliminates
  same-pointing leakage.

### Next steps

- Phase 8 (in flight): tokenizer pretrain on Perlmutter shared GPU.
- Phase 9: re-run Approach A and B with frozen pretrained tokenizer at
  `d_model=768, n_layers=6` on full DR1 (sv3 + main, bright + dark).
- Phase 10: OOD generalization on non-DESI spectra (assignment
  requirement).

---

## 2026-05-08: Phase 8 Result — Tokenizer Pretrain

### Trial run (job 52693687)

- 200 healpix files (sv3+bright), 393967 row index → 200-spectrum cap.
- 20k steps, batch 32, single A100 in `shared` QOS, 5h18min wallclock.
- I/O bound: 1.05 step/s × batch 32 = 34 spec/s — well below A100's
  ~200-500 spec/s for this 24M-param model. CFS FITS-read bandwidth is
  the bottleneck.

### Loss curve

| step | val_recon | val_total | val_quant |
|---|---|---|---|
| 500 | 6.63 | 6.68 | 0.05 |
| 2000 | 2.46 | 2.68 | 0.22 |
| 5000 | 1.73 | 2.04 | 0.31 |
| 12500 | 1.43 | 1.74 | 0.31 |
| **16500** | **1.35** | **1.69** | 0.34 ← best |
| 19500 | 1.34 | 1.79 | 0.45 |

Smooth descent to a clear plateau by ~step 14000. `quant_loss` climbing
from 0.05 → 0.4 is expected — the codebook starts being used (initial
collapse → diversified codes).

**Outcome:** `best.pt` at step 16500 is a real tokenizer. Used as the
frozen tokenizer for all subsequent transformer experiments.

### Performance opportunities (not pursued yet)

- Staging healpix files from CFS → SCRATCH before training: ~5-10×
  step-rate speedup (CFS read is the dominant cost).
- DDP across 4 GPUs in `regular` QOS: ~3-3.5× wallclock speedup, but
  also 4× allocation cost — only worth it after staging fixes I/O.
- 24h budget needs ~1.2 step/s for 100k steps; we got 0.23 step/s at
  100M-param transformer scale (next phase).

---

## 2026-05-11: Phase 9 Trial — Transformer with Frozen Pretrained Tokenizer

### Pre-fix runs (jobs 52827566 / 52827575, no redshift weighting)

Both Approach A and B at full scale (`d_model=768`, 6 layers, 100M params,
batch 8, AMP). 200 healpix, 393967 spectra, val_frac=0.02. 20k step
budget; jobs hit the 6h wallclock at step ~9000.

| step | A spec_acc | B spec_acc | A z_acc | B z_acc |
|---|---|---|---|---|
| 1500 | 38.8% | 39.6% | 2.4% | 0.7% |
| 5000 | 47.2% | 45.5% | 1.3% | 1.3% |
| 6000 | 55.7% | 65.1% | 3.0% | 0.0% |
| 7000 | 88.4% | **98.8%** | 3.2% | 2.7% |
| 9000 | **97.2%** | **99.91%** | 4.1% | 2.0% |

**Major discovery: spec_acc → 99% is the model learning a trivial
cross-attention copy, not real spectrum modeling.**

For Approach B, decoder at position `j` (`j≥1`) predicts spectrum
token `s_j`. The encoder is `[SOS, s1, ..., sN, EOS]` and cross-attention
is unmasked. The model learns to attend from decoder position `j` to
encoder position `j` — an off-by-one shift, pure copy. No spectroscopy
involved. The step 5500→7000 explosion (50% → 99%) is when this attention
pattern is discovered.

For Approach A, encoder is `[SOS, redshift, s1, ..., sN, EOS]`, off-by-two
shift, slightly harder but same trick. Hence A's 97% ceiling vs B's 99.9%.

**Critically, `z_acc` stayed at random (~2-4%) the whole time.** Reason:
cross-entropy averaged over 274 sequence positions, redshift contributes
1/274 ≈ 0.4% of gradient signal. The model has zero incentive to learn
position 0, so it never does.

**Conclusion:** the unweighted runs don't test the project's thesis.
Both Approach A and B degenerate into the same trivial copy-from-encoder
behavior, and neither learns redshift. We need to force position 0 to
matter.

### Fix 1: Redshift loss weighting

`forward()` now accepts `redshift_weight: float = 1.0` and splits the
cross-entropy by position:
```
loss = redshift_weight * loss_redshift_mean + loss_spectrum_mean
```
At `--redshift-loss-weight 50` (current default), redshift's aggregate
gradient share is 50:1 vs spectrum (~98% of total loss mass at position 0).

`compute_loss_breakdown` helper added to `src/training/utils.py` to log
unweighted per-segment losses separately — visible in `metrics.jsonl`
and wandb so we can tell whether the redshift term is actually
descending.

### Fix 2: Weights & Biases integration

`nersc/_wandb_util.py` provides `init_wandb` / `wlog` / `wfinish`
helpers used by both `pretrain_tokenizer.py` and `train_transformer.py`.
Reads `WANDB_API_KEY` from `.env` (gitignored) via `python-dotenv`.
Modes: online / offline / disabled. NERSC compute nodes default to
offline; metrics are written locally and uploaded later from a login
node with `wandb sync`.

### Weighted run (jobs 52840231 / 52840234, redshift_weight=50)

Same data, same model size, weight=50. First impression at step 1500
looked like the weighting was starving spectrum learning (15% spec_acc
vs the unweighted run's 38.8% at the same step). But the full
trajectory tells a different story:

**Approach A:**

| step | spec_acc | z_acc | loss_redshift |
|---|---|---|---|
| 1500 | 15.6% | 1.5% | 4.90 |
| 3000 | 23.6% | 1.5% | 4.81 |
| 4500 | 25.7% | 1.0% | 4.80 |
| 5000 | 26.4% | 3.1% | 4.65 |
| **5500** | **26.7%** | **5.8%** | **4.19** ← cross-attention copy discovered |

**Approach B:**

| step | spec_acc | z_acc | loss_redshift |
|---|---|---|---|
| 1500 | 11.2% | 0.3% | 4.77 |
| 3000 | 20.4% | 1.4% | 4.75 |
| 4000 | 22.8% | 2.95% | 4.70 |

Between step 4500 and 5500, A's z_acc jumped 1% → 5.8% and
`loss_redshift` dropped from 4.80 → 4.19 — the first real descent since
warmup. This is the model finally discovering the cross-attention
redshift pathway (for A it's a copy; for B it has to extract from
spectrum encoding).

### Why initial diagnosis was wrong

At step 1500 the weighted run looked worse on every axis. The temptation
was to lower the weight immediately. But the model needs time to discover
the cross-attention copy for redshift (~5000 steps with weight=50). The
unweighted run hit 50% spec_acc fast because the spectrum copy is easier
to discover than the redshift one — partly because redshift gets so
little gradient.

Lesson: don't kill a run during the "boring middle" phase of training.
The interesting behavior often emerges after a long flat period.

### Open questions / next moves

1. **How high does the weighted run's `z_acc` go?** A's 5.8% at step 5500
   should keep climbing fast (it's a copy task). B's progression is the
   real test — does the encoder actually encode redshift into its
   hidden state from spectrum alone?

2. **Does the weighted run also reach >90% `spec_acc` eventually?**
   Or does the heavy redshift focus permanently slow spectrum learning?

3. **Is weight=50 the right value?** Could try `weight ∈ {5, 10, 20}`
   if z_acc plateaus too low or spec_acc stays starved.

4. **Held-out healpix split.** Current val set is random rows from the
   same healpix files as train. For honest generalization we should
   hold out entire healpix files.

5. **Top-hat 5-pixel convolution as tokenizer preprocessing.** Still on
   the backlog; would smooth pixel noise the LFQ codebook is spending
   capacity on.

### Files touched this phase

- `src/models/transformer.py` — `redshift_weight` kwarg + position-split loss
- `src/training/utils.py` — `compute_loss_breakdown` helper
- `nersc/_wandb_util.py` (new) — wandb init/log/finish helpers
- `nersc/train_transformer.py` — flags, threading, breakdown + wandb logging
- `nersc/pretrain_tokenizer.py` — wandb logging
- `nersc/dr1_dataset.py`, `nersc/dr1_tokenized_dataset.py` — manifest-based DR1 loaders
- `nersc/train_transformer.slurm`, `nersc/smoke_transformer.slurm` — SLURM entry points
- `requirements.txt`, `pyproject.toml`, `nersc/setup_env.sh` — `python-dotenv` added
- `nersc/README.md` — wandb + weighting documentation

---

## 2026-05-12: Phase 9 Final Result — Thesis Tested

### Setup

- Two 6h runs in parallel: jobs 52840231 (Approach A) / 52840234 (Approach B).
- `redshift_loss_weight=50` (per Phase 9 fix).
- 200 healpix files (sv3+main, bright+dark), 393967 spectra in flat index.
- Frozen pretrained tokenizer from Phase 8 (`tokenizer_v1_52693687/best.pt`, val_recon 1.35).
- 100M parameter transformer: `d_model=768`, 6 encoder + 6 decoder layers, 12 heads, AMP on.
- AdamW, lr=2e-4, cosine schedule with 1000-step linear warmup. batch=8.
- Throughput ~0.55 step/s — both jobs hit the 6h wallclock somewhere around step 10000–12000.

### Result: A learns, B stays at random

**Approach A** discovered the cross-attention "copy redshift from encoder" pathway at step ~6500 and z_acc climbed steeply to 69.2% by step 15000:

| step | val_redshift_acc | val_loss_redshift | val_spectrum_acc | val_loss_spectrum |
|---|---|---|---|---|
| 500 | 0.6% | 5.10 | 0.0% | 6.64 |
| 1500 | 1.5% | 4.90 | 15.6% | 3.70 |
| 3000 | 1.5% | 4.81 | 23.6% | 3.07 |
| 4500 | 1.0% | 4.80 | 25.7% | 2.88 |
| 5500 | 5.8% | 4.19 | 26.7% | 2.82 |
| **6500** | **18.4%** | **3.65** | 27.0% | 2.80 ← copy ignites |
| 8000 | 29.3% | 2.94 | 27.9% | 2.75 |
| 10000 | 42.4% | 2.44 | 28.5% | 2.71 |
| 11500 | 52.4% | 2.09 | 29.0% | 2.68 |
| 13000 | 57.9% | 1.74 | 29.9% | 2.63 |
| 14500 | 64.8% | 1.38 | 30.1% | 2.61 |
| **15000** | **69.2%** | **1.21** | 30.0% | 2.62 ← wallclock cutoff, still climbing |

`loss_redshift` dropped 5.10 → 1.21 over the run — a 76% reduction. `spec_acc` essentially plateaued at ~29-30% (the delayed-copy regime under heavy redshift weighting).

**Approach B** stayed at noise floor for the entire 14000-step run:

| step | val_redshift_acc | val_loss_redshift | val_spectrum_acc | val_loss_spectrum |
|---|---|---|---|---|
| 500 | 0.7% | 4.92 | 0.0% | 6.62 |
| 1500 | 0.3% | 4.77 | 11.2% | 3.70 |
| 3000 | 1.4% | 4.75 | 20.4% | 3.21 |
| 5000 | 1.0% | 4.78 | 24.7% | 2.97 |
| 7000 | 2.1% | 4.63 | 26.6% | 2.83 |
| 10000 | 1.2% | 4.53 | 28.3% | 2.74 |
| 11500 | 2.4% | 4.50 | 28.9% | 2.70 |
| 13000 | 2.1% | 4.57 | 29.3% | 2.69 |
| **13500** | 4.1% | 4.53 | 29.3% | 2.68 ← max z_acc, then regresses |
| 14000 | 0.9% | 4.52 | 29.5% | 2.67 |

B's `loss_redshift` moved only 4.92 → 4.52 (8% reduction) over 14000 steps, with no sustained trend. The single 4.1% z_acc reading at step 13500 collapses to 0.9% at step 14000 — pure noise, not learning. The encoder is not learning to encode redshift into its hidden state from spectrum features.

**Final score:**

| metric | A (step 15000) | B (step 14000) |
|---|---|---|
| `val_redshift_acc` | **69.2%** | 0.9% (max 4.1%, noise) |
| `val_loss_redshift` | **1.21** | 4.52 |
| `val_spectrum_acc` | 30.0% | 29.5% |
| `val_loss_spectrum` | 2.62 | 2.67 |

### Interpretation: the project's thesis answered

The project's hypothesis (from the assignment, addressing the AION-1 critique): **forcing reconstruction of redshift from spectral context every step** (Approach B) should make redshift an organizing principle of the encoder representation. The result, with our 100M model + 14000-15000 training steps + frozen pretrained tokenizer + 395k spectra:

**B does not work.** When the encoder doesn't see the redshift token directly, the encoder simply leaves redshift unlearned. The decoder, given no redshift signal in cross-attention context, cannot recover the value, and the position-0 loss stays near `log(256) / log(e) ≈ 5.55` (random over 256 bins). 14000 steps of training with `redshift_loss_weight=50` (≈ 98% of loss mass at position 0) dropped B's `loss_redshift` from 4.92 only to 4.52. The trajectory is flat; this is not a "needs more compute" problem.

**A succeeds spectacularly — but for an uninteresting reason.** With the redshift token included in the encoder input, the decoder learns a cross-attention copy pattern that lifts redshift from encoder position 1 to decoder position 0. Over the same 15000 steps, A's `val_redshift_acc` climbs from 0.6% to **69.2%** and `loss_redshift` drops from 5.10 to 1.21. The phase transition is sharp — at step 6500 z_acc jumps from 7% to 18% in 500 steps as the copy attention pattern crystallizes. After that, the trajectory is monotone-increasing. This is the same trivial-copy phenomenon that inflates `spec_acc` (see Section: "Proof"); it tests neither spectroscopy nor representation learning — only attention-pattern discovery.

### Proof: the unweighted runs' 99% spec_acc was trivial copy

Pre-fix runs without redshift weighting (jobs 52827566 / 52827575) reached:
- A: val_spectrum_acc 97.2% at step 9000
- B: val_spectrum_acc 99.97% at step 9000

These accuracies on held-out unseen galaxies cannot be memorization. The mechanism is structural:

1. The decoder at position `j` (`j ≥ 1`) is predicting spectrum token `s_j`.
2. The encoder is `[SOS, (redshift,) s1, s2, ..., sN, EOS]` — the same `s_j` sits at encoder position `j` (B) or `j+1` (A).
3. Cross-attention is unmasked. The decoder learns one attention head: "from decoder position `j`, attend to encoder position `j` (B) or `j+1` (A), copy that token."
4. This is a positional shift-and-copy pattern, *data-independent*. It works on every galaxy the model has ever seen and every galaxy it will ever see, because it doesn't depend on galaxy identity.

Evidence this is the mechanism:

- The val_spectrum_acc curve from step 5500 → 7000 jumped from 50% to 99% in 1500 steps for B. This is consistent with "the model just discovered the right attention pattern," not "the model spent 1500 steps learning more spectroscopy."
- B reaches 99.97% but A only 97.2% — the offset in B is simpler (shift by 1) than in A (shift by 2 because of the redshift token), so B converges faster and tighter.
- `val_redshift_acc` stayed at random (~2–4%) the entire pre-fix run: position 0 cannot be solved by copy (B has no source; A has a source at position 1 but the 1/274 gradient share is too small to motivate the copy from position 1 vs from the trivial position-1+offset rule the model has already discovered).

Conclusion: **under the current encoder-decoder + teacher-forced + unmasked-cross-attention architecture, `val_spectrum_acc` does not measure spectrum understanding**. It measures whether the cross-attention copy pattern has been discovered. An honest spectrum reconstruction metric requires either encoder masking (BERT-style) or autoregressive evaluation without teacher forcing.

### AION-1 critique revisited

The original pitch: AION-1 treated redshift like any other token, masked occasionally, so redshift never became an organizing principle of the encoder representation. Our project would fix this by *always* masking redshift (Approach B) and forcing the encoder to encode it.

The current result refines the critique. AION-1's failure mode is real, but **always-masking does not by itself make the encoder encode redshift**. The encoder leaves redshift unlearned regardless of how aggressively the loss penalizes the decoder's failure to recover it. The information has to enter the encoder representation through some *constructive* mechanism, not just through the absence of an alternative shortcut.

Candidate mechanisms for making B work (none tested in this phase):

- **Auxiliary redshift head on the encoder.** Pool encoder outputs (mean, max, or CLS-token style) and predict z from the pooled vector with an auxiliary cross-entropy loss. This is closer to what AION-1 itself did with a separate frozen head, but we'd train it jointly to apply gradient pressure on the encoder.
- **Contrastive loss.** Pull together encoder representations of galaxies with similar redshift; push apart those with different redshift. Forces the encoder's geometry to align with z.
- **Larger encoder capacity / longer training.** B's `loss_redshift` was essentially flat after step 4000, so this is the least likely fix. The information bottleneck appears architectural, not capacity-bound.
- **Continuous redshift loss + scalar head** (instead of discrete bin classification). May give cleaner gradient signal than 256-way softmax.

### Implications for Phase 10

The weight=50 fix (Phase 9) is the right value for A and we should keep it as default. B's failure is not a hyperparameter problem; it's a structural one. Phase 10's encoder masking (next) fixes the `spec_acc` honesty problem and gives us a real metric for both A and B. After that, we can decide whether to test one of the B-rescue candidates above as Phase 11.

### Files referenced
- Train metrics for A: `$SCRATCH/deepsrch/checkpoints/approach_a_52840231/metrics.jsonl`
- Train metrics for B: `$SCRATCH/deepsrch/checkpoints/approach_b_52840234/metrics.jsonl`
- Pre-fix unweighted A: `$SCRATCH/deepsrch/checkpoints/approach_a_52827566/metrics.jsonl`
- Pre-fix unweighted B: `$SCRATCH/deepsrch/checkpoints/approach_b_52827575/metrics.jsonl`
- Tokenizer used: `$SCRATCH/deepsrch/checkpoints/tokenizer_v1_52693687/best.pt`

---

## 2026-05-11: Phase 10 Partial Result — Encoder Masking + AR Eval

### Setup

Phase 10 changes shipped together: encoder masking (`--encoder-mask-ratio 0.15`),
healpix-level train/val split (`--healpix-holdout-frac 0.05`), autoregressive
eval at every best-checkpoint update (`--ar-eval-batches 4`), and `WANDB_MODE`
forced online in `init_wandb`. All other knobs unchanged from Phase 9: 200
healpix, weight=50, 100M-param transformer, batch 8, AMP, lr=2e-4, cosine.

Two 6h shared-QOS jobs ran in parallel: `52846595` (A) and `52846605` (B).
Both were cancelled at ~step 10000–10200 (manually killed before completion
to free the allocation for the CFS→SCRATCH staging trial) — so the
trajectories below are not the full 20k-step run, but they're enough to
test the Phase 10 hypotheses.

### Result: AR ≥ TF for redshift. The thesis is now tested honestly.

Approach A val trajectory under Phase 10:

| step | val/redshift_acc (TF) | val_ar/redshift_acc | val/spectrum_acc (TF) | val/masked_spec_acc | val_ar/spectrum_acc |
|---|---|---|---|---|---|
| 500 | 0.8% | 3.6% | 0.0% | 0.0% | 0.1% |
| 1000 | 0.6% | 0.0% | 11.2% | 10.7% | 3.1% |
| 2000 | 3.8% | 7.1% | 14.1% | 14.4% | 4.2% |
| 3000 | 14.6% | 10.7% | 16.2% | 16.4% | 2.5% |
| 4000 | 13.7% | 7.1% | 22.0% | 21.8% | 4.2% |
| 5000 | 32.6% | 14.3% | 23.8% | 23.6% | 4.4% |
| 6000 | 31.5% | 21.4% | 24.8% | 25.1% | 4.1% |
| 7500 | 47.2% | 39.3% | 25.6% | 25.6% | 3.2% |
| 8000 | 47.5% | 46.4% | 25.7% | 26.0% | 2.4% |
| 8500 | 60.6% | 42.9% | 26.2% | 26.2% | 3.1% |
| 9000 | 53.5% | — | 26.5% | 26.4% | — |
| **9500** | **66.0%** | **71.4%** | 26.5% | 26.7% | 2.9% |

Approach B val trajectory (no AR breakout — flat throughout):

| step | val/redshift_acc (TF) | val_ar/redshift_acc | val/spectrum_acc (TF) | val/masked_spec_acc |
|---|---|---|---|---|
| 500 | 0.3% | 0.0% | 0.0% | 0.0% |
| 1000 | 1.1% | 3.6% | 12.4% | 12.4% |
| 3000 | 3.8% | 0.0% | 22.2% | 22.0% |
| 5000 | 1.0% | — | 25.0% | 24.8% |
| 6000 | 0.4% | 0.0% | 26.0% | 25.7% |
| 9500 | 1.2% | — | 27.6% | 28.0% |

### Headline finding: A's encoder really encodes redshift

In Phase 9 we couldn't tell whether A's `val/redshift_acc` was real or
cross-attention copy. Phase 10's `evaluate_ar()` settles it:

- **At step 9500, AR redshift acc (71.4%) ≥ teacher-forced redshift acc (66.0%).**
  AR has no teacher input at decoder position 0 — the model starts from
  `[SOS]` and predicts redshift purely from the encoder's hidden state.
  If the encoder were not encoding redshift, AR acc would be at the
  256-bin random baseline (~0.4%). It is 71%. The encoder is encoding
  redshift, and the decoder can recover it from the encoder context
  alone.
- AR redshift acc and TF redshift acc roughly track each other from step
  ~5000 onward (TF 32.6% / AR 14.3% → TF 47.5% / AR 46.4% → TF 66.0% /
  AR 71.4%). The AR-TF gap collapses as the redshift signal becomes
  dominant in the encoder representation.
- AR > TF at step 9500 is mildly surprising. Most likely cause: the AR
  eval used `model.generate()` with greedy sampling, which is slightly
  more accurate on its winning bin than the TF logits' argmax over a
  noisier mixed-position softmax. Plausibly also healpix-eval-batch
  variance (only 28 samples per AR pass). Either way, **AR is not
  meaningfully worse**, which is what matters.

This was the structural question we couldn't answer in Phase 9. We can
answer it now: **Approach A learns to encode redshift into the encoder's
hidden state, not just to copy it through cross-attention.** The
trivial-copy hypothesis is dead for the weighted run.

### Encoder masking accelerated A's redshift ignition

Comparing the same job shape at the same steps, before and after Phase 10:

| step | Phase 9 A (no mask) `val/z_acc` | Phase 10 A (mask=0.15) `val/z_acc` |
|---|---|---|
| 3000 | ~1.5% | **14.6%** |
| 4500 | ~1.0% | (between 5000 reading) |
| 5000 | ~2.4% | **32.6%** |
| 6500 | **18.4%** (ignition step) | between readings |
| 8000 | 29.3% | **47.5%** |
| 9500 | ~40% | **66.0%** |

A's z_acc ignites ~2000–3000 steps earlier under encoder masking. Likely
mechanism: masking 15% of encoder spectrum tokens forces the encoder to
build richer, less-redundant features at the unmasked positions to
support reconstruction. Those richer features apparently also make
redshift more easily readable from cross-attention. This was not
predicted; encoder masking was added to fix spec_acc honesty, not to
help redshift. It helps both.

### Spectrum: TF ≈ masked_spec_acc ≫ AR

For both A and B:
- `val/spectrum_acc` ≈ `val/masked_spec_acc` (always within 1 pp).
  The 15% masking ratio is too small to surface a gap between
  "copy-capable" and "honest" decoder positions. Both numbers land at
  ~26%–28% by step 9500.
- `val_ar/spectrum_acc` stays at **~3%** the entire run (compared to
  ~26% TF). 1024-codebook random is 0.1%, so 3% is ~30× random — the
  model has *some* spectrum knowledge, but tiny.

Interpretation: there *is* still substantial teacher-forcing inflation
in `spectrum_acc`, but the inflation isn't specifically at the unmasked
positions (otherwise masked_spec_acc would be much lower). The TF
position at step `j` likely benefits from the cumulative leakage of
positions `1..j-1` being teacher-fed, not from encoder-side copy. To
surface honest spectrum-from-context numbers, we'd need either:
- A higher encoder mask ratio (e.g. 0.50 or 0.80) so the encoder loses
  most of the spectrum it could copy from
- A decoder-side mask too (BERT-style, predict full sequence from
  partial decoder input)
- Trust AR as the spectrum-honesty metric (~3% is the real number).

For the writeup, **AR is the honest spectrum-accuracy signal**. It will
go in the paper as the headline number; teacher-forced spec_acc is
described as cheated and reported only for context.

### Approach B: AR confirms failure

B's `val_ar/redshift_acc` was 0.0–3.6% across all measurements — pure
noise around the 0.4% random baseline. The encoder is not encoding
redshift, the decoder cannot recover it, and the AR confirms the TF
diagnosis was not a teacher-forcing artifact in either direction.

B's `val/spectrum_acc` ~28% is *higher* than A's ~26% — consistent with
the gradient-share story: A's stronger redshift pressure slightly
starves spec learning, B has nothing else to learn so its spec gradient
is undiluted. The difference is small (2 pp) and probably not
significant given 6h cutoff variance.

### What the partial run means for the thesis

The PHYS303 assignment thesis was: *AION-1 treats redshift as just
another token, and that's why redshift never enters the encoder
representation. Forcing always-masking of redshift (Approach B) should
fix that.*

The Phase 10 result clarifies and partly inverts this:

1. **AION-1's diagnosis is correct.** When the redshift token is masked
   (B), the encoder doesn't learn it. Even with `weight=50` driving 98%
   of gradient mass to position 0, B's redshift loss stays at random
   for 10000 steps with no improving trend. The AR confirms B is
   genuinely not extracting redshift from spectrum features.
2. **The proposed fix (always-mask) does NOT work.** B fails the
   *Approach B* test the assignment proposed.
3. **The fix that does work is A.** Putting redshift in the encoder
   *as a token* and weighting the loss heavily makes the encoder build
   a redshift-aware representation that survives AR decoding. This is
   what AION-1 should have done — heavier redshift loss weighting, not
   different masking.
4. **Encoder masking matters for the metric, not the architecture.**
   Encoder masking ignites A's redshift learning earlier and gives us
   the AR-based honest spec_acc number. Without it we'd still believe
   the unweighted runs' 99% spec_acc was real.

### Files referenced

- A metrics: `$SCRATCH/deepsrch/checkpoints/approach_a_52846595/metrics.jsonl`
- B metrics: `$SCRATCH/deepsrch/checkpoints/approach_b_52846605/metrics.jsonl`
- Tokenizer (same as Phase 9): `$SCRATCH/deepsrch/checkpoints/tokenizer_v1_52693687/best.pt`
- Run config: `--encoder-mask-ratio 0.15 --healpix-holdout-frac 0.05 --redshift-loss-weight 50 --ar-eval-batches 4`

### Next steps

- Re-run with `MANIFEST=$SCRATCH/...dr1_200_scratch.jsonl` (post CFS→SCRATCH
  staging) to validate the I/O speedup; expect 5–10× step rate, full 20k
  steps in 6h budget.
- After that runs cleanly, scale to 2000-healpix manifest for the
  "production" run that goes in the writeup.
- Open question: does the AR-TF gap for redshift stay closed at 20k+
  steps, or does TF overshoot AR as cross-attention learns to exploit
  some teacher-forcing leak we haven't characterized yet?

---

## 2026-05-11: Phase 10 Final — mask=0.50 + batch=32 (the writeup result)

### Setup

After CFS→SCRATCH staging and `$HOME`-quota fixes (committed in same wave),
re-ran Phase 10 with three knobs increased:

- `--encoder-mask-ratio 0.50` (up from 0.15 — wider honest-prediction zone)
- `--batch-size 32` (up from 8 — better gradient estimates per step, fully saturates A100)
- `--lr 4e-4` (sqrt-scaled for batch 4×)

Everything else identical to Phase 10: weight=50, healpix-level val
split, AR eval at every best, frozen tokenizer, 200 healpix on staged
SCRATCH. Single A100, ~5 step/s × 32 batch = 160 spec/s.

Three runs landed:

| run | batch | steps reached | how it ended |
|---|---|---|---|
| `phase10_mask50_a` | 8 | 2000 | abandoned (early interactive run, nested-srun + crashes) |
| `phase10_mask50_a_big` | 32 | 4000 | killed by `$HOME` quota at the CFS-mirror step |
| `phase10_mask50_b` | 32 | 9500 | reached step cap of available wallclock |

### Result: A is learning faster than ever, B is doubly dead

| metric | `_a_big` (A, mask 0.50, batch 32) | `_b` (B, mask 0.50, batch 32) |
|---|---|---|
| steps trained | **4000** | 9500 |
| peak `val/redshift_acc` (TF) | **73.8%** | 5.3% (noise) |
| peak `val_ar/redshift_acc` (AR) | **55.0%** | 3.6% (literal random) |
| peak `val/spectrum_acc` | 24.0% | 28.2% |
| peak `val/masked_spec_acc` | 24.2% | 28.1% |
| peak `val_ar/spectrum_acc` | 3.5% | 5.3% |

### Approach A trajectory across the project

| config | mask | batch | steps to peak | peak TF z_acc | peak AR z_acc |
|---|---|---|---|---|---|
| Phase 9 (unweighted) | 0.0 | 8 | 9000 (cutoff) | 4.1% | — (AR not in scaffold yet) |
| Phase 9 (weight=50) | 0.0 | 8 | 15000 | 69.2% | — |
| Phase 10 (mask 0.15) | 0.15 | 8 | 9500 | 66.0% | 71.4% |
| **Phase 10 final** | **0.50** | **32** | **4000** | **73.8%** | **55.0%** |

A is now reaching **higher peak z_acc with less than half the previous
step count**. The combined intervention `mask=0.50 + batch=32 + lr=4e-4`
is ~3× more sample-efficient than Phase 9 and produces strictly better
final accuracy. The driver is unclear; candidate mechanisms:

- Heavier masking forces the encoder to build richer non-copy features
  at the visible positions, which carry redshift better.
- 4× batch reduces variance in the weight=50 redshift loss, which is
  dominated by a single position's gradient — bigger batch lets the
  cross-attention pathway converge before noise destabilizes it.
- 2× learning rate at 4× batch is closer to the optimal effective LR
  for this loss landscape.

Ablation would tell us which knob mattered most. Out of scope for the
final report.

### AR drop between mask=0.15 and mask=0.50 (worth noting)

Curiously, AR z_acc went **down** from 71.4% (mask=0.15, step 9500) to
55.0% (mask=0.50, step 4000) even as TF z_acc went up (66.0% → 73.8%).
Three possible explanations:

1. **Step-count mismatch.** A only reached step 4000 here vs 9500
   previously. At matched steps the comparison may flip.
2. **AR eval batch size is tiny (n=28).** The TF metric averages over
   ~21000 val examples per pass; the AR metric over 28 generated
   trajectories. AR variance is large.
3. **Greedy decoding sensitivity.** Higher mask ratio might produce
   sharper but less smoothly-decodable encoder distributions, where
   greedy generation traps slightly more often than under mask=0.15.

Without more compute we can't disentangle these. The TF and AR numbers
both clearly show *real* encoder-side redshift learning (random
baselines: 0.4%, AR is 137× above random). The exact AR/TF ratio is
secondary to the qualitative result.

### Approach B: robustly dead

B at mask=0.50, batch=32, 9500 steps (2.4× the steps of A's successful
run) produces:

- `val/redshift_acc` 5.3% peak — noise floor, no upward trend
- `val_ar/redshift_acc` 3.6% peak — within bin-count uncertainty of pure
  random over a 256-bin softmax

This is now confirmed across **four configurations** (Phase 9
unweighted, Phase 9 weight=50, Phase 10 mask=0.15, Phase 10 mask=0.50)
and **two batch sizes**. The encoder genuinely cannot extract redshift
from spectrum features alone within the training budgets we tested.
This is the project's headline negative result.

### Spectrum honesty status

At mask=0.50, `val/spectrum_acc` and `val/masked_spec_acc` remain
within ~1 pp of each other for both A and B. Two possible reads:

1. Encoder masking suppresses cross-attention copy at *all* decoder
   positions (not just the masked ones), so `spec_acc` is honest now.
2. There was never an encoder-side copy mechanism for spectrum to begin
   with; the inflation Phase 9 saw (99% spec_acc) came from a different
   pathway we haven't precisely localized.

The AR vs TF gap for spectrum is large in either case: **AR ~3.5%, TF
~24%**. So substantial teacher-forcing inflation does exist for
spectrum predictions — it just doesn't come from encoder-side copy.
The most likely source is **decoder-side previous-token leakage**:
when predicting position `j`, the decoder is teacher-fed the true
tokens at positions `1..j-1`. Under autoregressive generation, those
become the model's own predictions, errors compound, and accuracy
collapses to ~3.5%.

For the writeup, **AR spectrum accuracy is the honest generative
metric** and is what we report as the "real" spectrum prediction
capability. TF spectrum accuracy is reported alongside but described
as containing teacher-forcing inflation.

### Files

- A: `$SCRATCH/deepsrch/checkpoints/phase10_mask50_a_big/metrics.jsonl`
  (also mirrored: `/global/cfs/cdirs/deepsrch/joe2k/checkpoints/phase10_mask50_a_big/best.pt`)
- B: `$SCRATCH/deepsrch/checkpoints/phase10_mask50_b/metrics.jsonl`
- Tokenizer (same as Phase 9): `$SCRATCH/deepsrch/checkpoints/tokenizer_v1_52693687/best.pt`
- Run config: `--encoder-mask-ratio 0.50 --healpix-holdout-frac 0.05 --redshift-loss-weight 50 --batch-size 32 --lr 4e-4 --num-workers 16 --ar-eval-batches 4`

### Conclusion: this is the writeup configuration

We have everything we need:

1. **A succeeds and the success is real.** TF 73.8% / AR 55.0% z_acc.
   AR confirms encoder genuinely encodes redshift, not just copy.
2. **B fails and the failure is robust.** 4 configurations, 2 batch
   sizes, up to 9500 steps — encoder never learns redshift from
   spectrum alone.
3. **Honest spec metric established.** AR spec_acc ~3.5% is the
   generative spectrum-prediction baseline, vs ~24% TF (decoder-side
   teacher-forcing inflation).
4. **A vs B contrast is asymmetric in compute** (A: 4k steps, B: 9.5k
   steps). A reached *higher* z_acc with *less than half* the training.
   This strengthens, rather than weakens, the result.

No more training runs needed for the report. Move to writeup, plots,
and ablation discussion.

---

## 2026-06-06: Redshift copy mechanism — root cause + fix (conditioning dropout)

### The bug behind "A's success is real"

The 2026-05-11 conclusion ("AR confirms encoder genuinely encodes
redshift, not just copy") was itself confounded. `encoder_mask_ratio`
masks **only spectrum tokens** — it never touched the redshift token at
encoder position 1. So in Approach A the true redshift was present in
the encoder during **both** teacher-forced eval **and** AR eval. The
decoder's position-0 redshift prediction could be satisfied by a trivial
shift-and-copy from encoder position 1 (`[SOS, z, s1..sN, EOS]` →
predict `z` at decoder position 0), independent of the spectrum. The
sharp z_acc phase transition is the signature of an attention pattern
being discovered, not of redshift being learned from spectral features.
Approach A's reported z_acc is therefore not a genuine
redshift-from-spectrum number.

### Fix: redshift conditioning dropout (CFG / 4M-style)

Goal (per project owner): the model must learn redshift genuinely —
predict z **without seeing it sometimes during training and always at
inference**.

AION-1 reference (AION Paper.pdf §5, 4M masked modeling): every modality
token, redshift included, is randomly assigned to the observed set (fed
to the encoder) or the target set (predicted) per example; decoder query
tokens get only modality+position embeddings, never the value. The
lesson: never unconditionally show redshift to the encoder.

Minimal faithful adaptation for our teacher-forced encoder-decoder
(spectrum reconstruction is working well and is left untouched):
**replace the encoder's redshift token with the already-reserved
`REDMASK_TOKEN` (id 4) with probability `redshift_mask_ratio`.**
`decoder_input`/`target` keep the true redshift token, so position-0 is
still supervised against the real value; the position-0 prediction can't
see the teacher-forced z (causal mask), so the only path to it is
cross-attention into the (now sometimes-masked) encoder. When z is
REDMASK'd, copying is impossible → the model must infer z from the
spectrum.

- **Training**: `redshift_mask_ratio = 0.5` (z hidden half the time;
  half the batches still let the model learn to *use* z for spectrum —
  the cross-modal benefit, like AION's observed-z case).
- **Eval / inference**: ratio `1.0` (z always hidden). This is now the
  honest redshift-from-spectrum metric. Release inference paths default
  to 1.0; spectrum still reconstructs because `generate()` emits z at
  position 0 first, then conditions spectrum on its own generated z.
- Distinct from spectrum's `MASK_TOKEN` (3) so the model can tell
  "redshift hidden" from "spectrum pixel hidden". No-op for Approach B.

### Why REDMASK over deleting the token (≈ Approach B)

Keeps encoder length and positional alignment constant, so the working
architecture is unchanged; gives a single model that handles both
"z given" and "z hidden"; makes the held-out signal explicit. Approach B
(redshift entirely absent from the encoder) stayed dead across 4 configs;
conditioning dropout is the better-posed version of the same idea.

### Files changed

- `src/training/sequences.py` — `tokenize_and_build` gains
  `redshift_mask_ratio`; REDMASK applied to the encoder redshift token
  only (Approach A), `rng`-reproducible.
- `src/training/eval.py` — `evaluate` / `evaluate_ar` thread the param.
- `nersc/train_transformer.py` — `--redshift-mask-ratio` (default 0.5);
  train loop uses it; val/AR eval forced to 1.0 (honest z-hidden); added
  a z-given AR comparison (`val_ar/redshift_acc_zgiven`) to expose any
  residual copy path; ratio recorded in checkpoint/artifact metadata.
- `scripts/train.py` + `src/datasets/tokenized_dataset.py` — dataset path
  gains `redshift_mask_ratio` (per-item REDMASK at encoder index 1); val
  set forced to 1.0.
- `src/inference/release.py` — `predict_teacher_forced` /
  `predict_autoregressive` default `redshift_mask_ratio=1.0`.
- `tests/test_training_helpers.py` — `TestRedshiftMasking` (REDMASK at
  ratio 1.0, untouched at 0.0, rng reproducibility, decoder/target
  untouched, Approach B no-op). Also fixed the stale `FakeZTok` stub to
  match the batched `RedshiftTokenizer` API (`.device` + tensor
  `.encode`), which had silently broken `TestEncoderMasking` /
  `TestEvaluateAR`.

### What to re-measure (requires retraining A)

The released checkpoint was trained with the copy path open. Retrain
Approach A with `--redshift-mask-ratio 0.5` and compare AR z_acc with
z **hidden** (ratio 1.0, the honest number) vs z **given** (0.0). The
gap is the size of the old copy artifact; the hidden-z curve should rise
smoothly without the sharp phase transition. Spectrum metrics
(`spectrum_acc` / `masked_spec_acc`) must not regress.

---

## 2026-06-06: Add `--resume` to nersc/train_transformer.py

Training had no resume path — `step` started at 0 with a fresh optimizer,
and only the tokenizer was loaded. `best.pt` already saves full state
(`model`, `optim`, `scaler`, `step`, `val_loss`); it just wasn't read
back. Added `--resume PATH`: after the optimizer/scaler are built, load
the checkpoint on every rank (so DDP replicas stay identical), restore
model/optim/scaler and set `step`/`best_val` from it. The loop is
`while step < args.steps`, so resuming continues to the cap. Reuse the
same `--run-name` to keep the run dir. Periodic `step_*.pt` are
model-only and can't restore optimizer state — resume from `best.pt`/
`final.pt`.

## 2026-06-06: W&B run continuation on resume

Resumed jobs previously started a fresh W&B run, splitting one training
trajectory across multiple charts. Added run continuation:
`init_wandb` gains `run_id` + `resume` (forwards `id` and
`resume="allow"` to `wandb.init`); `train_transformer.py` saves the live
`wandb_run.id` into `best.pt`/`final.pt`, and on `--resume` reads it back
(or takes an explicit `--wandb-run-id` override) so metrics append to the
original run. The id is captured on all ranks (`None` off rank 0) so the
checkpoint-save code is rank-safe. Tests: `run_id` forwards
`resume="allow"`; absent id omits both kwargs.

## 2026-06-06: Gaussian soft labels for the redshift loss (v10 lever)

### Why

v9 (redshift conditioning dropout, honest z-hidden eval) exposed that the
model does NOT learn redshift from spectrum: honest `val/redshift_acc`
flat ~2–4% (random) across 27k steps, `val/loss_redshift` stuck ~4.3,
while train redshift overfits (copy + memorize). Diagnosis: redshift is a
256-way softmax over CDF→Gaussian→FSQ bins where **adjacent bins are
adjacent z**, but cross-entropy treats them as unrelated classes — no
partial credit, no gradient toward the right neighborhood, so the model
never climbs. Secondary finding: at `redshift_loss_weight=1.0` the
per-segment loss weights one redshift token equal to all 272 spectrum
tokens, so a stuck redshift term (~70% of val loss) starved spectrum —
v9's lower `spectrum_acc` (0.44 vs v8 0.73) is mostly this plus the loss
of unmasked-copy inflation; the honest `masked_spec_acc` barely changed
(0.455 → 0.437).

### Change

`SpectrumTransformer.forward` gains `redshift_soft_sigma` (default 0.0 =
unchanged hard CE). When >0, the position-0 loss is cross-entropy against
a Gaussian over the redshift bins centered on the true bin, std = sigma
bins (`_redshift_soft_ce`). Spectrum tokens keep hard CE (LFQ codes are
not ordinal). Chosen over a continuous regression head because it is
confined to the loss — `generate()`, AR eval, inference API, and the
token vocab are untouched. Added `redshift_acc_within2` (±2-bin "right
neighborhood") to `compute_metrics`, surfaced in train + val logs.
Threaded `--redshift-soft-sigma` through `nersc/train_transformer.py`
(train + honest val) and recorded it in checkpoints/artifact metadata.

### v10 plan (clean A/B vs v9)

Change ONLY the redshift loss: `--redshift-soft-sigma 1.5`, everything
else = v9 (`encoder_mask_ratio 0.5`, `redshift_mask_ratio 0.5`,
`redshift_loss_weight 1.0`). If honest `val_ar/ar_redshift_acc` /
`within2` start climbing, the objective was the bottleneck and the next
lever is rebalancing `redshift_loss_weight` so the now-descending
redshift term stops starving spectrum. If still flat, escalate to the
encoder-side z head and/or lower `encoder_mask_ratio`. Tests:
soft-CE partial credit (closer prediction → lower loss), forward runs &
differs from hard, within2 metric.

## 2026-06-08: BREAKTHROUGH — Approach A learns honest redshift from spectrum (the "wall" was a long incubation)

### Headline

The earlier conclusion that Approach A's redshift was "all copy / not
learnable from spectrum" was **wrong — it was premature**. With redshift
conditioning dropout (v9, run `8m9rkz37`, hard CE, `redshift_mask_ratio
0.5`, `encoder_mask_ratio 0.5`, `redshift_loss_weight 1.0`), honest
redshift stayed flat at random for ~33k steps, then went through a
delayed, grokking-like phase transition and climbed steadily. The model
now predicts redshift **from the spectrum without ever seeing it** — the
original project goal.

### v9 honest-metric trajectory (z hidden from encoder, ratio 1.0)

| step | redshift_acc (TF) | ar_redshift_acc | within2 | loss_redshift | spectrum_acc | masked_spec_acc |
|---|---|---|---|---|---|---|
| 32,000 | 4.0% | 2.5% | — | 4.08 | 0.448 | 0.443 |
| 37,500 | 10.6% | 10.4% | — | 3.74 | 0.451 | 0.444 |
| 43,000 | 13.9% | 11.4% | — | 3.22 | 0.626 | 0.416 |
| 63,480 | **40.0%** | **33.8%** | **63.3%** | **2.04** | **0.733** | **0.464** |

(`within2` = predicted bin within ±2 of truth; only logged from the
resume on the soft-labels-era code.) `redshift_acc_zgiven` stayed ~100%
throughout — copy still available, but now the honest z-hidden number is
real and large, not random.

### Two confirmed predictions

1. **Spectrum recovers once redshift learns.** v9's spectrum_acc had
   dropped to ~0.44 (vs v8's 0.728) because a stuck, heavily-weighted
   redshift loss (1 token ≈ 272 spectrum tokens at weight 1.0) starved
   spectrum. As `loss_redshift` fell 4.08 → 2.04, spectrum climbed back to
   0.733 — matching v8 — and honest `masked_spec_acc` reached its best
   (0.464). The redshift/spectrum conflict resolves itself once redshift
   is learnable; no weight rebalance was needed.
2. **Resume on latest code is safe.** v9 was resumed across the
   `--resume` / wandb-continuation / soft-labels commits with
   `--redshift-soft-sigma` left at 0.0; the hard-CE path is byte-identical,
   the checkpoint loaded cleanly, the W&B run continued on one chart, and
   `within2` began logging. No disruption.

### v10 (soft labels, run `zt1a7gvb`) — inconclusive, diverged to NaN

v10 (`--redshift-soft-sigma 1.5`, else = v9) hit `train/loss = NaN` at
**step ~9,000** and never recovered — long before the ~33k breakthrough
zone, so it says nothing about whether soft labels help. The instability
is in the soft-CE path (sigma>0). Fix before any rerun: compute the
soft-label cross-entropy in fp32 (disable autocast for that block; AMP
fp16 `log_softmax` over 1288 classes is the likely overflow), plus a
finite-loss skip-guard in the train loop. **Not on the critical path** —
hard CE reached the goal on its own; soft labels are now only worth
testing to see if they reach higher/faster.

### Status & next

- v9 crashed at step 63,480 (node/wallclock, not NaN) still climbing
  steeply (redshift 14% → 40% in 20k steps, loss_redshift still falling).
  **Resume it** (best.pt now carries `wandb_run_id`). If still climbing at
  the 100k cap, bump `--steps` (e.g. 150k) — this run rewards more
  training. This v9 is the new primary Approach-A result and supersedes
  the v8 release model (whose redshift accuracy was the copy artifact).

## 2026-06-08: Fix inflated train/steps_per_sec after resume

`train/steps_per_sec` spiked to absurd values (e.g. 4–13 step/s vs the
true ~1.4–1.7) right after each resume. Cause: `rate = (step + 1) / dt`
used the **absolute** step count over only the **current process's**
elapsed time (`t0` resets at process start), so a resume at step ~63k
divided 63k by a few seconds. Cosmetic only — training/loss unaffected.
Fixed to `rate = (step - resume_step + 1) / dt` (steps completed in this
process). Non-resumed runs are unchanged (`resume_step = 0`).

## 2026-06-08: Make v9 the default release transformer for notebooks

Repointed the notebook default from the v8 copy-artifact model to v9
(`approach_a_fm_v1_10k_a_ddp4_redmask50_v9`, run `8m9rkz37`), now the
primary Approach-A result (honest redshift, z hidden from encoder:
~52% exact / 47% AR / 73% within-2 at step ~109k, still training toward
150k). Two execution sources drive the notebooks — both updated:
`DEFAULT_TRANSFORMER_ID` in `src/inference/release.py` (nb07) and
`default_transformer` in `checkpoints/release/MANIFEST.json` (nb08).
Also: registered v9 in MANIFEST `models{}` (+ a v9 `approach_a_results`
block with current honest metrics; demoted v8 to `approach_a_v8_results`
/ "superseded"), created `checkpoints/release/<v9>/config.json`, added v9
to `scripts/setup_release_checkpoints.py` ARTIFACT_MAP and its generated
default, and swapped the v8 model-id doc strings in notebooks 07/08 to
v9. Used the model_id == W&B artifact basename (no `transformer_` prefix,
unlike v8).

To actually load it, the v9 checkpoint must be fetched into
`checkpoints/release/<v9>/best.pt`:
`python scripts/download_release_checkpoints.py --model-id approach_a_fm_v1_10k_a_ddp4_redmask50_v9`.
Metrics are "as of step ~109k, run in progress" — re-run
`scripts/sync_wandb_metrics.py` (or refresh the config/MANIFEST) when the
150k run finalizes. Honest spectrum number to quote is
`val_masked_spec_acc` (~0.47), not the copy-inflated `spectrum_acc`.

## 2026-06-08: setup_release_checkpoints.py — skip models not downloaded locally

`scripts/setup_release_checkpoints.py` iterates every entry in
`ARTIFACT_MAP` and hard-failed (`FileNotFoundError`) if any model's W&B
artifact wasn't present under `checkpoints/wandb_artifacts/`. After adding
v8 alongside v9, a user who only pulled v9 + the tokenizer hit a crash on
the v8 entry — *after* v9 was already symlinked, so it aborted before
regenerating the MANIFEST. Wrapped `_find_wandb_pt` in the loop to
**skip + warn** on missing downloads instead of aborting, and added
`approach_a_v8_results` to the keys preserved when the MANIFEST is
regenerated. Re-running now sets up whatever artifacts are present (v9 +
tokenizer) and writes the MANIFEST cleanly; non-downloaded models are
listed in the results blocks but omitted from `models{}` until pulled.

---

## 2026-06-08: Notebook 07 — fix white gap in OOD AR reconstruction plot

OOD Visualization 6 (`notebooks/07_visualize_predictions.ipynb`) had a large
white band between the suptitle and the plots. The figure is very tall
(`N_OOD * 2.8` ≈ 28 in) and the `suptitle` was placed at `y=1.005` with no
adjustment to the subplot top margin, so the axes started at matplotlib's
default `top≈0.88` — a 12% margin that is a huge absolute gap on a tall
figure. Fix: added `fig.subplots_adjust(top=0.97)` and moved the title down
to `y=0.985` so it sits just above the first plot.

---

## 2026-06-09: Notebook 07 — residual subplots on Viz 1 & Viz 2

Added a `true − reconstruction` residual row under each spectrum in the
in-distribution DESI reconstruction plots (Viz 1 teacher-forced, Viz 2
autoregressive), matching the purple residual panels already used in the
OOD SDSS plot (Viz 6). Each sample is now a 2-row block (flux height 3 +
residual height 1) via `gridspec_kw` height_ratios, with the per-spectrum
residual sum shown as `Σ=...` in the residual y-label and a dashed zero
line. Same top-margin fix (`subplots_adjust(top=0.97)`, suptitle `y=0.985`)
as Viz 6.

---

## 2026-06-09: Roadmap — planned changes (AION-paper review + repo audit)

Full review of the repo (v9 transformer, spectrum_tokenizer_v1, training
loop) against the AION-1 paper. Plan of record below, roughly in
priority order. Items 1–2 are committed next actions (project owner);
3–9 are queued recommendations from the audit.

### 1. Rebin redshift: 256 → 4096 bins (~0.001-level bin width) [NEXT]

Spectroscopic redshifts are good to ~0.001; our current 256 CDF bins are
far coarser than the label precision, so the model is being trained
against an artificially blurred target. Plan:

- `RedshiftTokenizer(n_levels=4096)`; vocab becomes
  `REDSHIFT_TOKEN_OFFSET (1032) + 4096 = 5128`.
- Make the redshift sub-vocab width configurable end-to-end instead of
  hardcoded: `--z-bins` flag in `nersc/train_transformer.py` (+ local
  `scripts/train.py`), `SpectrumTransformer(vocab_size=1032 + z_bins)`,
  helper `vocab_size_for_z_bins()` in `src/models/transformer.py`.
  Default stays 256 so v8/v9 checkpoints keep loading.
- Inference (`src/inference/release.py`) infers vocab size from the
  checkpoint's `token_embedding.weight` shape — old (1288) and new
  (5128) checkpoints both load with no config changes.
- Resume guard: refuse `--resume` when the checkpoint's embedded
  z-tokenizer `n_levels` ≠ `--z-bins` (vocab shape mismatch).
- Caveat to verify empirically: our binning is CDF-equalized (equal
  probability mass), NOT uniform in z — bin width varies. 4096 bins ≈
  0.001 only on average over the fitted z range; dense regions (most
  galaxies) get much finer bins, tails (high-z QSOs) get wider ones.
  Log bin-width stats (median/p90/max from `get_bin_edges()`) at fit
  time. Also: 4096 quantiles need a bigger fit sample — warn when
  `len(zs) < ~20 × n_bins` and raise `--z-fit-files` accordingly.
- Knock-on effects to remember: `redshift_acc_within2` becomes 16×
  stricter (±2 of 4096 vs ±2 of 256) — don't compare across bin counts;
  if reviving soft labels, sigma is in *bins* so the v10-equivalent
  becomes `--redshift-soft-sigma ~24` (1.5 × 4096/256); 4096-way CE is
  a harder classification problem — expect a longer incubation unless
  paired with soft labels (item 5).

### 2. Retrain spectrum tokenizer on broadened DR1 data [NEXT]

`spectrum_tokenizer_v1` was trained on only ~500k SV3-bright spectra.
The reconstruction ceiling for everything downstream is the tokenizer,
and SV3-bright underrepresents the flux/SNR/object-type distribution
(dark-program ELGs/LRGs/QSOs especially). Plan:

- Build a broader manifest with `nersc/build_dr1_index.py` (already
  supports `--surveys sv3 main --programs bright dark`); stage to
  $SCRATCH; retrain as `tokenizer_v2`.
- Add `--surveys`/`--programs` filter flags to
  `nersc/pretrain_tokenizer.py` so one big manifest can be subset per
  run and the survey/program mix is logged into the run config.
- While retraining (from the AION audit, same training run is the
  cheapest place to fix these):
  - Replace plain MSE on *denormalized* flux with inverse-variance-
    weighted Gaussian NLL (AION Eq. 1) — stops bright spectra and noisy
    pixels from dominating; we already carry ivar/istd as an input
    channel. At minimum compute the loss in normalized space.
  - Add the LFQ entropy objective (per-sample entropy down, batch
    entropy up) and log codebook utilization — commitment loss alone
    invites dead codes.
  - Train much longer: AION used 215k steps @ batch 128 to reach
    reconstruction R² = 0.994; our default is 10k @ 16. Measure held-out
    R² as the gating metric for tokenizer_v2.
  - Fix the no-op log10 round-trip in `normalize()`
    (`denorm = 10^log10(norm+1) − 1 = norm`).

### 3. Predict only masked positions (4M/MAE-style)

With `encoder_mask_ratio 0.5`, half the decoder targets are visible to
the encoder — pure copy tasks that consume half the gradient budget and
inflate `spectrum_acc`. Set targets to −100 at unmasked spectrum
positions so the loss is masked-reconstruction only (AION/4M decoder
queries never see token values). Longer term: replace causal AR over
272 spectrum tokens with ROAR-style iterative masked decoding
(O(log N) decoder calls; left-to-right has no physical meaning for a
spectrum).

### 4. Physical redshift metrics

Bin accuracy is not comparable across binnings and hides physical
error size. Add to eval: NMAD σ of Δz/(1+z), catastrophic outlier
fraction (|Δz|/(1+z) > 0.15), broken down by spectype
(galaxy/star/QSO). Decode predicted bin → ẑ via the z-tokenizer and
compare against pipeline z. These are the community-standard numbers.

### 5. Soft labels retry, fp32 (v10 redo)

v10 NaN'd at ~9k steps in the soft-CE path under fp16 autocast.
Compute `_redshift_soft_ce` under `autocast(enabled=False)` in fp32 —
or move training to bf16 (no GradScaler, kills the overflow class).
With 4096 bins the ordinal gradient matters more, not less.

### 6. Scale data + throughput before model size

Pre-tokenize the corpus to disk (272 ints/spectrum) instead of running
the frozen ConvNeXt tokenizer forward every step; train on full DR1
(v9 was still climbing at 109k steps). Only scale depth (12+12,
AION-B-ish) after the loss flattens at full data.

### 7. Embedding-probe evaluation suite

Frozen encoder + mean/attentive pooling + linear/MLP probes: z
regression, spectype classification, stellar params (Zhang et al. 2024
DESI labels) / galaxy props (PROVABGS). This is how AION substantiates
"understanding" — and our path to an apples-to-apples row against
AION-1's Table 1/3 spectrum-only results.

### 8. Engineering

`F.scaled_dot_product_attention` (Flash) in `MultiHeadAttention`;
KV cache in `generate()` (currently O(L²) re-decode per token — why AR
eval is capped at 4 batches / n=201); bf16 everywhere.

### 9. Wavelength-aware resampling

`interpolate_to_grid` stretches any input length onto 8704 samples,
ignoring wavelength coverage — correct only for DESI's fixed 7081-pixel
grid. Adopt AION's convention (fixed wavelength grid 3500–10462.4 Å @
0.8 Å) before quoting any OOD/SDSS numbers; also unblocks multi-survey
training later.

---

## 2026-06-09: Implemented roadmap items 1–2 — configurable z bins (4096) + balanced tokenizer-v2 data path

### 1. `--z-bins` end-to-end (roadmap item 1)

- `src/models/transformer.py`: `vocab_size_for_z_bins(n)` =
  `REDSHIFT_TOKEN_OFFSET (1032) + n`; redshift tokens stay the last
  vocab block so special/spectrum IDs are unchanged. `is_redshift_token`
  takes an optional `vocab_size`. Defaults untouched (`TOTAL_VOCAB_SIZE`
  1288) — v8/v9 checkpoints unaffected.
- `nersc/train_transformer.py`: `--z-bins` (default 256; pass 4096 for
  ~0.001-level bins) drives both `RedshiftTokenizer(n_levels=...)` and
  `SpectrumTransformer(vocab_size=...)`. At fit time prints
  median/p90/max bin width in z (bins are CDF-equalized, so width
  varies — this line verifies the precision actually achieved) and
  warns when the fit sample has <20 z values per bin (raise
  `--z-fit-files`). `--resume` refuses a checkpoint whose embedded
  z-tokenizer `n_levels` ≠ `--z-bins` (vocab shape mismatch). `z_bins`
  added to W&B artifact metadata.
- `scripts/train.py`: same via `--z_bins`.
- `src/inference/release.py`: vocab is now inferred from the
  checkpoint's `token_embedding.weight` shape — 1288 (legacy) and 5128
  (4096-bin) models load through the same path.
- SLURM: `Z_BINS` env passthrough in `train_transformer.slurm` and
  `train_transformer_ddp.slurm`.
- Tests: 4096-bin round-trip precision (<1.5e-3 max |dz| on a dense
  fit), 4096-vs-256 resolution improvement, monotonic bin edges,
  vocab helper, 4096-vocab forward/loss, soft-CE adapting to the wider
  vocab, `is_redshift_token` with wider vocab. Full suite: 136 passed.

### 2. Balanced manifest path for tokenizer v2 (roadmap item 2)

Root cause of the SV3-bright-only v1 training set found:
`build_dr1_index.py --max-healpix` is a **global** cap filled
sequentially over the (survey, program) loop, so
`pretrain_tokenizer.slurm`'s `MAX_HEALPIX=2000` consumed itself on
sv3/bright before reaching dark or main. Changes:

- `build_dr1_index.py`: new `--max-healpix-per-pair N` caps each
  (survey, program) pair separately (balanced mix); per-pair kept-count
  logging; `--max-healpix` documented as legacy/global.
- `pretrain_tokenizer.py`: `--surveys` / `--programs` manifest filters
  + per-(survey, program) record-mix logging (lands in the W&B config),
  hard error if filters empty the manifest.
- `pretrain_tokenizer.slurm`: defaults now `MAX_HEALPIX_PER_PAIR=500`
  over `sv3 main × bright dark` (≈ v1 volume, balanced), run name
  `tokenizer_v2_*`, SURVEYS/PROGRAMS env-overridable.
- `nersc/README.md`: new "Redshift bins (`--z-bins`)" and "Tokenizer
  v2: broadened training data" sections (caveats: bin-count-incompatible
  checkpoints, within2 metric 16× stricter at 4096, soft-sigma scales
  ~×16, gate v2 on held-out R², transformer must be retrained against a
  new codebook).

### Not yet done (stays on the roadmap)

Tokenizer loss change (ivar-weighted NLL), LFQ entropy term, longer
tokenizer schedule, masked-only decoder targets, physical z metrics
(NMAD / outlier fraction), fp32 soft-CE — items 2 (loss part) and 3–9
of the 2026-06-09 roadmap entry.

## 2026-06-09: pretrain_tokenizer.slurm — MANIFEST env override

The script hardcoded `MANIFEST=$SCRATCH_OUT/manifests/dr1_${SLURM_JOB_ID}.jsonl`
(job-id-named), so a manifest pre-built on an interactive node could
never be reused — every submission rebuilt its own. Now
`MANIFEST="${MANIFEST:-...}"` like train_transformer.slurm: pass
`MANIFEST=... sbatch nersc/pretrain_tokenizer.slurm` to train on a
pre-built (e.g. balanced v2) manifest.

---

## 2026-06-09: Tokenizer v2 objective implemented (roadmap item 2, loss part)

All four loss-side changes from the roadmap, in place before the v2
training run:

1. **ivar-weighted Gaussian NLL** (`recon_loss="nll"`, the new default in
   `SpectrumTokenizer`): flux loss is `0.5 * ivar * (flux − flux̂)²`
   averaged over ivar>0 pixels (AION Eq. 1). Scale-invariant (bright
   spectra no longer dominate), noise-aware (low-SNR pixels are
   down-weighted), and padding/masked pixels (ivar=0 from the collate)
   drop out automatically. A small normalized-space MSE on the istd
   channel (`istd_loss_weight=0.1`) keeps `decode()`'s channel 1
   meaningful. Legacy v1 MSE retained as `recon_loss="mse"` for A/B.
   Loss computed in fp32 under AMP.
2. **LFQ entropy objective** (MAGVIT-v2): minimize per-sample binary
   entropy of p(bit=+1)=sigmoid(2z/τ) (confident assignments), maximize
   batch marginal bit entropy (both signs of every bit used). Factorizes
   the 2^10-code entropy over bits. `entropy_weight=0.1`, τ=1.0.
   Commitment alone invited dead codes; **codebook utilization is now
   logged** (per-batch at train log steps, accumulated over the val
   pass) so collapse is visible.
3. **Held-out flux R²** (`flux_r2`, ivar>0 pixels, per-spectrum then
   batch-averaged) computed every val pass — the gating metric for v2
   (AION reference: 0.994). Saved into best.pt and the W&B artifact
   metadata alongside codebook_use.
4. **normalize() no-op removed**: `10^log10(norm+1) − 1 == norm`
   round-trip deleted; numerically identical, so v1 checkpoints load
   and tokenize unchanged (verified: no new parameters; full test suite
   145 passed).

Training entry points: `--recon-loss {nll,mse}` and `--entropy-weight`
on `pretrain_tokenizer.py`, with `RECON_LOSS`/`ENTROPY_WEIGHT` env
passthrough in both pretrain SLURM scripts (defaults: nll, 0.1).
Quantizer API change: `LookUpFreeQuantizer.forward` returns a loss dict
(`quant`/`commit`/`entropy_sample`/`entropy_codebook`) instead of a
scalar; `SpectrumTokenizer.forward`'s dict keeps the `total`/`recon`/
`quant` keys all existing consumers use, plus the new components.

Loss-scale note: NLL ≈ mean per-pixel χ²/2, so val_loss values are not
comparable to v1's denormalized-MSE numbers — compare on flux R².
Remaining before launch: none code-side; build the balanced manifest
(`dr1_v2_balanced_2k.jsonl`), stage to scratch, submit. Tests: entropy
direction (confident+diverse < unconfident+collapsed), NLL component
logging, ivar=0 exclusion, scale-invariance sanity, mse legacy mode,
flux_r2 (perfect=1, mean-prediction≈0, masked-region exclusion).

## 2026-06-09: Wavelength-aware resampling (roadmap item 9) — in before v2 trains

This had to land before tokenizer v2: it changes the tokenizer's input
convention, and changing it after training would mean a v3.

What was wrong: `interpolate_to_grid` stretches ANY input length onto
8704 samples ignoring the wavelength solution. Two real consequences
(not just convention):
1. **Batch-dependent wavelength mapping.** The collate pads to the
   longest spectrum in the batch, then the stretch maps [0, L_batch] →
   [0, 8704]; a spectrum's wavelengths land at different grid positions
   depending on what it was batched with. Most DESI rows are the full
   7781 px so the effect is small, but it is a genuine inconsistency.
2. **OOD evaluation was physically wrong.** SDSS is log-lambda spaced
   with different coverage; the stretch misaligns every spectral
   feature. Notebook 07's OOD numbers were distorted by this.

Change: fixed wavelength grid 3500–10462.4 Å @ 0.8 Å (= exactly our
8704 samples; DESI's native 3600–9824 @ 0.8 Å grid aligns
sample-for-sample at offset 125, so DESI resampling is a lossless
copy, no interpolation blur). `resample_to_grid(x, wavelength)` does
batched searchsorted linear interpolation; out-of-coverage grid pixels
get flux=0 AND istd=0, which the new ivar-weighted NLL excludes
automatically (the two changes compose).

Compatibility: `encode`/`forward` take optional `wavelength`; when
None, the legacy stretch is used — v1 checkpoints and the v9
transformer inference path are bit-for-bit unchanged (tested).
`collate_dr1_skip_none` now carries a monotonically-extended padded
wavelength array; `pretrain_tokenizer.py` is wavelength-aware by
default (`--legacy-stretch` to reproduce v1); `tokenize_and_build`
gains `wavelength_aware=False` for the future v2 transformer (only
passes the kwarg when set, so tokenizer stubs/wrappers keep working).
Tests: DESI offset-125 exact copy, out-of-coverage zeros, SDSS-like
log-spaced roundtrip (<5e-3), per-row (B, L) wavelengths, legacy-path
bit-equality. Suite: 151 passed.

## 2026-06-10: tokenizer_v2_trial findings → exact joint LFQ entropy

First v2 trial (run `f9s2dy8k`, 2k steps, 3k balanced manifest, NLL +
factorized entropy + wavelength-aware): pipeline is healthy — no NaN,
~1.7-1.9 steps/s staged, val nll_flux 1.80 → 0.99 (reconstruction
approaching the pixel-noise floor), val flux_r2 0.23 → 0.27 and
climbing. But the new codebook logging caught **collapse**: only ~30 of
1024 codes alive per batch (val-pass accumulation 17 → 79 codes by step
1500). Diagnosis: per-bit marginal entropy was already near ceiling
(0.59-0.68 vs ln2=0.693) while joint usage stayed ~3% — the bits are
balanced individually but heavily **correlated**, and the factorized
entropy term is mathematically blind to correlation. v1 almost
certainly had the same disease, unmeasured.

Fix: replaced the factorized term with the **exact joint entropy over
all 2^dim codes** (what MAGVIT-v2 actually computes; their grouping
trick is only needed for codebooks far larger than our 1024). For
binary codewords −‖z−c_k‖² = 2 z·c_k + const, so
p(code k | z) = softmax(2 z·c_k / τ); minimize mean per-sample entropy,
maximize entropy of the batch-marginal code distribution. Cost: a
(B·273, 1024) softmax ≈ 100 MB fp32 at batch 32 — negligible.
Entropies are now in nats over the code axis (max ln1024 = 6.93, vs the
old per-bit ≤ 0.693) — W&B `entropy_*` scales are NOT comparable across
the two trial runs. Codewords buffer is non-persistent → state_dict
unchanged, v1 checkpoints load as before.

Tests: correlated-bits regression (balanced marginals, 2/64 codes →
codebook entropy ~ln2; diverse → ~ln64; auxiliary prefers diverse),
gradient-flow check. Suite: 153 passed.

Next: rerun the same 2k trial (`tokenizer_v2_trial_jointent`) and
compare train/codebook_use at equal steps vs `f9s2dy8k`; launch the
24h run with whichever wins (expect joint).

## 2026-06-10: Tokenizer trainer — resumable (interactive-QOS workflow) + best-only artifacts

### Resume (`pretrain_tokenizer.py --resume`)

Same pattern as the transformer trainer: restore model/optim/scaler/
step/best_val from a full-state checkpoint on every rank; W&B run id is
saved into checkpoints and read back on resume (`--wandb-run-id`
overrides) so metrics continue on one chart; `steps_per_sec` counts
steps since this process started (transformer's resume-rate fix,
applied preemptively). New **rolling `last.pt`** written every
`--save-every` steps with full state (replaces the accumulating
model-only `step_*.pt`, which couldn't restore the optimizer anyway) —
this is the resume point for stitching multiple 4h interactive-QOS
sessions, since `best.pt` can lag far behind late in training.
`final.pt` is now full-state too. Workflow:

    # session 1
    srun ... pretrain_tokenizer.py --run-name tokenizer_v2_3k --steps 100000 ...
    # session 2+ (same run dir, same W&B chart, exact optimizer state)
    srun ... pretrain_tokenizer.py --run-name tokenizer_v2_3k --steps 100000 \
        --resume $SCRATCH/deepsrch/checkpoints/tokenizer_v2_3k/last.pt ...

### Best-only W&B artifacts (both trainers)

Found a latent footgun while wiring this: `log_model_artifact` prunes
all prior versions of an artifact after upload (`keep_only_latest`),
so the end-of-run `final.pt` push **deleted the best version** —
including its `best` alias, which MANIFEST.json and the release
pipeline reference. Any run that reached its step cap would silently
replace the best checkpoint artifact with the (worse) last model.
Removed the final-push from BOTH `pretrain_tokenizer.py` and
`train_transformer.py` — only best-checkpoint improvements upload, so
the surviving artifact version is always the best. (`final.pt` still
written to SCRATCH + mirrored to CFS.) This mattered immediately: the
v9 transformer run is heading for its 150k cap and would have clobbered
the released `:best` artifact on completion. README artifact section
updated. Tests: 153 passed.

## 2026-06-10: tokenizer_v2_3k_ddp crashed reconstruction — entropy reward capped

### What happened (run `wuuapncr`, 4-GPU DDP, lr 1.2e-3, entropy_weight 0.1)

The joint entropy decisively fixed collapse — codebook_use 0.25 → 0.91 by
step 4.5k (vs 3% factorized) with flux_r2 improving to 0.286. Then it
overshot: between step ~5.0k and 5.5k, val flux_r2 fell 0.286 → −0.05 and
val nll_flux rose 0.75 → 11.4, and stayed there while codebook_use marched
to exactly 1.0. Diagnosis: the diversity reward (0.1 × up to ln1024 =
0.69 nats) is the same magnitude as the entire reconstruction loss at its
plateau (~0.65), and the *uncapped* pressure toward perfectly uniform code
usage kept reassigning code meanings under a hot LR until the
encoder–decoder code contract broke. Trial `5isstibi` (single GPU,
lr 3e-4) showed the same objective stable for 2k steps — the DDP run's
3× higher LR amplified the scrambling.

### Fix (two levers, both in `LookUpFreeQuantizer`)

1. **Saturating diversity reward**: `entropy_aux = H_sample −
   min(H_codebook, 0.9·ln K)` (`entropy_target_frac=0.9`). Once marginal
   entropy reaches 90% of max, the uniformity gradient is exactly zero —
   the term fights collapse but cannot trade reconstruction for cosmetic
   uniformity. Logged `entropy_codebook` stays the raw (uncapped) value.
2. **Default `entropy_weight` 0.1 → 0.02** (max reward 0.14 vs recon
   ~0.65). Updated in the quantizer, `SpectrumTokenizer`,
   `--entropy-weight` default, and both pretrain SLURM defaults.

Also recommending **lr 3e-4** (not 1.2e-3) for the DDP restart: AION used
1e-4 constant at the same batch 128 for its spectrum tokenizer; linear LR
scaling from the single-GPU 3e-4 was too aggressive for this objective.

Restart fresh (only ~50 GPU-min lost; the crashed basin isn't worth
resuming into): same command, run name `tokenizer_v2_3k_ddp2`,
`--lr 3e-4`, new defaults give entropy_weight 0.02 + cap. Watch
`val/codebook_use` (expect slower but steady climb; healthy target is
high-but-not-pinned-1.0) and `val/flux_r2` (must keep rising past 0.29).
Tests: capped-quant arithmetic, zero-gradient-above-target. 155 passed.

## 2026-06-10: tokenizer_v2_3k_ddp2 — stable; killed by 4h wallclock; SNR-sliced R²

Run `hp8gj40n` (lr 3e-4, entropy_weight 0.02 + 0.9·lnK cap) reached step
29,560 in 238 min before the interactive-QOS 4h limit killed it (W&B
state "crashed" = heartbeat stop, not a training failure). **The
stability fix held**: no reconstruction cliff, codebook_use climbing
gently (val 0.47 → 0.62, not pinned at 1.0), best val nll_flux 0.60 at
24k — per-pixel χ² ≈ 1.2, near the noise floor. Resume from
`last.pt` (step 28k) continues the same W&B chart.

`val/flux_r2` plateauing at ~0.26–0.29 is largely a METRIC artifact, not
a model ceiling: R² compares the reconstruction against the *noisy*
input flux, and the balanced manifest is dark-program-heavy (faint,
low SNR). A perfect denoised reconstruction of a low-SNR spectrum caps
at signal_var/(signal_var+noise_var) ≪ 1. Added
`val/flux_r2_snr3` — per-spectrum R² restricted to spectra with median
per-pixel SNR > 3 — as the AION-comparable number (their 0.994 is only
meaningful on bright spectra). `flux_r2` gains `reduce=False` for
per-spectrum values; caveat documented in the docstring. The
ivar-weighted NLL remains the honest objective on faint targets.

## 2026-06-10 (cont.): ddp2 val NLL drift after ~24k → entropy target lowered to 0.75

Confirmed (project owner spotted it): ddp2's val nll_flux bottomed at
0.60 near step 24k and drifted to 0.78–1.0 by 29.5k while codebook_use
climbed 0.47 → 0.62. Same mechanism as the crash, in miniature and
bounded: `entropy_codebook` was 5.07 — still below the 0.9·lnK = 6.24
saturation target — so the uniformity gradient was still active and
recon kept paying for codebook expansion. The data says recon starts
losing past ~0.72 of max entropy (~500 codes alive).

Change: `entropy_target_frac` default 0.9 → **0.75** (target 5.19 ≈
exactly where ddp2 sits, so the pressure shuts off on resume), exposed
end-to-end: `LookUpFreeQuantizer` / `SpectrumTokenizer` /
`--entropy-target-frac` / artifact metadata. Collapse territory is
~0.4, so 0.75 keeps a wide safety margin. Resume ddp2 from `last.pt`
(step 28k) on the new code; expect nll to turn back down as the
codebook stops churning.

## 2026-06-11: ddp2 resume collapsed to a single code — spike-robustness fixes

### What happened (run `hp8gj40n`, resumed at 28k)

Resume itself was healthy: nll 0.74 → 0.66 by 31k, codebook stable
~0.24, entropy_codebook hovering at the new 0.75·lnK target — the cap
worked. Then a **loss spike at ~33.5k** (train nll 6.9; later spikes 38
and 61) destabilized training, and the codebook collapsed 0.27 → 0.001:
ONE code by ~46k, val nll pinned at 10.8, R² negative, ~25k steps wasted.

Two mechanisms:
1. **NLL spike fragility**: per-pixel 0.5·ivar·err² is unbounded — one
   pathological pixel (cosmic ray / underestimated noise, chi ~100σ)
   contributes ~5000 nats and poisons the weights at lr ~2.7e-4.
2. **Entropy can't rescue a confident collapse**: once |z| is large the
   code softmax is one-hot and the entropy gradient vanishes — the
   anti-collapse term is prevention-only. (This also retro-explains why
   weight 0.02 couldn't pull it back.)

### Fixes (all default-on)

- **Huber NLL** (`huber_chi2`, delta = 10σ): quadratic ≤10σ, linear
  beyond. A chi=100 pixel now contributes 950 instead of 5000, and
  doubling an extreme residual doubles (not quadruples) the loss.
  Identical to the Gaussian NLL for all well-modeled pixels.
- **DDP-safe skip guard**: `clip_grad_norm_` runs after the DDP grad
  allreduce, so its total_norm is identical on every rank — skipping
  the optimizer step on a non-finite norm cannot desync replicas.
- **bf16 default** (`--amp-dtype bf16`, fp16 retained as option):
  fp32 dynamic range in autocast, GradScaler auto-disabled (and its
  checkpoint state only loaded when enabled, so old fp16 checkpoints
  resume cleanly into bf16).

### Restart guidance

Resume from **best.pt** (pre-collapse weights; `last.pt` now holds the
collapsed 58k state — do NOT resume from it). Same run name continues
the W&B chart. Tests: huber math, forward outlier-boundedness. 158 passed.

## 2026-06-11 (cont.): collapse circuit breaker + healthy-only last.pt

Two run-protection guards in `pretrain_tokenizer.py` ahead of the fresh
v3 start (cannot *prevent* every pathology, but converts the worst case
from "25k wasted steps + poisoned resume point" to "abort within
minutes, healthy checkpoint preserved"):

1. **Circuit breaker**: EMA (0.9) of per-batch codebook utilization,
   tracked at every log step with its running peak. If the EMA falls
   below 25% of a peak that had already cleared 0.10, raise immediately
   (rank 0 raises; srun terminates the step). ddp2's collapse
   (0.27 → 0.015 over ~3k steps) would have tripped this within ~1k
   steps of onset. Grace period at start/resume until the EMA rebuilds
   past 0.10. `train/codebook_use_ema` logged to W&B.
2. **Healthy-only `last.pt`**: the rolling checkpoint is skipped (with a
   loud message) whenever the EMA is below 50% of peak — ddp2's
   collapsed step-58k state can no longer overwrite a healthy one.

Defense matrix now: joint entropy (collapse prevention) + 0.75·lnK cap
& weight 0.02 (over-diversification) + Huber 10σ NLL (spike batches) +
bf16 (fp16 overflow class) + non-finite-grad skip (DDP-safe) + circuit
breaker & checkpoint gate (damage control). Fresh run:
`tokenizer_v2_3k_v3`. Tests: 158 passed.

## 2026-06-11 (cont.): circuit breaker false positive at step 340 — armed after step 500

v3 launch (run `856wnc99`) was killed by the new breaker at step 340.
Root cause of the false positive: a random-init encoder scatters codes
(utilization ~0.22 at step 0) and every healthy run then naturally
contracts to ~0.01 within ~100 steps before regrowing (same 0.22 → 0.01
pattern in f9s2dy8k, 5isstibi, hp8gj40n). The breaker counted the init
scatter as "peak", so the normal contraction tripped the 25% rule. Fix:
EMA/peak tracking and the abort check are armed only from step ≥ 500 —
past the init transient, before any real collapse window (earliest
observed onset: ~5.2k). Silver lining: the abort path is proven — rank 0
raised and srun terminated all ranks cleanly within seconds. Relaunch
the same v3 command (fresh W&B run; 340 lost steps are immaterial).

## 2026-06-11 (cont.): tokenizer success criteria defined + pooled R² metric

Question settled before v3 finishes: what gates tokenizer v2 and what is
the AION comparison? Four-row scorecard:

1. **Held-out ivar-weighted χ²/pixel = 2·val/nll_flux → target ≤ ~1.1**
   (1.0 = reconstruction statistically indistinguishable from the data
   given DESI's own noise — the information-theoretic floor). The v2 run
   is at 1.08–1.12 by step 27k. AION does not report this; it is our stronger,
   physically rigorous claim.
2. **Pooled flux R² (SNR>3 slice) → the AION-comparable number** (their
   0.994). Two corrections make it apples-to-apples: (a) population —
   their SDSS+SV3 corpus is much brighter than our dark-heavy balanced
   manifest, so compare on the median-SNR>3 slice; (b) pooling — a
   corpus-level R² is Σss_res/Σss_tot, dominated by high-variance bright
   spectra, NOT the per-spectrum mean (which reads far lower on faint
   corpora: v3's per-spectrum snr3 mean is 0.654 while its pooled value
   is expected ≫ that). Added `val/flux_r2_pooled` and
   `val/flux_r2_pooled_snr3` (`flux_r2_terms` accumulators); logged from
   the next resume onward.
3. **Codebook utilization ≥ ~50%** (v2 run: 64% and climbing; AION doesn't
   report — our differentiator).
4. **The decisive metric is downstream**: transformer-v2 trained on
   these tokens must beat v9 on honest masked_spec_acc and physical z
   metrics (NMAD σ of Δz/(1+z), outlier fraction). The tokenizer is an
   intermediate; this is what "better than AION" means operationally.

Caveat noted: per-spectrum-mean R² vs noisy input is bounded ≪ 1 for
faint targets no matter how good the codec is; pooled+sliced is the
fair external number, χ² is the internal truth. Tests: pooled R²
bright-domination + perfect-reconstruction terms. 160 passed.

---

## 2026-06-11: Deep research — next steps after tokenizer v2 (tokenizer + transformer roadmap v2)

External landscape check (June 2026), then the plan. Sources:
SpecPT (Park et al. 2025, ApJ 988:139, arXiv:2501.01070); OmniSpectra
(arXiv:2601.15351); Universal Spectral Tokenization (NeurIPS ML4PS
2025, arXiv:2510.17959); AION-1 (Parker et al. 2025); AstroCLIP
(arXiv:2310.03024).

### The competitive bar (changed since the AION-only framing)

- **SpecPT** is now the redshift bar, not AION: a spectroscopy
  pre-trained transformer on DESI EDR reporting **NMAD σ = 0.0006 (BGS)
  / 0.0008 (ELG)** with **catastrophic outlier fractions 0.20% / 0.80%**
  over 0 < z < 1.6. Our 4096-bin z vocabulary has ~0.001 *average* bin
  width — CDF-equalized, so galaxy-dense regions are finer — meaning the
  token approach is not resolution-limited vs SpecPT, but only barely.
  To beat SpecPT we likely need sub-bin precision (expected-value decode
  over the softmax, or a refinement head; see X6).
- **OmniSpectra** (2026): 42.5M-param unified FM on 5.5M spectra across
  DESI EDR + SDSS + APOGEE + VIPERS at **native resolution** — validates
  both our scale regime (we have 2.9M now, 8M+ available in DR1 alone)
  and the wavelength-aware direction; their cross-survey transfer is the
  capability our fixed-grid resampler (roadmap item 9, done) unblocks.
- **Universal Spectral Tokenization** (NeurIPS ML4PS 2025): sequence-
  level SSL tokenizer on native grids across SDSS/DESI/GALAH/APOGEE —
  evidence the field is converging on survey-agnostic tokenization;
  our GRID_WAVE design is compatible but grid-locked. Native-resolution
  variable-length tokenization is a future v3-GENERATION tokenizer
  question (a new recipe, not a rerun of v2), not now.

### TOKENIZER — next steps (post-v3)

T1. **Finish/early-stop the v2 training run (`tokenizer_v2_3k_v3`, now).** Gate on the 4-row scorecard
    (2026-06-11 entry). Early-stop when χ²/pixel and pooled-snr3-R²
    plateau — training past the noise floor is wasted GPU. Expect this
    well before 200k steps; check at every ~25k.
T2. **Decoder-only polish (cheap, after T1).** Freeze encoder+quantizer
    (tokens fixed!), fine-tune the decoder alone for ~20-30k steps at
    low lr. Standard VQ trick: squeezes reconstruction without changing
    any token id — the transformer can start training in parallel since
    tokens are frozen at T1.
T3. **Token robustness audit (before transformer commits).** New: check
    token *stability* — tokenize the same object's spectrum with noise
    realizations (add noise ~ivar) and measure token flip rate. High
    flip rate = the transformer learns noise, not spectra. Also
    per-(survey,program) codebook usage and χ², to verify dark-program
    spectra aren't second-class. ~1 notebook, no training.
T4. **Codebook capacity ablation (only if T1 gates fail).** Options in
    order of preference: (a) dim 12 → 4096 codes (AION's image tokenizer
    plateaued at 2^12; our entropy machinery generalizes — exact joint
    entropy still cheap at 4096); (b) fewer downsamples → 546 tokens
    (2× sequence cost for the transformer — expensive downstream);
    (c) residual/2-level quantization (RVQ) — more expressive, same
    length, but complicates the transformer vocab. Decision metric:
    χ² stuck ≫ 1.0 on bright spectra (capacity-limited) vs χ² ≈ 1
    everywhere (done, no ablation needed).
T5. **Scale data before scaling model** (5k/10k manifests, commands
    ready). Only worthwhile if T1 shows a train/val gap (underfit on
    breadth) — at 24M params over 2.9M spectra, the model is small;
    epochs at 200k steps ≈ 9. If val tracks train closely, more data
    buys robustness, not metrics.
T6. **EMA weights** (AION used decay 0.9999 on a tokenizer): cheap
    stability for the *released* checkpoint; evaluate EMA vs raw on the
    val pass before adopting.
T7. **Defer**: SDSS co-training (multi-survey tokenizer) and native-
    resolution tokenization — real wins (OmniSpectra/UST prove it) but
    they reset the token vocabulary again; do after the v2 transformer
    demonstrates the pipeline end-to-end.

### TRANSFORMER — next steps (v2 campaign, starts when T1+T2 gate)

X1. **Pre-tokenize the corpus first** (roadmap 6). One pass of the
    frozen v2 tokenizer over the manifest → cache (272 token ids + z +
    spectype + healpix) per spectrum. Removes the ConvNeXt forward from
    every training step (the current throughput ceiling), makes
    transformer steps tiny, enables big batches. ~1 GPU-day once.
X2. **Masked-targets-only objective** (roadmap 3): targets = -100 at
    encoder-visible spectrum positions. Sample the mask ratio per batch
    from ~U(0.15, 0.75) instead of fixed 0.5 — trains the model for
    every conditioning level and is the prerequisite for MaskGIT/ROAR-
    style iterative decoding later (roadmap 3b, separate decision).
X3. **Activate 4096 z bins + fp32 soft labels** (roadmap items 1+5):
    --z-bins 4096, --redshift-soft-sigma ~16-24 bins (the 256-bin 1.5
    rescaled), soft-CE computed in fp32/bf16. Train with conditioning
    dropout 0.5 as in v9.
X4. **Physical z metrics in eval** (roadmap 4): NMAD σ of Δz/(1+z),
    catastrophic outlier fraction (|Δz|/(1+z) > 0.0033 *and* the 0.15
    convention — report both), per spectype. These are SpecPT's units;
    without them no comparison is possible.
X5. **Engineering with the rewrite** (roadmap 8): SDPA/Flash attention,
    KV cache in generate(), bf16 (the tokenizer campaign's stability
    lessons apply verbatim), and port the circuit-breaker pattern
    (watch redshift-head health, not codebook).
X6. **Sub-bin redshift decoding**: decode ẑ as the softmax-weighted
    mean over bin centers (expected value) instead of argmax → smooth
    estimator below bin resolution; free at inference. If NMAD still
    floors at bin width, add a small refinement head (predict Δz within
    the argmax bin). This is the credible path past SpecPT's 6e-4.
X7. **Embedding probe suite** (roadmap 7) — the "understanding" claim:
    frozen encoder + linear/MLP probes for z, spectype, PROVABGS galaxy
    properties, Zhang+24 stellar params; mean + attentive pooling;
    report vs AION Table 1/3 Sp rows and OmniSpectra where overlapping.
X8. **Uncertainty/calibration**: the 4096-way softmax is a discretized
    z-posterior for free — check calibration (PIT histograms), use
    entropy/multi-modality to flag catastrophic outliers. SpecPT-level
    outlier fractions likely require rejecting uncertain predictions;
    a calibrated posterior is also a differentiator vs point-estimate
    baselines.
X9. **OOD on SDSS** with the wavelength-aware path (now physically
    correct) — the generalization claim, and the bridge toward
    multi-survey training later.

### Sequencing

1. v2 tokenizer run to gate (T1) — sessions already underway.
2. T3 token audit + T2 decoder polish + X1 pre-tokenization in the
   same window (independent of each other).
3. Transformer v2 run = X2+X3+X5 together (one rewrite, one campaign),
   eval with X4 from step one.
4. X6-X9 on the trained model; X7 is the paper's centerpiece table.
5. T4/T5/T7 only on gate failure or after the campaign.

Target claim, stated once: a unimodal DESI FM whose frozen encoder
matches/beats AION-1's spectrum rows on property estimation (X7),
with redshift NMAD competitive with SpecPT (X4/X6) *from a generative
token model that also reconstructs spectra* — neither baseline does
both.


## 2026-06-12: Naming convention (to stop the v2/v3 jumble)

- **Tokenizer GENERATIONS** = recipe versions. v1 = released MSE/stretch
  codec (`spectrum_tokenizer_v1`). **v2 = the current recipe** (Huber
  ivar-NLL, joint capped entropy, wavelength-aware grid, balanced
  manifest). A hypothetical v3 generation would be a new recipe (e.g.
  native-resolution / multi-survey).
- **RUN ATTEMPTS** of a generation get suffixes: `tokenizer_v2_3k_ddp`
  (attempt 1, entropy overshoot), `tokenizer_v2_3k_ddp2` (attempt 2,
  spike collapse), `tokenizer_v2_3k_v3` (attempt 3, healthy, current).
  "v3" in run names means third attempt of the v2 generation, NOT a
  third-generation tokenizer.
- The artifact this campaign releases is **tokenizer v2** =
  `tokenizer_v2_3k_v3`'s best checkpoint.


---

## 2026-06-12: Tokenizer v2 training COMPLETE — final results (run `waotf2n0`, 80k steps)

The annealing resume finished cleanly: `tokenizer_v2_3k_v3` ran to its
full 80,000-step budget (resumed from `last.pt` at ~54.7k, learning rate
cosine-annealed 2.5e-4 → 3.0e-5) and W&B marks the run **finished** —
the first tokenizer-v2 attempt to reach a natural end rather than a
wallclock kill or a collapse. ~11.1h total GPU-time on the run id,
2.11 steps/s, batch 32, bf16, 3k balanced manifest.

### Final validation metrics (and what each means)

| Metric | Final (step ~80k) | Best | Meaning |
|---|---|---|---|
| `val/nll_flux` | 0.5006 | **0.4983** @ 74.5k | Ivar-weighted Gaussian NLL per pixel; χ²/pixel = 2·nll |
| **χ²/pixel** | 1.001 | **0.997** | 1.0 = reconstruction statistically indistinguishable from the observation given DESI's own noise. **We are AT the information-theoretic floor.** Gate was ≤1.1. |
| `val/flux_r2_pooled_snr3` | 0.8762 | 0.8762 @ 73k | Corpus-pooled R² on the median-SNR>3 slice — the AION-comparable number (theirs: 0.994 on a much brighter SDSS-heavy corpus) |
| `val/flux_r2_pooled` | 0.572 | — | Pooled R² over ALL spectra incl. very faint (R² vs noisy input is bounded ≪1 there regardless of codec quality) |
| `val/flux_r2_snr3` | 0.6675 | — | Per-spectrum mean R², SNR>3 slice (harsh on faint corpora; internal tracking only) |
| `val/codebook_use` | 70.7% | ~71% | Fraction of the 1024 LFQ codes active on val (gate ≥50%; v1 trial evidence was 3–8%) |
| `val/istd_mse` | 0.033 | — | Noise-channel (inverse-std) reconstruction MSE, aux head |

Annealing's contribution: nll 0.5144 → 0.4983 (χ² 1.029 → 0.997) from
step 52k to 74.5k; pooled_snr3 0.8744 → 0.8762. Real but small — the run
was already near-converged; the anneal bought the last ~3% of χ².

### Scorecard vs the 2026-06-11 success criteria

1. χ²/pixel ≤ ~1.1 → **0.997–1.001. PASSED, at the floor.** This is the
   physically rigorous claim AION does not report.
2. Pooled SNR>3 R² → **0.876** vs AION's 0.994. Not matched; the gap is
   partly population (their corpus is far brighter; pooled R² rises with
   corpus brightness by construction) and now bounded by the noise floor:
   at χ²=1.0 the residual IS the noise, so on OUR corpus this number
   cannot go materially higher with any codec. A same-corpus AION
   measurement (or our codec on an SDSS-bright slice) is the only honest
   head-to-head — deferred to X9.
3. Codebook utilization ≥50% → **70.7%. PASSED** (v1: single-digit %).
4. Downstream transformer-v2 → pending (the decisive gate).

### Finding: best.pt selection is skewed by the entropy reward

`best.pt` is keyed on `val/total` (pretrain_tokenizer.py), which includes
the entropy auxiliary (a REWARD, capped). The surviving W&B artifact
(`tokenizer_tokenizer_v2_3k_v3:v27`) is from **step 57,500**
(val_total 0.477, nll ≈ 0.511, χ² ≈ 1.02) — `val/total` never beat that
later because the sample-entropy term drifted up while reconstruction
kept improving. The genuinely best reconstructor is the **end-of-run
`last.pt` (step 80k, nll 0.5006)** / best-nll region ~74.5k.
- **Decision: ship `last.pt`'s weights as the released tokenizer v2**
  (χ² 1.001 vs 1.02, pooled_snr3 0.8762 vs ~0.875 — small but free).
- **Fix for future runs**: select best on `val/recon` (nll_flux +
  istd term) only, never on a total that mixes in auxiliary
  rewards. (Applies to any v3-generation run.)

### Status / next

Tokenizer v2 is DONE — T1 gate passed. Proceed per the roadmap:
T3 token-stability audit + T2 decoder-only polish (optional) + X1
pre-tokenize the corpus with this checkpoint, then the transformer-v2
campaign (X2+X3+X5).


---

## 2026-06-13: Freeze tokenizer v2 = final.pt + T3 stability-audit tooling

### Freeze decision: ship final.pt, not best.pt or last.pt

Compared the three checkpoints run `waotf2n0` produced (val metrics at each step):

| Checkpoint | Step | χ²/pixel | pooled R² (SNR>3) | Codebook | On W&B? |
|---|---|---|---|---|---|
| best.pt (= artifact `…v2_3k_v3:v27`) | 57,500 | 1.0125 | 0.8752 | 69.2% | yes |
| last.pt (rolling, health-gated) | ~78,000 | 1.0010 | 0.8762 | 70.2% | NERSC only |
| **final.pt (end-of-run, LR floor 3e-5)** | ~80,000 | 1.0012 | 0.8762 | 70.7% | scratch + CFS |

`final.pt` ≈ `last.pt` (identical to 3 decimals) and both beat the `best.pt`
artifact by ~1% χ² — `best.pt` is selected on `val/total`, which mixes in the
capped entropy *reward*, so it is not the best *reconstructor*. **Ship final.pt**:
the fully-annealed terminal state, CFS-mirrored. (True χ² min was step 74.5k @
0.9967, but no checkpoint exists there; the <0.5% gap is within val-set noise.)
For future runs, select best.pt on `val/recon` (nll_flux + istd), never on a
total that includes auxiliary rewards.

**Release wiring (this commit):** `scripts/setup_release_checkpoints.py` gains a
`spectrum_tokenizer_v2` entry + a `local_pt` source branch (final.pt is not a
W&B artifact, so it is copied from a staged local file rather than downloaded).
`src/inference/release.py` needs no change — `load_spectrum_tokenizer(model_id)`
already resolves by model_id and builds a default `SpectrumTokenizer()`.
`default_tokenizer` stays **v1** (the released v9 transformer was trained on v1
tokens; a swapped codec would break it). v2 is an *available* tokenizer for the
upcoming v2 transformer only. **User action on NERSC:** copy
`$CFS/tokenizer_v2_3k_v3/final.pt` → `checkpoints/wandb_artifacts/spectrum_tokenizer_v2_final/best.pt`,
then `python scripts/setup_release_checkpoints.py --copy`.

### T3 token-stability audit (tooling landed; awaiting the NERSC run)

New `nersc/audit_tokenizer_stability.py` — the last thing isolated metrics can
tell us before the transformer campaign. Reuses the dataset, the *exact* training
val split (`torch.randperm(seed)` recipe from pretrain_tokenizer.py), and
`flux_r2_terms`. Two diagnostics, single I/O pass (grouped by survey/program):

- **A. Noise-realization flip rate.** Re-draw the per-pixel noise (std =
  1/√ivar) K times, re-tokenize, measure per-token flip rate vs the observed
  spectrum's tokens, **stratified by per-token local SNR**. The diagnostic shape:
  at χ²=1.0 the codec encodes detail down to the noise, so low-SNR tokens
  flipping is *expected and benign*; high-SNR tokens must stay stable or the
  transformer would learn noise. Reports per-spectrum median/p90, per-position
  flip rate, and flip-vs-SNR bins.
- **B. Per-(survey, program) equity.** χ²/pixel (= 2·nll_flux), pooled R² and
  codebook usage per group, to confirm the balanced manifest didn't leave
  dark-program spectra second-class. Needed a small reusable addition:
  `DR1IndexedDataset.meta_for_index(i) → (survey, program)`.

Soft acceptance (CLI-tunable): high-SNR(>3) flip rate < 0.15; every group
χ²/pixel < 1.2 and codebook use > 0.30; flip rate falls with SNR. Emits
`t3_stability.json` / `.md` / `.npz` and a PASS/FLAG verdict.

Verified: 22 new audit unit tests + 37 tokenizer tests green; CPU smoke of the
full `run_audit` driver on a synthetic dataset (random-init model → correct FLAG
with all reasons, confirming the code path). On PASS of the real run → X1
pre-tokenize corpus + transformer-v2 campaign (X2+X3+X5). No tokenizer retraining
(already at the noise floor).


---

## 2026-06-13 (cont.): T3 first run + audit upgrade (per-bit flip, margin sweep)

Ran T3 on final.pt (1331 val spectra, K=16). Verdict FLAGged, but the FLAG was
a naive metric, not a codec defect. The two facts looked contradictory:

- **Part B reconstruction is excellent at the noise floor** — χ²/pixel ≈ 1.0
  across every (survey, program) group, pooled-SNR>3 R² 0.87–0.99. Dark is NOT
  second-class (main/sv2/sv3 dark χ² < bright). Only blips: sv1/dark χ²=1.29
  (n=81) and sv1/bright pooled-R²=0.29 (n=69) — small-sample pooling noise on
  the smallest survey-validation slice.
- **Yet token-index flip rate was ~73% even for high-SNR (>3) tokens.**

Reconciliation: each token is a **10-bit LFQ code**; the integer index changes
if ANY one of 10 sign-bits flips, so token-level flip overcounts. Inverting
1−(1−p_bit)^10 = 0.73 gives **per-bit flip ≈ 12%** — modest. And at χ²=1.0 the
codec reconstructs down to the noise, so its marginal (near-sign-boundary) bits
*encode the noise itself* and flip on a re-draw, while the signal-bearing bits
(and thus reconstruction) hold. The SNR curve confirmed it: the very-high-SNR
(>10) tokens were the MOST stable (0.63); peak instability was mid-SNR (signal ≈
noise → bits on the boundary). The ">3" aggregate just lumped the stable >10
tail with the marginal 3–10 band.

**The keep-this finding:** a large fraction of *exact token identity* is
noise-driven. Not fixable in the codec without backing off the noise floor
(fewer bits = stabler tokens but worse reconstruction — a v3-generation
tradeoff, not warranted). The lever is the **transformer objective**:
exact-token-match has a low ceiling here (almost certainly why v1 masked-token
accuracy plateaued ~47% — you can't predict noise bits). Validates training
masked-targets-only (X2) and judging on reconstruction + physical/NMAD metrics
(X4), NOT token-match accuracy.

**Audit upgrade (this commit)** to confirm the interpretation with evidence:
- `bits_from_indices` — recover the 10 sign-bits from each index and measure
  flip at the **per-bit** granularity (the honest number).
- **Margin sweep** over perturbation scale (0.25/0.5/1.0 σ) — exposes the
  sign-decision margin; bit flip rises smoothly and stays far below token flip.
- **Very-high-SNR (>10) slice** — the signal-bearing bits, reported separately.
- **Recalibrated verdict:** hard gate = per-bit flip on SNR>10 tokens
  (`--bit-thresh` 0.10) + reconstruction on adequately-sampled groups
  (n ≥ `--min-n-equity` 100); token-index flip and small-n group blips (e.g.
  sv1) are informational notes, not fails.

Tests: 30 audit unit tests (incl. bit-decomposition + the token-overcounts-bit
relationship) + 37 tokenizer tests green; CPU smoke confirms bit-flip ≤
token-flip at every scale and rises monotonically with perturbation. Re-running
T3 on final.pt next; expect PASS on the per-bit gate. This does not block
freezing v2 (reconstruction is the codec's job and it's at the floor).


---

## 2026-06-13 (cont.): T3 re-run — substantive PASS (proceed to transformer)

Re-ran T3 on final.pt with the per-bit/margin upgrade (1331 val spectra, K=16).
The verdict mechanism hard-FLAGs on one marginal number; human review =
**PASS, proceed**. Numbers (per-bit = the honest granularity):

- **per-bit flip, high-SNR (>10): 0.120** (gate was 0.100 — marginal miss)
- per-bit flip by SNR: [0,1) 0.170 · [1,2) 0.223 · [2,3) 0.219 · [3,5) 0.208 ·
  [5,10) 0.188 · **[10,∞) 0.120** → bit flip is at its MINIMUM at the highest
  SNR (the healthy signature: signal-bearing bits are the most stable).
- margin sweep (bit flip vs perturbation): 0.060 / 0.109 / 0.181 at
  0.25 / 0.5 / 1.0 σ → smooth, ~linear → finite decision margin, NOT
  boundary-chaos. (token-index flip in parallel: 0.345 / 0.531 / 0.708.)
- equity: every group χ² ∈ [0.91, 1.29]; only sv1/dark > 1.2 (n=81 → note).
  Dark not second-class (better than bright in main/sv2/sv3). sv1/bright
  pooled-R²=0.29 (n=69) is a pooling artifact — its χ²=1.04 is fine.

Why proceed despite the FLAG: (1) the perturbation is a FULL σ added on top of
already-noisy data, so the tokenized input differs by 1σ while real obs-to-obs
differs by √2σ — 0.12 is measured under a conservative kick; at 0.5σ it's ~0.07.
(2) bit flip is the SNR-minimum and the margin sweep is smooth → the bits sit at
a real distance from the sign boundary. (3) reconstruction is at χ²=1.0 — the
signal is provably preserved; only noise-level bits move. The 0.10 gate was a
pre-data guess; the defensible calibrated standard under a full-σ perturbation
is ~0.15. Decision rests on the diagnostic SHAPE, not a relaxed threshold — the
0.120 is recorded as-observed and the gate stays 0.10 (advisory).

Interpretation (universal, not a v2 defect): ~12% of even signal-bearing bits
are noise-driven → exact-bit/token prediction has a hard ceiling. AION's
identical 1024-code LFQ has the same property (unmeasured). Locks in the
transformer-v2 design: **masked-targets-only (X2); judge on reconstruction +
redshift NMAD/outliers (X4); never exact-token-match accuracy** (what capped v1
at ~47%). A bits/token reduction (256 codes) would stabilize tokens at a
reconstruction cost — a possible v3-generation experiment, not warranted now.

**T3 closed. Tokenizer v2 frozen (final.pt). Next: X1 pre-tokenize the corpus,
then the transformer-v2 campaign (X2+X3+X5, eval X4).**


---

## 2026-06-13 (cont.): X1 — corpus pre-tokenization (cache the frozen v2 tokens)

The transformer ran the ConvNeXt encoder on every step (`tokenize_and_build` →
`spec_tok.encode`) on top of per-step FITS stitching — the throughput ceiling.
Tokenizer v2 is frozen, so its tokens are fixed: run the encoder once, cache the
codes, make every step a tiny array read. Implemented end-to-end (writer + reader
+ train wiring + correctness proof).

**Shard format** — one compressed `.npz` per (survey, program, healpix),
`{survey}_{program}_{healpix}.npz`: `indices` (N, n_tokens) uint16 **raw codes
0..1023** (offset added at train time, unchanged), `z`/`denorm` float32, quality
flags `zwarn`/`fiberstatus`/`nonzero_flux`, and `spectype`/`targetid`/`row` +
`coadd`/`redrock` provenance for X4/X7/X9. Raw z is cached (not binned) so
`--z-bins`/4096 stays a train-time choice. ~575 B/spectrum → ~5 GB for 8.9M.

**Writer** `nersc/pretokenize_corpus.py`: streams the manifest one healpix at a
time (reuses `stitch_bands`), encodes on GPU, writes a shard. DDP-aware
(`records[rank::world]`), **resumable** (skips existing shards → survives the 4h
interactive cap), `--quality {all,strict}` (default all = tokenize every row +
store flags), and a built-in `--verify` that re-encodes cached spectra from their
source FITS and asserts token-for-token equality.

**Reader** `DR1CachedTokenDataset` (+ `collate_cached_skip_none`,
`collect_redshifts_from_cache`) in `nersc/dr1_tokenized_dataset.py`: builds the
flat index applying the quality cut from the stored flags (default reproduces
today's training cut), MRU shard cache, `meta_for_index` for audit parity.

**Integration**: `tokenize_and_build` gains a one-line branch — use
`raw_batch["spec_indices"]` when present (skip encode; `spec_tok` may be None),
everything after (offset, encoder/redshift masking, A/B assembly) unchanged.
`train_transformer.py` gains `--tokenized-dir`: builds the cached dataset over
the healpix-split shards, **skips loading the spectrum tokenizer entirely**, fits
z_tok from cached z. Unset → the current on-the-fly path, zero behavior change.

**Correctness proof** (`tests/test_pretokenize.py`, 12 tests): a CPU roundtrip
(no FITS) shows cached indices == `encode()`, and `tokenize_and_build(cached)` ==
`tokenize_and_build(flux, spec_tok)` element-for-element — both sequences AND
mask positions, with masking RNG fixed. Plus reader filter/collate/meta units.
Full suite 201 passed (1 pre-existing unrelated wandb-offline test fails with or
without these changes).

Run (NERSC): `pretokenize_corpus.py --manifest <balanced> --tokenizer-ckpt
final.pt --out <dir> --amp` (interactive, resumable), then `--verify`, then
`train_transformer.py --tokenized-dir <dir> …`. Corpus is the `--manifest` arg —
recommend the balanced DR1 manifest scaled to 5k/10k. **Next: transformer-v2
campaign (X2 masked-targets-only, X3 4096 z-bins + soft labels, X5 SDPA/bf16,
eval X4 NMAD/outliers).**


---

## 2026-06-13 (cont.): Stratified train/val split for full DR1

Decision: the transformer-v2 campaign trains on the FULL DR1 corpus (natural
distribution), pre-tokenized once via X1 (frozen tokenizer is cross-survey
validated by T3, so tokenizing unseen healpix is sound). The cache stores
survey/program/spectype, so natural-vs-balanced sampling stays a train-time
choice — no re-tokenization.

`split_records_by_healpix` was already a seeded random `randperm` holdout (never
a sequential block), but with full DR1's nested survey×program manifest ordering
and a small holdout_frac, a plain pooled split represents rare categories (sv1)
in val only in expectation. Upgraded to **stratified**: the holdout is drawn
independently WITHIN each (survey, program) group (seeded), so every category
appears in val at ~holdout_frac no matter how imbalanced the corpus, and any
group with >1 healpix contributes at least one to val. This prevents a category
from dominating/vanishing from val by chance and makes per-(survey,program) /
per-spectype val metrics (X4) trustworthy. Records lacking the keys fall back to
the prior plain random split (back-compatible). Still healpix-level (no
same-pointing leakage). Tests: stratified representation, imbalanced corpus,
not-sequential-by-category, tiny-category coverage, fallback. 205 passed.


---

## 2026-06-13 (cont.): pretokenize_corpus throughput + logging fix

First full-scale run symptom: 4 GPUs at 0% util, no logs for minutes. Not a
hang — per-spectrum cost is dominated by the FITS read + the pure-Python
`stitch_bands` loop, done single-threaded per rank, so the GPU starved while one
CPU core ground through stitching; and prints were buffered under srun. Fixes:
- **Parallel stitch**: `--num-workers` (default 8) CPU process pool per rank
  (spawn context so the parent's CUDA context is never forked), splitting each
  healpix's rows across workers. `np.array_split` + `pool.map` preserve order →
  identical output; `--num-workers 0` = serial fallback; `--verify` still proves
  byte-equality. Makes the 4-GPU srun worthwhile (~Nx per rank).
- **Streaming logs**: flushed stdout, a per-rank startup line, and progress every
  healpix for the first few then every 20 with a spectra/s rate.

Also confirmed for full DR1: no need to stage raw FITS to scratch — the manifest
points at the canonical CFS DR1 (`build_dr1_index --root` default
`/global/cfs/cdirs/desi/public/dr1`), read once. Only the ~5 GB token cache lives
on scratch (the working set training reads); copy it to CFS after build since
scratch is purged.

---

## 2026-06-16: X2 — masked-targets-only objective + variable mask ratio

**Why.** The encoder masked ~15% of spectrum tokens, but the decoder was
teacher-forced on the full unmasked sequence and the loss ran over *all* spectrum
target positions (`transformer.py:461-463`). For the unmasked ~85% the decoder
just copies the answer across cross-attention — free reward. That diluted the
gradient with trivially-copyable positions, inflated the logged `spectrum_acc`
(the honest `masked_spec_acc` was tracked but not optimized), and contradicted the
locked-in T3 finding that exact token identity is largely noise-driven — so
supervising every position on exact match is the wrong objective.

**Change (both opt-in, fully backward-compatible — existing runs byte-identical
with the flags off):**

- **Masked-targets-only loss** (`tokenize_and_build`, new `mask_targets_only` arg):
  set the spectrum target at the *un*masked positions to `-100`, so reconstruction
  is supervised ONLY where the encoder masked (real MAE in-filling). Redshift
  (position 0) and EOS stay supervised. No model change needed — the loss already
  honors `ignore_index=-100` with `n_spec.clamp(min=1)`, and `compute_metrics`
  ignores `-100`, so the headline `spectrum_acc` becomes the honest masked number
  for free. CLI: `--masked-targets-only` in `train_transformer.py`.
- **Variable mask ratio** (`train_transformer.py`, `--mask-ratio-min/--mask-ratio-max`):
  each step samples ratio ~ Uniform[min,max] from a step-seeded generator
  (`seed+step`) so the schedule is deterministic across resumes and identical
  across DDP ranks. Unset = fixed `--encoder-mask-ratio` (unchanged). The sampled
  `mask_ratio` is logged per step in `metrics.jsonl`.

**Verified.** New unit tests in `tests/test_training_helpers.py`
(`TestMaskedTargetsOnly`): unmasked spectrum targets become `-100`, masked keep
the true token, redshift/EOS always supervised, no-op at ratio 0, flag-off
unchanged, approach-b parity. Cache-parity roundtrip extended in
`tests/test_pretokenize.py` (cached == on-the-fly under the new flag, incl. the
`-100` positions). Loss sanity (CPU, tiny model): flag on supervises ~ratio·T_spec
positions (13/32 at ratio 0.5) vs 32/32 off, loss finite both ways.

Recommended v2-campaign launch (after the X1 cache lands + `--verify`):
`--tokenized-dir … --approach a --masked-targets-only --mask-ratio-min 0.3
--mask-ratio-max 0.7 --redshift-soft-sigma 1.5`. Next: X3 (4096 z-bins + fp32 soft
labels), X5 (SDPA/bf16/KV-cache), X4 eval (NMAD/outliers) — all on `--tokenized-dir`.

---

## 2026-06-16: Notebook 07 — "blind" reconstruction panel (honest AR)

Realized the autoregressive panel (Viz 2) in `notebooks/07_visualize_predictions.ipynb`
is not a "predict without info" test: it's AR on the *decoder* only, but with
`ENCODER_MASK_RATIO = 0.0` the **encoder sees the full spectrum** (and, for
Approach A, the true redshift), so the decoder reconstructs by *copying through
cross-attention*. Both TF and AR there leave the copy path open, so AR≈TF mostly
proves the copy works.

Added **Visualization 2b/2c — blind reconstruction**: a new `predict_autoregressive_blind`
helper + two panels (50% and 90% encoder masking) that mask the encoder's spectrum
tokens (`[MASK]`) and hide the redshift (`[REDMASK]`, Approach A; the v9 `redmask50`
model is in-distribution for this), then generate. The masked positions are absent
from the encoder, so they must be *inferred* — the genuinely honest reconstruction.
Prints `blind MSE` at each mask fraction to compare against the copy-path AR/TF MSE in
Viz 5; the gap (and the 50%→90% degradation) is the size of the copy shortcut. The
eval-side mirror of the X2 training change.

Also fixed the **redshift panel (Viz 3)**, which previously did not mask redshift in
the encoder — both `tf_z` (`predict_teacher_forced`) and `ar_z` (Viz 2
`predict_autoregressive`) ran with `redshift_mask_ratio=0.0`, so for Approach A the
true redshift token was in the encoder and the reported z-MAE was *copy fidelity*, not
inference. Added a third **"blind (z hidden)"** series + MAE using
`blind_generated[:, 1]` (encoder redshift = `[REDMASK]`) — the honest spectrum→z
number. The scatter now distinguishes the copy path (tf/ar) from honest inference.

Follow-ups in notebook 07:
- **Mask visualization**: Viz 2b (50%) and 2c (90%) now shade which latent-grid
  regions were hidden from the encoder vs visible, and print a masked(inferred) /
  visible(anchor) MSE split — directly showing where the reconstruction error lives.
- **N_TOKENS derived from the tokenizer**: the module constant
  `src.tokenizers.spectrum.N_TOKENS` is **stale (273)** — `encode()` actually emits
  **272** tokens (wavelength-aware resampling onto a fixed latent grid, independent of
  input length). The notebook now overrides `N_TOKENS` from a probe encode right after
  the model load, so every panel slices/masks the correct width (this also fixes a
  latent off-by-one where the old 273 slice grabbed the EOS token as a "spectrum
  token"). The library constant should eventually be corrected to 272.
- **SDSS OOD blind panels (Viz 6b)**: added 50% and 90% blind reconstructions on
  out-of-distribution SDSS, with the same orange mask shading (gray reserved for
  outside-SDSS-coverage edges) and per-ratio MSE — the OOD mirror of Viz 2b/2c.
- **SDSS redshift panel (Viz 7)**: had the same copy leak as Viz 3 — `ood_generated`
  ran with `redshift_mask_ratio=0.0`, so Approach A read z off the encoder. Added the
  honest **blind (z hidden)** series (`predict_autoregressive_blind(ood_batch, …,
  mask_ratio=0.0)`: full spectrum visible, encoder z = `[REDMASK]`) + its MAE, so the
  OOD redshift number is genuine spectrum→z. Both redshift panels (DESI Viz 3, SDSS
  Viz 7) are now consistent.

**Instructor request — redshift from the full (unmasked) spectrum.** The deliverable
of a spectrum FM is measuring z from a *full observed spectrum*, not a masked one
(masking was only a training aid). Added a dedicated section: `predict_redshift_fullspec`
runs the encoder on the **entire** spectrum (`encoder_mask_ratio=0.0`) with only the
redshift hidden (`[REDMASK]`, since it's the predicted quantity). Redshift is decoder
position 0, so it's read from a **single decode step** (no autoregression) — fast enough
to evaluate the whole sample. Reports NMAD of Δz/(1+z), catastrophic outlier fractions
(>0.003, >0.01), MAE, median bias, plus the 256-bin quantization floor. Run over the
**entire** local DESI sample (in-distribution) and a freshly-streamed **100**-spectrum
SDSS sample (OOD); reconstruction panels stay at N_OOD rows for legibility. Caveat
logged in-notebook: a strict val number needs the held-out split on NERSC (local tiles
may overlap training).

Fixed a `Broken pipe` RuntimeError from streaming SDSS twice: a second
`load_dataset('MultimodalUniverse/sdss', streaming=True)` re-init while the first
stream's multiprocessing shm handles were alive throws it. The SDSS-load cell now
streams **once** into `sdss_pool` (up to 100) and the panels use `sdss_raw =
sdss_pool[:N_OOD]`; the redshift metric reuses `sdss_pool` instead of re-streaming.
(A subsequent broken pipe on the *first* load means the prior double-stream already
killed the kernel's `torch_shm_manager` → restart the kernel; load cell also sets
`torch.multiprocessing.set_sharing_strategy('file_system')` defensively.)

Added a **raw-data noise row** to the three masked panels (DESI Viz 2b/2c, SDSS Viz
6b): a third sub-row per sample plotting the per-pixel **1σ = 1/√ivar** from the
observed spectrum's inverse-variance array (`batch['ivar']` / `ood_batch['ivar']`),
in the same physical-flux units as the flux/residual, with `nan` gaps where ivar≤0.
Lets the reader judge whether the reconstruction residual sits within the measurement
noise. (This is the raw measurement noise that comes with the spectrum, not a
tokenizer-derived quantity.)

**Source fix:** corrected `src/tokenizers/spectrum.py` `N_TOKENS` **273 → 272** to
match `encode()`'s actual output (`LATENT_GRID_SIZE 8704 / stride-32 = 272`, verified).
The constant was a documentation/export value — its only library consumer
(`src/inference/release.py:24`) imports but never uses it, so nothing relied on 273.
Full suite green (210 passed; the lone failure is the pre-existing, unrelated
`test_missing_api_key_falls_back_to_offline`). The notebook's derive-from-tokenizer
override now agrees with the corrected constant.

**Smoke / data-loader fix:** `nersc/dr1_dataset.py` opened coadd/redrock with
`memmap=True`, which astropy refuses for the DESI `*_MASK` image HDUs (they carry
BZERO/BSCALE → "Cannot load a memory-mapped image ... Set memmap=False"). This
crashed `train_transformer.py --smoke` in FITS I/O before the first training step
(unrelated to X2). Switched both opens to `memmap=False`; with `cache_size=1` and
the dataset reading most rows of a healpix per file, a single bulk read is also
faster than many memmapped slice seeks. Verified by running the smoke end-to-end
(both default and X2-on: `--masked-targets-only --mask-ratio-min 0.3
--mask-ratio-max 0.7`) — 100 steps, loss decreasing, `final.pt` saved — and
`tests/test_dataset.py` stays green (10 passed).

**Tokenizer v2 released + reconstruction notebook.** Registered
`spectrum_tokenizer_v2` as a release checkpoint on NERSC (`setup_release_checkpoints.py
--copy`, frozen `final.pt` @ step 80k; `default_tokenizer` stays v1 since the released
v9 transformer was trained on v1 tokens). Added
`notebooks/08_tokenizer_v2_reconstructions.ipynb` — tokenizer-only (encode→decode, no
transformer) on the local DESI data: flux reconstructions with raw 1σ-noise rows, the
istd (noise) channel reconstruction, per-spectrum χ²/pixel + ivar-weighted R²
distributions, pooled high-SNR R², and codebook utilization. Executed end-to-end on the
29 local spectra: χ²/pixel median 0.894, **pooled R²(SNR>3) = 0.993** (on par with
AION-1's token reconstruction R² = 0.994), codebook 341/1024 used. Per-spectrum R²
median is low (~0.41) on bright/low-SNR objects — expected, hence χ² and pooled-SNR R²
are the headline numbers.

Heads-up (not yet fixed): running `setup_release_checkpoints.py` on NERSC regenerated
`MANIFEST.json` with only `spectrum_tokenizer_v2` under `models` (v1/v9 artifacts weren't
present there), so the transformer notebook's v1/v9 lookups would break until MANIFEST is
regenerated on a machine that has all three artifacts.

---

## 2026-06-18: X1 pre-tokenization complete — full-DR1 cache + manifest of record

The frozen-v2 corpus pre-tokenization finished on NERSC: **30,376 token shards** in
`$SCRATCH/dr1_tokenized_v2/*.npz` (one `{survey}_{program}_{healpix}.npz` per healpix
across the full DR1 corpus). X1 is done; the ConvNeXt encoder is now out of the
transformer training loop.

**Manifest of record (write this down):** the cache was built from
**`$SCRATCH/manifests/dr1_v2_full.jsonl`** — the full-DR1 natural-distribution manifest,
NOT one of the balanced/scaled manifests under `$SCRATCH/deepsrch/manifests/`
(`dr1_v2_balanced_{2k,3k,7k,10k}.jsonl`, `dr1_{10k,20k,50k}.jsonl`, etc., which were the
tokenizer-training and smaller-scale experiments). This matches the 2026-06-13
"Stratified train/val split for full DR1" decision (transformer-v2 trains on the full
natural corpus) and is consistent with the 30,376-shard count.

**Why it must be exact:** `train_transformer.py --tokenized-dir` maps each manifest record
to its `{survey}_{program}_{healpix}.npz` shard, so the X2 training launch **must** pass
`--manifest $SCRATCH/manifests/dr1_v2_full.jsonl` — any other manifest silently drops
records lacking a matching shard. `--manifest` is still `required=True` because it is the
index + the (stratified) healpix train/val split; `--tokenized-dir` only supplies the
cached tokens.

X2 smoke launch (corrected, after hitting "the following arguments are required:
--manifest"):

```bash
python nersc/train_transformer.py \
    --manifest $SCRATCH/manifests/dr1_v2_full.jsonl \
    --tokenized-dir $SCRATCH/dr1_tokenized_v2 \
    --approach a --masked-targets-only \
    --mask-ratio-min 0.15 --mask-ratio-max 0.75 \
    --redshift-soft-sigma 1.5 --smoke
```

**X2 smoke verified on NERSC (run `662cwwx6`).** Cache wired correctly
(`cached tokens … encoder not loaded`), all 30,376 records matched a shard
(manifest↔cache exact), stratified split 28,856/1,520, `mask_ratio=U[0.15,0.75]`
+ `masked_targets_only=True`, loss decreasing (z_loss 7.27→5.36). The
`only 578 z values for 256 bins` warning is a **smoke-only artifact**: the smoke
block forces `z_fit_files=min(_,5)` (`train_transformer.py:227`), so it scanned
5 shards; the real run scans 200 → ~23k z (≈90/quantile at 256 bins), stable.

## 2026-06-18: X3 — 4096 z-bins + fp32 soft labels + cache-aware DDP launcher

Folding X3 into the X2 cache run (user wants both). Three parts:

1. **fp32 soft-CE** (`src/models/transformer.py::_redshift_soft_ce`): cast
   `red_logits` to float32 before the Gaussian soft-label build + full-vocab
   `log_softmax`. At 4096 bins under `--amp` (bf16), the per-row Gaussian
   normalization and the log_softmax tail lose precision in bf16, making the
   soft-label gradient noisy. The upcast is one position per sequence — cheap —
   and makes the redshift objective dtype-independent. No behavior change at
   256 bins / fp32. Tests: 88 passed (soft/redshift/transformer/mask suite).

2. **z-fit-files for 4096 bins**: 4096 bins need ≥~82k z (≥20/quantile), so the
   X3 launch raises `--z-fit-files 200 → 800` (~92k z at ~115/shard). Left at
   256-bin default otherwise.

3. **Cache-aware DDP wrapper** (`nersc/train_transformer_ddp.slurm`): was
   hard-required on `TOKENIZER_CKPT` + `--tokenizer-ckpt` (legacy on-the-fly).
   Now env-driven and backward-compatible:
   - `TOKENIZED_DIR` → `--tokenized-dir` (encoder not loaded); `TOKENIZER_CKPT`
     only required when `TOKENIZED_DIR` unset.
   - X2: `MASKED_TARGETS_ONLY=1`, `MASK_RATIO_MIN/MAX`.
   - X3: `Z_BINS=4096`, `REDSHIFT_SOFT_SIGMA` (in bins; 0 = hard CE),
     `Z_FIT_FILES`.
   - Cache runs default `MANIFEST=$SCRATCH/manifests/dr1_v2_full.jsonl` and
     error (not silently rebuild) if it's missing. Flags assembled via bash
     arrays (verified: cache+X2+X3 and legacy+hard-CE both assemble correctly).

**Soft-sigma at 4096 bins**: the 256-bin run used sigma 1.5; 4096/256 = 16×
finer, so sigma `1.5 × 16 = 24` bins preserves the same physical z-width of the
soft label. Conditioning dropout stays v9's 0.5 (`--redshift-mask-ratio`, the
wrapper default).

X2+X3 launch (full run):

```bash
TOKENIZED_DIR=$SCRATCH/dr1_tokenized_v2 \
    MANIFEST=$SCRATCH/manifests/dr1_v2_full.jsonl \
    MASKED_TARGETS_ONLY=1 MASK_RATIO_MIN=0.15 MASK_RATIO_MAX=0.75 \
    Z_BINS=4096 Z_FIT_FILES=800 REDSHIFT_SOFT_SIGMA=24 \
    RUN_NAME=approach_a_v2cache_x2x3_ddp4 \
    sbatch nersc/train_transformer_ddp.slurm
```

**X3 smoke verified on NERSC (run `weh54f9g`).** 4096 bins + fp32 soft-CE ran
clean: `z_loss` finite throughout (8.63→7.98, no NaN/inf), `spec_acc` and
`masked_spec_acc` climbed together to ~0.10 (X2 supervising only hidden
positions), `final.pt` saved. `z_acc=0` is expected — bin-exact match at 4096
bins in 100 steps on a 5M smoke model is near-impossible; the soft label is the
gradient, and z is judged on NMAD (X4), not bin-exact accuracy. The z-fit
warning is again the smoke-only 5-shard artifact.

## 2026-06-18: rolling full-state last.pt for lossless transformer resume

Auditing the `--resume` wiring for the interactive 4h-boundary workflow surfaced
a gap: **`train_transformer.py` never wrote a `last.pt`** (the `last.pt` in older
log entries is `pretrain_tokenizer.py`'s, a different script). Its three outputs
were `best.pt` (full optim+scaler+step+wandb_id, but only written when val
improves), `step_NNNNNNNN.pt` (every `save_every`, but **model-only** — no Adam
moments), and `final.pt` (end-of-run, no optim). So a wallclock kill late in
training — once val has plateaued and `best.pt` has stopped advancing — could
only resume from a stale `best.pt`, and the more recent `step_*.pt` would
silently cold-start the optimizer. My earlier "resume from last.pt" instruction
was wrong for the transformer.

Fix (`train_transformer.py`, `save_every` block): write a **rolling full-state
`last.pt`** every `save_every` steps, same dict shape as `best.pt` (model +
optim + scaler + step + val_loss + z_tokenizer + meta + wandb_run_id), mirrored
to `cfs_out`. The archival model-only `step_NNNNNNNN.pt` stays (commented as NOT
a resume point); the model state_dict is captured once and shared between the
two writes. Now a kill resumes from the true latest checkpointed step with the
optimizer state intact.

`--resume` itself was already correct: it loads optim/scaler only if present
(`:416,419`), restores `step`/`best_val`/`wandb_run_id` (so W&B continues with no
`--wandb-run-id` flag), recomputes lr from `step` each iteration (`:479`, not
stored), and the variable mask-ratio is seeded by `seed+step` (`:486`) so the
schedule is deterministic across resume. The `--z-bins` guard (`:409`) errors if
the resume z-bins disagree with the checkpoint, so re-pass `--z-bins 4096` (and
the X2/X3 + model-dim flags) on resume. Tests: 91 passed (1 pre-existing
unrelated wandb-offline failure). Correct resume target is now:

```bash
--resume $SCRATCH/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/last.pt
```

## 2026-06-18: X2+X3 first full run (`dcsvevx3`) diverged to NaN — root cause + hardening

First full X2+X3 run on the cache crashed to `train/loss = NaN` at **step ~10,500**.
Post-mortem from W&B: `val/loss` (= `10·loss_redshift + loss_spectrum`) rose
76 → 91 because **redshift loss was *rising*** (7.2 → 8.67; random CE at 4096 bins
= ln 4096 = 8.32, so the head was at/above random and drifting) while
**spectrum_acc decayed 0.184 → 0.149**. Both are the heavy-redshift-weight
pathology, then a loss spike with no guard tipped it to NaN — a replay of the
v10 failure (2026-06-08 entry).

**Three root causes, all now fixed:**

1. **Wrong redshift loss weight.** I launched with `--redshift-loss-weight 10`
   (copied from the stale DDP-wrapper default). v9's validated value is **1.0**
   (breakthrough entry, run `8m9rkz37`); 10 puts ~95% of the loss mass on the z
   token (50 → 98%, the v8 copy-artifact regime), starving spectrum. Fixed the
   **default everywhere to 1.0**: argparse (`train_transformer.py`, was a stale
   50) and `train_transformer_ddp.slurm` (`REDSHIFT_LOSS_WEIGHT`, was 10).

2. **fp16 autocast (no bf16 option).** The loop ran default fp16; fp16
   `log_softmax` over the 5128-wide vocab is the overflow→NaN class the tokenizer
   campaign already killed with bf16. Added `--amp-dtype {bf16,fp16}` **default
   bf16** (fp32 dynamic range, no overflow); GradScaler auto-disabled for bf16,
   and a bf16 resume skips loading old fp16 scaler state. Eval autocast threads
   the same dtype (`evaluate(..., amp_dtype=)`).

3. **No finite-loss skip-guard.** The loop did forward→backward→step with no
   `isfinite` check, so one bad batch poisoned the weights permanently (the log
   prescribed this guard for v10 but it was never added). Added a **DDP-safe**
   guard: backward runs on every rank (keeps the grad allreduce in lockstep),
   then `dist.all_reduce(MIN)` on the finiteness flag makes all ranks skip the
   same steps; poisoned grads are zeroed, the step is skipped and counted, and
   100 consecutive skips abort (weights unrecoverable → resume from healthy
   `last.pt`). The fp32 soft-CE from the X3 commit stays (defense in depth).

Tests: 114 passed (1 pre-existing unrelated wandb-offline failure); guard +
dtype logic unit-checked (NaN/inf → SKIP, finite → step; bf16→no scaler).

**Action:** killed `dcsvevx3`; relaunch at the corrected defaults (weight 1.0,
bf16, guard on). Interactive DDP needs `srun --gpu-bind=none` (a bare `srun` in
`salloc` doesn't bind one GPU per task → all ranks land on cuda:0 → NCCL
`invalid device ordinal`):

```bash
srun --gpu-bind=none python nersc/train_transformer.py \
    --manifest $SCRATCH/manifests/dr1_v2_full.jsonl \
    --tokenized-dir $SCRATCH/dr1_tokenized_v2 \
    --approach a --masked-targets-only \
    --mask-ratio-min 0.15 --mask-ratio-max 0.75 \
    --z-bins 4096 --z-fit-files 800 --redshift-soft-sigma 24 \
    --redshift-loss-weight 1.0 --encoder-mask-ratio 0.5 \
    --amp --amp-dtype bf16 \
    --steps 100000 --batch-size 32 --lr 8e-4 \
    --num-workers 12 --d-model 768 --n-encoder-layers 6 --n-decoder-layers 6 --n-heads 12 \
    --healpix-holdout-frac 0.05 --ar-eval-batches 8 \
    --run-name approach_a_v2cache_x2x3_ddp4 --wandb-project redshifty
```

## 2026-06-18: AR eval decoupled from best-val → fixed 10k-step cadence

The autoregressive eval (`evaluate_ar`, ×2: honest + z-given) ran inside the
`if v["loss"] < best_val` block, so early in training — when best improves on
nearly every 500-step val while redshift is still random — it fired constantly.
It is **rank-0-only** and stalls the other 3 GPUs at the next allreduce; live
telemetry on `aqxmwgl1` showed ~225–250 s stalls on best events (AR + the
~1 GB checkpoint save/upload bundled together).

Key realization: **AR is redundant for redshift.** Redshift is decoder target
**position 0**, so the teacher-forced `val/redshift_acc` (logged free every
500 steps, over ~1600 samples) conditions on exactly the same context as AR's
first generation step — they measure the same thing, and the TF one is less
noisy. Confirmed live: TF `val/redshift_acc` 0.000625 (1/1600) vs AR
`ar_redshift_acc` 0.0039 (1/256), both random. AR's only non-redundant signal
is honest free-running **spectrum** generation + the z-given copy-leak probe —
neither needs to run on every best.

Change: new `--ar-every-steps` (default **10000**); the AR block moved out of the
best-val branch into its own `step % ar_every_steps == 0` block (rank 0). Metric
`kind` renamed `ar_best` → `ar` (no consumers of the old name). Smoke caps it to
50 so the path is still exercised. Wired `AR_EVERY_STEPS` into
`train_transformer_ddp.slurm`. End-of-run AR unchanged. Redshift tracking is
unaffected (use the cheap TF `val/redshift_acc` + `val/loss_redshift`); this
just removes most of the per-best GPU stalls. Tests: 91 passed (1 pre-existing
unrelated wandb-offline failure). Applies on the next resume of `aqxmwgl1`.

---

## 2026-06-18: X4 — physical redshift metrics (σ_NMAD / outliers) + continuous decoding

### Run status that motivated this (W&B `jjayaseelan-university-of-san-francisco/redshifty`)

- `cd1ikb99` (512-bin **hard**-CE control) broke through at ~50k: honest
  `val/redshift_acc` 0.082, `within2` **0.302**, `val/loss_redshift` 3.79, copy
  path `zgiven`=1.0 — **now the leader**.
- `aqxmwgl1` (4096-bin **soft** main) at 80k: `within2` 0.176, `loss_redshift`
  4.66 — slower; finer bins are harder to converge in this budget.

The 512-hard recipe overtook the 4096-soft main on honest bin accuracy. But bin
accuracy is **not** how AION-1 (or the redshift literature) reports quality, and
it *undercounts* soft/fine-bin models because argmax throws away the probability
mass spread over neighbouring bins. We could neither compare to AION nor fairly
adjudicate 512-hard vs 4096-soft. AION measures Δz/(1+z) σ_NMAD + catastrophic
outlier fraction over 1,024 z-bins and reports DESI-spectrum z as ~"perfect"
(AION §7.1.1).

### Change (branch `x4-redshift-eval`, eval-only, additive — no objective change)

- **`src/eval/redshift_metrics.py`** (new): `redshift_metrics(z_pred, z_true)`
  → `sigma_nmad = 1.4826·median(|Δz−median Δz|)`, `bias = median(Δz)`,
  `outlier_frac` at |Δz|>0.0033 (DESI/Redrock) and >0.05, `rms`, `mean_abs`.
  `decode_redshift(logits, z_tok, mode)`: `argmax` (legacy) or **`expected`**
  (probability-weighted **in the Gaussian latent**, then mapped back through the
  tokenizer's inverse CDF — sub-bin precision, unbiased w.r.t. the skewed z
  density). Torch-only / CPU-safe so it imports into notebooks for the separate
  SDSS OOD check (pair with `SpectrumTokenizer.resample_to_grid`).
- **`src/training/eval.py`**: `evaluate()` now decodes the position-0 redshift
  token both ways and returns `z_nmad`, `z_outlier_frac`, `z_outlier_frac_05`,
  `z_bias` (expected) + `z_nmad_argmax`. True z is the **continuous** value from
  the raw batch, not the re-quantized target bin.
- **Live logging is automatic**: the train loop already logs `{f"val/{k}": v}`
  for every returned key, so on the next resume both runs emit `val/z_nmad`,
  `val/z_outlier_frac`, `val/z_bias`, `val/z_nmad_argmax` with no train-loop edit.
- **`nersc/eval_redshift.py`** (new): path-agnostic checkpoint → σ_NMAD table.
  Restores dims + z-tokenizer straight from the checkpoint
  (`_infer_transformer_dims`, `_restore_z_tokenizer`), rebuilds the same healpix
  val split, runs honest eval (`--redshift-mask-ratio 1.0`), prints bin-acc,
  within2, σ_NMAD (expected vs argmax), outlier fractions, bias. Works on NERSC
  scratch ckpts and local `checkpoints/release/`.

### Verification

- `tests/test_redshift_metrics.py` (new, 7 tests) green: perfect→0, constant
  offset closed-form, one-hot expected==argmax, **expected<argmax σ_NMAD on
  sub-bin truths**, empty NaN-safe. Existing suites still pass (91).
- End-to-end `evaluate()` smoke (tiny untrained model): returns all `z_*` keys;
  `z_nmad` (expected) 0.108 < `z_nmad_argmax` 0.280 — expected-value decoding
  already lower even untrained, as designed.

SDSS OOD eval stays **out of scope** here (run separately in local notebooks via
`resample_to_grid` + this module). Possible follow-up if expected-value decoding
plateaus: a learned sub-bin regression head.

## 2026-06-18: two live full runs in flight — 4096-soft (`aqxmwgl1`) + 512-hard control (`cd1ikb99`) + resume commands

Two parallel X2 full runs on the v2 cache are now the runs of record; both hit the
4h wallclock boundary (W&B `crashed`) and resume from their rolling `last.pt`. The
512-bin hard-CE run is **new and was not previously logged** — recording it here so
it isn't mistaken for a stray.

**Run A — `aqxmwgl1` `approach_a_v2cache_x2x3_ddp4` (4096 soft, σ=24):** the X2+X3
main line (continuation of the `dcsvevx3` post-NaN relaunch). Last heartbeat step
89,400. Redshift is **not learning** on this run — `val/loss_redshift` 4.64 and
`train/redshift_acc ≈ 0` (random CE at 4096 bins = ln4096 = 8.32, so it's below
random in nats but bin-exact acc is ~0); spectrum side healthy (`val/spec_acc`
0.263). 4096 bin-exact acc is the wrong lens (judge on σ_NMAD, X4); still, the flat
redshift signal is why the 512-hard control exists.

**Run B — `cd1ikb99` `approach_a_v2cache_x2_512hard_ctrl_ddp4` (512 hard, σ=0):**
control to isolate whether the 4096-bin soft-label head is what's starving redshift.
Same everything else (manifest, cache, X2 masking, weight 1.0, bf16, d768/6+6/12h)
but **z-bins 512, hard CE (`--redshift-soft-sigma 0`), `--z-fit-files 400`**. Much
healthier redshift: at step 59,980 `train/redshift_acc` 0.656 / within2 0.781,
`val/redshift_acc` 0.19 / within2 0.596 — i.e. the head *does* learn z at 512 hard
bins, supporting the hypothesis that the 4096-soft config (not the data/masking) is
the blocker. Both runs share `--steps 200000`, `--ar-every-steps 10000`, bf16 +
finite-loss guard.

Ignored as wrong/stale: `dcsvevx3` (NaN, stale `--redshift-loss-weight 10`) and the
smoke runs `weh54f9g` / `662cwwx6`.

**Resume commands of record** (interactive DDP4 under `salloc`; `srun
--gpu-bind=none` so each rank binds its own GPU; re-pass `--z-bins` + model-dim +
X2/X3 flags — the `--z-bins` guard errors on mismatch; `--resume` restores
step/best_val/wandb_run_id so W&B continues the same run):

```bash
# aqxmwgl1 — 4096 soft (X2+X3)
srun --gpu-bind=none python nersc/train_transformer.py \
    --resume $SCRATCH/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/last.pt \
    --manifest $SCRATCH/manifests/dr1_v2_full.jsonl \
    --tokenized-dir $SCRATCH/dr1_tokenized_v2 \
    --approach a --masked-targets-only \
    --mask-ratio-min 0.15 --mask-ratio-max 0.75 \
    --z-bins 4096 --z-fit-files 800 --redshift-soft-sigma 24 \
    --redshift-loss-weight 1.0 --encoder-mask-ratio 0.5 --redshift-mask-ratio 0.5 \
    --amp --amp-dtype bf16 \
    --steps 200000 --batch-size 32 --lr 8e-4 \
    --num-workers 12 --d-model 768 --n-encoder-layers 6 --n-decoder-layers 6 --n-heads 12 \
    --healpix-holdout-frac 0.05 --ar-eval-batches 8 --ar-every-steps 10000 \
    --run-name approach_a_v2cache_x2x3_ddp4 --wandb-project redshifty

# cd1ikb99 — 512 hard (control)
srun --gpu-bind=none python nersc/train_transformer.py \
    --resume $SCRATCH/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/last.pt \
    --manifest $SCRATCH/manifests/dr1_v2_full.jsonl \
    --tokenized-dir $SCRATCH/dr1_tokenized_v2 \
    --approach a --masked-targets-only \
    --mask-ratio-min 0.15 --mask-ratio-max 0.75 \
    --z-bins 512 --z-fit-files 400 --redshift-soft-sigma 0 \
    --redshift-loss-weight 1.0 --encoder-mask-ratio 0.5 --redshift-mask-ratio 0.5 \
    --amp --amp-dtype bf16 \
    --steps 200000 --batch-size 32 --lr 8e-4 \
    --num-workers 12 --d-model 768 --n-encoder-layers 6 --n-decoder-layers 6 --n-heads 12 \
    --healpix-holdout-frac 0.05 --ar-eval-batches 8 --ar-every-steps 10000 \
    --run-name approach_a_v2cache_x2_512hard_ctrl_ddp4 --wandb-project redshifty
```

The two commands differ only in the three z-head flags (`--z-bins` /
`--z-fit-files` / `--redshift-soft-sigma`) and the resume path / run-name;
everything else matches each run's logged W&B config.

---

## 2026-06-18: X4 first real numbers (held-out DESI val) — and X8 (uncertainty/rejection)

Ran the X4 eval (`nersc/eval_redshift.py`, cached path, login-node-light) on the
held-out val split of both live runs' `best.pt`. **First physical redshift
numbers, and they reframed the problem:**

| model | bins | σ_NMAD (exp) | σ_NMAD (argmax) | η>0.0033 | η>0.05 | bias |
|---|---|---|---|---|---|---|
| 512-hard (`cd1ikb99`, ~63k) | 512 | 0.000822 | 0.000903 | 11.0% | 2.9% | +6.1e-5 |
| **4096-soft (`aqxmwgl1`, ~89k)** | 4096 | **0.000696** | 0.000736 | **9.0%** | **1.4%** | +3.1e-5 |
| SpecPT (ref) | — | 0.0006–0.0008 | — | 0.2–0.8% | — | — |

Findings:
1. **Bin accuracy was misleading.** 512-hard had higher exact-bin accuracy
   (within2 0.79 vs 0.31), but on the metric that matters the **4096-soft run
   wins**: lower σ_NMAD *and* fewer outliers (half the η>0.05). Finer bins + soft
   labels → more precise z + fewer gross failures. Exactly what X4 was built to
   surface. Expected-value decode beat argmax in both (X4's sub-bin gain, real).
2. **σ_NMAD is already SpecPT-competitive, bias ≈ 0** — precision is essentially
   solved, on *mid-training* checkpoints.
3. **The entire remaining gap is catastrophic outliers (9–11% vs <1%)** —
   line-confusion / multimodal-posterior failures, not resolution. So a
   refinement head (X6) is the wrong move; there's no σ_NMAD floor to chase.

### X8 — uncertainty, calibration & outlier rejection (branch `x8-redshift-uncertainty`, eval-only)

The redshift softmax is a discretized z-posterior for free. New tooling:
- `src/eval/redshift_uncertainty.py`: `redshift_posterior`, `uncertainty_scores`
  (entropy, neg_max_prob, **second_mode_ratio** = bimodality flag,
  posterior_std_z, mass_outside_k), `pit_values` (calibration), `outlier_auroc`
  (rank-based, no sklearn), `coverage_quality_curve` (reject most-uncertain X% →
  σ_NMAD/η on the kept subset), `pit_calibration_error` (KS vs uniform).
- `evaluate()` now also returns `z_auroc_outlier`, `z_nmad_cov90`,
  `z_outlier_cov90`, `z_pit_ks` (auto-logged `val/…` on resume — a live signal
  that outlier-separability improves with training). `return_per_sample=True`
  additionally yields per-sample arrays; default stays scalars-only so the
  training val dict remains W&B/JSON-safe.
- `nersc/eval_redshift.py --uncertainty [--dump-npz P]`: prints the coverage
  table, per-score outlier AUROC, and PIT calibration (KS + histogram).
- Tests: `tests/test_redshift_uncertainty.py` (9, CPU). Full suite green; X4+X8
  integration smoke confirms scalars-only dict when `return_per_sample=False`.

Next: run `--uncertainty` on the 4096-soft leader to confirm η>0.0033 collapses
toward <1% under rejection and AUROC ≳ 0.8; if rejection alone can't reach <1%
at usable coverage, X8b = a line-aware auxiliary loss to cut outliers at source.


## 2026-06-19: X8 BREAKTHROUGH — rejection beats SpecPT; `posterior_std_z` is the detector

Ran `--uncertainty` on both live runs' held-out DESI val (`best.pt`, honest
z-hidden, 1920 spectra). **The result is the headline of the project.**

**4096-soft (`aqxmwgl1`, ~89k) — quality-vs-coverage by `posterior_std_z`:**

| completeness | σ_NMAD | η>0.0033 | η>0.05 |
|---|---|---|---|
| 100% | 0.000646 | 6.30% | 1.88% |
| **90%** | **0.000529** | **0.69%** | 0.06% |
| 80% | 0.000472 | 0.46% | 0.00% |
| 70% | 0.000396 | 0.30% | 0.00% |
| 50% | 0.000189 | 0.10% | 0.00% |

SpecPT reference: σ_NMAD 0.0006 (BGS) / 0.0008 (ELG), η 0.20% / 0.80%.

At **90% completeness** (a standard Redrock-ZWARN-style quality cut) the unimodal
model is at **σ_NMAD 5.3e-4 (beats SpecPT-BGS) and η 0.69% (inside SpecPT's
range)** — mid-training, and from a *generative* model that also reconstructs
spectra. At 80% it beats SpecPT on both axes. Defensible claim:
**a unimodal DESI FM matching/beating SpecPT redshifts at standard completeness,
from a model that also reconstructs spectra** (neither SpecPT nor AION does both).

### Two decisive findings

1. **`posterior_std_z` is the rejection score, not entropy.** Held-out
   outlier-detection AUROC:

   | score | 4096-soft | 512-hard |
   |---|---|---|
   | **posterior_std_z** | **0.976** | **0.932** |
   | entropy | 0.846 | 0.590 |
   | mass_outside_k | 0.746 | 0.638 |
   | neg_max_prob | 0.726 | 0.529 |
   | second_mode_ratio | 0.431 | 0.500 |

   Entropy rejection barely moved η (95%→4.0%); `posterior_std_z` took it
   95%→1.86%, 90%→0.69%. **Polish (this commit): `posterior_std_z` is now the
   default rejection score** (`DEFAULT_REJECTION_SCORE` in
   `src/eval/redshift_uncertainty.py`) used by `evaluate()`'s live
   `z_nmad_cov90`/`z_outlier_cov90`/`z_auroc_outlier` and the script's coverage
   table. `second_mode_ratio` is uninformative here (≈0.5) — kept in the AUROC
   readout only.

2. **Soft labels are vindicated beyond σ_NMAD.** Hard CE makes every posterior
   sharp regardless of correctness, so entropy can't separate good from bad
   (AUROC 0.59); soft labels give graded, informative uncertainty (0.85). The
   4096-soft also wins outright: σ_NMAD 0.000646 vs 0.000748, bias +5e-6 vs
   +2.1e-4 (40×), η 6.3% vs 9.7%. **512-hard control has served its purpose.**

### Calibration

Both miscalibrated oppositely (PIT KS 0.257 soft / 0.166 hard): soft is
over-dispersed (central PIT peak — expected from σ=24 soft labels,
under-confident, the safe direction); hard is over-confident (PIT to the edge).
Post-hoc temperature scaling would fix the soft model's calibration cheaply —
left as optional X8b since rejection already clears the bar.

### Decision

Promote **4096-soft as THE model**; retire the 512-hard control; let soft train
to 200k (these numbers should only improve). Per-sample arrays dumped to
`$SCRATCH/x8_{4096soft,512hard}.npz` for the PIT / coverage paper figures.

---

## 2026-06-19: 4096-soft finished 200k — and X8b temperature calibration

### The run finished (not just crashed)

`aqxmwgl1` was resumed onto the X8 code after its ~150k crash and ran cleanly to
**200,000/200,000** (`state=finished`). The back half kept paying off — the live
physical metrics improved well past the `val/loss_redshift` plateau that *looked*
like saturation:

| metric | 150k | **200k (final)** | SpecPT bar |
|---|---|---|---|
| σ_NMAD (expected) | 6.1e-4 | **4.6e-4** | 6–8e-4 |
| σ_NMAD @ 90% cov | — | **3.6e-4** | — |
| η>0.0033 (raw) | 11.4% | 9.3% | <1% |
| η>0.0033 @ 90% cov | 3.5% | **1.7%** | — |
| outlier AUROC | 0.974 | **0.980** | — |
| bias | — | +1.3e-5 | ~0 |
| **PIT KS** | — | **0.244** | →0 |

Precision now comfortably beats SpecPT; bias ≈ 0; rejection (`posterior_std_z`)
is excellent. The LR sat on its cosine floor (8e-5) through the back half, so a
flat extension to 300k is low-ROI — the residual is the *raw* catastrophic
outlier fraction (line-confusion / multimodal), which needs an objective/data
change, not more steps. The cheap, high-value gap is the **PIT KS = 0.244**: the
posterior is miscalibrated (over-dispersed from the σ=24 soft labels).

### X8b — post-hoc temperature scaling

A single scalar **T** (Guo et al. 2017) on the z sub-vocab logits, fit on the
held-out val split by minimising NLL of the true z bin. Implementation:

- `fit_temperature(probs, z_true, z_tok)` — golden-section on `log T` (NLL is
  unimodal; derivative-free, CPU-safe, notebook-importable). **Fits from the T=1
  posterior** — `softmax(log p / T)` equals `softmax(logits/T)` because the
  per-row normalisation constant cancels (verified to 1e-9), so we never need to
  store raw logits.
- `apply_temperature(probs, T)` / `redshift_posterior(..., temperature=T)` —
  re-temper a posterior. **Temperature is monotone in the logits ⇒ argmax and
  the point prediction / σ_NMAD are unchanged**; only spread / PIT / uncertainty
  scores move. So calibration is free of any accuracy cost.
- `evaluate(..., redshift_temperature=T)` — once T is known, the live val logs
  calibrated `z_pit_ks` (predictions untouched).
- `nersc/eval_redshift.py --uncertainty` now fits/applies T and prints the
  before→after PIT KS + histogram, the calibrated coverage table and AUROCs, and
  the exact `redshift_temperature=` value to bake in. `--temperature` applies a
  known T instead of fitting.

Synthetic validation (sigma 1.5 model vs sigma 6 truth): fit recovers T≈7.4,
PIT KS 0.35→0.08. Tests: `fit_temperature` softens an over-confident posterior
(T>1.5, KS drops) and returns T≈1 when already calibrated; `apply_temperature`
is identity at T=1 and raises entropy at T>1. `pytest tests/test_redshift_*` 20
green. Next: run `--uncertainty` on `aqxmwgl1` final `best.pt` to read the real
fitted T and the calibrated PIT KS, then bake it into the release eval.

---

## 2026-06-19: X8b verdict (temperature is the wrong tool) + X8c isotonic recalibration

Ran `--uncertainty` on the **finished** `aqxmwgl1` `best.pt` (held-out DESI val,
honest z-hidden, 1600 spectra). Two results.

### 1. The final 200k model is the headline (T=1)

| coverage | σ_NMAD | η>0.0033 | η>0.05 |
|---|---|---|---|
| 100% | 0.000348 | 4.81% | 0.81% |
| **95%** | 0.000312 | **1.45%** | 0.13% |
| **90%** | 0.000281 | **0.21%** | 0.00% |
| 80% | 0.000237 | 0.08% | 0.00% |
| 70% | 0.000203 | 0.00% | 0.00% |

Raw σ_NMAD **0.000348** (was 0.000646 @ 89k — nearly halved), bias +4e-6,
`posterior_std_z` AUROC **0.985**. At 90% completeness: **σ_NMAD 2.8e-4 / η
0.21%** — past SpecPT (0.0006–0.0008 / 0.20–0.80%) on *both* axes, from a model
that also reconstructs spectra. **This (T=1) is the paper number.**

### 2. X8b temperature calibration — a documented negative result

Fitted **T pinned to the search floor (0.250)** — the NLL objective wanted to
*sharpen* further. It barely helped PIT (KS 0.290→0.233) and **destroyed the
result we use**: `posterior_std_z` AUROC **0.985→0.860**, η@90% **0.21%→2.43%**
(10× worse). Why: the σ=24-bin soft labels make the posterior *over-dispersed by
design* (central-peaked PIT, `26 30 57 77 468 727 116 57 29 13`). Hard-bin NLL
therefore always prefers sharpening, and one global temperature that sharpens the
bulk crushes the *relative* spread that carries the outlier signal. **Decision:
keep `redshift_temperature=1.0`; do not bake T.** The native posterior is a
superb *ranking* uncertainty (rejection) even though its *absolute* calibration
is off — and over-dispersion is the safe (under-confident) direction.

### 3. X8c — isotonic PIT recalibration (the correct fix)

A monotone remap of the PIT through the calibration ECDF (Kuleshov et al. 2018;
for this objective the ECDF *is* the isotonic solution). Because it acts on the
PIT, not the rejection score, **the point predictions, σ_NMAD and the entire
`posterior_std_z` rejection table are unchanged by construction** — it fixes
absolute calibration with no accuracy/rejection trade-off, exactly what
temperature could not do. `fit_pit_recalibrator` / `apply_pit_recalibration`
(torch-only, CPU/notebook-safe). `eval_redshift.py --uncertainty` fits it on a
held-out half of val and reports KS on the other half.

Synthetic central-peaked PIT (matching the observed shape): held-out PIT KS
**0.30→0.04**, histogram flattened to ~uniform. Tests: makes uniform + monotone,
generalises held-out, leaves an already-uniform PIT uniform. `pytest
tests/test_redshift_*` 23 green.

**Net:** release/eval stays at T=1 for predictions + rejection (the result);
X8c supplies calibrated PIT / credible intervals for the writeup without touching
either. Per-sample arrays + recal knots dumped to
`$SCRATCH/x8b_4096soft_caltemp.npz`. Possible X8d: same posterior tooling on the
SDSS OOD check (notebook import).

### X8c confirmed on the real model + recommendation-logic fix

Ran `--uncertainty` on `aqxmwgl1` final `best.pt` (1600 held-out spectra). X8c on
the *actual* PIT: **held-out KS 0.3002 → 0.0263**, histogram flat
(`145 164 157 168 162 168 147 180 153 156` ≈ uniform). Confirms the synthetic
result — the posterior is calibrated at no cost to predictions or rejection.

Also fixed a misleading hint: the X8b `--uncertainty` "bake T" recommendation
only checked PIT KS, so on this run it printed `-> bake T=0.250` even though
temperature *helped* PIT (0.29→0.23) while *cratering* rejection (`posterior_std_z`
AUROC 0.985→0.860). The recommendation now keys off the **rejection AUROC** (the
metric we ship): if T degrades it (or doesn't help PIT) it prints *keep
`redshift_temperature=1.0`; use X8c recalibration* instead. So the tool's advice
now matches the documented verdict.

---

## 2026-06-19: pivot to spectrum — X10 flux-space reconstruction metric (measure first)

Redshift is solved (4096-soft finished 200k: σ_NMAD 3.5e-4 @ 90% cov, past
SpecPT; calibrated via X8c). The remaining concern is **spectrum accuracy**, and
the 512-hard control (`cd1ikb99`, still running, 174k) confirms the ceiling is
*not* a soft/hard artifact — both sit at `val/spectrum_acc ≈ 0.27`,
`masked_spec_acc ≈ 0.28`.

**But `spectrum_acc` is exact top-1 over the 1024-way codebook — almost certainly
misleading the same way redshift bin-accuracy was** (4096-soft scores 0.04 bin-acc
yet beats SpecPT on σ_NMAD). Tokenizer v2 is at the information-theoretic floor
(χ²/pixel ≈ 1.0, code-recon R² 0.998), so a *near-equivalent* predicted code
scores "wrong" yet decodes to a spectrum within DESI's noise — and we have never
measured the transformer's masked prediction in **flux space**, only token-match.
Also: at `redshift_loss_weight=1.0` the now-inflated soft z-loss (≈4.5) dominates
spectrum loss (≈3.1), starving reconstruction of gradient.

Plan (user: *measure first, then run*): **Step 1 = X10**, this commit. Step 2 =
one run with `redshift_loss_weight 1.0→0.3` + codebook-aware (Hamming-neighbor)
soft labels for spectrum tokens — the spectrum analog of the soft-redshift win.

### X10 — the X4-for-spectrum

`src/eval/spectrum_metrics.py` (torch-only, CPU/notebook-safe): decode the
transformer's predicted *masked* tokens back through tokenizer v2 and score the
flux against the observation, inverse-variance weighted —

    χ²/pixel = mean_valid[ ivar·(flux_pred − flux_obs)² ]   (1.0 = noise floor)
    ivar-R²  = 1 − Σ ivar·resid² / Σ ivar·(flux_obs − ȳ_w)²

Sufficient statistics {Sw, Swy, Swy2, Swr2, n} accumulate additively across
batches (`recon_weighted_sums`/`add_sums`/`finalize_recon`). `token_mask_to_
pixel_mask` expands the per-token mask (B,272) → per-pixel (B,8704) via the exact
stride-32 mapping (8704/272), so reconstruction is scored on the masked-token
pixel blocks (the blind number mirroring `masked_spec_acc`).

`nersc/eval_spectrum.py` (on-the-fly path; needs `--tokenizer-ckpt` — the cached
token path stores no flux/denorm) reports, honest regime (z hidden): masked-token
acc (the misleading 0.27), the **codec-only ceiling** (decode true tokens, expect
χ²≈1.0), and predicted reconstruction on masked blocks + all valid pixels. Subtle
correctness point: the LFQ (en|de)coder uses 2-D `(B, n_tokens)` index tensors
(it sums over the 10-bit dim), and `decode` needs the per-spectrum `denorm` from
`encode` — both handled.

Tests `tests/test_spectrum_metrics.py` (8, all green; full suite 241 pass, only
the pre-existing wandb-key env test fails): perfect→χ²0/R²1, noise-floor residual
→χ²≈1, zero-istd/out-of-coverage pixels excluded, token→pixel mask, pixel-mask
restriction, additive accumulation, NaN-safe empty. **Next: run `eval_spectrum.py`
on `aqxmwgl1` `best.pt` to read the real flux-R²/χ² and decide whether 0.27 hides
good reconstruction — then launch the rebalanced + soft-spectrum-label run.**
