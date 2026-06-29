# `nersc/zr/` — redshift eval & plotting harness (recovered from NERSC scratch)

Scratch scripts that lived under `$SCRATCH/zr/` on Perlmutter (and a few staged in
`/tmp` locally) used to evaluate and visualize blind-redshift performance of the
Spectrum-FM transformers (V2 / V3 / V4) on DESI DR1 and SDSS. Collected here so they
are version-controlled. They expect to run on NERSC with `module load pytorch` and the
repo on `PYTHONPATH`; paths inside are hard-coded to `$SCRATCH` (`/pscratch/sd/j/joe2k`).

## Blind-redshift eval (z hidden, full spectrum) → `zr_<tag>_<split>.npz`
- `zr_eval.py` — core blind-z evaluator; writes `z_pred`/`z_true` per split (`full`/`val`/`sdss`).
- `zr_merge.py` — merge sharded eval outputs.
- `patch_zr_eval.py` — one-off patch applied to `zr_eval.py`.
- `run_all2.sh`, `run_all4.sh`, `run_model.sh`, `run_worker.sh`, `run_val.sh`,
  `run_val4.sh`, `val_workers.sh` — multi-GPU launch/shard wrappers for the above.

## Per-object-type eval (BGS/LRG/ELG/QSO) → `type_<tag>_r*.npz`
- `zr_type_eval.py` — **per-type harness**: one pass = blind-z (full spectrum, z hidden)
  + z-hidden 50%-masked recon; reads spectype per shard. (Used for the V3/V4 per-type tables.)
- `zr_type_analyze.py` — aggregate `type_*.npz` into the per-type σ_NMAD / R² / η / flux table.
- `zr_type_pooled.py`, `zr_type_table_full.py`, `zr_type_outliers.py` — pooled stats,
  full table render, and per-type outlier breakdown variants.
- `build_typemap.py` — build `typemap.npz` (targetid → BGS/LRG/ELG/QSO from DESI target bits).
- `add_types.py` — attach spectype/class columns to eval npz.
- `run_all_type.sh`, `run_type_worker.sh` — 4-GPU launch wrappers.

## Plots
- `joint_val.py` / `joint_train.py` — **parity + count-marginals** "joint" plot
  (RdBu Δz/(1+z) coloring, true-z top hist, predicted-z right hist) used on deck slides
  42/43. `joint_val.py` is the generalized variant (any npz/label/split); it produced
  `plots/plot_joint_v4_val.png`.
- `zr_plot.py` — 2×2 (V2/V3) parity + error-histogram figure per split.
- `zr_plot6.py` — single-model parity + error-hist per split.

## Outlier analysis
- `decompose_outliers.py` — decompose catastrophic-outlier population by type/cause.

## SDSS few-shot fine-tuning helpers (OOD)
- `sdss_smoke.sh`, `sdss_transition.sh`, `sdss_verify.sh`, `sdss_done_poll.sh`,
  `sdss_ft_check{,2,3}.sh` — launch / poll / verify SDSS few-shot fine-tuning runs.
- `select_examples.py`, `run_select.sh`, `run_select_5000.sh` — pick representative
  spectra for the qualitative montages.
- `render_montages.py`, `render_montages_5000.sh` — render the per-bucket spectrum montages.

## Train-shard / monitors (one-off)
- `run_train4.sh`, `run_train_wrap.sh`, `zr_train_shard.sh` — sharded train-split eval wrappers.
- `v4_monitor.sh`, `dec_monitor.sh` — poll loops that watched the V4 and decoupled-mask runs.

> Polished, actively-maintained launchers live one level up in `nersc/` (e.g.
> `decab_analyze.py`, `run_decab_eval.sh`, `finetune_zhead.py`). The files here are the
> as-run scratch originals, kept for provenance/reproducibility.
