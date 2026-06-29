#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train SpectrumTransformer on DR1 with a pretrained tokenizer.

Thin CLI wrapper. All non-NERSC-specific training logic lives in
`src/training/` so it's reusable from notebooks / tests / `scripts/`
without dragging in DR1 paths. This file only handles:
  - argparse + smoke override
  - DR1 manifest loading (NERSC-specific filesystem paths)
  - SpectrumTokenizer load + RedshiftTokenizer fit
  - Healpix-level train/val partitioning of records
  - The actual training loop using helpers from src/training/

Usage (from inside a SLURM job; see train_transformer.slurm):

    python nersc/train_transformer.py \\
        --manifest $SCRATCH/deepsrch/manifests/dr1_smoke.jsonl \\
        --tokenizer-ckpt $SCRATCH/deepsrch/checkpoints/<run>/best.pt \\
        --approach a \\
        --steps 50000 \\
        --batch-size 8 \\
        --amp
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# Core logic — moved to src/training/ for reusability.
from src.models.transformer import SpectrumTransformer, vocab_size_for_z_bins  # noqa: E402
from src.tokenizers.redshift import RedshiftTokenizer  # noqa: E402
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.training.data_split import split_records_by_healpix  # noqa: E402
from src.training.eval import evaluate, evaluate_ar  # noqa: E402
from src.training.sequences import lr_at, tokenize_and_build  # noqa: E402
from src.training.utils import (  # noqa: E402
    compute_loss_breakdown,
    compute_masked_metrics,
    compute_metrics,
)
from src.training.wandb_util import init_wandb, log_model_artifact, wfinish, wlog  # noqa: E402

# NERSC-specific DR1 loaders.
from dr1_dataset import (  # noqa: E402
    DR1IndexedDataset,
    collate_dr1_skip_none,
    load_manifest,
)
from dr1_tokenized_dataset import (  # noqa: E402
    DR1CachedTokenDataset,
    collate_cached_skip_none,
    collect_redshifts,
    collect_redshifts_from_cache,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train SpectrumTransformer on DR1")
    # Data
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tokenizer-ckpt", type=Path, default=None,
                   help="Pretrained SpectrumTokenizer .pt (best.pt or final.pt). "
                        "Required for on-the-fly tokenization; unused with "
                        "--tokenized-dir (the encoder isn't loaded).")
    p.add_argument("--tokenized-dir", type=Path, default=None,
                   help="Directory of pre-tokenized shards from "
                        "pretokenize_corpus.py (X1). When set, training reads "
                        "cached token ids instead of running the ConvNeXt "
                        "encoder every step; the spectrum tokenizer is not loaded.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a full-state checkpoint (best.pt or "
                        "final.pt): restores model, optimizer, scaler, step "
                        "and best_val so training continues where it stopped. "
                        "Reuse the SAME --run-name to keep the same run dir. "
                        "Note: periodic step_*.pt are model-only and cannot "
                        "restore optimizer state.")
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--healpix-holdout-frac", type=float, default=0.05,
                   help="Fraction of HEALPIX FILES (not rows) to reserve "
                        "for validation. Avoids same-pointing leakage.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--z-fit-files", type=int, default=200,
                   help="How many redrock files to scan when fitting RedshiftTokenizer")
    p.add_argument("--z-bins", type=int, default=256,
                   help="Number of redshift quantization bins (CDF-equalized). "
                        "256 is the legacy default (v8/v9); 4096 gives "
                        "~0.001-level bin width, matching spectroscopic z "
                        "precision. Grows the vocab to 1032 + z_bins, so "
                        "checkpoints with different --z-bins are incompatible.")
    p.add_argument("--z-gaussian-range", type=float, default=3.0,
                   help="FSQ half-range for the redshift tokenizer. The top bin "
                        "decodes to the Phi(range) percentile of the fitted z, so "
                        "this sets the max emittable z (the prediction ceiling). "
                        "Default 3.0 (Phi=0.99865 -> z~2.1 on DESI); z-v2 uses 4.0 "
                        "(-> z~4.4) to lift the QSO ceiling. See RESEARCH_LOG Z1.")

    # Approach
    p.add_argument("--approach", choices=["a", "b"], required=True)

    # Model
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-encoder-layers", type=int, default=6)
    p.add_argument("--n-decoder-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.1)

    # Optim
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                   help="Autocast dtype when --amp is set. bf16 (default) has "
                        "fp32 dynamic range so it cannot overflow to inf/NaN the "
                        "way fp16 log_softmax over a large vocab does (the v10 "
                        "NaN class); GradScaler is auto-disabled for bf16. fp16 "
                        "retained for back-compat (re-enables GradScaler).")
    p.add_argument("--redshift-loss-weight", type=float, default=1.0,
                   help="Multiplier on the position-0 (redshift) loss term. "
                        "1.0 is the validated v9 value: it balances redshift "
                        "against the 272 spectrum tokens. Higher (10, 50) puts "
                        ">=95%% of the loss mass on z, which starves spectrum and "
                        "reproduces the v8 stuck/copy-artifact regime.")
    p.add_argument("--encoder-mask-ratio", type=float, default=0.15,
                   help="Fraction of encoder spectrum positions to replace "
                        "with [MASK]. BERT-style. 0.0 disables; 0.15 is "
                        "BERT canonical. Forces honest spectrum reconstruction. "
                        "Ignored when --mask-ratio-min/--mask-ratio-max are set "
                        "(the ratio is then sampled per step from that range).")
    p.add_argument("--mask-ratio-min", type=float, default=None,
                   help="Lower bound for per-step variable encoder mask ratio. "
                        "When both --mask-ratio-min and --mask-ratio-max are "
                        "set, each step samples ratio ~ Uniform[min, max] "
                        "(MAE-style schedule). Unset = fixed "
                        "--encoder-mask-ratio. Try 0.3 with --mask-ratio-max 0.7.")
    p.add_argument("--mask-ratio-max", type=float, default=None,
                   help="Upper bound for per-step variable encoder mask ratio. "
                        "See --mask-ratio-min.")
    p.add_argument("--masked-targets-only", action="store_true",
                   help="MAE-style spectrum objective: supervise reconstruction "
                        "ONLY at encoder-masked positions (unmasked spectrum "
                        "targets become ignore_index). Stops the decoder earning "
                        "reward by copying visible tokens, and makes the logged "
                        "spectrum_acc the honest masked number. Redshift stays "
                        "supervised. Needs encoder masking > 0. Default off "
                        "(supervise every spectrum position = legacy behavior).")
    p.add_argument("--redshift-mask-ratio", type=float, default=0.5,
                   help="Per-sample probability of replacing the encoder's "
                        "redshift token with [REDMASK] (Approach A only). "
                        "Redshift conditioning dropout: forces the model to "
                        "predict z from the spectrum instead of copying it. "
                        "0.0 disables (z always visible = copy path open); "
                        "0.5 = z hidden half the time. Eval/inference always "
                        "hide z (ratio 1.0). Ignored when --decouple-masks is set "
                        "(the mode split governs z visibility instead).")
    p.add_argument("--decouple-masks", action="store_true",
                   help="Approach A only. Replace the two INDEPENDENT masks "
                        "(which let a predict-z example also have a half-masked "
                        "spectrum) with a per-sample three-mode split: Z-task "
                        "(full spectrum, predict z), recon+z-shown, recon+z-hidden. "
                        "Redshift is supervised ONLY in the Z-task, always from a "
                        "full spectrum. Needs --masked-targets-only and an encoder "
                        "mask > 0. Default off (legacy independent masks).")
    p.add_argument("--blind-z-frac", type=float, default=0.5,
                   help="With --decouple-masks: fraction of samples that are the "
                        "Z-task (full spectrum, predict z). The rest are recon.")
    p.add_argument("--recon-z-shown-frac", type=float, default=0.5,
                   help="With --decouple-masks: within the recon samples, fraction "
                        "that SHOW the true z to the encoder. 0.0 => z hidden "
                        "everywhere (pure 'always predict z, never given' FM "
                        "objective; recon eval is then exactly in-distribution).")
    p.add_argument("--two-pass-val", action="store_true",
                   help="Log validation as TWO clean passes — val_z/* (full "
                        "spectrum, z hidden) and val_recon/* (spectrum masked, z "
                        "hidden) — instead of one half-masked pass. best.pt is "
                        "selected on the val_z (redshift) loss. Implied by "
                        "--decouple-masks; set it on the independent-mask control "
                        "arm too so the two arms log comparable val_z numbers.")
    p.add_argument("--redshift-soft-sigma", type=float, default=0.0,
                   help="If >0, the redshift (position-0) loss uses Gaussian "
                        "soft labels with this std (in bins) instead of hard "
                        "256-way cross-entropy. Gives partial credit / an "
                        "ordinal gradient toward the right z neighborhood. "
                        "0.0 keeps hard CE; ~1.5 is a good starting point.")
    p.add_argument("--ar-eval-batches", type=int, default=4,
                   help="Number of batches to run through autoregressive "
                        "eval at end-of-run and on the AR cadence.")
    p.add_argument("--ar-every-steps", type=int, default=10000,
                   help="Run the (expensive, rank-0-only) autoregressive eval "
                        "every N steps, decoupled from best-val. Redshift is "
                        "target position 0, so the cheap teacher-forced "
                        "val/redshift_acc already equals the AR redshift number "
                        "over more samples; AR's unique value is honest "
                        "free-running spectrum generation + the z-given "
                        "copy-leak probe, which don't need to run on every best.")

    # Logging
    p.add_argument("--run-name", type=str, default="approach_a")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--wandb-project", type=str, default="redshifty")
    p.add_argument("--wandb-run-id", type=str, default=None,
                   help="Continue an existing W&B run by id (resume='allow'). "
                        "Overrides the id saved in --resume's checkpoint. "
                        "Leave unset to auto-continue the checkpoint's run.")
    p.add_argument("--push-wandb-artifact", action="store_true", default=True,
                   help="Upload best.pt to wandb as an Artifact on each best "
                        "update (best only; final is never pushed). "
                        "Disable with --no-push-wandb-artifact.")
    p.add_argument("--no-push-wandb-artifact", action="store_false",
                   dest="push_wandb_artifact")
    p.add_argument("--scratch-out", type=Path,
                   default=Path(os.environ.get("SCRATCH", "/tmp")) / "deepsrch")
    p.add_argument("--cfs-out", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)

    p.add_argument("--smoke", action="store_true",
                   help="Tiny config: 100 steps, 200 spectra, smaller model")
    return p.parse_args()


def _records_to_shards(records, tokenized_dir):
    """Map manifest records to existing pre-tokenized shard paths.

    Same naming as pretokenize_corpus.shard_name (survey_program_healpix.npz).
    Silently drops records whose shard hasn't been written yet, so a partially
    pre-tokenized corpus still trains on whatever is cached.
    """
    out = []
    for r in records:
        p = Path(tokenized_dir) / f"{r['survey']}_{r['program']}_{r['healpix']}.npz"
        if p.exists():
            out.append(p)
    return out


def main():
    args = parse_args()
    if args.smoke:
        args.steps = 100
        args.max_spectra = 200
        args.batch_size = min(args.batch_size, 4)
        args.val_every = 50
        args.save_every = 100
        args.log_every = 10
        args.num_workers = 0
        args.warmup = 20
        args.d_model = 256
        args.n_encoder_layers = 2
        args.n_decoder_layers = 2
        args.n_heads = 8
        args.z_fit_files = min(args.z_fit_files, 5)
        args.ar_eval_batches = 1
        args.ar_every_steps = min(args.ar_every_steps, 50)

    # Variable mask-ratio schedule: both bounds must be given together and form
    # a valid sub-interval of [0, 1]. When unset, training uses the fixed
    # --encoder-mask-ratio (unchanged behavior).
    args.variable_mask = args.mask_ratio_min is not None or args.mask_ratio_max is not None
    if args.variable_mask:
        if args.mask_ratio_min is None or args.mask_ratio_max is None:
            raise ValueError(
                "--mask-ratio-min and --mask-ratio-max must be set together"
            )
        if not 0.0 <= args.mask_ratio_min <= args.mask_ratio_max <= 1.0:
            raise ValueError(
                f"need 0 <= mask-ratio-min ({args.mask_ratio_min}) <= "
                f"mask-ratio-max ({args.mask_ratio_max}) <= 1"
            )

    # Decoupled three-mode masking presupposes Approach A (encoder z token),
    # MAE supervision (so Z-task rows supervise nothing in the spectrum), and a
    # non-zero encoder mask (so the recon modes have masked positions to learn).
    if args.decouple_masks:
        if args.approach != "a":
            raise ValueError("--decouple-masks requires --approach a")
        if not args.masked_targets_only:
            raise ValueError("--decouple-masks requires --masked-targets-only")
        eff_max_mask = args.mask_ratio_max if args.variable_mask else args.encoder_mask_ratio
        if not eff_max_mask or eff_max_mask <= 0.0:
            raise ValueError(
                "--decouple-masks needs an encoder mask > 0 "
                "(set --encoder-mask-ratio or --mask-ratio-min/max)"
            )

    # Detect DDP: either explicit RANK env var (from SLURM script exports)
    # or SLURM's native SLURM_PROCID (from interactive srun).
    is_distributed = "RANK" in os.environ or "SLURM_PROCID" in os.environ
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
        # PyTorch env:// rendezvous requires RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        # When srun uses --gpus-per-task=N, each task sees only its
        # bound GPU(s) as cuda:0.., so local_rank may exceed device_count.
        # When all GPUs are visible to every task, local_rank is the
        # actual device index. Pick whichever applies.
        cuda_idx = local_rank if local_rank < torch.cuda.device_count() else 0
        torch.cuda.set_device(cuda_idx)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{cuda_idx}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_desc = (f"U[{args.mask_ratio_min},{args.mask_ratio_max}]"
                 if args.variable_mask else f"{args.encoder_mask_ratio}")
    print(f"[setup] rank={rank}/{world_size} device={device} approach={args.approach} "
          f"steps={args.steps} mask_ratio={mask_desc} "
          f"masked_targets_only={args.masked_targets_only} "
          f"redshift_mask_ratio={args.redshift_mask_ratio} "
          f"redshift_weight={args.redshift_loss_weight} "
          f"amp={args.amp}/{args.amp_dtype}")
    run_dir = args.scratch_out / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    with (run_dir / "config.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, f, indent=2)

    # Manifest
    print(f"[data] loading manifest {args.manifest}")
    records = load_manifest(args.manifest)
    print(f"[data] {len(records)} healpix records")

    # Healpix-level train/val split — no same-pointing leakage.
    train_records, val_records = split_records_by_healpix(
        records, holdout_frac=args.healpix_holdout_frac, seed=args.seed,
    )
    print(f"[data] healpix split: {len(train_records)} train, "
          f"{len(val_records)} val (frac={args.healpix_holdout_frac})")

    cached = args.tokenized_dir is not None

    # Pretrained spectrum tokenizer. On the cached path the encoder is never
    # loaded — tokens come from the shards, so there's no ConvNeXt in memory and
    # no per-step encode (the whole point of X1).
    if cached:
        spec_tok = None
        print(f"[tok] cached tokens from {args.tokenized_dir} — encoder not loaded")
    else:
        if args.tokenizer_ckpt is None:
            print("ERROR: --tokenizer-ckpt is required without --tokenized-dir",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[tok] loading spectrum tokenizer {args.tokenizer_ckpt}")
        spec_tok = SpectrumTokenizer().to(device)
        ckpt = torch.load(args.tokenizer_ckpt, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        spec_tok.load_state_dict(sd)
        spec_tok.eval()
        for p in spec_tok.parameters():
            p.requires_grad_(False)

    # Fit redshift tokenizer on a sample of TRAIN redshifts (from the cache when
    # available, else by scanning redrock files).
    if cached:
        train_shards = _records_to_shards(train_records, args.tokenized_dir)
        val_shards = _records_to_shards(val_records, args.tokenized_dir)
        print(f"[tok] fitting redshift tokenizer from cached z ({len(train_shards)} shards)")
        zs = collect_redshifts_from_cache(train_shards, max_files=args.z_fit_files)
    else:
        print(f"[tok] fitting redshift tokenizer on up to {args.z_fit_files} redrock files")
        zs = collect_redshifts(train_records, max_files=args.z_fit_files)
    print(f"[tok]   gathered {len(zs)} z values, min={zs.min():.4f} max={zs.max():.4f}")
    if len(zs) < 20 * args.z_bins:
        print(f"[tok]   WARN: only {len(zs)} z values for {args.z_bins} bins "
              f"(<20 per quantile); raise --z-fit-files for stable bin edges")
    z_tok = RedshiftTokenizer(n_levels=args.z_bins, gaussian_range=args.z_gaussian_range)
    z_tok.fit(zs)
    # Bin-width report: bins are CDF-equalized (equal probability mass), so
    # width varies with the z density — verify the precision actually achieved.
    edges = z_tok.get_bin_edges()
    widths = (edges[1:] - edges[:-1]).abs()
    print(f"[tok]   z_bins={args.z_bins} bin width in z: "
          f"median={widths.median():.5f} p90={widths.quantile(0.9):.5f} "
          f"max={widths.max():.5f}")

    # Build separate datasets for each partition. The cached path reads
    # pre-tokenized shards (no FITS, no encode); both paths apply the same
    # quality cut and produce batches tokenize_and_build consumes identically.
    val_cap = None if args.max_spectra is None else max(50, args.max_spectra // 10)
    if cached:
        collate = collate_cached_skip_none
        train_ds = DR1CachedTokenDataset(
            train_shards, require_good_zwarn=True, require_nonzero_flux=True,
            max_spectra=args.max_spectra)
        val_ds = DR1CachedTokenDataset(
            val_shards, require_good_zwarn=True, require_nonzero_flux=True,
            max_spectra=val_cap)
    else:
        collate = collate_dr1_skip_none
        train_ds = DR1IndexedDataset(
            train_records, require_good_zwarn=True, require_nonzero_flux=True,
            max_spectra=args.max_spectra)
        val_ds = DR1IndexedDataset(
            val_records, require_good_zwarn=True, require_nonzero_flux=True,
            max_spectra=val_cap)
    print(f"[data] train_ds={len(train_ds)} val_ds={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    # Model — vocab grows with the redshift sub-vocab width.
    model = SpectrumTransformer(
        vocab_size=vocab_size_for_z_bins(args.z_bins),
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,} (~{n_params/1e6:.1f}M)")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # bf16 has fp32 dynamic range and needs no loss scaling; only fp16 uses the
    # GradScaler. Disabling it for bf16 also means old fp16 scaler state is not
    # loaded on a bf16 resume (see the resume block).
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = args.amp and args.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Optional resume: restore model/optim/scaler/step/best_val from a
    # full-state checkpoint. Every rank loads the same file, keeping DDP
    # replicas identical.
    resume_step = 0
    resume_best = float("inf")
    resume_wandb_id = None
    if args.resume is not None:
        rckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ck_bins = (rckpt.get("z_tokenizer") or {}).get("n_levels")
        if ck_bins is not None and int(ck_bins) != args.z_bins:
            raise ValueError(
                f"--resume checkpoint was trained with {ck_bins} redshift bins "
                f"but --z-bins={args.z_bins}; the vocab sizes differ. "
                f"Resume with --z-bins {ck_bins} or start a fresh run.")
        (model.module if is_distributed else model).load_state_dict(rckpt["model"])
        if "optim" in rckpt:
            optim.load_state_dict(rckpt["optim"])
        if "scaler" in rckpt and use_scaler:
            scaler.load_state_dict(rckpt["scaler"])
        resume_step = int(rckpt.get("step", 0))
        resume_best = float(rckpt.get("val_loss", float("inf")))
        resume_wandb_id = rckpt.get("wandb_run_id")
        if rank == 0:
            print(f"[resume] {args.resume} -> step={resume_step} "
                  f"best_val={resume_best:.4f} wandb_id={resume_wandb_id}")

    # Continue the same W&B run on resume: explicit --wandb-run-id wins,
    # else the id saved in the checkpoint.
    wandb_run_id = args.wandb_run_id or resume_wandb_id

    # Wandb init — rank 0 only
    wandb_run = None
    if rank == 0:
        wandb_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        wandb_config.update({
            "n_params": n_params,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_manifest_records": len(records),
            "n_train_records": len(train_records),
            "n_val_records": len(val_records),
        })
        wandb_dir = args.scratch_out / "wandb" / args.run_name
        wandb_run = init_wandb(
            mode=args.wandb_mode,
            project=args.wandb_project,
            run_name=args.run_name,
            config=wandb_config,
            out_dir=wandb_dir,
            run_id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
        )

    # W&B run id to persist in checkpoints so later resumes continue this run.
    wandb_id_to_save = wandb_run.id if wandb_run is not None else wandb_run_id

    step = resume_step
    epoch = 0
    best_val = resume_best
    nonfinite_skips = 0       # total optimizer steps skipped on a non-finite loss
    consecutive_skips = 0     # run of back-to-back skips (poison detector)
    MAX_CONSEC_SKIPS = 100    # abort if the weights are unrecoverably NaN
    t0 = time.time()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)
    model.train()

    while step < args.steps:
        try:
            raw = next(train_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            raw = next(train_iter)
        if raw is None:
            continue

        for g_ in optim.param_groups:
            g_["lr"] = lr_at(step, args.lr, args.warmup, args.steps)

        # Per-step encoder mask ratio. With --mask-ratio-min/max, sample
        # Uniform[min, max] from a step-seeded generator so the schedule is
        # deterministic and identical across resumes and DDP ranks (all ranks
        # share seed+step). Otherwise use the fixed ratio.
        if args.variable_mask:
            mr_gen = torch.Generator().manual_seed(args.seed + step)
            mask_ratio = (
                args.mask_ratio_min
                + (args.mask_ratio_max - args.mask_ratio_min)
                * torch.rand((), generator=mr_gen).item()
            )
        else:
            mask_ratio = args.encoder_mask_ratio

        enc, dec, tgt, mask_pos = tokenize_and_build(
            raw, spec_tok, z_tok, args.approach, device,
            encoder_mask_ratio=mask_ratio,
            redshift_mask_ratio=args.redshift_mask_ratio,
            mask_targets_only=args.masked_targets_only,
            decouple_masks=args.decouple_masks,
            blind_z_frac=args.blind_z_frac,
            recon_z_shown_frac=args.recon_z_shown_frac,
        )

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
            logits, loss = model(enc, dec, targets=tgt,
                                 redshift_weight=args.redshift_loss_weight,
                                 redshift_soft_sigma=args.redshift_soft_sigma)
        # Backward is called on every rank (keeps DDP's grad allreduce in
        # lockstep), then all ranks collectively decide whether the step is
        # safe to take. A single non-finite loss (overflow / pathological
        # batch) would otherwise apply NaN grads and poison the weights for the
        # rest of the run — exactly the v10 failure. The reduce makes the skip
        # decision identical on every rank so no replica desyncs.
        scaler.scale(loss).backward()
        finite = torch.isfinite(loss).all().to(torch.float32)
        if is_distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() < 1.0:
            optim.zero_grad(set_to_none=True)   # drop the poisoned grads
            nonfinite_skips += 1
            consecutive_skips += 1
            if rank == 0:
                print(f"  WARN: non-finite loss at step {step}; skipped update "
                      f"(skip #{nonfinite_skips}, {consecutive_skips} in a row)")
            if consecutive_skips >= MAX_CONSEC_SKIPS:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive non-finite losses — the "
                    f"weights are poisoned; aborting. Resume from the last "
                    f"healthy last.pt and/or lower --lr.")
            step += 1
            continue
        consecutive_skips = 0
        if args.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()

        if rank == 0 and step % args.log_every == 0:
            with torch.no_grad():
                m = compute_metrics(logits, tgt)
                b = compute_loss_breakdown(logits, tgt)
                mm = (compute_masked_metrics(logits, tgt, mask_pos)
                      if mask_pos is not None else
                      {"masked_spec_acc": float("nan"), "n_masked": 0})
            dt = time.time() - t0
            # Throughput of THIS process: steps since (re)start / elapsed.
            # Use (step - resume_step) so a resume at a high absolute step
            # doesn't inflate the rate by counting pre-resume steps over only
            # this process's wall time.
            rate = (step - resume_step + 1) / max(dt, 1e-6)
            msg = {
                "kind": "train", "step": step, "lr": optim.param_groups[0]["lr"],
                "loss": float(loss.item()),
                **m, **b,
                "masked_spec_acc": mm["masked_spec_acc"],
                "n_masked": mm["n_masked"],
                "mask_ratio": mask_ratio,
                "steps_per_sec": rate, "elapsed_s": dt,
            }
            print(f"[step {step:6d}] loss={msg['loss']:.4f} "
                  f"z_loss={b['loss_redshift']:.3f} spec_loss={b['loss_spectrum']:.3f} "
                  f"z_acc={m['redshift_acc']:.3f} spec_acc={m['spectrum_acc']:.3f} "
                  f"masked_acc={mm['masked_spec_acc']:.3f} "
                  f"{rate:.1f} step/s")
            with metrics_path.open("a") as f:
                f.write(json.dumps(msg) + "\n")
            wlog(wandb_run, {
                "train/loss": msg["loss"],
                "train/loss_redshift": b["loss_redshift"],
                "train/loss_spectrum": b["loss_spectrum"],
                "train/loss_total_unweighted": b["loss_total"],
                "train/lr": msg["lr"],
                "train/overall_acc": m["overall_acc"],
                "train/redshift_acc": m["redshift_acc"],
                "train/redshift_acc_within2": m["redshift_acc_within2"],
                "train/spectrum_acc": m["spectrum_acc"],
                "train/masked_spec_acc": mm["masked_spec_acc"],
                "train/steps_per_sec": rate,
            }, step=step)

        if rank == 0 and step > 0 and step % args.val_every == 0:
            # z is hidden (ratio 1.0) so the redshift metrics reflect the honest
            # inference-time, predict-z-from-spectrum case.
            if args.decouple_masks or args.two_pass_val:
                # Two CLEAN passes matching the decoupled objective + inference:
                #   val_z    : full spectrum, z hidden -> redshift metrics (headline)
                #   val_recon: spectrum masked, z hidden -> reconstruction metrics
                # (vs the legacy single half-masked pass, which computed the
                # redshift number from a partially-missing spectrum.)
                vz = evaluate(
                    model.module if is_distributed else model,
                    val_loader, spec_tok, z_tok, args.approach, device,
                    args.amp, args.redshift_loss_weight,
                    encoder_mask_ratio=0.0,
                    redshift_mask_ratio=1.0,
                    redshift_soft_sigma=args.redshift_soft_sigma,
                    amp_dtype=amp_dtype,
                )
                vr = evaluate(
                    model.module if is_distributed else model,
                    val_loader, spec_tok, z_tok, args.approach, device,
                    args.amp, args.redshift_loss_weight,
                    encoder_mask_ratio=args.encoder_mask_ratio,
                    redshift_mask_ratio=1.0,
                    redshift_soft_sigma=args.redshift_soft_sigma,
                    amp_dtype=amp_dtype,
                )
                model.train()
                v = vz  # headline = redshift task; drives best.pt selection below
                print(f"[val   {step:6d}] "
                      + "z:" + " ".join(f"{k}={vz[k]:.4f}" for k in vz)
                      + " | recon:" + " ".join(f"{k}={vr[k]:.4f}" for k in vr))
                with metrics_path.open("a") as f:
                    f.write(json.dumps({"kind": "val", "step": step,
                                        **{f"val_z_{k}": vv for k, vv in vz.items()},
                                        **{f"val_recon_{k}": vv for k, vv in vr.items()}}) + "\n")
                wlog(wandb_run, {**{f"val_z/{k}": vv for k, vv in vz.items()},
                                 **{f"val_recon/{k}": vv for k, vv in vr.items()}}, step=step)
            else:
                v = evaluate(
                    model.module if is_distributed else model,
                    val_loader, spec_tok, z_tok, args.approach, device,
                    args.amp, args.redshift_loss_weight,
                    encoder_mask_ratio=args.encoder_mask_ratio,
                    redshift_mask_ratio=1.0,
                    redshift_soft_sigma=args.redshift_soft_sigma,
                    amp_dtype=amp_dtype,
                )
                model.train()
                print(f"[val   {step:6d}] " + " ".join(f"{k}={v[k]:.4f}" for k in v))
                with metrics_path.open("a") as f:
                    f.write(json.dumps({"kind": "val", "step": step,
                                        **{f"val_{k}": vv for k, vv in v.items()}}) + "\n")
                wlog(wandb_run, {f"val/{k}": vv for k, vv in v.items()}, step=step)
            if v["loss"] < best_val:
                best_val = v["loss"]
                p = run_dir / "best.pt"
                torch.save({
                    "step": step,
                    "model": model.module.state_dict() if is_distributed else model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "val_loss": best_val,
                    "z_tokenizer": {
                        "sorted_z": z_tok._sorted_z.cpu(),
                        "n_levels": z_tok.n_levels,
                        "gaussian_range": z_tok.gaussian_range,
                    },
                    "tokenizer_ckpt_path": str(args.tokenizer_ckpt),
                    "approach": args.approach,
                    "encoder_mask_ratio": args.encoder_mask_ratio,
                    "redshift_mask_ratio": args.redshift_mask_ratio,
                    "redshift_soft_sigma": args.redshift_soft_sigma,
                    "wandb_run_id": wandb_id_to_save,
                }, p)
                print(f"  *** new best val_loss={best_val:.4f} -> {p}")
                if args.cfs_out is not None:
                    try:
                        args.cfs_out.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, args.cfs_out / "best.pt")
                    except OSError as e:
                        print(f"  WARN: cfs_out mirror failed ({e}); "
                              f"SCRATCH best.pt is safe at {p}")

                if args.push_wandb_artifact and wandb_run is not None:
                    log_model_artifact(
                        wandb_run, p,
                        name=f"approach_{args.approach}_{args.run_name}",
                        aliases=["best", f"step_{step}"],
                        metadata={
                            "val_loss": best_val,
                            "step": step,
                            "approach": args.approach,
                            "encoder_mask_ratio": args.encoder_mask_ratio,
                            "redshift_mask_ratio": args.redshift_mask_ratio,
                            "redshift_soft_sigma": args.redshift_soft_sigma,
                            "redshift_loss_weight": args.redshift_loss_weight,
                            "z_bins": args.z_bins,
                            "full_state": True,
                        },
                    )

        if rank == 0 and step > 0 and step % args.ar_every_steps == 0:
            # Honest autoregressive eval on its own sparse cadence, decoupled
            # from best-val. Redshift is target position 0, so the cheap
            # teacher-forced val/redshift_acc already equals the AR redshift
            # number (over more samples) — AR's unique value is honest
            # free-running SPECTRUM generation plus the z-given copy-leak probe.
            # It is rank-0-only and stalls the other ranks at the next
            # allreduce, so it runs rarely (default every 10k steps).
            ar = evaluate_ar(
                model.module if is_distributed else model,
                val_loader, spec_tok, z_tok, args.approach, device,
                max_batches=args.ar_eval_batches,
                encoder_mask_ratio=args.encoder_mask_ratio,
                redshift_mask_ratio=1.0,
            )
            # z-given AR for comparison (exposes any residual copy path).
            ar_zgiven = evaluate_ar(
                model.module if is_distributed else model,
                val_loader, spec_tok, z_tok, args.approach, device,
                max_batches=args.ar_eval_batches,
                encoder_mask_ratio=args.encoder_mask_ratio,
                redshift_mask_ratio=0.0,
            )
            model.train()
            print(f"  [ar {step:6d}] " + " ".join(f"{k}={ar[k]}" for k in ar)
                  + f" | zgiven_redshift_acc={ar_zgiven['ar_redshift_acc']}")
            with metrics_path.open("a") as f:
                f.write(json.dumps({"kind": "ar", "step": step,
                                    **{f"val_ar_{k}": vv for k, vv in ar.items()},
                                    "val_ar_redshift_acc_zgiven": ar_zgiven["ar_redshift_acc"]}) + "\n")
            wlog(wandb_run, {**{f"val_ar/{k}": vv for k, vv in ar.items()},
                             "val_ar/redshift_acc_zgiven": ar_zgiven["ar_redshift_acc"]}, step=step)

        if rank == 0 and step > 0 and step % args.save_every == 0:
            msd = model.module.state_dict() if is_distributed else model.state_dict()
            # Archival model-only snapshot (handy for picking a specific step,
            # but NOT a resume point — no optimizer state).
            p = run_dir / f"step_{step:08d}.pt"
            torch.save({"step": step, "model": msd}, p)
            # Rolling FULL-STATE checkpoint for lossless resume. Unlike best.pt
            # (which only advances when val improves) this is written every
            # save_every regardless, so a wallclock kill late in training
            # resumes from the true latest step with Adam moments + scaler +
            # step + wandb id intact. Same dict shape as best.pt.
            last = run_dir / "last.pt"
            torch.save({
                "step": step,
                "model": msd,
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "val_loss": best_val,
                "z_tokenizer": {
                    "sorted_z": z_tok._sorted_z.cpu(),
                    "n_levels": z_tok.n_levels,
                    "gaussian_range": z_tok.gaussian_range,
                },
                "tokenizer_ckpt_path": str(args.tokenizer_ckpt),
                "approach": args.approach,
                "encoder_mask_ratio": args.encoder_mask_ratio,
                "redshift_mask_ratio": args.redshift_mask_ratio,
                "redshift_soft_sigma": args.redshift_soft_sigma,
                "wandb_run_id": wandb_id_to_save,
            }, last)
            print(f"  ckpt -> {p}  (+ rolling last.pt @ step {step})")
            if args.cfs_out is not None:
                try:
                    args.cfs_out.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(last, args.cfs_out / "last.pt")
                except OSError as e:
                    print(f"  WARN: cfs_out last.pt mirror failed ({e}); "
                          f"SCRATCH last.pt is safe at {last}")

        step += 1

    if rank == 0:
        p = run_dir / "final.pt"
        torch.save({
            "step": step,
            "model": model.module.state_dict() if is_distributed else model.state_dict(),
            "z_tokenizer": {
                "sorted_z": z_tok._sorted_z.cpu(),
                "n_levels": z_tok.n_levels,
                "gaussian_range": z_tok.gaussian_range,
            },
            "tokenizer_ckpt_path": str(args.tokenizer_ckpt),
            "approach": args.approach,
            "encoder_mask_ratio": args.encoder_mask_ratio,
            "redshift_mask_ratio": args.redshift_mask_ratio,
            "redshift_soft_sigma": args.redshift_soft_sigma,
            "wandb_run_id": wandb_id_to_save,
        }, p)
        print(f"[done] final -> {p}  best_val_loss={best_val:.4f}")
        if args.cfs_out is not None:
            try:
                args.cfs_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, args.cfs_out / "final.pt")
            except OSError as e:
                print(f"  WARN: cfs_out final mirror failed ({e}); "
                      f"SCRATCH final.pt is safe at {p}")

        # NOTE: no final.pt artifact push — only the best checkpoint goes to
        # W&B. log_model_artifact prunes prior versions of the artifact, so
        # a final push would DELETE the best version (the one MANIFEST.json
        # and the release pipeline reference as :best).

        ar_final = evaluate_ar(
            model.module if is_distributed else model,
            val_loader, spec_tok, z_tok, args.approach, device,
            max_batches=args.ar_eval_batches,
            encoder_mask_ratio=args.encoder_mask_ratio,
            redshift_mask_ratio=1.0,
        )
        print(f"[ar_final {step:6d}] " + " ".join(f"{k}={ar_final[k]}" for k in ar_final))
        with metrics_path.open("a") as f:
            f.write(json.dumps({"kind": "ar_final", "step": step,
                                **{f"val_ar_{k}": vv for k, vv in ar_final.items()}}) + "\n")
        wlog(wandb_run, {f"val_ar/{k}": vv for k, vv in ar_final.items()}, step=step)

        wfinish(wandb_run)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
