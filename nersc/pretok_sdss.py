#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-tokenize SDSS spectra once with the frozen v2 spectrum tokenizer.

Both transformers (V2, V3) consume the SAME frozen tokenizer, so SDSS spectrum
tokenization is model-independent — do it once, reuse for every model and shot count.

Worker mode (one process per GPU, strided over the path list):
    python nersc/pretok_sdss.py --tokenizer-ckpt <tok.pt> \
        --paths $SCRATCH/sdss_ft/train_paths.txt \
        --out   $SCRATCH/sdss_ft/sdss_train.r{SHARD}.npz \
        --num-shards 4 --shard-id {SHARD}

Merge mode (caps to --max-good good rows, in deterministic shard order):
    python nersc/pretok_sdss.py --merge \
        --shard-glob '$SCRATCH/sdss_ft/sdss_train.r*.npz' \
        --out $SCRATCH/sdss_ft/sdss_train.npz --max-good 5000

Output npz fields (mirror the DR1 token cache): spec_indices (N,272) int16,
z (N,) f32, denorm (N,) f32, zwarn (N,) i8 (all 0 — good-zwarn enforced at read).
"""
from __future__ import annotations

import argparse
import glob as globmod
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.utils.sdss import SDSSSpectrumDataset, collate_sdss_skip_none  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def merge(args):
    parts = sorted(globmod.glob(args.shard_glob))
    if not parts:
        raise SystemExit(f"[pretok] no shards match {args.shard_glob}")
    idx = np.concatenate([np.load(p)["spec_indices"] for p in parts])
    z = np.concatenate([np.load(p)["z"] for p in parts])
    den = np.concatenate([np.load(p)["denorm"] for p in parts])
    n = len(z)
    if args.max_good and n > args.max_good:
        idx, z, den = idx[: args.max_good], z[: args.max_good], den[: args.max_good]
    np.savez(args.out, spec_indices=idx.astype(np.int16), z=z.astype(np.float32),
             denorm=den.astype(np.float32),
             zwarn=np.zeros(len(z), np.int8))
    print(f"[pretok] merged {len(parts)} shards -> {args.out}  N={len(z)} "
          f"(from {n} good)", flush=True)


def worker(args):
    if torch.cuda.is_available():
        gpu = args.gpu_id if args.gpu_id is not None else 0
        gpu = gpu % torch.cuda.device_count()
        torch.cuda.set_device(gpu)
        dev = torch.device(f"cuda:{gpu}")
    else:
        dev = torch.device("cpu")
    print(f"[pretok] cuda_avail={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()} -> dev={dev}", flush=True)
    paths = [ln.strip() for ln in Path(args.paths).read_text().splitlines() if ln.strip()]
    shard_paths = paths[args.shard_id:: args.num_shards]
    print(f"[pretok] shard {args.shard_id}/{args.num_shards}: {len(shard_paths)} files "
          f"dev={dev}", flush=True)

    spec_tok = SpectrumTokenizer().to(dev).eval()
    tck = torch.load(args.tokenizer_ckpt, map_location=dev, weights_only=False)
    spec_tok.load_state_dict(tck.get("model", tck) if isinstance(tck, dict) else tck)
    for pm in spec_tok.parameters():
        pm.requires_grad_(False)

    ds = SDSSSpectrumDataset(shard_paths, require_good_zwarn=True,
                             require_nonzero_flux=True)
    print(f"[pretok] shard {args.shard_id}: {len(ds)} good spectra after filtering",
          flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_sdss_skip_none,
                        pin_memory=dev.type == "cuda")

    IDX, Z, DEN = [], [], []
    t0 = time.time(); n = 0
    with torch.no_grad():
        for b in loader:
            if b is None:
                continue
            flux = b["flux"].to(dev, non_blocking=True)
            ivar = b["ivar"].to(dev, non_blocking=True)
            wave = b["wavelength"].to(dev, non_blocking=True)
            istd = torch.sqrt(ivar.clamp(min=1e-10))
            x = torch.stack([flux, istd], dim=1)  # (B, 2, L)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                indices, denorm = spec_tok.encode(x, wavelength=wave)
            if indices.dim() == 3:
                indices = indices[:, 0, :]
            IDX.append(indices.to(torch.int16).cpu().numpy())
            Z.append(b["z"].numpy())
            DEN.append(denorm.float().cpu().numpy().reshape(-1))
            n += len(b["z"])
            if n % 2048 < args.batch_size:
                print(f"[pretok] shard {args.shard_id}: {n} {n/(time.time()-t0):.0f}/s",
                      flush=True)

    out = {"spec_indices": np.concatenate(IDX) if IDX else np.zeros((0, 272), np.int16),
           "z": np.concatenate(Z) if Z else np.zeros((0,), np.float32),
           "denorm": np.concatenate(DEN) if DEN else np.zeros((0,), np.float32)}
    np.savez(args.out, **out)
    print(f"[pretok] DONE shard {args.shard_id}: N={n} -> {args.out} "
          f"{n/(time.time()-t0):.0f}/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shard-glob")
    ap.add_argument("--max-good", type=int, default=None)
    ap.add_argument("--tokenizer-ckpt", type=Path)
    ap.add_argument("--paths", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--gpu-id", type=int, default=None,
                    help="Explicit CUDA device for this shard (all GPUs visible).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    if args.merge:
        merge(args)
    else:
        if not args.tokenizer_ckpt or not args.paths:
            raise SystemExit("[pretok] worker mode needs --tokenizer-ckpt and --paths")
        worker(args)


if __name__ == "__main__":
    main()
