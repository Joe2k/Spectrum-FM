#!/usr/bin/env python3
"""
Pre-tokenize the DR1 corpus with the frozen v2 spectrum tokenizer (X1).
======================================================================
Runs the ConvNeXt+LFQ encoder ONCE over the manifest and caches the discrete
token ids per spectrum, so transformer training becomes a fast array read
instead of a per-step encode + FITS stitch (the current throughput ceiling).

Output: one compressed `.npz` per (survey, program, healpix) under --out, named
`{survey}_{program}_{healpix}.npz`, with parallel arrays over the coadd rows:
    indices (N, n_tokens) uint16   raw LFQ codes 0..1023 (offset added at train time)
    z, denorm               float32
    zwarn int16 · fiberstatus int32 · nonzero_flux int8   (quality flags)
    spectype <U8 · targetid int64 · row int32             (eval / joins)
    survey, program, healpix, coadd, redrock              scalars (provenance)

Resumable: skips shards that already exist (unless --overwrite), so the
~1 GPU-day pass survives the 4h interactive cap. DDP-aware: each rank takes
records[rank::world_size]. `--verify` re-encodes cached spectra on the fly and
asserts token equality against the shards.

Interactive single-GPU example:
    python nersc/pretokenize_corpus.py \
        --manifest $SCRATCH/manifests/dr1_v2_balanced_3k_scratch.jsonl \
        --tokenizer-ckpt $CFS/tokenizer_v2_3k_v3/final.pt \
        --out $SCRATCH/dr1_tokenized_v2 --amp
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

# Per-spectrum cost is dominated by the FITS read + the pure-Python stitch,
# not the GPU encode, so logs are flushed and progress is reported per-healpix.
print = partial(print, flush=True)  # noqa: A001 — unbuffer under srun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa: E402
from src.utils.data import stitch_bands  # noqa: E402
from dr1_dataset import load_manifest  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Pre-tokenize DR1 with the frozen tokenizer")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tokenizer-ckpt", type=Path, required=True,
                   help="frozen tokenizer checkpoint (v2 final.pt)")
    p.add_argument("--out", type=Path, required=True, help="shard output directory")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (matches training)")
    p.add_argument("--quality", choices=["all", "strict"], default="all",
                   help="all = tokenize every row + store flags; strict = drop "
                        "zwarn!=0 / bad-fiberstatus / all-zero-flux before tokenizing")
    p.add_argument("--surveys", nargs="*", default=None)
    p.add_argument("--programs", nargs="*", default=None)
    p.add_argument("--num-workers", type=int, default=8,
                   help="CPU processes per rank for the FITS-read + stitch "
                        "(the bottleneck; the GPU encode is cheap). 0 = serial.")
    p.add_argument("--overwrite", action="store_true",
                   help="re-tokenize shards that already exist")
    p.add_argument("--verify", action="store_true",
                   help="verify mode: re-encode cached spectra and assert equality")
    p.add_argument("--verify-shards", type=int, default=10,
                   help="how many shards to sample in --verify")
    p.add_argument("--verify-rows", type=int, default=16,
                   help="how many rows per shard to re-encode in --verify")
    return p.parse_args()


def load_tokenizer(ckpt_path, device):
    model = SpectrumTokenizer().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _stitched_rows(coadd, redrock):
    """Yield (stitched_flux, stitched_ivar, stitched_wave, nonzero) per row.

    Mirrors DR1IndexedDataset.__getitem__'s band reading + stitch_bands call.
    Band wavelength grids are fixed within a coadd, so the stitched length is
    constant across rows (they stack cleanly into one array).
    """
    b_wave = coadd["B_WAVELENGTH"].data
    r_wave = coadd["R_WAVELENGTH"].data
    z_wave = coadd["Z_WAVELENGTH"].data
    b_flux, r_flux, z_flux = coadd["B_FLUX"].data, coadd["R_FLUX"].data, coadd["Z_FLUX"].data
    b_ivar, r_ivar, z_ivar = coadd["B_IVAR"].data, coadd["R_IVAR"].data, coadd["Z_IVAR"].data
    b_mask, r_mask, z_mask = coadd["B_MASK"].data, coadd["R_MASK"].data, coadd["Z_MASK"].data
    n = b_flux.shape[0]
    for row in range(n):
        st = stitch_bands(
            [b_wave, r_wave, z_wave],
            [b_flux[row], r_flux[row], z_flux[row]],
            [b_ivar[row], r_ivar[row], z_ivar[row]],
            [b_mask[row] != 0, r_mask[row] != 0, z_mask[row] != 0],
        )
        nonzero = float(np.abs(b_flux[row]).sum()
                        + np.abs(r_flux[row]).sum()
                        + np.abs(z_flux[row]).sum()) > 0.0
        yield st["flux"], st["ivar"], st["wavelength"], nonzero


@torch.no_grad()
def _encode_rows(model, flux_arr, ivar_arr, wave_arr, device, amp, batch_size):
    """Encode stacked rows (N, L) → (N, n_tokens) uint16 indices + (N,) denorm."""
    out_idx, out_den = [], []
    n = flux_arr.shape[0]
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        flux = torch.from_numpy(flux_arr[s:e]).to(device)
        ivar = torch.from_numpy(ivar_arr[s:e]).to(device)
        wave = torch.from_numpy(wave_arr[s:e]).to(device)
        istd = torch.sqrt(ivar.clamp(min=1e-10))
        x = torch.stack([flux, istd], dim=1)
        with torch.amp.autocast("cuda", enabled=amp):
            idx, denorm = model.encode(x, wavelength=wave)
        out_idx.append(idx.cpu().numpy().astype(np.uint16))
        out_den.append(denorm.float().cpu().numpy().astype(np.float32))
    return np.concatenate(out_idx, axis=0), np.concatenate(out_den, axis=0)


def shard_name(rec) -> str:
    return f"{rec['survey']}_{rec['program']}_{rec['healpix']}.npz"


def _stitch_chunk(arg):
    """CPU-only worker: stitch a slice of coadd rows. Imports no torch and
    touches no GPU, so it is safe under a spawn ProcessPool while the parent
    process holds the CUDA context. Re-opens the coadd (memmap) per call.
    """
    coadd_path, rows = arg
    out = []
    with fits.open(coadd_path, memmap=True) as coadd:
        b_wave = coadd["B_WAVELENGTH"].data
        r_wave = coadd["R_WAVELENGTH"].data
        z_wave = coadd["Z_WAVELENGTH"].data
        bf, rf, zf = coadd["B_FLUX"].data, coadd["R_FLUX"].data, coadd["Z_FLUX"].data
        bi, ri, zi = coadd["B_IVAR"].data, coadd["R_IVAR"].data, coadd["Z_IVAR"].data
        bm, rm, zm = coadd["B_MASK"].data, coadd["R_MASK"].data, coadd["Z_MASK"].data
        for row in rows:
            st = stitch_bands(
                [b_wave, r_wave, z_wave],
                [bf[row], rf[row], zf[row]],
                [bi[row], ri[row], zi[row]],
                [bm[row] != 0, rm[row] != 0, zm[row] != 0],
            )
            nz = float(np.abs(bf[row]).sum() + np.abs(rf[row]).sum()
                       + np.abs(zf[row]).sum()) > 0.0
            out.append((st["flux"].astype(np.float32), st["ivar"].astype(np.float32),
                        st["wavelength"].astype(np.float32), bool(nz)))
    return out


def _stitch_all_rows(coadd_path, n, pool, num_workers):
    """Stitch all n rows (the bottleneck), in order, parallel across `pool`.

    np.array_split preserves order and pool.map preserves order, so the
    concatenated result is in original row order.
    """
    rows = list(range(n))
    if pool is None or num_workers <= 0 or n <= 1:
        chunks = [_stitch_chunk((coadd_path, rows))]
    else:
        splits = [list(c) for c in np.array_split(rows, min(num_workers, n)) if len(c)]
        chunks = list(pool.map(_stitch_chunk, [(coadd_path, c) for c in splits]))
    flux_rows, ivar_rows, wave_rows, nonzero = [], [], [], []
    for chunk in chunks:
        for f, iv, wv, nz in chunk:
            flux_rows.append(f); ivar_rows.append(iv); wave_rows.append(wv); nonzero.append(nz)
    return flux_rows, ivar_rows, wave_rows, np.array(nonzero, dtype=np.int8)


def write_shard(rec, model, device, args, pool=None):
    with fits.open(rec["redrock"], memmap=True) as redrock, \
            fits.open(rec["coadd"], memmap=True) as coadd:
        rs = redrock["REDSHIFTS"].data
        fm = coadd["FIBERMAP"].data
        z = rs["Z"].astype(np.float32)
        zwarn = rs["ZWARN"].astype(np.int16)
        spectype = rs["SPECTYPE"].astype("<U8")
        targetid = fm["TARGETID"].astype(np.int64)
        fstat = fm["COADD_FIBERSTATUS"].astype(np.int32)
    n = len(z)

    flux_rows, ivar_rows, wave_rows, nonzero = _stitch_all_rows(
        str(rec["coadd"]), n, pool, args.num_workers)
    row = np.arange(n, dtype=np.int32)

    keep = np.ones(n, dtype=bool)
    if args.quality == "strict":
        keep = (zwarn == 0) & (fstat == 0) & nonzero.astype(bool)

    idx_keep = np.nonzero(keep)[0]
    if idx_keep.size == 0:
        return 0  # nothing to write for this healpix
    flux_arr = np.stack([flux_rows[i] for i in idx_keep])
    ivar_arr = np.stack([ivar_rows[i] for i in idx_keep])
    wave_arr = np.stack([wave_rows[i] for i in idx_keep])

    indices, denorm = _encode_rows(
        model, flux_arr, ivar_arr, wave_arr, device, args.amp, args.batch_size)

    shard = args.out / shard_name(rec)
    np.savez_compressed(
        shard,
        indices=indices,
        z=z[idx_keep], denorm=denorm,
        zwarn=zwarn[idx_keep], fiberstatus=fstat[idx_keep],
        nonzero_flux=nonzero[idx_keep],
        spectype=spectype[idx_keep], targetid=targetid[idx_keep],
        row=row[idx_keep],
        survey=np.array(rec["survey"]), program=np.array(rec["program"]),
        healpix=np.array(int(rec["healpix"])),
        coadd=np.array(str(rec["coadd"])), redrock=np.array(str(rec["redrock"])),
    )
    return int(indices.shape[0])


def verify(model, device, args):
    """Re-encode a sample of cached spectra from their source FITS and compare."""
    rng = np.random.default_rng(0)
    shards = sorted(args.out.glob("*.npz"))
    if not shards:
        print("[verify] no shards found", file=sys.stderr)
        return 1
    sample = rng.choice(len(shards), size=min(args.verify_shards, len(shards)), replace=False)
    n_checked = n_mismatch = 0
    for si in sample:
        with np.load(shards[si], allow_pickle=False) as d:
            cached_idx = d["indices"]
            rows = d["row"]
            coadd_path = str(d["coadd"])
            redrock_path = str(d["redrock"])
        coadd = fits.open(coadd_path, memmap=True)
        redrock = fits.open(redrock_path, memmap=True)
        try:
            stitched = list(_stitched_rows(coadd, redrock))
        finally:
            coadd.close()
            redrock.close()
        pick = rng.choice(len(rows), size=min(args.verify_rows, len(rows)), replace=False)
        flux_arr = np.stack([stitched[rows[i]][0].astype(np.float32) for i in pick])
        ivar_arr = np.stack([stitched[rows[i]][1].astype(np.float32) for i in pick])
        wave_arr = np.stack([stitched[rows[i]][2].astype(np.float32) for i in pick])
        re_idx, _ = _encode_rows(model, flux_arr, ivar_arr, wave_arr,
                                 device, args.amp, args.batch_size)
        for k, i in enumerate(pick):
            n_checked += 1
            if not np.array_equal(re_idx[k], cached_idx[i]):
                n_mismatch += 1
    print(f"[verify] checked {n_checked} spectra across {len(sample)} shards: "
          f"{n_mismatch} mismatch")
    return 0 if n_mismatch == 0 else 2


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_tokenizer(args.tokenizer_ckpt, device)
    print(f"[setup] device={device} amp={args.amp} quality={args.quality}")

    if args.verify:
        return verify(model, device, args)

    records = load_manifest(args.manifest)
    if args.surveys:
        records = [r for r in records if r.get("survey") in set(args.surveys)]
    if args.programs:
        records = [r for r in records if r.get("program") in set(args.programs)]

    # DDP rank striping (each rank writes a disjoint set of shards).
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    mine = records[rank::world]
    print(f"[setup] rank {rank}/{world}: {len(mine)} of {len(records)} healpix")

    # CPU pool for the FITS-read + stitch bottleneck (spawn so the parent's
    # CUDA context is never forked into the workers).
    pool = None
    if args.num_workers > 0:
        pool = ProcessPoolExecutor(max_workers=args.num_workers,
                                   mp_context=mp.get_context("spawn"))
    print(f"[rank {rank}] starting: {args.num_workers} stitch workers, "
          f"batch {args.batch_size}", flush=True)

    import time
    t0 = time.time()
    n_done = n_spec = n_skip = 0
    try:
        for i, rec in enumerate(mine):
            shard = args.out / shard_name(rec)
            if shard.exists() and not args.overwrite:
                n_skip += 1
                continue
            n = write_shard(rec, model, device, args, pool=pool)
            n_done += 1
            n_spec += n
            # Log early and often so progress is visible under srun.
            if n_done <= 3 or (i + 1) % 20 == 0 or i + 1 == len(mine):
                rate = n_spec / max(time.time() - t0, 1e-6)
                print(f"[rank {rank}] {i + 1}/{len(mine)} healpix · "
                      f"{n_done} written ({n_spec} spectra, {rate:.0f}/s) · "
                      f"{n_skip} skipped")
    finally:
        if pool is not None:
            pool.shutdown()
    print(f"[rank {rank}] done: {n_done} shards / {n_spec} spectra written, "
          f"{n_skip} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
