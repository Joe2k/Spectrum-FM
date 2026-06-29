#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build plate-disjoint train/test path lists for legacy SDSS few-shot fine-tuning.

Legacy SDSS DR17 spec-lite coadds live at
  <root>/<RUN2D>/spectra/lite/<PLATE>/spec-<PLATE>-<MJD>-<FIBER>.fits

We split BY PLATE (never by spectrum) so the held-out test set shares no plate with
the fine-tune set — the SDSS analogue of the DESI healpix split, preventing
near-duplicate leakage. Quality filtering (ZWARN==0, nonzero flux) is applied later
in pretok_sdss.py via SDSSSpectrumDataset; here we just over-provision file paths from
disjoint plate pools so enough survive filtering.

Usage:
    python nersc/build_sdss_lists.py \
        --root /global/cfs/cdirs/sdss/data/sdss/dr17/sdss/spectro/redux \
        --out-dir $SCRATCH/sdss_ft \
        --n-train-files 8000 --n-test-files 35000 --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def find_plate_dirs(root: Path, run2d_glob: str):
    """Return a list of per-plate lite dirs: <root>/<RUN2D>/spectra/lite/<PLATE>/."""
    plate_dirs = []
    for run2d in sorted(root.glob(run2d_glob)):
        lite = run2d / "spectra" / "lite"
        if not lite.is_dir():
            continue
        for pl in sorted(lite.iterdir()):
            if pl.is_dir():
                plate_dirs.append(pl)
    return plate_dirs


def collect_until(plate_dirs, target_files, glob="spec-*.fits"):
    """Walk plate dirs (in given order) accumulating file paths until target_files."""
    out, used_plates = [], []
    for pl in plate_dirs:
        files = sorted(str(p) for p in pl.glob(glob))
        if not files:
            continue
        out.extend(files)
        used_plates.append(pl.name)
        if len(out) >= target_files:
            break
    return out, used_plates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/global/cfs/cdirs/sdss/data/sdss/dr17/sdss/spectro/redux"))
    ap.add_argument("--run2d-glob", default="*",
                    help="Glob for RUN2D dirs under root (e.g. '26' or '10*' or '*').")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train-files", type=int, default=8000,
                    help="Over-provisioned (pretok caps good rows to --n-train).")
    ap.add_argument("--n-test-files", type=int, default=35000)
    ap.add_argument("--glob", default="spec-*.fits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plate_dirs = find_plate_dirs(args.root, args.run2d_glob)
    print(f"[lists] found {len(plate_dirs)} plate dirs under {args.root} "
          f"(run2d='{args.run2d_glob}')", flush=True)
    if not plate_dirs:
        raise SystemExit("[lists] no plate dirs — check --root/--run2d-glob on NERSC.")

    # Deterministic plate shuffle, then disjoint train/test plate pools.
    rng = random.Random(args.seed)
    rng.shuffle(plate_dirs)
    # Take test plates from the front, train plates from the back → guaranteed disjoint.
    test_files, test_plates = collect_until(plate_dirs, args.n_test_files, args.glob)
    train_files, train_plates = collect_until(
        list(reversed(plate_dirs)), args.n_train_files, args.glob)
    # Safety: ensure no plate appears in both (front vs back can't overlap unless pools
    # are huge; assert anyway).
    overlap = set(train_plates) & set(test_plates)
    if overlap:
        raise SystemExit(f"[lists] PLATE overlap between train/test: {sorted(overlap)[:5]}…")

    (args.out_dir / "train_paths.txt").write_text("\n".join(train_files) + "\n")
    (args.out_dir / "test_paths.txt").write_text("\n".join(test_files) + "\n")
    print(f"[lists] train: {len(train_files)} files from {len(train_plates)} plates", flush=True)
    print(f"[lists] test : {len(test_files)} files from {len(test_plates)} plates", flush=True)
    print(f"[lists] wrote {args.out_dir}/train_paths.txt, test_paths.txt", flush=True)


if __name__ == "__main__":
    main()
