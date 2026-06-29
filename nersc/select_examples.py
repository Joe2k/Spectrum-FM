#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select example spectra by blind-z prediction category, for montage plots.

Two datasets:
  --dataset val  : DESI DR1 val split (cached tokens), identity=targetid + coadd.
  --dataset sdss : SDSS zero-shot on test_paths (on-the-fly), identity=file path.

Both models (V2 512-hard, V3 4096-soft) are scored. Per (model, category) we pick
up to --n-per rows, then read the RAW observed flux for those rows and save a
compact npz for offline montage rendering.

Categories (per model):
  cat1_predz0     : |z_pred| < 0.02
  cat2_overpred   : z_pred < 0.5 and (z_pred - z_true) > 0.15 (predicted >> true)
  cat3_underpred  : 0.5 < z_true < 1 and (z_true - z_pred) > 0.15 (true >> predicted)
  cat4_ceiling    : z_pred near model ceiling (>=q999-0.12) and z_true > z_pred+0.3
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))

from src.eval.redshift_metrics import decode_redshift           # noqa: E402
from src.inference.release import _infer_transformer_dims, _restore_z_tokenizer  # noqa: E402
from src.models.transformer import SpectrumTransformer          # noqa: E402
from src.tokenizers.spectrum import SpectrumTokenizer           # noqa: E402
from src.training.sequences import tokenize_and_build           # noqa: E402
from src.training.data_split import split_records_by_healpix    # noqa: E402
from src.utils.data import stitch_bands                         # noqa: E402
from src.utils.sdss import read_sdss_spec_fits                  # noqa: E402

CATS = ["cat1_predz0", "cat2_overpred", "cat3_underpred", "cat4_ceiling"]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    dims = _infer_transformer_dims(sd)
    model = SpectrumTransformer(**dims).to(device); model.load_state_dict(sd); model.eval()
    z_tok = _restore_z_tokenizer(ckpt)
    return model, z_tok


@torch.no_grad()
def predict_blind(model, z_tok, batches, spec_tok, device, wavelength_aware):
    """batches: iterable of raw dicts. Returns z_pred np array (concat)."""
    out = []
    for raw in batches:
        enc, dec, tgt, _ = tokenize_and_build(
            raw, spec_tok, z_tok, "a", device,
            encoder_mask_ratio=0.0, redshift_mask_ratio=1.0,
            wavelength_aware=wavelength_aware)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            logits, _ = model(enc, dec, targets=tgt)
        red = logits[:, 0, :].float()
        out.append(decode_redshift(red, z_tok, mode="expected").cpu().numpy())
    return np.concatenate(out)


def select_categories(z_true, z_pred, n_per, seed):
    rng = np.random.default_rng(seed)
    ceil = float(np.quantile(z_pred, 0.999))
    masks = {
        "cat1_predz0":    np.abs(z_pred) < 0.02,
        "cat2_overpred":  (z_pred < 0.5) & ((z_pred - z_true) > 0.15) & (z_true < z_pred),
        "cat3_underpred": (z_true > 0.5) & (z_true < 1.0) & ((z_true - z_pred) > 0.15),
        "cat4_ceiling":   (z_pred >= ceil - 0.12) & (z_true > z_pred + 0.3),
    }
    picks = {}
    for name, m in masks.items():
        idx = np.nonzero(m)[0]
        if len(idx) > n_per:
            idx = np.sort(rng.choice(idx, n_per, replace=False))
        picks[name] = idx
        print(f"    {name:16s}: {len(np.nonzero(m)[0]):7d} avail -> {len(idx)} picked", flush=True)
    return picks


# ----------------------------- VAL (DESI) ----------------------------------- #
def _build_val_ds(args):
    from dr1_dataset import load_manifest
    from dr1_tokenized_dataset import DR1CachedTokenDataset
    from train_transformer import _records_to_shards
    records = load_manifest(args.manifest)
    _, val_records = split_records_by_healpix(records, holdout_frac=0.05, seed=42)
    ds = DR1CachedTokenDataset(_records_to_shards(val_records, args.tokenized_dir),
                               require_good_zwarn=True, require_nonzero_flux=True)
    return ds, val_records


def run_val_worker(args):
    """One GPU scores a strided subset of val rows; saves preds + identity shard.

    torchrun-aware: if LOCAL_RANK/RANK/WORLD_SIZE are set, derive shard/gpu from
    them (independent workers, no dist comm) and write to <shard-out-prefix>.r{rank}.npz.
    """
    import os
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        args.gpu_id = int(lr)
        args.shard_id = int(os.environ["RANK"])
        args.num_shards = int(os.environ["WORLD_SIZE"])
        if args.shard_out_prefix:
            args.shard_out = Path(f"{args.shard_out_prefix}.r{args.shard_id}.npz")
    from dr1_tokenized_dataset import collate_cached_skip_none
    from torch.utils.data import DataLoader, Subset
    if torch.cuda.is_available():
        gpu = (args.gpu_id or 0) % torch.cuda.device_count()
        torch.cuda.set_device(gpu); device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    ds, _ = _build_val_ds(args)
    N = len(ds)
    idxs = list(range(args.shard_id, N, args.num_shards))
    print(f"[val.w{args.shard_id}] dev={device} subset={len(idxs)}/{N}", flush=True)

    tid = np.empty(len(idxs), np.int64); zt = np.empty(len(idxs), np.float64)
    survey = np.empty(len(idxs), object); program = np.empty(len(idxs), object)
    healpix = np.empty(len(idxs), np.int64)
    cache = {}; t0 = time.time()
    for k, gi in enumerate(idxs):
        si, r = ds.flat_index[gi]
        if si not in cache:
            dd = np.load(ds.shard_paths[si]); cache = {si: (dd["targetid"], dd["z"])}
        d_tid, d_z = cache[si]
        s, p, hp = ds.shard_meta[si]
        tid[k] = int(d_tid[r]); zt[k] = float(d_z[r]); survey[k] = s; program[k] = p; healpix[k] = hp
    print(f"[val.w{args.shard_id}] identity {time.time()-t0:.0f}s", flush=True)

    loader = DataLoader(Subset(ds, idxs), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_cached_skip_none,
                        pin_memory=True)
    preds = {}
    for tag, ck in (("v2", args.ckpt_v2), ("v3", args.ckpt_v3)):
        model, z_tok = load_model(ck, device); t0 = time.time()
        preds[tag] = predict_blind(model, z_tok, _iter_loader(loader, device), None, device, False)
        print(f"[val.w{args.shard_id}] {tag} {time.time()-t0:.0f}s n={len(preds[tag])}", flush=True)
        del model; torch.cuda.empty_cache()
    np.savez(args.shard_out, z_pred_v2=preds["v2"], z_pred_v3=preds["v3"], z_true=zt,
             targetid=tid, survey=survey.astype(str), program=program.astype(str), healpix=healpix)
    print(f"[val.w{args.shard_id}] saved -> {args.shard_out}", flush=True)


def run_val_merge(args):
    """Gather shard preds, select categories globally, read raw coadd flux, save."""
    import glob as g
    from astropy.io import fits
    from dr1_dataset import load_manifest
    _, val_records = split_records_by_healpix(
        load_manifest(args.manifest), holdout_frac=0.05, seed=42)
    parts = sorted(g.glob(str(args.merge_glob)))
    print(f"[val.merge] {len(parts)} shards", flush=True)
    cat = lambda k: np.concatenate([np.load(p, allow_pickle=True)[k] for p in parts])
    zp_v2 = cat("z_pred_v2"); zp_v3 = cat("z_pred_v3"); z_true = cat("z_true")
    tid = cat("targetid"); survey = cat("survey"); program = cat("program"); healpix = cat("healpix")
    print(f"[val.merge] total rows {len(z_true)}", flush=True)

    recmap = {}
    for rec in val_records:
        recmap[(str(rec.get("survey")), str(rec.get("program")), int(rec.get("healpix", -1)))] = rec

    preds = {"v2": zp_v2, "v3": zp_v3}
    rows = []
    for tag in ("v2", "v3"):
        print(f"[val.merge] selecting {tag}", flush=True)
        picks = select_categories(z_true, preds[tag], args.n_per, args.seed)
        for c, idxs in picks.items():
            for i in idxs:
                key = (str(survey[i]), str(program[i]), int(healpix[i]))
                rec = recmap.get(key)
                spec = _read_desi_raw(rec, int(tid[i]), fits) if rec else None
                if spec is None:
                    continue
                rows.append(dict(model=tag, cat=c, targetid=int(tid[i]),
                                 z_true=float(z_true[i]), z_pred=float(preds[tag][i]),
                                 flux=spec["flux"], wave=spec["wave"],
                                 info=f"{survey[i]}/{program[i]}/{healpix[i]}"))
    _save_rows(rows, args.out)


def _read_desi_raw(rec, targetid, fits):
    try:
        with fits.open(rec["coadd"], memmap=False) as h:
            fm = h["FIBERMAP"].data
            tids = np.asarray(fm["TARGETID"])
            where = np.nonzero(tids == targetid)[0]
            if len(where) == 0:
                return None
            row = int(where[0])
            st = stitch_bands(
                [h["B_WAVELENGTH"].data, h["R_WAVELENGTH"].data, h["Z_WAVELENGTH"].data],
                [h["B_FLUX"].data[row], h["R_FLUX"].data[row], h["Z_FLUX"].data[row]],
                [h["B_IVAR"].data[row], h["R_IVAR"].data[row], h["Z_IVAR"].data[row]],
                [h["B_MASK"].data[row] != 0, h["R_MASK"].data[row] != 0, h["Z_MASK"].data[row] != 0])
        return {"flux": st["flux"].astype(np.float32), "wave": st["wavelength"].astype(np.float32)}
    except Exception as e:  # noqa: BLE001
        print(f"      [warn] raw read failed tid={targetid}: {e}", flush=True)
        return None


# ----------------------------- SDSS ----------------------------------------- #
def run_sdss(args, device):
    paths = [ln.strip() for ln in Path(args.sdss_paths).read_text().splitlines() if ln.strip()]
    if args.max_sdss:
        paths = paths[: args.max_sdss]
    print(f"[sdss] scanning {len(paths)} files ...", flush=True)
    recs = []  # (path, flux, ivar, wave, z)
    t0 = time.time()
    for k, p in enumerate(paths):
        r = read_sdss_spec_fits(p)
        if r is None or r["zwarn"] != 0 or not np.isfinite(r["z"]):
            continue
        if not np.any(np.abs(r["flux"]) > 0):
            continue
        recs.append((p, r["flux"].astype(np.float32), r["ivar"].astype(np.float32),
                     r["wavelength"].astype(np.float32), float(r["z"])))
        if (k + 1) % 2000 == 0:
            print(f"[sdss] {k+1}/{len(paths)} read, kept {len(recs)} ({(k+1)/(time.time()-t0):.0f}/s)", flush=True)
    print(f"[sdss] kept {len(recs)} good spectra", flush=True)
    z_true = np.array([r[4] for r in recs], np.float64)

    spec_tok = SpectrumTokenizer().to(device).eval()
    tck = torch.load(args.tokenizer_ckpt, map_location=device, weights_only=False)
    spec_tok.load_state_dict(tck.get("model", tck) if isinstance(tck, dict) else tck)
    for pm in spec_tok.parameters():
        pm.requires_grad_(False)

    def batches():
        B = args.batch_size
        for s in range(0, len(recs), B):
            chunk = recs[s:s + B]
            L = max(len(r[1]) for r in chunk)
            def pad(a, fill=0.0):
                if len(a) == L: return a
                o = np.full(L, fill, a.dtype); o[:len(a)] = a; return o
            def padw(w):
                if len(w) == L: return w
                step = w[-1] - w[-2] if len(w) >= 2 else 1.0
                ext = w[-1] + step * np.arange(1, L - len(w) + 1)
                return np.concatenate([w, ext]).astype(np.float32)
            flux = torch.from_numpy(np.stack([pad(r[1]) for r in chunk]))
            ivar = torch.from_numpy(np.stack([pad(r[2], 0.0) for r in chunk]))
            wave = torch.from_numpy(np.stack([padw(r[3]) for r in chunk]))
            z = torch.tensor([r[4] for r in chunk], dtype=torch.float32)
            yield {"flux": flux.to(device), "ivar": ivar.to(device),
                   "wavelength": wave.to(device), "z": z.to(device)}

    preds = {}
    for tag, ck in (("v2", args.ckpt_v2), ("v3", args.ckpt_v3)):
        model, z_tok = load_model(ck, device)
        print(f"[sdss] {tag} blind-z ...", flush=True); t0 = time.time()
        preds[tag] = predict_blind(model, z_tok, batches(), spec_tok, device, True)
        print(f"[sdss] {tag} done {time.time()-t0:.0f}s (n={len(preds[tag])})", flush=True)
        del model; torch.cuda.empty_cache()

    rows = []
    for tag in ("v2", "v3"):
        print(f"[sdss] selecting {tag}", flush=True)
        picks = select_categories(z_true, preds[tag], args.n_per, args.seed)
        for cat, idxs in picks.items():
            for i in idxs:
                p, flux, ivar, wave, z = recs[i]
                rows.append(dict(model=tag, cat=cat, targetid=Path(p).stem.replace("spec-", ""),
                                 z_true=float(z), z_pred=float(preds[tag][i]),
                                 flux=flux, wave=wave, info=Path(p).name))
    _save_rows(rows, args.out)


# ----------------------------- shared --------------------------------------- #
def _iter_loader(loader, device):
    for raw in loader:
        if raw is None:
            continue
        yield {"spec_indices": raw["spec_indices"].to(device), "z": raw["z"].to(device)}


def _save_rows(rows, out):
    if not rows:
        print("[save] NO ROWS selected!", flush=True); return
    K = len(rows)
    Lmax = max(len(r["flux"]) for r in rows)
    flux = np.zeros((K, Lmax), np.float32); wave = np.zeros((K, Lmax), np.float32)
    ln = np.zeros(K, np.int32)
    for j, r in enumerate(rows):
        n = len(r["flux"]); flux[j, :n] = r["flux"]; wave[j, :n] = r["wave"]; ln[j] = n
    np.savez(out,
             model=np.array([r["model"] for r in rows]),
             cat=np.array([r["cat"] for r in rows]),
             targetid=np.array([str(r["targetid"]) for r in rows]),
             info=np.array([str(r["info"]) for r in rows]),
             z_true=np.array([r["z_true"] for r in rows], np.float32),
             z_pred=np.array([r["z_pred"] for r in rows], np.float32),
             flux=flux, wave=wave, length=ln)
    print(f"[save] wrote {K} spectra -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["val", "sdss"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-v2", type=Path, required=True)
    ap.add_argument("--ckpt-v3", type=Path, required=True)
    ap.add_argument("--tokenizer-ckpt", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--tokenized-dir", type=Path)
    ap.add_argument("--sdss-paths", type=Path)
    ap.add_argument("--max-sdss", type=int, default=15000)
    ap.add_argument("--n-per", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    # 4-GPU sharding for val
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--shard-out", type=Path, default=None)
    ap.add_argument("--shard-out-prefix", type=str, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--merge-glob", type=str, default=None)
    args = ap.parse_args()
    if args.dataset == "val":
        if args.merge:
            run_val_merge(args)
        else:
            run_val_worker(args)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[init] device={device}", flush=True)
        run_sdss(args, device)


if __name__ == "__main__":
    main()
