#!/usr/bin/env python3
"""Stage A: fine-tune V3 from best.pt with the z-v2 (high-z) tokenizer + high-z
oversampling, to lift the QSO z>2 ceiling and cut QSO catastrophic outliers.

Fully spectrum-blind (no class token). The z bins are vocab slots in the shared
lm_head; keeping 4096 bins means NO architecture change — swapping in z-v2
(gaussian_range=4.0) only remaps bin<->z, and this fine-tune re-fits the z-bin
output/embedding weights. Body warm-starts from best.pt.

Trains on the DESI DR1 TRAIN healpix split (cached tokens, same split as the
original run: holdout 0.05 / seed 42), evaluates blind redshift PER TYPE on the
held-out VAL split (BGS/LRG/ELG/QSO via the DESI_TARGET typemap + spectype),
reporting the QSO ceiling (max z_pred, #>2.13) and gross-eta — the regression
guard watches BGS/LRG.

Ablation (run as parallel 4-GPU jobs):
  --oversample-hi 1   -> z-v2 tail only
  --oversample-hi 20  -> z-v2 + high-z oversampling

DDP: srun -n4 --gpus-per-task=1 (reads SLURM_PROCID/LOCALID).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))

from src.eval.redshift_metrics import decode_redshift  # noqa: E402
from src.inference.release import _infer_transformer_dims  # noqa: E402
from src.models.transformer import SOS_TOKEN, SpectrumTransformer  # noqa: E402
from src.tokenizers.redshift import RedshiftTokenizer  # noqa: E402
from src.training.sequences import lr_at, tokenize_and_build  # noqa: E402
from src.training.data_split import split_records_by_healpix  # noqa: E402
from dr1_dataset import load_manifest  # noqa: E402
from dr1_tokenized_dataset import (  # noqa: E402
    DR1CachedTokenDataset, collate_cached_skip_none, collect_redshifts_from_cache)


# --------------------------------------------------------------------------- DDP
def setup_ddp():
    if "RANK" in os.environ or "SLURM_PROCID" in os.environ:
        local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
        os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29512")
        cuda_idx = local if local < torch.cuda.device_count() else 0
        torch.cuda.set_device(cuda_idx)
        dist.init_process_group(backend="nccl")
        return True, rank, world, torch.device(f"cuda:{cuda_idx}")
    return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _records_to_shards(records, tokenized_dir):
    out = []
    for r in records:
        p = Path(tokenized_dir) / f"{r['survey']}_{r['program']}_{r['healpix']}.npz"
        if p.exists():
            out.append(p)
    return out


class OversampledCache(Dataset):
    """DR1CachedTokenDataset with high-z rows repeated `mult` times.

    Pre-scans z+spectype per kept row (one cheap read per shard), then replicates
    rows with z>hi_thresh (and optionally QSO spectype) `mult` times so the starved
    high-z tail gets gradient. mult=1 -> identity (z-v2 tail-only ablation).
    """

    def __init__(self, base: DR1CachedTokenDataset, hi_thresh=1.5, mult=1, include_qso=True):
        self.base = base
        if mult <= 1:
            self.expanded = np.arange(len(base), dtype=np.int64)
            self.n_hi = 0
            return
        # group flat_index rows by shard, read z/spectype once per shard
        z = np.empty(len(base), np.float32)
        is_qso = np.zeros(len(base), bool)
        by_shard = {}
        for fi, (si, r) in enumerate(base.flat_index):
            by_shard.setdefault(si, []).append((fi, r))
        for si, rows in by_shard.items():
            with np.load(base.shard_paths[si], allow_pickle=False) as d:
                zz = d["z"]; sty = d["spectype"] if "spectype" in d.files else None
                for fi, r in rows:
                    z[fi] = zz[r]
                    if sty is not None:
                        is_qso[fi] = (str(sty[r]) == "QSO")
        hi = z > hi_thresh
        if include_qso:
            hi = hi | is_qso
        self.n_hi = int(hi.sum())
        reps = np.where(hi, mult, 1).astype(np.int64)
        self.expanded = np.repeat(np.arange(len(base), dtype=np.int64), reps)

    def __len__(self):
        return len(self.expanded)

    def __getitem__(self, j):
        return self.base[int(self.expanded[j])]


def load_val_arrays(val_shards, cap, typemap_path):
    """Scan val shards into CPU arrays (idx, z, cls) for per-type blind-z eval."""
    tm = np.load(typemap_path, allow_pickle=True)
    t2c = dict(zip(tm["targetid"].tolist(), tm["cls"].tolist()))
    IDX, Z, CLS = [], [], []
    n = 0
    for p in val_shards:
        with np.load(p, allow_pickle=False) as d:
            zwarn = d["zwarn"] if "zwarn" in d.files else np.zeros(len(d["z"]), np.int16)
            nzf = d["nonzero_flux"] if "nonzero_flux" in d.files else np.ones(len(d["z"]), np.int8)
            keep = (zwarn == 0) & nzf.astype(bool)
            idx = d["indices"][keep].astype(np.int64)
            z = d["z"][keep].astype(np.float32)
            tid = d["targetid"][keep] if "targetid" in d.files else np.full(keep.sum(), -1)
            sty = d["spectype"][keep] if "spectype" in d.files else np.array(["?"] * int(keep.sum()))
            cls = np.array([t2c.get(int(t), "OTHER") for t in tid], dtype="U6")
            cls[(cls == "OTHER") & (sty.astype(str) == "QSO")] = "QSO"
        IDX.append(idx); Z.append(z); CLS.append(cls); n += len(z)
        if n >= cap:
            break
    return (torch.from_numpy(np.concatenate(IDX)[:cap]),
            torch.from_numpy(np.concatenate(Z)[:cap]),
            np.concatenate(CLS)[:cap])


@torch.no_grad()
def blind_z(core, idx, z, z_tok, device, batch=512):
    core.eval()
    ZP = []
    for s in range(0, len(z), batch):
        e = min(len(z), s + batch)
        rb = {"spec_indices": idx[s:e].to(device), "z": z[s:e]}
        enc, _, _, _ = tokenize_and_build(rb, None, z_tok, "a", device,
                                          encoder_mask_ratio=0.0, redshift_mask_ratio=1.0)
        sos = torch.full((enc.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = core(enc, sos)
        ZP.append(decode_redshift(logits[:, 0, :].float(), z_tok, mode="expected").cpu().numpy())
    return np.concatenate(ZP)


def per_type_report(zp, zt, cls, log):
    def m(zp, zt):
        dz = (zp - zt) / (1.0 + zt)
        nm = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
        return nm, float(np.mean(np.abs(dz) > 0.05)), float(np.mean(np.abs(dz) > 0.0033))
    nm, g, e = m(zp, zt)
    log(f"    ALL   N={len(zt):>7,} sigNMAD={nm:.5f} grossEta={g:.4f} eta={e:.3f}")
    for c in ("BGS", "LRG", "ELG", "QSO"):
        sel = cls == c
        if sel.sum() < 50:
            continue
        nm, g, e = m(zp[sel], zt[sel])
        extra = ""
        if c == "QSO":
            extra = f" | maxZpred={zp[sel].max():.3f} #>2.13={int((zp[sel] > 2.13).sum())}"
        log(f"    {c:4s}  N={int(sel.sum()):>7,} sigNMAD={nm:.5f} grossEta={g:.4f} eta={e:.3f}{extra}")
    return float(m(zp, zt)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tokenized-dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="best.pt (V3) to warm-start from")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--typemap", default="/pscratch/sd/j/joe2k/zr/typemap.npz")
    ap.add_argument("--n-train-shards", type=int, default=1500)
    ap.add_argument("--z-bins", type=int, default=4096)
    ap.add_argument("--gaussian-range", type=float, default=4.0)
    ap.add_argument("--oversample-hi", type=float, default=20.0, help="1 = off (tail-only)")
    ap.add_argument("--hi-z-thresh", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--encoder-mask-ratio", type=float, default=0.5)
    ap.add_argument("--redshift-weight", type=float, default=5.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-cap", type=int, default=60000)
    ap.add_argument("--z-fit-files", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    is_dist, rank, world, device = setup_ddp()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    # ---- data split (same as original run) ----
    records = load_manifest(Path(args.manifest))
    train_records, val_records = split_records_by_healpix(
        records, holdout_frac=0.05, seed=args.seed)
    train_shards = _records_to_shards(train_records, args.tokenized_dir)
    val_shards = _records_to_shards(val_records, args.tokenized_dir)
    # deterministic shard subset for a bounded Stage-A run
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_shards)
    train_shards = train_shards[: args.n_train_shards]
    log(f"[ft] {args.tag}: {len(train_shards)} train shards, {len(val_shards)} val shards")

    # ---- z-v2 tokenizer (gaussian_range widened; fit on cached TRAIN z) ----
    zs = collect_redshifts_from_cache(train_shards, max_files=args.z_fit_files)
    z_tok = RedshiftTokenizer(n_levels=args.z_bins, gaussian_range=args.gaussian_range)
    z_tok.fit(zs)
    edges = z_tok.get_bin_edges()
    log(f"[ft] z-v2: bins={args.z_bins} gr={args.gaussian_range} maxZ={edges[-1].item():.3f} "
        f"bins>=2:{int((edges[:-1] >= 2.0).sum())} (fit N={len(zs):,}, zmax={zs.max():.3f})")

    # ---- model warm-start from best.pt ----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    soft_sigma = float(ckpt.get("redshift_soft_sigma", 0.0))
    model = SpectrumTransformer(**_infer_transformer_dims(sd)).to(device)
    model.load_state_dict(sd)
    log(f"[ft] loaded {args.checkpoint} soft_sigma={soft_sigma} world={world}")

    # ---- val arrays for per-type eval (rank 0 only) ----
    if rank == 0:
        val_idx, val_z, val_cls = load_val_arrays(val_shards, args.eval_cap, args.typemap)
        val_zt = val_z.numpy()
        log(f"[ft] val eval set N={len(val_z):,}  "
            f"(QSO {int((val_cls=='QSO').sum()):,} / ELG {int((val_cls=='ELG').sum()):,})")

    # ---- train dataset (+ high-z oversampling) ----
    base = DR1CachedTokenDataset(train_shards, require_good_zwarn=True,
                                 require_nonzero_flux=True, cache_size=4)
    train_ds = OversampledCache(base, hi_thresh=args.hi_z_thresh,
                                mult=int(round(args.oversample_hi)))
    log(f"[ft] base N={len(base):,} -> oversampled N={len(train_ds):,} "
        f"(hi rows={getattr(train_ds, 'n_hi', 0):,}, x{int(round(args.oversample_hi))})")

    sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=4, collate_fn=collate_cached_skip_none,
                        drop_last=True, persistent_workers=True)

    if is_dist:
        model = DDP(model, device_ids=[device.index])
    core = model.module if is_dist else model
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(20, args.steps // 20)
    gen = torch.Generator(device=device)

    step = 0; t0 = time.time(); best_nm = float("inf")
    done = False
    for epoch in range(10_000):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for raw in loader:
            if raw is None:
                continue
            model.train()
            for g_ in optim.param_groups:
                g_["lr"] = lr_at(step, args.lr, warmup, args.steps)
            gen.manual_seed(1000 * step + rank)
            enc, dec, tgt, _ = tokenize_and_build(
                raw, None, z_tok, "a", device,
                encoder_mask_ratio=args.encoder_mask_ratio,
                redshift_mask_ratio=1.0, rng=gen)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, loss = model(enc, dec, targets=tgt,
                                redshift_weight=args.redshift_weight,
                                redshift_soft_sigma=soft_sigma)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); step += 1

            if step % args.eval_every == 0 or step == args.steps:
                if rank == 0:
                    zp = blind_z(core, val_idx, val_z, z_tok, device)
                    log(f"[ft] step {step}/{args.steps} loss {loss.item():.4f} "
                        f"({(time.time()-t0)/step:.2f}s/step)")
                    nm = per_type_report(zp, val_zt, val_cls, log)
                    if nm < best_nm:
                        best_nm = nm
                if is_dist:
                    dist.barrier()
            if step >= args.steps:
                done = True; break
        if done:
            break

    log(f"[ft] done {step} steps in {time.time()-t0:.0f}s  best_all_sigNMAD={best_nm:.5f}")

    # ---- final dump (rank 0) ----
    if rank == 0:
        zp = blind_z(core, val_idx, val_z, z_tok, device)
        np.savez(args.out_dir / f"zr_{args.tag}.npz",
                 z_pred=zp, z_true=val_zt, cls=val_cls)
        torch.save({"model": {k: v.detach().cpu() for k, v in core.state_dict().items()},
                    "z_tokenizer": {"sorted_z": z_tok._sorted_z.cpu(),
                                    "n_levels": z_tok.n_levels,
                                    "gaussian_range": z_tok.gaussian_range},
                    "redshift_soft_sigma": soft_sigma,
                    "ft": vars(args) | {"steps": step}},
                   args.out_dir / f"best_{args.tag}.pt")
        nm = per_type_report(zp, val_zt, val_cls, log)
        (args.out_dir / f"metrics_{args.tag}.json").write_text(json.dumps(
            {"tag": args.tag, "all_sigNMAD": nm, "oversample_hi": args.oversample_hi,
             "gaussian_range": args.gaussian_range, "steps": step}, indent=2))
        log(f"[ft] wrote best_{args.tag}.pt / zr_{args.tag}.npz / metrics_{args.tag}.json")

    if is_dist:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
