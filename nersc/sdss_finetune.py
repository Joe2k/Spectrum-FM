#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Few-shot fine-tune a DESI-trained transformer on legacy SDSS, with dual eval.

Standalone (does NOT extend train_transformer.py) because SDSS comes as a flat
pre-tokenized cache, and because we must NOT refit the redshift tokenizer (refitting
on SDSS would remap bins and destroy the pretrained redshift head) — instead we
RESTORE the checkpoint's bundled DESI z-tokenizer and keep it frozen.

Per (model, shots):
  * full-model fine-tune (fresh AdamW, low LR) on the first --shots rows of the SDSS
    train cache, joint objective (encoder_mask_ratio=0.15, redshift_mask_ratio=1.0,
    up-weighted redshift loss); V3 keeps its soft labels via the checkpoint's sigma.
  * checkpoint selected by best held-out redshift sigma_NMAD on a fixed subset.
  * FINAL eval on the full SDSS test cache (DDP-sharded), BOTH tasks:
      - blind redshift  (spectrum shown, z masked)  -> sigma_NMAD / eta / bias
      - masked recon    (50% spectrum + z masked)    -> token acc / flux RMS / flux R^2
  * dumps best.pt, metrics.json, and z_pred/z_true npz for the per-stage plots.

--shots 0 = eval-only (zero-shot baseline on the new legacy test set).

DDP: launch one process per GPU (srun -n4); auto-detects RANK/WORLD_SIZE/LOCAL_RANK
(or SLURM_PROCID/SLURM_LOCALID), same as train_transformer.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src.eval.redshift_metrics import decode_redshift  # noqa: E402
from src.inference.release import _infer_transformer_dims, _restore_z_tokenizer  # noqa: E402
from src.models.transformer import (  # noqa: E402
    SOS_TOKEN, SPECTRUM_TOKEN_OFFSET, SpectrumTransformer)
from src.tokenizers.spectrum import N_TOKENS, SpectrumTokenizer  # noqa: E402
from src.training.sequences import lr_at, tokenize_and_build  # noqa: E402

PXPT = 8704 // N_TOKENS  # 32 flux pixels per spectrum token


# --------------------------------------------------------------------------- DDP
def setup_ddp():
    is_dist = "RANK" in os.environ or "SLURM_PROCID" in os.environ
    if is_dist:
        local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29511")
        cuda_idx = local if local < torch.cuda.device_count() else 0
        torch.cuda.set_device(cuda_idx)
        dist.init_process_group(backend="nccl")
        return True, rank, world, torch.device(f"cuda:{cuda_idx}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, dev


# --------------------------------------------------------------------- datasets
class CacheDataset(Dataset):
    """First `limit` rows of a pre-tokenized SDSS cache (nested across shot counts)."""

    def __init__(self, npz_path, limit=None):
        d = np.load(npz_path)
        idx = d["spec_indices"].astype(np.int64)
        z = d["z"].astype(np.float32)
        if limit is not None:
            idx, z = idx[:limit], z[:limit]
        self.idx = torch.from_numpy(idx)
        self.z = torch.from_numpy(z)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        return self.idx[i], self.z[i]


def collate_cache(batch):
    idx = torch.stack([b[0] for b in batch])
    z = torch.stack([b[1] for b in batch])
    return {"spec_indices": idx, "z": z}


def metrics_z(zp, zt):
    zp = np.asarray(zp, float); zt = np.asarray(zt, float)
    dz = (zp - zt) / (1.0 + zt)
    nm = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
    eta = float(np.mean(np.abs(dz) > 0.0033))
    return nm, eta, float(np.median(dz))


# ------------------------------------------------------------------- eval (z)
@torch.no_grad()
def eval_redshift(core, idx, z, z_tok, device, batch=512, max_n=None):
    """Blind redshift: full spectrum, z masked, single-[SOS] decode. Returns zp, zt."""
    core.eval()
    n = len(z) if max_n is None else min(max_n, len(z))
    ZP, ZT = [], []
    for s in range(0, n, batch):
        e = min(n, s + batch)
        rb = {"spec_indices": idx[s:e].to(device), "z": z[s:e]}
        enc, _, _, _ = tokenize_and_build(rb, None, z_tok, "a", device,
                                          encoder_mask_ratio=0.0, redshift_mask_ratio=1.0)
        sos = torch.full((enc.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = core(enc, sos)
        zp = decode_redshift(logits[:, 0, :].float(), z_tok, mode="expected").cpu().numpy()
        ZP.append(zp); ZT.append(z[s:e].numpy())
    return np.concatenate(ZP), np.concatenate(ZT)


@torch.no_grad()
def eval_recon(core, spec_tok, idx, z, den, z_tok, device, batch=128, max_n=None):
    """Masked spectrum reconstruction (50% spec + z masked). Returns acc, rms, r2 lists."""
    core.eval()
    n = len(z) if max_n is None else min(max_n, len(z))
    ACC, RMS, R2 = [], [], []
    gen = torch.Generator(device=device)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        rb = {"spec_indices": idx[s:e].to(device), "z": z[s:e]}
        gen.manual_seed(12345 + s)
        enc, dec, tgt, mp = tokenize_and_build(rb, None, z_tok, "a", device,
                                               encoder_mask_ratio=0.5,
                                               redshift_mask_ratio=1.0, rng=gen)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = core(enc, dec, targets=tgt)
        spec_pred = logits[:, 1:1 + N_TOKENS, :].float().argmax(-1)
        spec_true = tgt[:, 1:1 + N_TOKENS]
        mp = mp.to(device)
        acc = ((spec_pred == spec_true) & mp).sum(1).float() / mp.sum(1).clamp(min=1)
        recon = torch.where(mp, spec_pred, spec_true)
        dn = den[s:e].reshape(-1, 1).to(device)
        f_r = spec_tok.decode((recon - SPECTRUM_TOKEN_OFFSET).clamp(0, 1023), dn)[:, 0, :]
        f_t = spec_tok.decode((spec_true - SPECTRUM_TOKEN_OFFSET).clamp(0, 1023), dn)[:, 0, :]
        pxm = mp.repeat_interleave(PXPT, dim=1)
        npx = pxm.sum(1).clamp(min=1).float()
        resid = ((f_t - f_r) * pxm)
        ssr = (resid ** 2).sum(1)
        rms = torch.sqrt(ssr / npx)
        mu = (f_t * pxm).sum(1) / npx
        sst = (((f_t - mu.unsqueeze(1)) * pxm) ** 2).sum(1).clamp(min=1e-12)
        r2 = 1.0 - ssr / sst
        ACC.append(acc.cpu().numpy()); RMS.append(rms.cpu().numpy()); R2.append(r2.cpu().numpy())
    return (np.concatenate(ACC) if ACC else np.zeros(0),
            np.concatenate(RMS) if RMS else np.zeros(0),
            np.concatenate(R2) if R2 else np.zeros(0))


def gather_concat(arr, is_dist):
    if not is_dist:
        return arr
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, arr)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer-ckpt", type=Path, required=True)
    ap.add_argument("--train-cache", type=Path, required=True)
    ap.add_argument("--test-cache", type=Path, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="model tag, e.g. v3 / v2")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--encoder-mask-ratio", type=float, default=0.15)
    ap.add_argument("--redshift-weight", type=float, default=3.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--sel-subset", type=int, default=4000,
                    help="held-out subset size for checkpoint selection")
    args = ap.parse_args()

    is_dist, rank, world, device = setup_ddp()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    # ---- load model + frozen tokenizers ----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model = SpectrumTransformer(**_infer_transformer_dims(sd)).to(device)
    model.load_state_dict(sd)
    z_tok = _restore_z_tokenizer(ckpt)               # DESI CDF, frozen (NOT refit)
    soft_sigma = float(ckpt.get("redshift_soft_sigma", 0.0))
    spec_tok = SpectrumTokenizer().to(device).eval()
    tck = torch.load(args.tokenizer_ckpt, map_location=device, weights_only=False)
    spec_tok.load_state_dict(tck.get("model", tck) if isinstance(tck, dict) else tck)
    for pm in spec_tok.parameters():
        pm.requires_grad_(False)
    log(f"[ft] {args.tag} shots={args.shots} soft_sigma={soft_sigma} world={world}")

    # ---- test cache (kept on CPU, sharded for final eval) ----
    td = np.load(args.test_cache)
    test_idx = torch.from_numpy(td["spec_indices"].astype(np.int64))
    test_z = torch.from_numpy(td["z"].astype(np.float32))
    test_den = torch.from_numpy(td["denorm"].astype(np.float32))
    sel_idx, sel_z = test_idx[: args.sel_subset], test_z[: args.sel_subset]

    def cpu_sd(m):
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    core = model
    best_sd = cpu_sd(model)               # CPU copy → device-safe DDP broadcast
    best_nmad = float("inf")

    # ---- fine-tune (skip entirely for shots == 0) ----
    if args.shots > 0:
        train_ds = CacheDataset(args.train_cache, limit=args.shots)
        sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=(sampler is None), sampler=sampler,
                            collate_fn=collate_cache, drop_last=False)
        steps_per_epoch = max(1, len(loader))
        total = steps_per_epoch * args.epochs
        warmup = max(5, total // 20)
        log(f"[ft] train N={len(train_ds)} steps/epoch={steps_per_epoch} total={total}")

        if is_dist:
            model = DDP(model, device_ids=[device.index])
        core = model.module if is_dist else model
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        gen = torch.Generator(device=device)
        step = 0; t0 = time.time()
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for raw in loader:
                model.train()
                for g_ in optim.param_groups:
                    g_["lr"] = lr_at(step, args.lr, warmup, total)
                gen.manual_seed(1000 * step + rank)
                enc, dec, tgt, _ = tokenize_and_build(
                    raw, None, z_tok, "a", device,
                    encoder_mask_ratio=args.encoder_mask_ratio,
                    redshift_mask_ratio=1.0, rng=gen)
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    _, loss = model(enc, dec, targets=tgt,
                                    redshift_weight=args.redshift_weight,
                                    redshift_soft_sigma=soft_sigma)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                step += 1
                if step % args.eval_every == 0 or step == total:
                    if rank == 0:
                        zp, zt = eval_redshift(core, sel_idx, sel_z, z_tok, device)
                        nm, eta, _ = metrics_z(zp, zt)
                        improved = nm < best_nmad
                        if improved:
                            best_nmad = nm
                            best_sd = cpu_sd(core)
                        log(f"[ft] step {step}/{total} loss {loss.item():.4f} "
                            f"sel_nmad {nm:.5f} eta {eta:.3f}"
                            f"{'  *best' if improved else ''}")
                    if is_dist:
                        dist.barrier()
        log(f"[ft] training done in {time.time()-t0:.0f}s  best_sel_nmad={best_nmad:.5f}")
        optim.zero_grad(set_to_none=True)
        del optim
        if is_dist:
            model = core            # drop DDP wrapper; eval uses unwrapped core
        torch.cuda.empty_cache()    # free training activations before the recon eval

    # ---- load best weights everywhere, run FINAL full sharded eval ----
    if is_dist:
        bsd = [best_sd if rank == 0 else None]
        dist.broadcast_object_list(bsd, src=0)
        best_sd = bsd[0]
    core.load_state_dict(best_sd)

    my = slice(rank, len(test_z), world)
    zp_l, zt_l = eval_redshift(core, test_idx[my], test_z[my], z_tok, device)
    acc_l, rms_l, r2_l = eval_recon(core, spec_tok, test_idx[my], test_z[my],
                                    test_den[my], z_tok, device)
    zp = gather_concat(zp_l, is_dist); zt = gather_concat(zt_l, is_dist)
    acc = gather_concat(acc_l, is_dist); rms = gather_concat(rms_l, is_dist)
    r2 = gather_concat(r2_l, is_dist)

    if rank == 0:
        nm, eta, bias = metrics_z(zp, zt)
        res = {"tag": args.tag, "shots": args.shots, "n_test": int(len(zt)),
               "z_nmad": nm, "z_eta": eta, "z_bias": bias,
               "recon_token_acc": float(np.mean(acc)),
               "recon_flux_rms": float(np.median(rms)),
               "recon_flux_r2": float(np.median(r2)),
               "sel_nmad": best_nmad if args.shots > 0 else nm,
               "lr": args.lr, "epochs": args.epochs,
               "redshift_weight": args.redshift_weight}
        stem = f"{args.tag}_{args.shots}"
        (args.out_dir / f"metrics_{stem}.json").write_text(json.dumps(res, indent=2))
        np.savez(args.out_dir / f"zr_{stem}.npz", z_pred=zp, z_true=zt)
        if args.shots > 0:
            torch.save({"model": best_sd, "z_tokenizer": ckpt.get("z_tokenizer"),
                        "redshift_soft_sigma": soft_sigma, "shots": args.shots},
                       args.out_dir / f"best_{stem}.pt")
        log(f"[ft] FINAL {stem}: z_nmad={nm:.5f} eta={eta:.3f} bias={bias:+.5f} | "
            f"recon acc={res['recon_token_acc']:.3f} rms={res['recon_flux_rms']:.4f} "
            f"r2={res['recon_flux_r2']:.3f}  (N={len(zt)})")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
