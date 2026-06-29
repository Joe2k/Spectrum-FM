from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path("/global/homes/j/joe2k/Spectrum-FM")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nersc"))

from src.inference.release import _infer_transformer_dims, _restore_z_tokenizer  # noqa
from src.models.transformer import SpectrumTransformer, SOS_TOKEN, SPECTRUM_TOKEN_OFFSET  # noqa
from src.tokenizers.spectrum import SpectrumTokenizer, N_TOKENS  # noqa
from src.training.data_split import split_records_by_healpix  # noqa
from src.training.sequences import tokenize_and_build  # noqa
from src.eval.redshift_metrics import decode_redshift  # noqa
from dr1_dataset import load_manifest  # noqa
from train_transformer import _records_to_shards  # noqa

PXPT = 8704 // N_TOKENS  # 32 flux pixels per spectrum token
MASK_RATIO = 0.5


def good_rows(d):
    z = d["zwarn"] == 0
    nz = d["nonzero_flux"] == 1 if "nonzero_flux" in d.files else np.ones(len(d["z"]), bool)
    return z & nz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer-ckpt", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tokenized-dir", type=Path, required=True)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    model = SpectrumTransformer(**_infer_transformer_dims(sd)).to(dev).eval()
    model.load_state_dict(sd)
    z_tok = _restore_z_tokenizer(ck)
    spec_tok = SpectrumTokenizer().to(dev).eval()
    tck = torch.load(args.tokenizer_ckpt, map_location=dev, weights_only=False)
    spec_tok.load_state_dict(tck.get("model", tck) if isinstance(tck, dict) else tck)
    for q in spec_tok.parameters():
        q.requires_grad_(False)

    _, val = split_records_by_healpix(load_manifest(args.manifest),
                                      holdout_frac=args.holdout_frac, seed=42)
    shards = _records_to_shards(val, args.tokenized_dir)
    shards = shards[args.shard_id::args.num_shards]
    if args.max_shards:
        shards = shards[: args.max_shards]
    print(f"[type] {args.out.name} shards={len(shards)} dev={dev}", flush=True)

    gen = torch.Generator(device=dev)
    Z_P, Z_T, TID, STY, MACC, FRMS, FR2 = ([] for _ in range(7))
    t0 = time.time(); ns = 0
    with torch.no_grad():
        for si, sp in enumerate(shards):
            d = np.load(sp, allow_pickle=False)
            g = good_rows(d)
            idx = d["indices"][g].astype(np.int64)
            zt = d["z"][g].astype(np.float32)
            den = d["denorm"][g].astype(np.float32)            # (n,1)
            tid = d["targetid"][g].astype(np.int64)
            sty = d["spectype"][g].astype("U8")
            for s in range(0, len(zt), args.batch_size):
                e = s + args.batch_size
                ii = torch.from_numpy(idx[s:e]).to(dev)
                zz = torch.from_numpy(zt[s:e])
                dn = torch.from_numpy(den[s:e]).to(dev)
                rb = {"spec_indices": ii, "z": zz}
                # ---- redshift (z masked, full spectrum) ----
                enc, _, _, _ = tokenize_and_build(rb, None, z_tok, "a", dev,
                                                  encoder_mask_ratio=0.0, redshift_mask_ratio=1.0)
                sos = torch.full((enc.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lg, _ = model(enc, sos)
                zp = decode_redshift(lg[:, 0, :].float(), z_tok, mode="expected").cpu()
                # ---- spectrum reconstruction (50% encoder tokens masked) ----
                gen.manual_seed(1234 + si * 100000 + s)
                enc2, dec2, tgt2, mp = tokenize_and_build(
                    rb, None, z_tok, "a", dev, encoder_mask_ratio=MASK_RATIO,
                    redshift_mask_ratio=1.0, rng=gen)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lg2, _ = model(enc2, dec2, targets=tgt2)
                spec_pred = lg2[:, 1:1 + N_TOKENS, :].float().argmax(-1)   # global ids
                spec_true = tgt2[:, 1:1 + N_TOKENS]                        # global ids
                mp = mp.to(dev)
                # masked-token accuracy (per sample, over masked tokens)
                correct = (spec_pred == spec_true) & mp
                macc = correct.sum(1).float() / mp.sum(1).clamp(min=1).float()
                # flux decode: predicted tokens at masked, true elsewhere
                recon = torch.where(mp, spec_pred, spec_true)
                lfq_r = (recon - SPECTRUM_TOKEN_OFFSET).clamp(0, 1023)
                lfq_t = (spec_true - SPECTRUM_TOKEN_OFFSET).clamp(0, 1023)
                f_r = spec_tok.decode(lfq_r, dn)[:, 0, :]                  # (B,8704)
                f_t = spec_tok.decode(lfq_t, dn)[:, 0, :]
                pxm = mp.repeat_interleave(PXPT, dim=1)                    # (B,8704) bool
                resid = (f_t - f_r) * pxm
                n_px = pxm.sum(1).clamp(min=1).float()
                ss_res = (resid ** 2).sum(1)
                rms = torch.sqrt(ss_res / n_px)
                mu = (f_t * pxm).sum(1) / n_px
                ss_tot = (((f_t - mu.unsqueeze(1)) * pxm) ** 2).sum(1).clamp(min=1e-12)
                r2 = 1.0 - ss_res / ss_tot
                Z_P.append(zp.numpy()); Z_T.append(zt[s:e]); TID.append(tid[s:e]); STY.append(sty[s:e])
                MACC.append(macc.cpu().numpy()); FRMS.append(rms.cpu().numpy()); FR2.append(r2.cpu().numpy())
                ns += len(zz)
            if (si + 1) % 50 == 0:
                print(f"[type] {args.out.name} {ns} spectra {ns/(time.time()-t0):.0f}/s", flush=True)

    np.savez(args.out,
             z_pred=np.concatenate(Z_P), z_true=np.concatenate(Z_T),
             targetid=np.concatenate(TID), spectype=np.concatenate(STY),
             masked_acc=np.concatenate(MACC), flux_rms=np.concatenate(FRMS),
             flux_r2=np.concatenate(FR2))
    print(f"[type] DONE {args.out.name} N={ns} {ns/(time.time()-t0):.0f}/s", flush=True)


if __name__ == "__main__":
    main()
