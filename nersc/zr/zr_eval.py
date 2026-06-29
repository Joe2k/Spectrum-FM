from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

REPO = Path("/global/homes/j/joe2k/Spectrum-FM")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nersc"))

from src.inference.release import _infer_transformer_dims, _restore_z_tokenizer  # noqa
from src.models.transformer import SpectrumTransformer, SOS_TOKEN  # noqa
from src.tokenizers.spectrum import SpectrumTokenizer  # noqa
from src.training.data_split import split_records_by_healpix  # noqa
from src.training.sequences import tokenize_and_build  # noqa
from src.eval.redshift_metrics import decode_redshift  # noqa
from dr1_dataset import load_manifest  # noqa
from dr1_tokenized_dataset import DR1CachedTokenDataset, collate_cached_skip_none  # noqa
from train_transformer import _records_to_shards  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mode", choices=["desi", "sdss"], required=True)
    p.add_argument("--full-corpus", action="store_true")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--tokenized-dir", type=Path)
    p.add_argument("--tokenizer-ckpt", type=Path)
    p.add_argument("--sdss-dir", type=Path)
    p.add_argument("--sdss-manifest", type=Path, help="newline list of spec fits paths")
    p.add_argument("--glob", default="**/spec-*.fits")
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--train-split", action="store_true", help="use the 95%% train side of the healpix split")
    p.add_argument("--approach", default="a")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model = SpectrumTransformer(**_infer_transformer_dims(sd)).to(device)
    model.load_state_dict(sd)
    model.eval()
    z_tok = _restore_z_tokenizer(ckpt)

    spec_tok = None
    wave_aware = False
    if args.mode == "desi":
        records = load_manifest(args.manifest)
        if args.full_corpus:
            use = records
        elif args.train_split:
            use, _ = split_records_by_healpix(records, holdout_frac=args.holdout_frac, seed=42)
        else:
            _, use = split_records_by_healpix(records, holdout_frac=args.holdout_frac, seed=42)
        use = use[args.shard_id::args.num_shards]                  # shard by healpix record
        ds = DR1CachedTokenDataset(
            _records_to_shards(use, args.tokenized_dir),
            require_good_zwarn=True, require_nonzero_flux=True,
            max_spectra=args.max_spectra)
        collate = collate_cached_skip_none
    else:
        from src.utils.sdss import SDSSSpectrumDataset, collate_sdss_skip_none
        spec_tok = SpectrumTokenizer().to(device)
        tck = torch.load(args.tokenizer_ckpt, map_location=device, weights_only=False)
        spec_tok.load_state_dict(tck.get("model", tck) if isinstance(tck, dict) else tck)
        spec_tok.eval()
        for pm in spec_tok.parameters():
            pm.requires_grad_(False)
        if args.sdss_manifest is not None:
            paths = [ln.strip() for ln in args.sdss_manifest.read_text().splitlines() if ln.strip()]
            paths = paths[args.shard_id::args.num_shards]          # shard by file
            source = paths
        else:
            source = args.sdss_dir
        ds = SDSSSpectrumDataset(
            source, require_good_zwarn=True, require_nonzero_flux=True,
            max_spectra=args.max_spectra, glob=args.glob)
        collate = collate_sdss_skip_none
        wave_aware = True

    print(f"[zr] {args.out.name} mode={args.mode} full={args.full_corpus} "
          f"shard={args.shard_id}/{args.num_shards} N={len(ds)} dev={device}", flush=True)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda")

    zps, zts = [], []
    t0 = time.time(); ns = 0
    with torch.no_grad():
        for nb, batch in enumerate(loader):
            if batch is None:
                continue
            enc, _dec, _tgt, _ = tokenize_and_build(
                batch, spec_tok, z_tok, args.approach, device,
                wavelength_aware=wave_aware,
                encoder_mask_ratio=0.0, redshift_mask_ratio=1.0)
            # Redshift is decoder position 0 → a single [SOS] decode step suffices
            # (no need to run the decoder over the full 274-token spectrum).
            sos = torch.full((enc.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits, _ = model(enc, sos)
            red_logits = logits[:, 0, :].float()
            zp = decode_redshift(red_logits, z_tok, mode="expected").cpu()
            zt = batch["z"].detach().flatten().cpu().float()
            zps.append(zp); zts.append(zt)
            ns += len(zt)
            if (nb + 1) % 100 == 0:
                print(f"[zr] {args.out.name} {ns} spectra  {ns/(time.time()-t0):.0f}/s", flush=True)

    zp = torch.cat(zps).numpy()
    zt = torch.cat(zts).numpy()
    dz = (zp - zt) / (1.0 + zt)
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    np.savez(args.out, z_pred=zp, z_true=zt)
    print(f"[zr] DONE {args.out.name} N={len(zt)} nmad={nmad:.6f} "
          f"eta={np.mean(np.abs(dz)>0.0033):.3f} {len(zt)/(time.time()-t0):.0f}/s", flush=True)


if __name__ == "__main__":
    main()
