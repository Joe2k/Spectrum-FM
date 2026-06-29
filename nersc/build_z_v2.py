"""
Build the z-v2 redshift tokenizer (high-z tail support).
=======================================================

The released V2/V3 z-tokenizers use ``gaussian_range=3.0`` (CDF->Gaussian->FSQ),
so the top FSQ level decodes to the Phi(3.0)=99.865th percentile of the fitted z
sample -> a hard ceiling at z~=2.13 (V3) / 1.66 (V2). Anything rarer is unemittable,
which pins high-z QSOs at the ceiling (see RESEARCH_LOG 2026-06-28).

z-v2 widens ``gaussian_range`` to 4.0 (Phi(4)=0.99997), keeping the SAME 4096 bins
and the SAME empirical CDF fit sample. This extends the reachable z to ~4.4 (covers
99.994% of DESI objects) with hundreds of bins above z=2, while the low-z bin width
only grows ~0.00027 -> ~0.00035 (negligible vs the sub-bin expected-value decode at
sigma_NMAD ~3.5e-4). It is a parameter-free remap of the bin<->z mapping; the
transformer's z-bin output/embedding weights are then re-fit by the Stage-A
fine-tune. No architecture change (bin count unchanged).

The fit sample is taken, by default, from the bundled z-tokenizer of a release
checkpoint (same DESI z-distribution the original was fit on, incl. high-z QSOs up
to z~6). Optionally pass a raw .npy of corpus redshifts for a fresh/broader fit
(e.g. the Stage-B full run).

Output: a small ``.pt`` with the z_tokenizer state dict
``{n_levels, gaussian_range, sorted_z}`` consumable by ``_restore_z_tokenizer``-style
loaders (and by ``finetune_zhead.py``).
"""

from __future__ import annotations

import argparse

import torch

from src.inference.release import _restore_z_tokenizer
from src.tokenizers.redshift import RedshiftTokenizer


def _load_fit_sample(args) -> torch.Tensor:
    if args.z_npy:
        import numpy as np
        z = torch.from_numpy(np.load(args.z_npy)).float().flatten()
        print(f"[z-v2] fit sample from {args.z_npy}: N={len(z):,}")
        return z
    ck = torch.load(args.from_ckpt, map_location="cpu", weights_only=False)
    z = _restore_z_tokenizer(ck)._sorted_z.float().flatten()
    print(f"[z-v2] fit sample from bundled z-tokenizer of {args.from_ckpt}: N={len(z):,}")
    return z


def main():
    p = argparse.ArgumentParser(description="Build the z-v2 (high-z) redshift tokenizer.")
    p.add_argument("--from-ckpt", default="checkpoints/release/transformer_v3_4096soft/best.pt",
                   help="Release checkpoint whose bundled z-tokenizer supplies the fit sample.")
    p.add_argument("--z-npy", default=None,
                   help="Optional .npy of corpus redshifts to fit on instead of --from-ckpt.")
    p.add_argument("--n-levels", type=int, default=4096, help="Number of z bins (keep 4096).")
    p.add_argument("--gaussian-range", type=float, default=4.0, help="Widened FSQ range (z-v2=4.0).")
    p.add_argument("--out", default="checkpoints/z_tokenizer_v2.pt", help="Output .pt path.")
    args = p.parse_args()

    z = _load_fit_sample(args)
    zt = RedshiftTokenizer(n_levels=args.n_levels, gaussian_range=args.gaussian_range)
    zt.fit(z)

    edges = zt.get_bin_edges()
    n_above2 = int((edges[:-1] >= 2.0).sum())
    n_above3 = int((edges[:-1] >= 3.0).sum())
    print(f"[z-v2] n_levels={zt.n_levels} gaussian_range={zt.gaussian_range}")
    print(f"[z-v2] fit z range [{zt._min_z:.4f}, {zt._max_z:.4f}]  max decodable z={edges[-1].item():.4f}")
    print(f"[z-v2] bins >= z2.0: {n_above2}/{zt.n_levels}   >= z3.0: {n_above3}/{zt.n_levels}")

    # Round-trip sanity: high-z values must now survive encode->decode (not clamp).
    for ztest in (1.0, 2.0, 2.5, 3.0, 3.5):
        idx = zt.encode(torch.tensor([ztest]))
        zrec = zt.decode(idx).item()
        print(f"[z-v2]   z={ztest:>4}  -> bin {int(idx.item()):>5} -> decode {zrec:.4f}  (|err|={abs(zrec-ztest):.4f})")

    state = {"n_levels": zt.n_levels, "gaussian_range": zt.gaussian_range,
             "sorted_z": zt._sorted_z.clone().cpu()}
    torch.save({"z_tokenizer": state}, args.out)
    print(f"[z-v2] wrote {args.out}")


if __name__ == "__main__":
    main()
