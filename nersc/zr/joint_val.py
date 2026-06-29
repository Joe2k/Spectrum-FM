#!/usr/bin/env python
"""Parity + count-marginals "joint" plot for blind redshift on the DESI DR1 val
split (the slide-42 / slide-43 style). Generalized from joint_train.py to any
checkpoint's per-object eval npz (z_pred / z_true), any split label.

Usage:
  python joint_val.py --npz type_v4200_r0.npz --label "Transformer V4 (z-v2, high-z ceiling)" \
      --split "DESI DR1 val split" --out plots/plot_joint_v4_val.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

rng = np.random.default_rng(0)


def stats(zp, zt):
    dz = (zp - zt) / (1.0 + zt)
    nm = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    eta = float(np.mean(np.abs(dz) > 0.0033))
    return nm, eta, dz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--split", default="DESI DR1 val split")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    d = np.load(a.npz)
    zp, zt = d["z_pred"].astype(float), d["z_true"].astype(float)
    nm, eta, dz = stats(zp, zt)                      # full-sample stats
    hi = float(max(zt.max(), zp.max())) * 1.02
    bins = np.linspace(0, hi, 160)

    n = len(zt)
    idx = rng.choice(n, size=min(200_000, n), replace=False)
    zts, zps, dzs = zt[idx], zp[idx], dz[idx]

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(2, 3, width_ratios=[4, 1, 0.22], height_ratios=[1, 4],
                  wspace=0.04, hspace=0.04, figure=fig)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_cb = fig.add_subplot(gs[1, 2])

    sc = ax.scatter(zts, zps, c=np.clip(dzs, -0.01, 0.01), cmap="RdBu",
                    vmin=-0.01, vmax=0.01, s=6, alpha=0.4,
                    edgecolors="none", rasterized=True)
    ax.plot([0, hi], [0, hi], "k--", lw=1.1, alpha=0.7)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("true redshift z", fontsize=15)
    ax.set_ylabel("predicted z  (z hidden)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, cax=ax_cb)
    cb.set_label(r"$\Delta z/(1+z)$", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    ax_top.hist(zt, bins=bins, color="#6a8fd0", edgecolor="none")
    ax_top.set_ylabel("count", fontsize=20)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=18)
    ax_top.grid(alpha=0.25)
    ax_top.set_title(f"{a.label} — {a.split}, blind redshift\n"
                     f"N={n:,}   $\\sigma_{{NMAD}}$={nm:.5f}   "
                     f"$\\eta$>0.0033={eta:.1%}", fontsize=16)

    ax_right.hist(zp, bins=bins, orientation="horizontal",
                  color="#cd7a6a", edgecolor="none")
    ax_right.set_xlabel("count", fontsize=20)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelsize=18)
    ax_right.grid(alpha=0.25)

    fig.savefig(a.out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {a.out}  N={n:,} nmad={nm:.5f} eta={eta:.2%}")


if __name__ == "__main__":
    main()
