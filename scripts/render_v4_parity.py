#!/usr/bin/env python3
"""Render Transformer V4 (z-v2) blind-z parity PNGs from the per-type eval dump.

Input: zr_type_eval.py npz (z_pred/z_true/targetid/spectype for the full DR1 val
split, blind-z = full spectrum + z hidden) + typemap.npz (targetid -> BGS/LRG/ELG/QSO).
Outputs (plots/):
  v4_parity_by_type.png    2x2 hexbin parity per object type (sigma_NMAD + gross-eta)
  v4_parity_qso_ceiling.png  QSO parity zoom showing the z-v1 2.13 wall is gone

Usage: python scripts/render_v4_parity.py [--npz PATH] [--typemap PATH] [--out DIR]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ZV1_CEIL = 2.13   # old z-v1 (gaussian_range 3.0) reachable redshift ceiling
CLASSES = ["BGS", "LRG", "ELG", "QSO"]
ZMAX = {"BGS": 1.6, "LRG": 1.6, "ELG": 2.0, "QSO": 4.6}


def sigma_nmad(zp, zt):
    dz = (zp - zt) / (1.0 + zt)
    return 1.4826 * np.median(np.abs(dz - np.median(dz)))


def gross_eta(zp, zt):
    return float(np.mean(np.abs((zp - zt) / (1.0 + zt)) > 0.05))


def classes_of(tids, sty, t2c):
    out = np.array([t2c.get(int(t), "OTHER") for t in tids], dtype="U6")
    out[(out == "OTHER") & (sty == "QSO")] = "QSO"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/type_v4200_r0.npz")
    ap.add_argument("--typemap", default="/tmp/typemap.npz")
    ap.add_argument("--out", default="plots")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.npz)
    tm = np.load(a.typemap, allow_pickle=True)
    t2c = dict(zip(tm["targetid"].tolist(), tm["cls"].tolist()))
    zp, zt = d["z_pred"].astype(float), d["z_true"].astype(float)
    cl = classes_of(d["targetid"], d["spectype"], t2c)

    # ---- Figure 1: 2x2 per-type parity ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, c in zip(axes.ravel(), CLASSES):
        m = cl == c
        x, y = zt[m], zp[m]
        zmax = ZMAX[c]
        hb = ax.hexbin(x, y, gridsize=120, bins="log", cmap="viridis",
                       extent=(0, zmax, 0, zmax), mincnt=1)
        ax.plot([0, zmax], [0, zmax], "r--", lw=1, alpha=0.7)
        nm, eta = sigma_nmad(x, y), gross_eta(x, y)
        ax.set_title(f"{c}   (N={m.sum():,})", fontsize=12, weight="bold")
        ax.set_xlabel("redshift (truth)"); ax.set_ylabel("redshift (predicted, z hidden)")
        ax.set_xlim(0, zmax); ax.set_ylim(0, zmax)
        ax.text(0.04, 0.96, f"$\\sigma_{{NMAD}}$ = {nm:.5f}\n$\\eta_{{>0.05}}$ = {eta*100:.2f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
        fig.colorbar(hb, ax=ax, label="count", shrink=0.85)
    fig.suptitle("Transformer V4 (z-v2) — blind-z parity by object type  ·  DR1 val (full spectrum, z hidden)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    f1 = out / "v4_parity_by_type.png"
    fig.savefig(f1, dpi=150); plt.close(fig)
    print("wrote", f1)

    # ---- Figure 2: QSO high-z ceiling ----
    m = cl == "QSO"; x, y = zt[m], zp[m]
    fig = plt.figure(figsize=(12, 5.2))
    # left: parity with the old wall marked
    ax1 = fig.add_subplot(1, 2, 1)
    hb = ax1.hexbin(x, y, gridsize=130, bins="log", cmap="magma",
                    extent=(0, 4.6, 0, 4.6), mincnt=1)
    ax1.plot([0, 4.6], [0, 4.6], "c--", lw=1, alpha=0.8, label="1:1")
    ax1.axhline(ZV1_CEIL, color="lime", ls=":", lw=1.6, label=f"old z-v1 ceiling ({ZV1_CEIL})")
    ax1.axvline(ZV1_CEIL, color="lime", ls=":", lw=1.6)
    ax1.set_xlim(0, 4.6); ax1.set_ylim(0, 4.6)
    ax1.set_xlabel("QSO redshift (truth)"); ax1.set_ylabel("QSO redshift (predicted, z hidden)")
    ax1.set_title("QSO parity — high-z tail now recovered", fontsize=12, weight="bold")
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.colorbar(hb, ax=ax1, label="count", shrink=0.85)
    # right: marginal z distributions above the wall
    ax2 = fig.add_subplot(1, 2, 2)
    bins = np.linspace(0, 4.6, 70)
    ax2.hist(x, bins=bins, histtype="step", lw=1.8, color="k", label="truth")
    ax2.hist(y, bins=bins, histtype="step", lw=1.8, color="crimson", label="predicted")
    ax2.axvline(ZV1_CEIL, color="lime", ls=":", lw=1.6, label=f"old ceiling ({ZV1_CEIL})")
    fpt = np.mean(y > ZV1_CEIL); ftt = np.mean(x > ZV1_CEIL)
    ax2.set_xlabel("QSO redshift"); ax2.set_ylabel("count")
    ax2.set_title(f"high-z coverage:  pred {fpt*100:.1f}%  vs  truth {ftt*100:.1f}%  above {ZV1_CEIL}",
                  fontsize=11, weight="bold")
    ax2.legend(fontsize=9)
    fig.suptitle(f"Transformer V4 (z-v2) — QSO z-ceiling fix  ·  N={m.sum():,}  ·  max z_pred = {y.max():.2f}",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f2 = out / "v4_parity_qso_ceiling.png"
    fig.savefig(f2, dpi=150); plt.close(fig)
    print("wrote", f2)


if __name__ == "__main__":
    main()
