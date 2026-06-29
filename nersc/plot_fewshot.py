#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plots for the SDSS few-shot sweep.

Reads zr_{tag}_{shots}.npz + metrics_{tag}_{shots}.json from --dir and produces:
  * per-stage redshift plots: parity scatter (colored by Δz/(1+z)) + attached vertical
    error-distribution histogram — one per (tag, shots).
  * two-task learning curves vs #shots: redshift σ_NMAD & η; recon flux R² & RMS.

Usage: python nersc/plot_fewshot.py --dir $SCRATCH/sdss_ft/out --shots 0 500 1000 2000 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

LABEL = {"v3": "Transformer V3 (4096-soft)", "v2": "Transformer V2 (512-hard)"}
COLOR = {"v3": "C0", "v2": "C3"}


def stats(zp, zt):
    dz = (zp - zt) / (1.0 + zt)
    nm = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    eta = float(np.mean(np.abs(dz) > 0.0033))
    return nm, eta, dz


def parity_plot(d, tag, shots, out):
    z = np.load(d / f"zr_{tag}_{shots}.npz")
    zp, zt = z["z_pred"].astype(float), z["z_true"].astype(float)
    nm, eta, dz = stats(zp, zt)
    hi = float(max(zt.max(), zp.max())) * 1.02
    fig = plt.figure(figsize=(11, 7.2))
    gs = GridSpec(1, 3, width_ratios=[4.0, 1.15, 0.16], wspace=0.04,
                  left=0.07, right=0.93, top=0.88, bottom=0.10)
    ax, axh, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    sc = ax.scatter(zt, zp, c=np.clip(dz, -0.01, 0.01), cmap="RdBu", vmin=-0.01, vmax=0.01,
                    s=4, alpha=0.4, edgecolors="none", rasterized=True)
    ax.plot([0, hi], [0, hi], "k--", lw=0.9, alpha=0.7)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
    ax.set_xlabel("true redshift z"); ax.set_ylabel("predicted z  (z hidden)")
    ax.set_title("predicted vs true", fontsize=11); ax.grid(alpha=0.25)
    axh.hist(np.clip(dz, -0.02, 0.02), bins=140, orientation="horizontal", color="#3a5fcd", log=True)
    axh.axhline(0, color="k", lw=0.8, ls="--", alpha=0.7); axh.set_ylim(-0.02, 0.02)
    axh.set_xlabel("count (log)"); axh.set_title("error distribution", fontsize=11)
    axh.yaxis.set_label_position("right"); axh.yaxis.tick_right()
    axh.set_ylabel(r"$\Delta z/(1+z)$  (clip $\pm$0.02)"); axh.grid(alpha=0.25)
    cb = fig.colorbar(sc, cax=cax); cb.set_label(r"$\Delta z/(1+z)$")
    fig.suptitle(f"{LABEL[tag]} — SDSS blind redshift, {shots} shots\n"
                 f"N={len(zt):,}   σ_NMAD={nm:.5f}   η>0.0033={eta:.1%}", fontsize=13, y=0.98)
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out, flush=True)


def curves(d, tags, shots_list, out_prefix):
    M = {}
    for tag in tags:
        rows = []
        for s in shots_list:
            f = d / f"metrics_{tag}_{s}.json"
            if f.exists():
                rows.append(json.loads(f.read_text()))
        M[tag] = sorted(rows, key=lambda r: r["shots"])

    # redshift curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for tag in tags:
        xs = [r["shots"] for r in M[tag]]
        a1.plot(xs, [r["z_nmad"] for r in M[tag]], "o-", color=COLOR[tag], label=LABEL[tag])
        a2.plot(xs, [r["z_eta"] * 100 for r in M[tag]], "o-", color=COLOR[tag], label=LABEL[tag])
    a1.set_ylabel("σ_NMAD (lower better)"); a2.set_ylabel("η>0.0033  (%)")
    for a in (a1, a2):
        a.set_xlabel("# SDSS fine-tune spectra (shots)"); a.grid(alpha=0.3); a.legend()
    fig.suptitle("SDSS few-shot — blind redshift vs #shots", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_redshift.png", dpi=150); plt.close(fig)
    print("wrote", f"{out_prefix}_redshift.png", flush=True)

    # recon curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for tag in tags:
        xs = [r["shots"] for r in M[tag]]
        a1.plot(xs, [r["recon_flux_r2"] for r in M[tag]], "o-", color=COLOR[tag], label=LABEL[tag])
        a2.plot(xs, [r["recon_flux_rms"] for r in M[tag]], "o-", color=COLOR[tag], label=LABEL[tag])
    a1.set_ylabel("flux R² (median, masked px)"); a2.set_ylabel("flux RMS (median, masked px)")
    for a in (a1, a2):
        a.set_xlabel("# SDSS fine-tune spectra (shots)"); a.grid(alpha=0.3); a.legend()
    fig.suptitle("SDSS few-shot — masked spectrum reconstruction vs #shots", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_recon.png", dpi=150); plt.close(fig)
    print("wrote", f"{out_prefix}_recon.png", flush=True)

    # text table
    print("\nmodel  shots   z_nmad   eta     flux_r2  flux_rms  tok_acc")
    for tag in tags:
        for r in M[tag]:
            print(f"{tag:4} {r['shots']:6d}  {r['z_nmad']:.5f}  {r['z_eta']*100:5.1f}%  "
                  f"{r['recon_flux_r2']:6.3f}  {r['recon_flux_rms']:7.4f}  {r['recon_token_acc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--tags", nargs="+", default=["v3", "v2"])
    ap.add_argument("--shots", nargs="+", type=int, default=[0, 500, 1000, 2000, 5000])
    args = ap.parse_args()
    for tag in args.tags:
        for s in args.shots:
            if (args.dir / f"zr_{tag}_{s}.npz").exists():
                parity_plot(args.dir, tag, s, args.dir / f"plot_fewshot_{tag}_{s}.png")
    curves(args.dir, args.tags, args.shots, str(args.dir / "curve"))


if __name__ == "__main__":
    main()
