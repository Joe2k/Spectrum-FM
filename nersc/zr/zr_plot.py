import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(f):
    d = np.load(f)
    return d["z_pred"].astype(float), d["z_true"].astype(float)


def stats(zp, zt):
    dz = (zp - zt) / (1.0 + zt)
    m = np.median(dz)
    nm = 1.4826 * np.median(np.abs(dz - m))
    eta = np.mean(np.abs(dz) > 0.0033)
    return nm, eta, dz


def panel(ax_s, ax_h, zp, zt, title):
    nm, eta, dz = stats(zp, zt)
    lo = float(min(zt.min(), zp.min()))
    hi = float(max(zt.max(), zp.max()))
    ax_s.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, zorder=3)
    ax_s.scatter(zt, zp, s=2, alpha=0.04, c="C0", edgecolors="none", rasterized=True)
    ax_s.set_xlim(lo, hi)
    ax_s.set_ylim(lo, hi)
    ax_s.set_xlabel("true z")
    ax_s.set_ylabel("predicted z  (spectrum shown, z hidden)")
    ax_s.set_title(f"{title}\nN={len(zt):,}   σ_NMAD={nm:.5f}   η>0.0033={eta:.1%}",
                   fontsize=10)
    ax_s.grid(alpha=0.3)
    ax_h.hist(np.clip(dz, -0.02, 0.02), bins=140, color="C0", alpha=0.85)
    ax_h.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax_h.set_yscale("log")
    ax_h.set_xlabel(r"$\Delta z/(1+z)$  (clipped to $\pm$0.02)")
    ax_h.set_ylabel("count (log)")
    ax_h.grid(alpha=0.3)


def figure(v2f, v3f, suptitle, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    z2p, z2t = load(v2f)
    z3p, z3t = load(v3f)
    panel(axes[0, 0], axes[1, 0], z2p, z2t, "Transformer V2  (512-bin hard)")
    panel(axes[0, 1], axes[1, 1], z3p, z3t, "Transformer V3  (4096-bin soft)")
    fig.suptitle(suptitle, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out, flush=True)


base = sys.argv[1]
jobs = [
    ("zr_v2_full.npz", "zr_v3_full.npz",
     "Blind redshift prediction — full DR1 (train + val, spectrum shown / z masked)",
     "plot_full_dr1.png"),
    ("zr_v2_val.npz", "zr_v3_val.npz",
     "Blind redshift prediction — DR1 held-out val split (spectrum shown / z masked)",
     "plot_val_split.png"),
    ("zr_v2_sdss.npz", "zr_v3_sdss.npz",
     "Blind redshift prediction — SDSS out-of-distribution (spectrum shown / z masked)",
     "plot_sdss.png"),
]
for v2, v3, title, out in jobs:
    try:
        figure(f"{base}/{v2}", f"{base}/{v3}", title, f"{base}/{out}")
    except Exception as e:
        print(f"SKIP {out}: {type(e).__name__}: {e}", flush=True)
