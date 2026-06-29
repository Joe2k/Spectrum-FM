import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"v2": "Transformer V2 (512-bin hard)", "v3": "Transformer V3 (4096-bin soft)"}
DSLAB = {
    "full": "full DR1 (train + val)",
    "val": "DR1 held-out val split",
    "sdss": "SDSS out-of-distribution",
}


def load(f):
    d = np.load(f)
    return d["z_pred"].astype(float), d["z_true"].astype(float)


def stats(zp, zt):
    dz = (zp - zt) / (1.0 + zt)
    m = np.median(dz)
    nm = 1.4826 * np.median(np.abs(dz - m))
    eta = np.mean(np.abs(dz) > 0.0033)
    return nm, eta, dz


def one(base, tag, ds):
    zp, zt = load(f"{base}/zr_{tag}_{ds}.npz")
    nm, eta, dz = stats(zp, zt)
    fig, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(13, 5.5))
    lo = float(min(zt.min(), zp.min()))
    hi = float(max(zt.max(), zp.max()))
    ax_s.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, zorder=3)
    ax_s.scatter(zt, zp, s=2, alpha=0.04, c="C0", edgecolors="none", rasterized=True)
    ax_s.set_xlim(lo, hi)
    ax_s.set_ylim(lo, hi)
    ax_s.set_xlabel("true z")
    ax_s.set_ylabel("predicted z  (spectrum shown, z hidden)")
    ax_s.set_title("predicted vs true")
    ax_s.grid(alpha=0.3)
    ax_h.hist(np.clip(dz, -0.02, 0.02), bins=140, color="C0", alpha=0.85)
    ax_h.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax_h.set_yscale("log")
    ax_h.set_xlabel(r"$\Delta z/(1+z)$  (clipped to $\pm$0.02)")
    ax_h.set_ylabel("count (log)")
    ax_h.set_title("error distribution")
    ax_h.grid(alpha=0.3)
    fig.suptitle(f"{LABEL[tag]} — blind redshift on {DSLAB[ds]}\n"
                 f"N={len(zt):,}   σ_NMAD={nm:.5f}   η>0.0033={eta:.1%}",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{base}/plot_{tag}_{ds}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out, flush=True)


base = sys.argv[1]
for tag in ("v3", "v2"):
    for ds in ("full", "val", "sdss"):
        try:
            one(base, tag, ds)
        except Exception as e:
            print(f"SKIP {tag}_{ds}: {type(e).__name__}: {e}", flush=True)
