#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/tmp/zr_plots/examples"
NPZ = f"{SRC}/sel_sdss_5000_typed.npz"
DS_LABEL = "SDSS 5000-shot fine-tuned (25k plate-disjoint test)"
CAT_DESC = {
    "cat1_predz0":    "Predicted z ≈ 0",
    "cat2_overpred":  "Predicted z ≫ true  (z_pred < 0.5)",
    "cat3_underpred": "True z ≫ predicted  (0.5 < z_true < 1)",
    "cat4_ceiling":   "Predicted z at ceiling (≈2) ≪ true z",
}
CATS = ["cat1_predz0", "cat2_overpred", "cat3_underpred", "cat4_ceiling"]
MODEL_LABEL = {"v2": "Transformer V2 (512-hard)", "v3": "Transformer V3 (4096-soft)"}


def smooth(y, w=10):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


d = np.load(NPZ, allow_pickle=True)
model = d["model"].astype(str); cat = d["cat"].astype(str)
tid = d["targetid"].astype(str)
stype = d["spectype"].astype(str) if "spectype" in d.files else np.array(["?"] * len(tid))
zt = d["z_true"].astype(float); zp = d["z_pred"].astype(float)
flux = d["flux"]; wave = d["wave"]; length = d["length"].astype(int)
for m in ("v2", "v3"):
    for c in CATS:
        sel = np.nonzero((model == m) & (cat == c))[0]
        fig, axes = plt.subplots(2, 5, figsize=(22, 8.2))
        axes = axes.ravel()
        for k, ax in enumerate(axes):
            if k < len(sel):
                j = sel[k]; n = length[j]
                w = wave[j, :n]; f = smooth(flux[j, :n], 10)
                ax.plot(w, f, lw=0.6, color="#1f4e9c")
                ax.set_title(f"#{k+1}   {stype[j]}   id {tid[j]}\n"
                             f"$z_t$={zt[j]:.3f}  $z_p$={zp[j]:.3f}", fontsize=9)
                ax.tick_params(labelsize=7); ax.margins(x=0.01); ax.grid(alpha=0.2)
                ax.text(0.02, 0.93, f"{k+1}", transform=ax.transAxes, fontsize=11,
                        fontweight="bold", va="top", ha="left",
                        bbox=dict(boxstyle="circle,pad=0.25", fc="#ffe08a", ec="#444", lw=0.8))
            else:
                ax.axis("off")
        fig.suptitle(
            f"{DS_LABEL}  —  {MODEL_LABEL[m]}\n"
            f"{CAT_DESC[c]}   (n={len(sel)})   •   raw observed flux (10-px smoothed) vs wavelength (Å)",
            fontsize=14, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = f"{SRC}/sdss5000_{m}_{c}.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print("wrote", out, f"({len(sel)} spectra)")
