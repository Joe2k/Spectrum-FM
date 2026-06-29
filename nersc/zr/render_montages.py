#!/usr/bin/env python3
"""Render 2x5 montages of raw spectra per (dataset, model, category)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/tmp/zr_plots/examples"
CAT_DESC = {
    "cat1_predz0":    "Predicted z ≈ 0",
    "cat2_overpred":  "Predicted z ≫ true  (z_pred < 0.5)",
    "cat3_underpred": "True z ≫ predicted  (0.5 < z_true < 1)",
    "cat4_ceiling":   "Predicted z at ceiling (≈2) ≪ true z",
}
CATS = ["cat1_predz0", "cat2_overpred", "cat3_underpred", "cat4_ceiling"]
MODEL_LABEL = {"v2": "Transformer V2 (512-hard)", "v3": "Transformer V3 (4096-soft)"}
DS_LABEL = {"val": "DESI DR1 val split", "sdss": "SDSS zero-shot (25k plate-disjoint test)"}


def render(ds_key, npz_path):
    d = np.load(npz_path, allow_pickle=True)
    model = d["model"].astype(str); cat = d["cat"].astype(str)
    tid = d["targetid"].astype(str)
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
                    w = wave[j, :n]; f = flux[j, :n]
                    ax.plot(w, f, lw=0.5, color="#1f4e9c")
                    ax.set_title(f"id {tid[j]}\n$z_t$={zt[j]:.3f}  $z_p$={zp[j]:.3f}",
                                 fontsize=9)
                    ax.tick_params(labelsize=7)
                    ax.margins(x=0.01)
                    ax.grid(alpha=0.2)
                else:
                    ax.axis("off")
            fig.suptitle(
                f"{DS_LABEL[ds_key]}  —  {MODEL_LABEL[m]}\n"
                f"{CAT_DESC[c]}   (n={len(sel)})   •   raw observed flux vs wavelength (Å)",
                fontsize=14, y=0.99)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            out = f"{SRC}/{ds_key}_{m}_{c}.png"
            fig.savefig(out, dpi=120); plt.close(fig)
            print("wrote", out, f"({len(sel)} spectra)")


if __name__ == "__main__":
    Path(SRC).mkdir(parents=True, exist_ok=True)
    for ds_key in ("val", "sdss"):
        p = f"{SRC}/sel_{ds_key}.npz"
        if Path(p).exists():
            render(ds_key, p)
        else:
            print("MISSING", p)
