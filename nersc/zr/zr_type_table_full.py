import sys, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = sys.argv[1]
out = sys.argv[2]
tm = np.load(f"{base}/typemap.npz", allow_pickle=True)
t2c = dict(zip(tm["targetid"].tolist(), tm["cls"].tolist()))
CLASSES = ["BGS", "LRG", "ELG", "QSO"]
MODELS = [("v3", "V3 (4096-soft)"), ("v2", "V2 (512-hard)")]
KEYS = ["z_pred", "z_true", "targetid", "spectype", "masked_acc", "flux_rms", "flux_r2"]


def merge(tag):
    parts = sorted(glob.glob(f"{base}/type_{tag}_r*.npz"))
    return {k: np.concatenate([np.load(p)[k] for p in parts]) for k in KEYS}


def classes_of(tids, sty):
    out = np.array([t2c.get(int(t), "OTHER") for t in tids], dtype="U6")
    out[(out == "OTHER") & (sty == "QSO")] = "QSO"
    return out


def pooled_r2(rms, r2):
    ssr = rms.astype(np.float64) ** 2
    sst = ssr / np.clip(1.0 - r2.astype(np.float64), 1e-6, None)
    return 1.0 - ssr.sum() / sst.sum()


rows = []
for tag, label in MODELS:
    a = merge(tag)
    cl = classes_of(a["targetid"], a["spectype"])
    for c in CLASSES:
        m = cl == c
        if m.sum() == 0:
            continue
        zp, zt = a["z_pred"][m], a["z_true"][m]
        dz = (zp - zt) / (1.0 + zt)
        nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
        ss_res = np.sum((zp - zt) ** 2)
        ss_tot = np.sum((zt - zt.mean()) ** 2)
        r2z = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rms, r2 = a["flux_rms"][m], a["flux_r2"][m]
        rows.append([label, c, int(m.sum()), nmad, r2z,
                     float(a["masked_acc"][m].mean()),
                     float(np.median(rms)),
                     float(np.median(r2)),
                     float(pooled_r2(rms, r2))])

hdr = ["model", "type", "N", "z σ_NMAD", "z R²", "mask-acc",
       "flux RMS", "flux R²\n(median)", "flux R²\n(pooled)"]
for r in rows:
    print(r[1], r[0][:2], f"med={r[7]:+.3f} pooled={r[8]:+.3f}")

fig, ax = plt.subplots(figsize=(14, 0.55 * len(rows) + 1.8))
ax.axis("off")
cell = [[r[0], r[1], f"{r[2]:,}", f"{r[3]:.5f}", f"{r[4]:.3f}",
         f"{r[5]:.3f}", f"{r[6]:.4f}", f"{r[7]:.3f}", f"{r[8]:.3f}"] for r in rows]
tbl = ax.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.9)
for j in range(len(hdr)):
    tbl[0, j].set_facecolor("#222"); tbl[0, j].set_text_props(color="w", weight="bold")
# highlight the new pooled column and the ELG sign-flip cells
pooled_col = len(hdr) - 1
for i in range(1, len(rows) + 1):
    tbl[i, pooled_col].set_facecolor("#e8f0ff")
for i, r in enumerate(rows, 1):
    if r[1] == "ELG":
        tbl[i, pooled_col - 1].set_facecolor("#ffd9d9")   # negative median
        tbl[i, pooled_col].set_facecolor("#d9f2d9")        # positive pooled
ax.set_title("Per-object-type performance — DR1 val split (z masked; spectrum 50% masked)\n"
             "Reconstruction flux R²: median (per-spectrum) vs pooled (Σss_res/Σss_tot, variance-weighted)",
             fontsize=12, pad=14)
fig.tight_layout()
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
