import sys, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = sys.argv[1]
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
    out[(out == "OTHER") & (sty == "QSO")] = "QSO"   # spectype fallback
    return out


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
        rows.append([label, c, int(m.sum()), nmad, r2z,
                     float(a["masked_acc"][m].mean()),
                     float(np.median(a["flux_rms"][m])),
                     float(np.median(a["flux_r2"][m]))])

hdr = ["model", "type", "N", "z σ_NMAD", "z R²", "mask-acc", "flux RMS", "flux R²"]
line = f"{hdr[0]:14} {hdr[1]:4} {hdr[2]:>8} {hdr[3]:>10} {hdr[4]:>7} {hdr[5]:>8} {hdr[6]:>9} {hdr[7]:>8}"
print(line)
print("-" * len(line))
for r in rows:
    print(f"{r[0]:14} {r[1]:4} {r[2]:8d} {r[3]:10.5f} {r[4]:7.3f} {r[5]:8.3f} {r[6]:9.4f} {r[7]:8.3f}")

# table figure
fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 1.6))
ax.axis("off")
cell = [[r[0], r[1], f"{r[2]:,}", f"{r[3]:.5f}", f"{r[4]:.3f}",
         f"{r[5]:.3f}", f"{r[6]:.4f}", f"{r[7]:.3f}"] for r in rows]
tbl = ax.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
for j in range(len(hdr)):
    tbl[0, j].set_facecolor("#222"); tbl[0, j].set_text_props(color="w", weight="bold")
ax.set_title("Per-object-type performance — DR1 val split (z masked; spectrum 50% masked)\n"
             "Redshift: σ_NMAD + R²   |   Reconstruction: masked-token acc, flux RMS, flux R²",
             fontsize=12, pad=14)
fig.tight_layout()
fig.savefig(f"{base}/plot_by_type.png", dpi=150)
print("\nwrote", f"{base}/plot_by_type.png")
