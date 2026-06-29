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
    out[(out == "OTHER") & (sty == "QSO")] = "QSO"
    return out


def pooled_r2(rms, r2):
    # ss_res_i = rms_i^2 * n_px_i ; ss_tot_i = ss_res_i / (1 - r2_i)
    # n_px is pure-RNG (Bernoulli masking), independent of the spectrum -> cancels.
    ssr = rms.astype(np.float64) ** 2
    denom = np.clip(1.0 - r2.astype(np.float64), 1e-6, None)
    sst = ssr / denom
    return 1.0 - ssr.sum() / sst.sum()


rows = []
for tag, label in MODELS:
    a = merge(tag)
    cl = classes_of(a["targetid"], a["spectype"])
    for c in CLASSES:
        m = cl == c
        if m.sum() == 0:
            continue
        rms, r2 = a["flux_rms"][m], a["flux_r2"][m]
        rows.append([label, c, int(m.sum()),
                     float(np.median(rms)),
                     float(np.median(r2)),
                     float(pooled_r2(rms, r2))])

hdr = ["model", "type", "N", "flux RMS", "flux R² (median)", "flux R² (pooled)"]
print(f"{hdr[0]:14} {hdr[1]:4} {hdr[2]:>8} {hdr[3]:>9} {hdr[4]:>16} {hdr[5]:>16}")
print("-" * 74)
for r in rows:
    print(f"{r[0]:14} {r[1]:4} {r[2]:8d} {r[3]:9.4f} {r[4]:16.3f} {r[5]:16.3f}")
