import sys, glob, numpy as np
base = sys.argv[1]
tm = np.load(f"{base}/typemap.npz", allow_pickle=True)
t2c = dict(zip(tm["targetid"].tolist(), tm["cls"].tolist()))
CLASSES = ["BGS", "LRG", "ELG", "QSO"]
MODELS = [("v3", "V3 (4096-soft)"), ("v2", "V2 (512-hard)")]
KEYS = ["z_pred", "z_true", "targetid", "spectype"]


def merge(tag):
    parts = sorted(glob.glob(f"{base}/type_{tag}_r*.npz"))
    return {k: np.concatenate([np.load(p)[k] for p in parts]) for k in KEYS}


def classes_of(tids, sty):
    out = np.array([t2c.get(int(t), "OTHER") for t in tids], dtype="U6")
    out[(out == "OTHER") & (sty == "QSO")] = "QSO"
    return out


for tag, label in MODELS:
    a = merge(tag)
    cl = classes_of(a["targetid"], a["spectype"])
    print(f"=== {label} ===")
    for c in CLASSES:
        m = cl == c
        if m.sum() == 0:
            continue
        zp, zt = a["z_pred"][m].astype(float), a["z_true"][m].astype(float)
        dz = (zp - zt) / (1.0 + zt)
        sig = 1.4826 * np.median(np.abs(dz - np.median(dz)))   # per-type sigma_NMAD
        frac5 = float(np.mean(np.abs(dz) > 5.0 * sig))
        # also report the fixed 0.0033 catastrophic for reference
        eta_fixed = float(np.mean(np.abs(dz) > 0.0033))
        print(f"  {c:4s} N={m.sum():>8d}  sigmaNMAD={sig:.5f}  5sig_thresh={5*sig:.4f}  "
              f"frac(>5sig)={frac5:.4%}   [eta>0.0033={eta_fixed:.2%}]")
