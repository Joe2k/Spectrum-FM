"""Step 0(b): decompose per-type redshift catastrophic outliers.

For QSO/ELG (the heavy-tail types), split the catastrophic outliers
(|dz/(1+z)| > 0.0033) into three buckets:
  (1) ceiling-limited : z_pred pinned near the model's max emittable z while
      z_true is higher (the z-tokenizer ceiling — fixed by z-v2).
  (2) line-alias       : (1+z_pred)/(1+z_true) matches a rest-line wavelength
      ratio lambda_i/lambda_j (the model locked onto the wrong line).
  (3) genuine          : everything else (real scatter / noise).

Reads the per-type arrays /pscratch/sd/j/joe2k/zr/type_{tag}_r*.npz
(z_pred, z_true, targetid, spectype) and the targetid->class typemap, exactly
like zr_type_outliers.py.
"""
import sys, glob
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "/pscratch/sd/j/joe2k/zr"
# model -> approx max emittable z (from the bundled z-tokenizer, gaussian_range=3.0)
CEIL = {"v3": 2.130, "v2": 1.655}
CLASSES = ["BGS", "LRG", "ELG", "QSO"]
KEYS = ["z_pred", "z_true", "targetid", "spectype"]

# Strong rest-frame lines (Angstrom) that drive spectroscopic redshifts.
LINES = {
    "Lya": 1215.67, "CIV": 1549.0, "CIII]": 1908.7, "MgII": 2799.0,
    "[OII]": 3727.0, "CaK": 3933.7, "CaH": 3968.5, "4000break": 4000.0,
    "Hbeta": 4861.0, "[OIII]": 5007.0, "Halpha": 6563.0,
}
LN = list(LINES.items())
# all ordered line-pair ratios lambda_i/lambda_j (i!=j) with a label
PAIR_RATIOS = []
for i in range(len(LN)):
    for j in range(len(LN)):
        if i == j:
            continue
        PAIR_RATIOS.append((LN[i][1] / LN[j][1], f"{LN[i][0]}->{LN[j][0]}"))


def merge(tag):
    parts = sorted(glob.glob(f"{BASE}/type_{tag}_r*.npz"))
    return {k: np.concatenate([np.load(p)[k] for p in parts]) for k in KEYS}


def classes_of(tids, sty):
    tm = np.load(f"{BASE}/typemap.npz", allow_pickle=True)
    t2c = dict(zip(tm["targetid"].tolist(), tm["cls"].tolist()))
    out = np.array([t2c.get(int(t), "OTHER") for t in tids], dtype="U6")
    out[(out == "OTHER") & (sty == "QSO")] = "QSO"
    return out


def alias_label(zp, zt, tol=0.006):
    """Closest line-pair ratio to (1+zp)/(1+zt); return (label, rel_err) or (None, _)."""
    R = (1.0 + zp) / (1.0 + zt)
    best, berr = None, 1e9
    for ratio, lab in PAIR_RATIOS:
        e = abs(R / ratio - 1.0)
        if e < berr:
            berr, best = e, lab
    return (best, berr) if berr < tol else (None, berr)


for tag in ("v3", "v2"):
    a = merge(tag)
    cl = classes_of(a["targetid"], a["spectype"])
    ceil = CEIL[tag]
    print(f"\n================  {tag.upper()}  (ceiling z={ceil})  ================")
    for c in ("QSO", "ELG"):
        m = cl == c
        zp = a["z_pred"][m].astype(float); zt = a["z_true"][m].astype(float)
        dz = (zp - zt) / (1.0 + zt)
        out = np.abs(dz) > 0.0033
        nout = int(out.sum())
        if nout == 0:
            print(f"  {c}: no outliers"); continue
        zpo, zto = zp[out], zt[out]
        # (1) ceiling-limited: pred near ceiling AND truth meaningfully above pred
        ceil_lim = (zpo >= ceil - 0.08) & (zto > zpo + 0.10)
        # (2) line-alias among the rest
        rest = ~ceil_lim
        labels = [alias_label(zpo[i], zto[i])[0] for i in np.nonzero(rest)[0]]
        is_alias = np.array([l is not None for l in labels])
        n_ceil = int(ceil_lim.sum())
        n_alias = int(is_alias.sum())
        n_gen = nout - n_ceil - n_alias
        N = int(m.sum())
        print(f"  {c}: N={N:,}  outliers={nout:,} ({100*nout/N:.2f}%)")
        print(f"      ceiling-limited : {n_ceil:>6,} ({100*n_ceil/nout:5.1f}% of outliers)")
        print(f"      line-alias      : {n_alias:>6,} ({100*n_alias/nout:5.1f}% of outliers)")
        print(f"      genuine/other   : {n_gen:>6,} ({100*n_gen/nout:5.1f}% of outliers)")
        # top alias channels
        from collections import Counter
        cc = Counter([l for l in labels if l is not None])
        if cc:
            top = ", ".join(f"{k} {v}" for k, v in cc.most_common(5))
            print(f"      top alias channels: {top}")
        # how many outliers have z_true above the OLD ceiling (z-v2 directly helps)
        n_above_ceil = int((zto > ceil).sum())
        print(f"      z_true > old ceiling: {n_above_ceil:,} ({100*n_above_ceil/nout:.1f}% of outliers)")
