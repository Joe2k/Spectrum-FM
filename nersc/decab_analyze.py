#!/usr/bin/env python
"""Aggregate per-type eval npz (from zr_type_eval.py) into the decoupled-mask A/B
decision table + V4 headline (QSO z-ceiling) report.

Adds, beyond zr_type_analyze.py: gross-outlier fraction eta (|dz|>0.05), and a
QSO z-ceiling check (max/percentile z_pred, frac z_pred>2.13, frac z_true>2.13).

Usage:
  python decab_analyze.py BASE TAG1 [TAG2 ...]
where each TAG has files BASE/type_<TAG>_r*.npz and typemap.npz lives in BASE.
"""
import sys, glob, numpy as np

base = sys.argv[1]
tags = sys.argv[2:]
tm = np.load(f"{base}/typemap.npz", allow_pickle=True)
t2c = dict(zip(tm["targetid"].tolist(), tm["cls"].tolist()))
CLASSES = ["BGS", "LRG", "ELG", "QSO"]
KEYS = ["z_pred", "z_true", "targetid", "spectype", "masked_acc", "flux_rms", "flux_r2"]
ZV2_CEIL = 2.13   # old z-v1 (gaussian_range=3.0) reachable ceiling


def merge(tag):
    parts = sorted(glob.glob(f"{base}/type_{tag}_r*.npz"))
    if not parts:
        raise SystemExit(f"no npz for tag '{tag}' in {base}")
    return {k: np.concatenate([np.load(p)[k] for p in parts]) for k in KEYS}


def classes_of(tids, sty):
    out = np.array([t2c.get(int(t), "OTHER") for t in tids], dtype="U6")
    out[(out == "OTHER") & (sty == "QSO")] = "QSO"
    return out


def metrics(zp, zt, fr2):
    dz = (zp - zt) / (1.0 + zt)
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    eta = float(np.mean(np.abs(dz) > 0.05))           # gross-outlier fraction
    ss_res = np.sum((zp - zt) ** 2)
    ss_tot = np.sum((zt - zt.mean()) ** 2)
    r2z = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return nmad, eta, r2z, float(np.median(fr2))


print("\n================ PER-TYPE  (z blind: full spectrum, z masked  |  recon: 50% spectrum masked, z hidden) ================")
hdr = ["tag", "type", "N", "z_sNMAD", "gross_eta", "z_R2", "recon_fluxR2"]
line = f"{hdr[0]:10} {hdr[1]:4} {hdr[2]:>8} {hdr[3]:>9} {hdr[4]:>9} {hdr[5]:>7} {hdr[6]:>12}"
print(line); print("-" * len(line))
store = {}
for tag in tags:
    a = merge(tag); store[tag] = a
    cl = classes_of(a["targetid"], a["spectype"])
    for c in CLASSES:
        m = cl == c
        if m.sum() == 0:
            continue
        nmad, eta, r2z, fr2 = metrics(a["z_pred"][m], a["z_true"][m], a["flux_r2"][m])
        print(f"{tag:10} {c:4} {int(m.sum()):8d} {nmad:9.5f} {eta:9.4f} {r2z:7.3f} {fr2:12.4f}")
    # pooled
    nmad, eta, r2z, fr2 = metrics(a["z_pred"], a["z_true"], a["flux_r2"])
    print(f"{tag:10} {'ALL':4} {len(a['z_pred']):8d} {nmad:9.5f} {eta:9.4f} {r2z:7.3f} {fr2:12.4f}")
    print("-" * len(line))

print("\n================ QSO z-CEILING CHECK (does z-v2 head now predict past the old 2.13 wall?) ================")
print(f"{'tag':10} {'N_QSO':>7} {'maxZpred':>9} {'p99Zpred':>9} {'frac_pred>2.13':>15} {'frac_true>2.13':>15} {'frac_pred>3':>12}")
for tag in tags:
    a = store[tag]
    cl = classes_of(a["targetid"], a["spectype"])
    m = cl == "QSO"
    if m.sum() == 0:
        continue
    zp, zt = a["z_pred"][m], a["z_true"][m]
    print(f"{tag:10} {int(m.sum()):7d} {zp.max():9.3f} {np.percentile(zp,99):9.3f} "
          f"{np.mean(zp>ZV2_CEIL):15.4f} {np.mean(zt>ZV2_CEIL):15.4f} {np.mean(zp>3.0):12.4f}")
print()
