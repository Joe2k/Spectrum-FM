import sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits
sys.path.insert(0, "/global/homes/j/joe2k/Spectrum-FM")
sys.path.insert(0, "/global/homes/j/joe2k/Spectrum-FM/nersc")
from src.training.data_split import split_records_by_healpix
from dr1_dataset import load_manifest

# desitarget / sv*_desi_mask primary bits (identical across main + SV for tracers)
QSO, LRG, ELG, BGS_ANY = (1 << 2), (1 << 0), (1 << 1), (1 << 60)


def classify(dt, bt):
    if dt & QSO:
        return "QSO"
    if dt & LRG:
        return "LRG"
    if dt & ELG:
        return "ELG"
    if (dt & BGS_ANY) or (bt != 0):
        return "BGS"
    return "OTHER"


def read_one(rec):
    try:
        with fits.open(rec["redrock"], memmap=True) as h:
            fm = h["FIBERMAP"]
            cols = fm.columns.names
            dcol = [c for c in cols if c.endswith("DESI_TARGET")][0]
            bcol = [c for c in cols if c.endswith("BGS_TARGET")]
            tid = np.asarray(fm.data["TARGETID"], dtype="int64")
            dt = np.asarray(fm.data[dcol], dtype="int64")
            bt = np.asarray(fm.data[bcol[0]], dtype="int64") if bcol else np.zeros(len(tid), "int64")
        return tid, dt, bt
    except Exception as e:
        print("skip", rec.get("redrock"), e, flush=True)
        return None


man = load_manifest("/pscratch/sd/j/joe2k/manifests/dr1_v2_full.jsonl")
_, val = split_records_by_healpix(man, holdout_frac=0.05, seed=42)
print("val healpix records:", len(val), flush=True)

tmap = {}
with ThreadPoolExecutor(max_workers=32) as ex:
    for r in ex.map(read_one, val):
        if r is None:
            continue
        tid, dt, bt = r
        for i in range(len(tid)):
            tmap[int(tid[i])] = classify(int(dt[i]), int(bt[i]))

tids = np.fromiter(tmap.keys(), dtype="int64", count=len(tmap))
cls = np.array(list(tmap.values()))
np.savez("/pscratch/sd/j/joe2k/zr/typemap.npz", targetid=tids, cls=cls)
u, c = np.unique(cls, return_counts=True)
print("typemap entries:", len(tids), dict(zip(u.tolist(), c.tolist())), flush=True)
