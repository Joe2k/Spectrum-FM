import sys, glob, numpy as np
base = sys.argv[1]
for tag in ("v2", "v3"):
    for ds in ("val", "full", "sdss"):
        parts = sorted(glob.glob(f"{base}/zr_{tag}_{ds}.r*.npz"))
        if not parts:
            print(f"[merge] MISSING {tag}_{ds}", flush=True)
            continue
        zp = np.concatenate([np.load(p)["z_pred"] for p in parts])
        zt = np.concatenate([np.load(p)["z_true"] for p in parts])
        np.savez(f"{base}/zr_{tag}_{ds}.npz", z_pred=zp, z_true=zt)
        print(f"[merge] {tag}_{ds}: {len(zt):,} from {len(parts)} shards", flush=True)
