#!/usr/bin/env python3
"""Read SDSS object CLASS/SUBCLASS for the selected 5000-shot example spectra."""
from pathlib import Path
import numpy as np
from astropy.io import fits

SC = "/pscratch/sd/j/joe2k"
sel = np.load(f"{SC}/examples/sel_sdss_5000.npz", allow_pickle=True)
paths = [ln.strip() for ln in Path(f"{SC}/sdss_ft/test_paths.txt").read_text().splitlines() if ln.strip()]
by_name = {Path(p).name: p for p in paths}

info = sel["info"].astype(str)
spectype = np.empty(len(info), object)
subclass = np.empty(len(info), object)
for i, name in enumerate(info):
    p = by_name.get(name)
    cls, sub = "?", ""
    if p:
        try:
            with fits.open(p, memmap=False) as h:
                sp = None
                for hn in ("SPECOBJ", "SPALL"):
                    try:
                        d = h[hn].data
                        if d is not None and "CLASS" in d.names:
                            sp = d; break
                    except (KeyError, IndexError, TypeError, AttributeError):
                        continue
                if sp is not None:
                    cls = str(np.asarray(sp["CLASS"]).ravel()[0]).strip()
                    if "SUBCLASS" in sp.names:
                        sub = str(np.asarray(sp["SUBCLASS"]).ravel()[0]).strip()
        except Exception as e:  # noqa: BLE001
            print("warn", name, e)
    spectype[i] = cls; subclass[i] = sub

out = dict(sel)
out["spectype"] = spectype.astype(str)
out["subclass"] = subclass.astype(str)
np.savez(f"{SC}/examples/sel_sdss_5000_typed.npz", **out)
# quick summary
m = sel["model"].astype(str); c = sel["cat"].astype(str)
import collections
for mm in ("v2", "v3"):
    for cc in ("cat1_predz0", "cat2_overpred", "cat3_underpred", "cat4_ceiling"):
        s = (m == mm) & (c == cc)
        print(mm, cc, dict(collections.Counter(spectype[s].astype(str))))
print("wrote", f"{SC}/examples/sel_sdss_5000_typed.npz")
