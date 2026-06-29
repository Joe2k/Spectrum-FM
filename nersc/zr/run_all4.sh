#!/bin/bash
# 4 GPUs: V3 on shards {0,1} (GPU 0,1), V2 on shards {0,1} (GPU 2,3). Then merge + plot.
module load pytorch/2.8.0
cd $HOME/Spectrum-FM
ZR=/pscratch/sd/j/joe2k/zr
SC=/pscratch/sd/j/joe2k
V3=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
V2=$SC/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt

echo "=== building SDSS path list (first 30k) ==="
python -c "
from pathlib import Path
import itertools
root = Path('/global/cfs/cdirs/sdss/data/sdss/dr17/eboss/spectro/redux/v5_13_2/spectra/lite')
ps = list(itertools.islice((str(p) for p in root.rglob('spec-*.fits')), 30000))
open('$ZR/sdss_paths.txt', 'w').write(chr(10).join(ps))
print('sdss paths:', len(ps))
"

echo "=== launching 4 workers ==="
CUDA_VISIBLE_DEVICES=0 bash $ZR/run_worker.sh v3 "$V3" 0 2 > $ZR/v3_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash $ZR/run_worker.sh v3 "$V3" 1 2 > $ZR/v3_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash $ZR/run_worker.sh v2 "$V2" 0 2 > $ZR/v2_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash $ZR/run_worker.sh v2 "$V2" 1 2 > $ZR/v2_1.log 2>&1 &
wait
echo "=== MERGE ==="
python $ZR/zr_merge.py $ZR
echo "=== PLOT ==="
python $ZR/zr_plot.py $ZR
echo "=== ALL DONE ==="
ls -la $ZR/plot_*.png
