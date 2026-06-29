#!/bin/bash
# Drives two models on two GPUs in parallel, then plots.
module load pytorch/2.8.0
cd $HOME/Spectrum-FM
ZR=/pscratch/sd/j/joe2k/zr
SC=/pscratch/sd/j/joe2k
V3=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
V2=$SC/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt

CUDA_VISIBLE_DEVICES=0 bash $ZR/run_model.sh v3 "$V3" > $ZR/v3.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=1 bash $ZR/run_model.sh v2 "$V2" > $ZR/v2.log 2>&1 &
P2=$!
echo "launched v3(pid $P3, GPU0) and v2(pid $P2, GPU1)"
wait $P3; echo "v3 exit $?"
wait $P2; echo "v2 exit $?"

echo "===== PLOT ====="
python $ZR/zr_plot.py $ZR
echo "===== ALL DONE ====="
ls -la $ZR/plot_*.png
