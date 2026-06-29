#!/bin/bash
# DESI TRAIN split blind-z, both models. Each model = srun -n4 --gpus-per-task=1
# (Slurm pins each shard to a distinct GPU; arg-sharded so SLURM_PROCID = shard-id).
# Sequential V3 then V2 so V3 is fully saved even if V2 runs near the wall.
module load pytorch/2.8.0
cd $HOME/Spectrum-FM
ZR=/pscratch/sd/j/joe2k/zr
SC=/pscratch/sd/j/joe2k
V3=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
V2=$SC/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt

echo "=== V3 train START $(date) ==="
srun -n4 -N1 --gpus-per-task=1 --cpus-per-task=16 bash $ZR/zr_train_shard.sh v3 "$V3"
echo "=== V3 train DONE $(date) ==="
ls -la $ZR/zr_v3_train.r*.npz
echo "=== V2 train START $(date) ==="
srun -n4 -N1 --gpus-per-task=1 --cpus-per-task=16 bash $ZR/zr_train_shard.sh v2 "$V2"
echo "=== V2 train DONE $(date) ==="
ls -la $ZR/zr_v2_train.r*.npz
echo "=== ALL DONE $(date) ==="
