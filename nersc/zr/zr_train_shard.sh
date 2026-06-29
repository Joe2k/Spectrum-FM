#!/bin/bash
# One srun task = one shard on its Slurm-assigned GPU. Args: TAG CKPT
ZR=/pscratch/sd/j/joe2k/zr
SC=/pscratch/sd/j/joe2k
MAN=$SC/manifests/dr1_v2_full.jsonl
TOKDIR=$SC/dr1_tokenized_v2
TAG=$1; CKPT=$2
S=${SLURM_PROCID:-0}
echo "[shard $S] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host=$(hostname)" > $ZR/${TAG}_train_${S}.log
python $ZR/zr_eval.py --mode desi --checkpoint "$CKPT" \
  --num-shards 4 --shard-id $S --num-workers 8 \
  --manifest $MAN --tokenized-dir $TOKDIR --train-split \
  --batch-size 1024 --out $ZR/zr_${TAG}_train.r${S}.npz \
  >> $ZR/${TAG}_train_${S}.log 2>&1
