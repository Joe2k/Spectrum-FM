#!/bin/bash
# Runs under `srun -n1` so all 4 GPUs are visible (device_count==4).
cd /global/homes/j/joe2k/Spectrum-FM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[gpus] visible=$(python -c 'import torch;print(torch.cuda.device_count())')"
SC=/pscratch/sd/j/joe2k
V2=$SC/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt
V3=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
MAN=$SC/manifests/dr1_v2_full.jsonl
TOK=$SC/dr1_tokenized_v2
for i in 0 1 2 3; do
  python nersc/select_examples.py --dataset val --out /dev/null \
    --ckpt-v2 $V2 --ckpt-v3 $V3 --manifest $MAN --tokenized-dir $TOK \
    --num-shards 4 --shard-id $i --gpu-id $i \
    --shard-out $SC/examples/val_shard.r$i.npz --batch-size 384 --num-workers 6 \
    > $SC/examples/val.w$i.log 2>&1 &
done
wait
echo "### WORKERS DONE ###"
