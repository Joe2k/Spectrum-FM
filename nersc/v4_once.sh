#!/bin/bash
# V4 — ONE 4h training chunk (z-v2, gr=4.0), resume-aware + singleton-guarded.
# NOT a loop: runs a single salloc that holds one job for up to 4h, then exits.
# The agent (Claude) re-invokes this each time the previous chunk ends, until
# step >= 200000. Singleton guard (named job v4zv2) means a double-invoke is safe.
RUN=approach_a_v2cache_x2x3_zv2_ddp4
SC=/pscratch/sd/j/joe2k
REPO=/global/homes/j/joe2k/Spectrum-FM
LAST=$SC/deepsrch/checkpoints/$RUN/last.pt

if squeue -u joe2k -h -o "%j" | grep -qx v4zv2; then
  echo "[once] a v4zv2 job is already queued/running — abort (singleton)"; exit 0
fi
RESUME=""; [ -f "$LAST" ] && RESUME="--resume $LAST"
echo "[once $(date)] launching v4zv2 (resume='$RESUME')"
salloc -N1 --gpus 4 -J v4zv2 -A m5374_g -C gpu -q interactive -t 240 \
  bash -lc "module load pytorch/2.8.0; cd $REPO; export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN; \
    srun -n1 --cpus-per-task=64 torchrun --nproc_per_node=4 --master_port=29400 \
      nersc/train_transformer.py $RESUME \
      --manifest $SC/manifests/dr1_v2_full.jsonl \
      --tokenized-dir $SC/dr1_tokenized_v2 \
      --approach a --masked-targets-only \
      --mask-ratio-min 0.15 --mask-ratio-max 0.75 \
      --z-bins 4096 --z-fit-files 800 --redshift-soft-sigma 24 --z-gaussian-range 4.0 \
      --redshift-loss-weight 1.0 --encoder-mask-ratio 0.5 --redshift-mask-ratio 0.5 \
      --amp --amp-dtype bf16 --steps 200000 --batch-size 32 --lr 8e-4 \
      --num-workers 12 --d-model 768 --n-encoder-layers 6 --n-decoder-layers 6 --n-heads 12 \
      --healpix-holdout-frac 0.05 --ar-eval-batches 8 --ar-every-steps 10000 \
      --run-name $RUN --wandb-project redshifty --wandb-mode online"
echo "[once $(date)] salloc returned (job ended)"
