#!/bin/bash
# Decoupled-masking A/B — ONE 4h chunk for ONE arm. Usage:
#   bash ab_decouple.sh ctrl       # independent masks (baseline)
#   bash ab_decouple.sh decouple   # three-mode decoupled masking
# Both arms: from scratch, z-v2 (gr=4.0), 50k steps (~2.4h @ 5.7 steps/s ⇒ one
# chunk). Singleton-guarded per arm + resume-aware (in case a chunk dies early).
# Identical to the V3/V4 recipe EXCEPT the masking objective, so the per-type
# blind-z + recon delta is attributable to masking alone.
set -u
ARM="${1:?usage: ab_decouple.sh ctrl|decouple}"
SC=/pscratch/sd/j/joe2k
REPO=/global/homes/j/joe2k/Spectrum-FM

case "$ARM" in
  ctrl)
    RUN=abmask_ctrl_zv2;  JOB=abctrl;  PORT=29411
    MASK_ARGS="--encoder-mask-ratio 0.5 --redshift-mask-ratio 0.5 --two-pass-val" ;;
  decouple)
    RUN=abmask_dec_zv2;   JOB=abdec;   PORT=29412
    MASK_ARGS="--encoder-mask-ratio 0.5 --decouple-masks --blind-z-frac 0.5 --recon-z-shown-frac 0.5" ;;
  decouple_v2)
    RUN=abmask_dec_v2_zv2;  JOB=abdecv2;  PORT=29413
    MASK_ARGS="--encoder-mask-ratio 0.5 --decouple-masks --blind-z-frac 0.5 --recon-z-shown-frac 0.5" ;;
  *) echo "unknown arm '$ARM' (want ctrl|decouple|decouple_v2)"; exit 2 ;;
esac

LAST=$SC/deepsrch/checkpoints/$RUN/last.pt
if squeue -u joe2k -h -o "%j" | grep -qx "$JOB"; then
  echo "[ab $ARM] a $JOB job is already queued/running — abort (singleton)"; exit 0
fi
RESUME=""; [ -f "$LAST" ] && RESUME="--resume $LAST"
echo "[ab $ARM $(date)] launching $JOB run=$RUN (resume='$RESUME')"

salloc -N1 --gpus 4 -J "$JOB" -A m5374_g -C gpu -q interactive -t 240 \
  bash -lc "module load pytorch/2.8.0; cd $REPO; export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN; \
    srun -n1 --cpus-per-task=64 torchrun --nproc_per_node=4 --master_port=$PORT \
      nersc/train_transformer.py $RESUME \
      --manifest $SC/manifests/dr1_v2_full.jsonl \
      --tokenized-dir $SC/dr1_tokenized_v2 \
      --approach a --masked-targets-only \
      --mask-ratio-min 0.15 --mask-ratio-max 0.75 \
      --z-bins 4096 --z-fit-files 800 --redshift-soft-sigma 24 --z-gaussian-range 4.0 \
      --redshift-loss-weight 1.0 \
      $MASK_ARGS \
      --seed 42 \
      --amp --amp-dtype bf16 --steps 50000 --batch-size 32 --lr 8e-4 \
      --num-workers 12 --d-model 768 --n-encoder-layers 6 --n-decoder-layers 6 --n-heads 12 \
      --healpix-holdout-frac 0.05 --ar-eval-batches 8 --ar-every-steps 10000 \
      --run-name $RUN --wandb-project redshifty --wandb-mode online"
echo "[ab $ARM $(date)] salloc returned (job ended)"
