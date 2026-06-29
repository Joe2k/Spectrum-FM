#!/bin/bash
# V4 — full fresh pretrain with the z-v2 tokenizer (gaussian_range=4.0, lifts the
# QSO z-ceiling). Auto-resumes across 4h interactive sessions: each iteration grabs
# a 4-GPU interactive node (-t 240) and resumes from the rolling last.pt; stops when
# the run reaches --steps. nohup this on a login node so it survives ssh logout.
#
#   nohup bash /pscratch/sd/j/joe2k/v4/v4_driver.sh > /pscratch/sd/j/joe2k/v4/nohup.out 2>&1 &
set -uo pipefail

RUN=approach_a_v2cache_x2x3_zv2_ddp4
SC=/pscratch/sd/j/joe2k
REPO=/global/homes/j/joe2k/Spectrum-FM
CKDIR=$SC/deepsrch/checkpoints/$RUN
LAST=$CKDIR/last.pt
TARGET=200000
LOG=$SC/v4/driver.log
mkdir -p $SC/v4
module load pytorch/2.8.0 2>/dev/null || true

step_now() {
  [ -f "$LAST" ] || { echo 0; return; }
  python - "$LAST" <<'PY' 2>/dev/null || echo 0
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)))
except Exception:
    print(0)
PY
}

echo "=== v4 driver START $(date) run=$RUN ===" >> $LOG
while true; do
  S=$(step_now)
  echo "[driver $(date)] step=$S / $TARGET" >> $LOG
  if [ "$S" -ge "$TARGET" ]; then
    echo "[driver $(date)] reached target, DONE" >> $LOG; touch $CKDIR/DONE; break
  fi
  RESUME=""
  [ -f "$LAST" ] && RESUME="--resume $LAST"
  echo "[driver $(date)] salloc (resume='$RESUME')" >> $LOG
  salloc -N1 --gpus 4 -A m5374_g -C gpu -q interactive -t 240 \
    bash -lc "module load pytorch/2.8.0; cd $REPO; \
      export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN; \
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
        --run-name $RUN --wandb-project redshifty --wandb-mode offline" >> $LOG 2>&1
  echo "[driver $(date)] salloc returned (rc=$?), looping in 15s" >> $LOG
  sleep 15
done
echo "=== v4 driver END $(date) ===" >> $LOG
