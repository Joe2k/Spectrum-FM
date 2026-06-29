#!/bin/bash
# SDSS few-shot fine-tuning sweep on ONE interactive 4-GPU node (DDP via torchrun).
#
#   salloc -N 1 --gpus 4 -A m5374_g -C gpu -q interactive -t 02:00:00
#   module load pytorch/2.8.0
#   bash nersc/run_sdss_ft.sh 2>&1 | tee $SCRATCH/sdss_ft/run.log
#
# Env overrides: REPO, ZR (work dir), TOK, CKPT_V3, CKPT_V2, SDSS_ROOT, RUN2D_GLOB.
set -euo pipefail

REPO=${REPO:-/global/homes/j/joe2k/Spectrum-FM}
ZR=${ZR:-$SCRATCH/sdss_ft}
OUT=$ZR/out
TOK=${TOK:-$SCRATCH/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt}
CKPT_V3=${CKPT_V3:-$SCRATCH/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt}
CKPT_V2=${CKPT_V2:-$SCRATCH/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt}
SDSS_ROOT=${SDSS_ROOT:-/global/cfs/cdirs/sdss/data/sdss/dr17/sdss/spectro/redux}
RUN2D_GLOB=${RUN2D_GLOB:-26}      # legacy SDSS-I/II (low-z); excludes BOSS v5_13_2
N_TRAIN=${N_TRAIN:-20000}
N_TEST=${N_TEST:-25000}
SHOTS=${SHOTS:-"0 500 1000 2000 5000 10000 20000"}

mkdir -p "$ZR" "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/nersc:${PYTHONPATH:-}"

echo "=== [1/4] build plate-disjoint path lists ==="
python nersc/build_sdss_lists.py --root "$SDSS_ROOT" --run2d-glob "$RUN2D_GLOB" \
    --out-dir "$ZR" --n-train-files $((N_TRAIN*2)) --n-test-files $((N_TEST*2))

echo "=== [2/4] pre-tokenize SDSS (4-way sharded) ==="
for split in train test; do
  for i in 0 1 2 3; do
    python nersc/pretok_sdss.py \
        --tokenizer-ckpt "$TOK" --paths "$ZR/${split}_paths.txt" \
        --out "$ZR/sdss_${split}.r${i}.npz" --num-shards 4 --shard-id $i --gpu-id $i &
  done
  wait
done
python nersc/pretok_sdss.py --merge --shard-glob "$ZR/sdss_train.r*.npz" \
    --out "$ZR/sdss_train.npz" --max-good $N_TRAIN
python nersc/pretok_sdss.py --merge --shard-glob "$ZR/sdss_test.r*.npz" \
    --out "$ZR/sdss_test.npz" --max-good $N_TEST

echo "=== [3/4] few-shot sweep (DDP-4 per run) ==="
run_one () {  # tag ckpt shots
  echo "--- $1 shots=$3 ---"
  torchrun --standalone --nproc_per_node=4 nersc/sdss_finetune.py \
      --checkpoint "$2" --tokenizer-ckpt "$TOK" \
      --train-cache "$ZR/sdss_train.npz" --test-cache "$ZR/sdss_test.npz" \
      --shots "$3" --out-dir "$OUT" --tag "$1"
}
for shots in $SHOTS; do
  run_one v3 "$CKPT_V3" "$shots"
  run_one v2 "$CKPT_V2" "$shots"
done

echo "=== [4/4] plots ==="
python nersc/plot_fewshot.py --dir "$OUT" --shots $SHOTS
echo "=== ALL DONE -> $OUT ==="
ls -la "$OUT"
