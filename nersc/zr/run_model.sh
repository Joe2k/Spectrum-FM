#!/bin/bash
# Per-model blind-redshift eval: val split, full corpus, SDSS OOD.
# Args: TAG CKPT   (runs on whatever GPU CUDA_VISIBLE_DEVICES pins)
set -uo pipefail
ZR=/pscratch/sd/j/joe2k/zr
SC=/pscratch/sd/j/joe2k
TOK=$SC/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
MAN=$SC/manifests/dr1_v2_full.jsonl
TOKDIR=$SC/dr1_tokenized_v2
SDSS=/global/cfs/cdirs/sdss/data/sdss/dr17/eboss/spectro/redux/v5_13_2/spectra/lite
TAG=$1; CKPT=$2; NW=16

echo "[$TAG] VAL  (GPU=$CUDA_VISIBLE_DEVICES)"
python $ZR/zr_eval.py --mode desi --checkpoint "$CKPT" --manifest $MAN \
  --tokenized-dir $TOKDIR --batch-size 1024 --num-workers $NW --out $ZR/zr_${TAG}_val.npz
echo "[$TAG] FULL (GPU=$CUDA_VISIBLE_DEVICES)"
python $ZR/zr_eval.py --mode desi --checkpoint "$CKPT" --manifest $MAN \
  --tokenized-dir $TOKDIR --batch-size 1024 --num-workers $NW --full-corpus --out $ZR/zr_${TAG}_full.npz
echo "[$TAG] SDSS (GPU=$CUDA_VISIBLE_DEVICES)"
python $ZR/zr_eval.py --mode sdss --checkpoint "$CKPT" --tokenizer-ckpt $TOK \
  --sdss-dir $SDSS --glob "**/spec-*.fits" --max-spectra 30000 \
  --batch-size 256 --num-workers $NW --out $ZR/zr_${TAG}_sdss.npz
echo "[$TAG] DONE"
