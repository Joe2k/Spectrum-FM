#!/bin/bash
# One shard of one model on one GPU: val, full corpus, SDSS.
# Args: TAG CKPT SHARD NSHARDS
set -uo pipefail
ZR=/pscratch/sd/j/joe2k/zr
SC=/pscratch/sd/j/joe2k
TOK=$SC/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
MAN=$SC/manifests/dr1_v2_full.jsonl
TOKDIR=$SC/dr1_tokenized_v2
TAG=$1; CKPT=$2; S=$3; N=$4; NW=8
C="--checkpoint $CKPT --num-shards $N --shard-id $S --num-workers $NW"

echo "[$TAG.$S] VAL  (GPU=$CUDA_VISIBLE_DEVICES)"
python $ZR/zr_eval.py --mode desi $C --manifest $MAN --tokenized-dir $TOKDIR \
  --batch-size 1024 --out $ZR/zr_${TAG}_val.r${S}.npz
echo "[$TAG.$S] FULL (GPU=$CUDA_VISIBLE_DEVICES)"
python $ZR/zr_eval.py --mode desi $C --manifest $MAN --tokenized-dir $TOKDIR \
  --batch-size 1024 --full-corpus --out $ZR/zr_${TAG}_full.r${S}.npz
echo "[$TAG.$S] SDSS (GPU=$CUDA_VISIBLE_DEVICES)"
python $ZR/zr_eval.py --mode sdss $C --tokenizer-ckpt $TOK \
  --sdss-manifest $ZR/sdss_paths.txt --max-spectra 999999 \
  --batch-size 256 --out $ZR/zr_${TAG}_sdss.r${S}.npz
echo "[$TAG.$S] DONE"
