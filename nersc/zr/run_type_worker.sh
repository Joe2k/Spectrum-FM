#!/bin/bash
set -uo pipefail
ZR=/pscratch/sd/j/joe2k/zr; SC=/pscratch/sd/j/joe2k
TOK=$SC/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
MAN=$SC/manifests/dr1_v2_full.jsonl; TOKDIR=$SC/dr1_tokenized_v2
TAG=$1; CKPT=$2; S=$3; N=$4
echo "[$TAG.$S] start (GPU=$CUDA_VISIBLE_DEVICES)"
python $ZR/zr_type_eval.py --checkpoint $CKPT --tokenizer-ckpt $TOK \
  --manifest $MAN --tokenized-dir $TOKDIR --num-shards $N --shard-id $S \
  --batch-size 512 --out $ZR/type_${TAG}_r${S}.npz
echo "[$TAG.$S] DONE"
