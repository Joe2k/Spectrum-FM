#!/bin/bash
# v450 (V4 @50k, z_tokenizer grafted) per-type eval, sharded 4-way across the node's
# GPUs, then emit the combined decoupled-mask A/B + headline report (needs dec50 &
# v4200 npz already present from run_decab_eval.sh).
set -uo pipefail
module load pytorch/2.8.0
cd "$HOME/Spectrum-FM"
ZR=/pscratch/sd/j/joe2k/zr; SC=/pscratch/sd/j/joe2k
TOK=$SC/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
MAN=$SC/manifests/dr1_v2_full.jsonl; TOKDIR=$SC/dr1_tokenized_v2
CK=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_zv2_ddp4/step_00050000_zt.pt

rm -f $ZR/type_v450_r*.npz
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python $ZR/zr_type_eval.py --checkpoint $CK \
    --tokenizer-ckpt $TOK --manifest $MAN --tokenized-dir $TOKDIR \
    --num-shards 4 --shard-id $g --batch-size 512 --out $ZR/type_v450_r$g.npz \
    > $ZR/decab_v450_s$g.log 2>&1 &
done
wait
echo "=== v450 shards done; combined analysis ==="
python $HOME/Spectrum-FM/nersc/decab_analyze.py $ZR dec50 v450 v4200 \
  | tee $ZR/decab_report.txt
echo "=== report at $ZR/decab_report.txt ==="
