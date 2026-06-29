#!/bin/bash
# Per-type eval for the decoupled-mask A/B + V4 headline, 3 checkpoints in parallel
# on one interactive 4-GPU node. Reuses zr/zr_type_eval.py (blind-z full-spectrum +
# z-hidden 50%-masked recon in a single pass) and the existing typemap.npz.
#   GPU0: DEC @50k   (abmask_dec_zv2/last.pt)            -> type_dec50_r0.npz
#   GPU1: V4  @50k   (..._zv2_ddp4/step_00050000.pt)     -> type_v450_r0.npz   (fair A/B)
#   GPU2: V4  @200k  (..._zv2_ddp4/last.pt)              -> type_v4200_r0.npz  (headline)
set -uo pipefail
module load pytorch/2.8.0
cd "$HOME/Spectrum-FM"
ZR=/pscratch/sd/j/joe2k/zr; SC=/pscratch/sd/j/joe2k
TOK=$SC/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
MAN=$SC/manifests/dr1_v2_full.jsonl; TOKDIR=$SC/dr1_tokenized_v2
V4=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_zv2_ddp4
DEC=$SC/deepsrch/checkpoints/abmask_dec_zv2

run() {  # tag ckpt gpu
  echo "[$1] start gpu=$3 ckpt=$2"
  CUDA_VISIBLE_DEVICES=$3 python $ZR/zr_type_eval.py --checkpoint "$2" \
    --tokenizer-ckpt $TOK --manifest $MAN --tokenized-dir $TOKDIR \
    --num-shards 1 --shard-id 0 --batch-size 512 --out $ZR/type_$1_r0.npz \
    > $ZR/decab_$1.log 2>&1
  echo "[$1] DONE rc=$?"
}

rm -f $ZR/type_dec50_r0.npz $ZR/type_v450_r0.npz $ZR/type_v4200_r0.npz
run dec50  "$DEC/last.pt"            0 &
run v450   "$V4/step_00050000.pt"   1 &
run v4200  "$V4/last.pt"            2 &
wait
echo "=== all workers done; analyzing ==="
python $HOME/Spectrum-FM/nersc/decab_analyze.py $ZR dec50 v450 v4200 \
  | tee $ZR/decab_report.txt
echo "=== report at $ZR/decab_report.txt ==="
