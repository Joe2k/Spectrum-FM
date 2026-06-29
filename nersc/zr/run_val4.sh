set -e
cd /global/homes/j/joe2k/Spectrum-FM
SC=/pscratch/sd/j/joe2k
rm -f $SC/examples/val_shard.r*.npz $SC/examples/val.w*.log
echo "=== VAL4 START $(date) ==="
salloc -N1 --gpus 4 -A deepsrch_g -C gpu -q interactive -t 240 bash -lc '
  module load pytorch/2.8.0
  SC=/pscratch/sd/j/joe2k
  V2=$SC/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt
  V3=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
  MAN=$SC/manifests/dr1_v2_full.jsonl ; TOK=$SC/dr1_tokenized_v2
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  srun -n1 bash -lc "
    cd /global/homes/j/joe2k/Spectrum-FM
    torchrun --standalone --nproc_per_node=4 nersc/select_examples.py --dataset val --out /dev/null \
      --ckpt-v2 $V2 --ckpt-v3 $V3 --manifest $MAN --tokenized-dir $TOK \
      --shard-out-prefix $SC/examples/val_shard --batch-size 384 --num-workers 6
  "
  echo "=== merging ==="
  cd /global/homes/j/joe2k/Spectrum-FM
  python nersc/select_examples.py --dataset val --out $SC/examples/sel_val.npz \
    --ckpt-v2 $V2 --ckpt-v3 $V3 --manifest $MAN --tokenized-dir $TOK \
    --merge --merge-glob "$SC/examples/val_shard.r*.npz" --n-per 10
  echo "### DONE ###"; ls -la $SC/examples/sel_val.npz
'
echo "=== VAL4 END $(date) ==="
