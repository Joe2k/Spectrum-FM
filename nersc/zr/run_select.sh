set -e
SC=/pscratch/sd/j/joe2k
TOK=$SC/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
V2=$SC/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt
V3=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
mkdir -p $SC/examples
cd /global/homes/j/joe2k/Spectrum-FM
echo "=== START $(date) ==="
salloc -N1 --gpus 1 -A deepsrch_g -C gpu -q interactive -t 60 bash -lc '
  module load pytorch/2.8.0
  cd /global/homes/j/joe2k/Spectrum-FM
  SC=/pscratch/sd/j/joe2k
  TOK=$SC/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
  V2=$SC/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt
  V3=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
  echo "### SDSS ###"
  python nersc/select_examples.py --dataset sdss --out $SC/examples/sel_sdss.npz \
    --ckpt-v2 $V2 --ckpt-v3 $V3 --tokenizer-ckpt $TOK \
    --sdss-paths $SC/sdss_ft/test_paths.txt --max-sdss 15000 --n-per 10
  echo "### VAL ###"
  python nersc/select_examples.py --dataset val --out $SC/examples/sel_val.npz \
    --ckpt-v2 $V2 --ckpt-v3 $V3 \
    --manifest $SC/manifests/dr1_v2_full.jsonl --tokenized-dir $SC/dr1_tokenized_v2 --n-per 10
  echo "### DONE ###"
  ls -la $SC/examples/
'
echo "=== END $(date) ==="
