set -e
SC=/pscratch/sd/j/joe2k
TOK=$SC/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
V2=$SC/sdss_ft/out/best_v2_5000.pt
V3=$SC/sdss_ft/out/best_v3_5000.pt
mkdir -p $SC/examples
cd /global/homes/j/joe2k/Spectrum-FM
echo "=== START $(date) ==="
salloc -N1 --gpus 1 -A deepsrch_g -C gpu -q interactive -t 240 bash -lc '
  module load pytorch/2.8.0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd /global/homes/j/joe2k/Spectrum-FM
  SC=/pscratch/sd/j/joe2k
  TOK=$SC/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
  V2=$SC/sdss_ft/out/best_v2_5000.pt
  V3=$SC/sdss_ft/out/best_v3_5000.pt
  echo "### SDSS 5000-shot FT ###"
  srun -n1 python nersc/select_examples.py --dataset sdss --out $SC/examples/sel_sdss_5000.npz \
    --ckpt-v2 $V2 --ckpt-v3 $V3 --tokenizer-ckpt $TOK \
    --sdss-paths $SC/sdss_ft/test_paths.txt --max-sdss 26000 --n-per 10 --batch-size 256
  echo "### DONE ###"
  ls -la $SC/examples/sel_sdss_5000.npz
'
echo "=== END $(date) ==="
