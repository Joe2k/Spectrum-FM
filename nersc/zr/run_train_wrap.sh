set -e
cd /global/homes/j/joe2k/Spectrum-FM
echo "=== ALLOC START $(date) ==="
salloc -N1 --gpus 4 -A deepsrch_g -C gpu -q interactive -t 240 bash -lc '
  bash /pscratch/sd/j/joe2k/zr/run_train4.sh
'
echo "=== ALLOC END $(date) ==="
