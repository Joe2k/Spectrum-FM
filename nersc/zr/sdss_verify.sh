ssh perlmutter 'salloc -N 1 --gpus 1 -A deepsrch_g -C gpu -q interactive -t 0:15:00 bash -lc "
set -e
module load pytorch/2.8.0 >/dev/null 2>&1
cd /global/homes/j/joe2k/Spectrum-FM
export PYTHONPATH=\$PWD:\$PWD/nersc:\$PYTHONPATH
Z=\$SCRATCH/sdss_ft_smoke
TOK=\$SCRATCH/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
CK=\$SCRATCH/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
rm -rf \$Z/out; mkdir -p \$Z/out
env -u SLURM_PROCID -u RANK python nersc/sdss_finetune.py --checkpoint \$CK --tokenizer-ckpt \$TOK --train-cache \$Z/sdss_train.npz --test-cache \$Z/sdss_test.npz --shots 64 --epochs 3 --eval-every 5 --sel-subset 100 --batch-size 16 --out-dir \$Z/out --tag v3
echo VERIFY_DONE; ls -la \$Z/out
"' 2>&1 | grep -vE "NOTICE|Laboratory|property|expectation|intercepted|authorized|recording|Unauthorized|disciplinary|stated|consent|^\*|Login|^\$|monitored|enforcement|By using|Any or all|conditions"
