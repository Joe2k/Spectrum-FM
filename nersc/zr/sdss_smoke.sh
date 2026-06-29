ssh perlmutter 'salloc -N 1 --gpus 4 -A deepsrch_g -C gpu -q interactive -t 0:25:00 bash -lc "
set -e
module load pytorch/2.8.0 >/dev/null 2>&1
cd /global/homes/j/joe2k/Spectrum-FM
export PYTHONPATH=\$PWD:\$PWD/nersc:\$PYTHONPATH
Z=\$SCRATCH/sdss_ft_smoke; rm -rf \$Z; mkdir -p \$Z/out
TOK=\$SCRATCH/deepsrch/checkpoints/tokenizer_v2_3k_v3/final.pt
CK=\$SCRATCH/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
echo \"### build_lists\"
python nersc/build_sdss_lists.py --run2d-glob 26 --out-dir \$Z --n-train-files 150 --n-test-files 300
echo \"### pretok\"
for split in train test; do CUDA_VISIBLE_DEVICES=0 python nersc/pretok_sdss.py --tokenizer-ckpt \$TOK --paths \$Z/\${split}_paths.txt --out \$Z/sdss_\${split}.r0.npz --num-shards 1 --shard-id 0; done
python nersc/pretok_sdss.py --merge --shard-glob \"\$Z/sdss_train.r*.npz\" --out \$Z/sdss_train.npz --max-good 100
python nersc/pretok_sdss.py --merge --shard-glob \"\$Z/sdss_test.r*.npz\" --out \$Z/sdss_test.npz --max-good 200
echo \"### finetune shots=64\"
env -u SLURM_PROCID -u RANK CUDA_VISIBLE_DEVICES=0 python nersc/sdss_finetune.py --checkpoint \$CK --tokenizer-ckpt \$TOK --train-cache \$Z/sdss_train.npz --test-cache \$Z/sdss_test.npz --shots 64 --epochs 3 --eval-every 5 --sel-subset 100 --batch-size 16 --out-dir \$Z/out --tag v3
echo SMOKE_DONE; ls -la \$Z/out
"' 2>&1 | grep -vE "NOTICE|Laboratory|property|expectation|intercepted|authorized|recording|Unauthorized|disciplinary|stated|consent|^\*|Login|^\$|monitored|enforcement|By using|Any or all|conditions"
