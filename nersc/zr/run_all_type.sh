#!/bin/bash
module load pytorch/2.8.0
cd $HOME/Spectrum-FM
ZR=/pscratch/sd/j/joe2k/zr; SC=/pscratch/sd/j/joe2k
V3=$SC/deepsrch/checkpoints/approach_a_v2cache_x2x3_ddp4/best.pt
V2=$SC/deepsrch/checkpoints/approach_a_v2cache_x2_512hard_ctrl_ddp4/best.pt
rm -f $ZR/type_v*_r*.npz $ZR/plot_by_type.png
echo "=== typemap build (CPU, bg) ==="; python $ZR/build_typemap.py > $ZR/typemap.log 2>&1 &
echo "=== 4 GPU workers ==="
CUDA_VISIBLE_DEVICES=0 bash $ZR/run_type_worker.sh v3 "$V3" 0 2 > $ZR/t_v3_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash $ZR/run_type_worker.sh v3 "$V3" 1 2 > $ZR/t_v3_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 bash $ZR/run_type_worker.sh v2 "$V2" 0 2 > $ZR/t_v2_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 bash $ZR/run_type_worker.sh v2 "$V2" 1 2 > $ZR/t_v2_1.log 2>&1 &
wait
echo "=== typemap ==="; tail -2 $ZR/typemap.log
echo "=== ANALYZE ==="; python $ZR/zr_type_analyze.py $ZR
echo "=== DONE ==="; ls -la $ZR/plot_by_type.png
