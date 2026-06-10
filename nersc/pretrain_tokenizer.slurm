#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH -J tok-pretrain
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Full tokenizer pretrain on a single A100 in shared QOS.
# `shared` lets us request just 1 GPU (vs a whole 4-GPU node) so we burn
# 1/4 the allocation hours. Max 48h wallclock; we use 24h.
#
# Configure SURVEYS / PROGRAMS / TARGET_SPECTRA below for the run scale
# you want.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

REPO=$(pwd)
SCRATCH_OUT="$SCRATCH/deepsrch"
RUN_NAME="${RUN_NAME:-tokenizer_v2_${SLURM_JOB_ID}}"

# ---------- run-scale knobs ----------
# v1 lesson: --max-healpix is a GLOBAL cap that build_dr1_index.py fills
# sequentially, so the v1 manifest was ~all sv3/bright (~500k spectra).
# Default to a per-(survey,program) cap so main/dark contribute equally.
SURVEYS="${SURVEYS:-sv3 main}"
PROGRAMS="${PROGRAMS:-bright dark}"
MAX_HEALPIX_PER_PAIR="${MAX_HEALPIX_PER_PAIR:-500}"  # 4 pairs x 500 ≈ v1 volume, balanced
MAX_HEALPIX="${MAX_HEALPIX:-}"       # legacy global cap; empty = unused
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# -------------------------------------

BUILD_FLAGS=""
[ -n "$MAX_HEALPIX" ] && BUILD_FLAGS="$BUILD_FLAGS --max-healpix $MAX_HEALPIX"
[ -n "$MAX_HEALPIX_PER_PAIR" ] && BUILD_FLAGS="$BUILD_FLAGS --max-healpix-per-pair $MAX_HEALPIX_PER_PAIR"

MANIFEST="$SCRATCH_OUT/manifests/dr1_${SLURM_JOB_ID}.jsonl"
# CFS mirror lives under the project CFS dir (multi-TB quota), not $HOME (40 GB).
DEFAULT_CFS_BASE="/global/cfs/cdirs/deepsrch/$USER/checkpoints"
CFS_OUT="${CFS_OUT:-$DEFAULT_CFS_BASE/$RUN_NAME}"

echo "[pretrain] run_name=$RUN_NAME"
echo "[pretrain] manifest=$MANIFEST  (surveys=$SURVEYS programs=$PROGRAMS"
echo "[pretrain]  max_healpix_per_pair=${MAX_HEALPIX_PER_PAIR:-none} max_healpix=${MAX_HEALPIX:-none})"
echo "[pretrain] cfs_out=$CFS_OUT"

if [ ! -s "$MANIFEST" ]; then
    echo "[pretrain] building manifest..."
    python "$REPO/nersc/build_dr1_index.py" \
        --root /global/cfs/cdirs/desi/public/dr1 \
        --surveys $SURVEYS \
        --programs $PROGRAMS \
        $BUILD_FLAGS \
        --out "$MANIFEST"
fi

srun -n 1 -c 32 -G 1 python "$REPO/nersc/pretrain_tokenizer.py" \
    --manifest "$MANIFEST" \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT/checkpoints" \
    --cfs-out "$CFS_OUT" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --amp

echo "[pretrain] done; best.pt mirrored to $CFS_OUT"
