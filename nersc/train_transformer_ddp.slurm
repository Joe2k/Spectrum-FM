#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH -J xform-ddp
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# 4-GPU DDP training of SpectrumTransformer on a full Perlmutter node.
# Regular QOS, 12h budget.  srun spawns 4 Python procs, each bound to
# 1 GPU via LOCAL_RANK.
#
# DATA SOURCE (pick one):
#   TOKENIZED_DIR  X1 token cache      -> reads cached tokens, encoder NOT loaded
#                                         (TOKENIZER_CKPT not needed)
#   TOKENIZER_CKPT pretrained best.pt  -> legacy on-the-fly tokenization
# OPTIONAL env:
#   APPROACH       a or b              (default: a)
#   MAX_HEALPIX    healpix cap         (default: 10000; ignored when MANIFEST set)
#   STEPS          optimizer steps     (default: 100000)
#   BATCH_SIZE     per-GPU batch       (default: 32)
#   LR             base lr             (default: 8e-4, linear-scaled 4x)
#   NUM_WORKERS    DataLoader workers  (default: 8)
#   D_MODEL        hidden size         (default: 768)
#   N_LAYERS       enc/dec layer count (default: 6)
#   N_HEADS        attention heads     (default: 12)
#   Z_BINS         redshift bins       (default: 256; X3 = 4096)
#   Z_FIT_FILES    shards to fit z-tok (default: 200; raise for 4096 bins)
#   REDSHIFT_SOFT_SIGMA  soft-label std in bins (default: 0 = hard CE)
#   MASKED_TARGETS_ONLY  1 = supervise only encoder-hidden positions (X2)
#   MASK_RATIO_MIN/MAX   per-batch U[min,max] encoder mask ratio (X2)
#
# Submit (X2+X3 on the X1 cache):
#   TOKENIZED_DIR=$SCRATCH/dr1_tokenized_v2 \
#       MANIFEST=$SCRATCH/manifests/dr1_v2_full.jsonl \
#       MASKED_TARGETS_ONLY=1 MASK_RATIO_MIN=0.15 MASK_RATIO_MAX=0.75 \
#       Z_BINS=4096 Z_FIT_FILES=800 REDSHIFT_SOFT_SIGMA=24 \
#       RUN_NAME=approach_a_v2cache_x2x3_ddp4 \
#       sbatch nersc/train_transformer_ddp.slurm
# Submit (legacy on-the-fly):
#   TOKENIZER_CKPT=$SCRATCH/deepsrch/checkpoints/<run>/best.pt \
#       MANIFEST=$SCRATCH/deepsrch/manifests/dr1_10k.jsonl \
#       APPROACH=a sbatch nersc/train_transformer_ddp.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

if [ -z "${TOKENIZED_DIR:-}" ] && [ -z "${TOKENIZER_CKPT:-}" ]; then
    echo "ERROR: set TOKENIZED_DIR (X1 cache) or TOKENIZER_CKPT (on-the-fly)"
    exit 1
fi

REPO=$(pwd)
APPROACH="${APPROACH:-a}"
SCRATCH_OUT="$SCRATCH/deepsrch"
RUN_NAME="${RUN_NAME:-approach_${APPROACH}_ddp4_${SLURM_JOB_ID}}"

# ---------- run-scale knobs ----------
SURVEYS="sv3 main"
PROGRAMS="bright dark"
MAX_HEALPIX="${MAX_HEALPIX:-10000}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-8e-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"
D_MODEL="${D_MODEL:-768}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-12}"
REDSHIFT_LOSS_WEIGHT="${REDSHIFT_LOSS_WEIGHT:-10.0}"
ENCODER_MASK_RATIO="${ENCODER_MASK_RATIO:-0.50}"
HEALPIX_HOLDOUT_FRAC="${HEALPIX_HOLDOUT_FRAC:-0.05}"
AR_EVAL_BATCHES="${AR_EVAL_BATCHES:-8}"
Z_BINS="${Z_BINS:-256}"
Z_FIT_FILES="${Z_FIT_FILES:-200}"
REDSHIFT_SOFT_SIGMA="${REDSHIFT_SOFT_SIGMA:-0}"
WANDB_MODE_ARG="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-redshifty}"
# -------------------------------------

# Default manifest: full-DR1 cache run uses dr1_v2_full; legacy run uses the
# MAX_HEALPIX-scaled manifest under deepsrch.
if [ -n "${TOKENIZED_DIR:-}" ]; then
    MANIFEST="${MANIFEST:-$SCRATCH/manifests/dr1_v2_full.jsonl}"
else
    MANIFEST="${MANIFEST:-$SCRATCH_OUT/manifests/dr1_${MAX_HEALPIX}.jsonl}"
fi
DEFAULT_CFS_BASE="/global/cfs/cdirs/deepsrch/$USER/checkpoints"
CFS_OUT="${CFS_OUT:-$DEFAULT_CFS_BASE/$RUN_NAME}"

echo "[train-ddp] approach=$APPROACH run=$RUN_NAME"
echo "[train-ddp] manifest=$MANIFEST  cfs_out=$CFS_OUT"
echo "[train-ddp] 4 GPUs, per-gpu batch=$BATCH_SIZE, effective batch=$((BATCH_SIZE * 4)), lr=$LR"

if [ ! -s "$MANIFEST" ]; then
    if [ -n "${TOKENIZED_DIR:-}" ]; then
        echo "[train-ddp] ERROR: cache run needs the same manifest used to"
        echo "            pre-tokenize ($MANIFEST not found). Set MANIFEST."
        exit 1
    fi
    echo "[train-ddp] building manifest..."
    python "$REPO/nersc/build_dr1_index.py" \
        --root /global/cfs/cdirs/desi/public/dr1 \
        --surveys $SURVEYS --programs $PROGRAMS \
        --max-healpix "$MAX_HEALPIX" \
        --out "$MANIFEST"
fi

# Global rendezvous vars (same for all tasks). RANK/LOCAL_RANK are
# per-task and must NOT be exported here — srun sets SLURM_PROCID /
# SLURM_LOCALID per child task, and the Python entrypoint reads those.
# Exporting RANK at batch level would propagate one value (typically 0)
# to all 4 srun children, causing every rank to think it's rank 0.
export WORLD_SIZE=${SLURM_NTASKS}
export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# NCCL knobs for Slingshot-11
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Data-source flags: cache (X1) skips the tokenizer; legacy loads it.
DATA_ARGS=()
if [ -n "${TOKENIZED_DIR:-}" ]; then
    DATA_ARGS+=(--tokenized-dir "$TOKENIZED_DIR" --z-fit-files "$Z_FIT_FILES")
    echo "[train-ddp] data=cache $TOKENIZED_DIR (encoder not loaded)"
else
    DATA_ARGS+=(--tokenizer-ckpt "$TOKENIZER_CKPT")
    echo "[train-ddp] data=on-the-fly tokenizer $TOKENIZER_CKPT"
fi

# X2: masked-targets-only + variable mask ratio (opt-in).
X2_ARGS=()
if [ "${MASKED_TARGETS_ONLY:-0}" = "1" ]; then
    X2_ARGS+=(--masked-targets-only)
fi
if [ -n "${MASK_RATIO_MIN:-}" ] && [ -n "${MASK_RATIO_MAX:-}" ]; then
    X2_ARGS+=(--mask-ratio-min "$MASK_RATIO_MIN" --mask-ratio-max "$MASK_RATIO_MAX")
fi

# X3: soft redshift labels (>0 enables; sigma in bins).
Z_ARGS=()
if awk "BEGIN{exit !($REDSHIFT_SOFT_SIGMA > 0)}"; then
    Z_ARGS+=(--redshift-soft-sigma "$REDSHIFT_SOFT_SIGMA")
fi

echo "[train-ddp] z_bins=$Z_BINS soft_sigma=$REDSHIFT_SOFT_SIGMA masked_targets_only=${MASKED_TARGETS_ONLY:-0} mask_ratio=${MASK_RATIO_MIN:-fixed}..${MASK_RATIO_MAX:-}"

srun python "$REPO/nersc/train_transformer.py" \
    --manifest "$MANIFEST" \
    "${DATA_ARGS[@]}" \
    "${X2_ARGS[@]}" \
    "${Z_ARGS[@]}" \
    --approach "$APPROACH" \
    --run-name "$RUN_NAME" \
    --scratch-out "$SCRATCH_OUT" \
    --cfs-out "$CFS_OUT" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --d-model "$D_MODEL" \
    --n-encoder-layers "$N_LAYERS" \
    --n-decoder-layers "$N_LAYERS" \
    --n-heads "$N_HEADS" \
    --redshift-loss-weight "$REDSHIFT_LOSS_WEIGHT" \
    --encoder-mask-ratio "$ENCODER_MASK_RATIO" \
    --healpix-holdout-frac "$HEALPIX_HOLDOUT_FRAC" \
    --ar-eval-batches "$AR_EVAL_BATCHES" \
    --z-bins "$Z_BINS" \
    --wandb-mode "$WANDB_MODE_ARG" \
    --wandb-project "$WANDB_PROJECT" \
    --amp

echo "[train-ddp] done; best/final mirrored to $CFS_OUT"
