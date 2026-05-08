#!/bin/bash
#SBATCH -A research
#SBATCH -J neurwin_paper
#SBATCH -p u22
#SBATCH --nodelist=gnode090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1%1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/neurwin_%A_%a.out
#SBATCH --error=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/neurwin_%A_%a.err
set -euo pipefail

# NeurWIN paper-config benchmark (compact 2-task array to fit QOS submit limit).
#
# 2 array tasks = 2 envs (line, recovering).
# Each task loops over 3 (N,V) configs and 3 seeds:
#   for each config:
#     1. Pretrain N independent NeurWIN arms (skipped if checkpoints exist).
#     2. For each seed: run main_neurwin_paper.py against the joint env,
#        log paper_out/{env}_N{N}_V{V}_neurwin_seed{1,2,3}.csv .
# => 9 eval runs per task, 18 total. Throttle %2 lets both tasks run together.

mkdir -p /home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs

ENV_PY=/ssd_scratch/eshaan.sharma/conda_envs/rl-project/bin/python
REPO=/home2/eshaan.sharma/Reinforcement-Learning-Project
OUT_DIR=$REPO/paper_out
mkdir -p "$OUT_DIR"

# Caches on /ssd_scratch
export HF_HOME=/ssd_scratch/eshaan.sharma/hf_cache
export PIP_CACHE_DIR=/ssd_scratch/eshaan.sharma/pip_cache
export TORCH_HOME=/ssd_scratch/eshaan.sharma/torch_cache
export XDG_CACHE_HOME=/ssd_scratch/eshaan.sharma/xdg_cache

# Cap thread pools.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ENV_NAMES=(line recovering)
ENV_DIRS=("$REPO/RMAB" "$REPO/recovering_bandits_rmab/recovering_RMAB")
ENV_NAME=${ENV_NAMES[$TASK_ID]}
ENV_DIR=${ENV_DIRS[$TASK_ID]}

NB_ARMS_LIST=(10 20 30)
BUDGET_LIST=(3 5 6)
SEEDS=(1 2 3)

MAX_STEPS=${MAX_STEPS:-12000}
WARMUP=${WARMUP:-1000}
PRETRAIN_SEED=${PRETRAIN_SEED:-87452}

echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID:-local} (array task ${TASK_ID})"
echo "Env:     $ENV_NAME"
echo "Steps:   $MAX_STEPS (warmup $WARMUP)"

cd "$ENV_DIR"

for CONFIG_IDX in 0 1 2; do
    NB_ARMS=${NB_ARMS_LIST[$CONFIG_IDX]}
    BUDGET=${BUDGET_LIST[$CONFIG_IDX]}
    PRETRAIN_DIR="$ENV_DIR/neurwin_training_results/arms_${NB_ARMS}_activate_${BUDGET}"

    echo "========================================"
    echo "[config] N=$NB_ARMS V=$BUDGET"
    echo "[config] pretrain dir: $PRETRAIN_DIR"

    LAST_EPISODE_CKPT="$PRETRAIN_DIR/arm_$((NB_ARMS-1))_40.pt"
    if [ -f "$LAST_EPISODE_CKPT" ]; then
        echo "[pretrain] checkpoints already present, skipping"
    else
        echo "[pretrain] training $NB_ARMS arms (seed $PRETRAIN_SEED) ..."
        "$ENV_PY" -u neurwin_train.py \
            --nb_arms "$NB_ARMS" \
            --budget "$BUDGET" \
            --seed "$PRETRAIN_SEED"
        echo "[pretrain] done."
    fi

    for SEED in "${SEEDS[@]}"; do
        LOG_CSV="$OUT_DIR/${ENV_NAME}_N${NB_ARMS}_V${BUDGET}_neurwin_seed${SEED}.csv"
        echo "[seed $SEED] -> $LOG_CSV"
        "$ENV_PY" -u main_neurwin_paper.py \
            --nb_arms "$NB_ARMS" \
            --budget "$BUDGET" \
            --seed "$SEED" \
            --max_steps "$MAX_STEPS" \
            --warmup "$WARMUP" \
            --reward_log "$LOG_CSV"
    done
done

echo "task ${TASK_ID} done"
