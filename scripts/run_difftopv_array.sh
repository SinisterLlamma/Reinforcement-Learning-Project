#!/bin/bash
#SBATCH -A research
#SBATCH -J difftopv_paper
#SBATCH -p u22
#SBATCH --nodelist=gnode090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=12:00:00
#SBATCH --array=0-5%2
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/difftopv_%A_%a.out
#SBATCH --error=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/difftopv_%A_%a.err
set -euo pipefail

# DiffTopV (End-to-End Differentiable Top-V) full paper-config benchmark.
#
# 6 array tasks = 2 envs x 3 (N,V) configs.
# Each task runs all 3 seeds sequentially => 18 total runs.
# Layout (linearised by index = env*3 + config):
#   env:    line, recovering
#   config: (N=10,V=3), (N=20,V=5), (N=30,V=6)
# Throttle %2 to stay well under QOSMaxSubmitJobPerUserLimit.

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

# Decode array index -> (env, config).
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

ENV_IDX=$(( TASK_ID / 3 ))                  # 0=line, 1=recovering
CONFIG_IDX=$(( TASK_ID % 3 ))               # 0,1,2 -> (10,3),(20,5),(30,6)

NB_ARMS_LIST=(10 20 30)
BUDGET_LIST=(3 5 6)
SEEDS=(1 2 3)
ENV_NAMES=(line recovering)
ENV_DIRS=("$REPO/RMAB" "$REPO/recovering_bandits_rmab/recovering_RMAB")

ENV_NAME=${ENV_NAMES[$ENV_IDX]}
ENV_DIR=${ENV_DIRS[$ENV_IDX]}
NB_ARMS=${NB_ARMS_LIST[$CONFIG_IDX]}
BUDGET=${BUDGET_LIST[$CONFIG_IDX]}

# DiffTopV-specific defaults (overridable via env vars at submission time)
BETA=${BETA:-5.0}
MAX_STEPS=${MAX_STEPS:-12000}
WARMUP=${WARMUP:-1000}

echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID:-local} (array task ${TASK_ID})"
echo "Env:     $ENV_NAME"
echo "Config:  N=$NB_ARMS V=$BUDGET"
echo "Beta:    $BETA"
echo "Steps:   $MAX_STEPS (warmup $WARMUP)"
echo "Seeds:   ${SEEDS[*]}  (run sequentially in this task)"

cd "$ENV_DIR"

for SEED in "${SEEDS[@]}"; do
    LOG_CSV="$OUT_DIR/${ENV_NAME}_N${NB_ARMS}_V${BUDGET}_difftopv_seed${SEED}.csv"
    echo "----"
    echo "[seed $SEED] -> $LOG_CSV"
    "$ENV_PY" -u main_DiffTopV.py \
        --nb_arms "$NB_ARMS" \
        --budget "$BUDGET" \
        --seed "$SEED" \
        --max_steps "$MAX_STEPS" \
        --warmup "$WARMUP" \
        --beta "$BETA" \
        --reward_log "$LOG_CSV"
done

echo "task ${TASK_ID} done"
