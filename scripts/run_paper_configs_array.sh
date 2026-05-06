#!/bin/bash
#SBATCH -A research
#SBATCH -J deeptop_c1_paper
#SBATCH -p u22
#SBATCH --nodelist=gnode091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=12:00:00
#SBATCH --array=0-35%2
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/paper_%A_%a.out
#SBATCH --error=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/paper_%A_%a.err
set -euo pipefail

# Joint-Summary-Augmented DeepTOP (C1) full paper-config benchmark.
#
# 36 tasks = 2 envs x 3 (N,V) configs x 2 modes (base/aug) x 3 seeds.
# Layout (linearised by index = env*18 + config*6 + mode*3 + seed_idx):
#   env:    line, recovering
#   config: (N=10,V=3), (N=20,V=5), (N=30,V=6)
#   mode:   base (baseline DeepTOP), aug (joint-summary)
#   seed:   1, 2, 3
#
# /ssd_scratch is node-local => pin to gnode091 where the env lives.
# Slurm logs go to /home2 (NFS) so they're written even if the node
# changes. %2 throttle stays under QOSMaxJobsPerUserLimit.

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

# Decode array index -> (env, config, mode, seed).
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

ENV_IDX=$(( TASK_ID / 18 ))                 # 0=line, 1=recovering
REM=$((  TASK_ID % 18 ))
CONFIG_IDX=$(( REM / 6 ))                   # 0,1,2 -> (10,3),(20,5),(30,6)
REM=$((  REM % 6 ))
MODE_IDX=$(( REM / 3 ))                     # 0=base, 1=aug
SEED_IDX=$(( REM % 3 ))                     # 0,1,2 -> seeds 1,2,3

NB_ARMS_LIST=(10 20 30)
BUDGET_LIST=(3 5 6)
SEEDS=(1 2 3)
ENV_NAMES=(line recovering)
ENV_DIRS=("$REPO/RMAB" "$REPO/recovering_bandits_rmab/recovering_RMAB")

ENV_NAME=${ENV_NAMES[$ENV_IDX]}
ENV_DIR=${ENV_DIRS[$ENV_IDX]}
NB_ARMS=${NB_ARMS_LIST[$CONFIG_IDX]}
BUDGET=${BUDGET_LIST[$CONFIG_IDX]}
SEED=${SEEDS[$SEED_IDX]}

if [ "$MODE_IDX" -eq 0 ]; then
    MODE_TAG="base"
    JOINT_FLAG=""
else
    MODE_TAG="aug"
    JOINT_FLAG="--joint-summary --summary_dim 100"
fi

LOG_CSV="$OUT_DIR/${ENV_NAME}_N${NB_ARMS}_V${BUDGET}_${MODE_TAG}_seed${SEED}.csv"

echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID:-local} (array task ${TASK_ID})"
echo "Env:     $ENV_NAME"
echo "Config:  N=$NB_ARMS V=$BUDGET"
echo "Mode:    $MODE_TAG"
echo "Seed:    $SEED"
echo "Out CSV: $LOG_CSV"

cd "$ENV_DIR"

"$ENV_PY" -u main_DeepTOP.py \
    --nb_arms "$NB_ARMS" \
    --budget "$BUDGET" \
    --seed "$SEED" \
    --max_steps 12000 \
    --warmup 1000 \
    --reward_log "$LOG_CSV" \
    $JOINT_FLAG
