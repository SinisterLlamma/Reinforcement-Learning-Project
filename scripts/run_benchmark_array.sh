#!/bin/bash
#SBATCH -A research
#SBATCH -J deeptop_c1
#SBATCH -p u22
#SBATCH --nodelist=gnode091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=08:00:00
#SBATCH --array=0-5%2
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/deeptop_c1_%A_%a.out
#SBATCH --error=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/deeptop_c1_%A_%a.err
set -euo pipefail

# /ssd_scratch is NODE-LOCAL ext4 -- the conda env at
# /ssd_scratch/eshaan.sharma/conda_envs/rl-project only exists on
# gnode091, hence the --nodelist pin above. Slurm log files go to
# /home2 (NFS) so they're written even if the node assignment changes.
# QOSMaxJobsPerUserLimit caps concurrent array tasks; %2 throttles us
# below that cap (adjust if your QoS allows more).

# Joint-Summary-Augmented DeepTOP (C1) benchmark, lineEnv N=10 V=3.
# Job-array layout: 6 tasks = 2 modes x 3 seeds.
#   id 0,1,2 -> baseline,         seeds 1,2,3
#   id 3,4,5 -> joint-summary,    seeds 1,2,3
# After all tasks finish, run:
#   python plot_benchmark.py
# from the repo root to aggregate the CSVs and produce the plot.

mkdir -p /ssd_scratch/eshaan.sharma/logs

ENV_PY=/ssd_scratch/eshaan.sharma/conda_envs/rl-project/bin/python
REPO=/home2/eshaan.sharma/Reinforcement-Learning-Project
OUT_DIR=$REPO/benchmark_out
mkdir -p "$OUT_DIR"

# Caches on /ssd_scratch
export HF_HOME=/ssd_scratch/eshaan.sharma/hf_cache
export PIP_CACHE_DIR=/ssd_scratch/eshaan.sharma/pip_cache
export TORCH_HOME=/ssd_scratch/eshaan.sharma/torch_cache
export XDG_CACHE_HOME=/ssd_scratch/eshaan.sharma/xdg_cache

# Each task gets 2 CPUs; cap thread pools so we don't oversubscribe.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

# Map array index -> (mode, seed)
SEEDS=(1 2 3)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if   [ "$TASK_ID" -lt 3 ]; then
    MODE_TAG="base"
    SEED=${SEEDS[$TASK_ID]}
    JOINT_FLAG=""
else
    MODE_TAG="aug"
    SEED=${SEEDS[$((TASK_ID - 3))]}
    JOINT_FLAG="--joint-summary --summary_dim 100"
fi

LOG_CSV="$OUT_DIR/rewards_${MODE_TAG}_seed${SEED}.csv"

echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID:-local} (array task ${TASK_ID})"
echo "Mode:    $MODE_TAG"
echo "Seed:    $SEED"
echo "Out CSV: $LOG_CSV"

cd "$REPO/RMAB"

"$ENV_PY" -u main_DeepTOP.py \
    --nb_arms 10 \
    --budget 3 \
    --seed "$SEED" \
    --max_steps 12000 \
    --warmup 1000 \
    --reward_log "$LOG_CSV" \
    $JOINT_FLAG
