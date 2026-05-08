#!/bin/bash
#SBATCH -A research
#SBATCH -J deeptopjreg_paper
#SBATCH -p u22
#SBATCH --nodelist=gnode090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=12:00:00
#SBATCH --array=0-5%2
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/deeptopjreg_%A_%a.out
#SBATCH --error=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/deeptopjreg_%A_%a.err
set -euo pipefail

# DeepTOP-J + decoupling regularisation (gated attention + adv normalisation).
# Writes to *_deeptopjreg_seed*.csv so it does NOT overwrite the previous
# *_deeptopj_seed*.csv runs. Both lines will appear on the comparison plot.
#
# 6 array tasks = 2 envs x 3 (N,V) configs, 3 seeds each (18 runs).

mkdir -p /home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs

ENV_PY=/ssd_scratch/eshaan.sharma/conda_envs/rl-project/bin/python
REPO=/home2/eshaan.sharma/Reinforcement-Learning-Project
OUT_DIR=$REPO/paper_out
mkdir -p "$OUT_DIR"

export HF_HOME=/ssd_scratch/eshaan.sharma/hf_cache
export PIP_CACHE_DIR=/ssd_scratch/eshaan.sharma/pip_cache
export TORCH_HOME=/ssd_scratch/eshaan.sharma/torch_cache
export XDG_CACHE_HOME=/ssd_scratch/eshaan.sharma/xdg_cache

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

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

ACTOR_HIDDEN=${ACTOR_HIDDEN:-128}
ACTOR_HEADS=${ACTOR_HEADS:-4}
ACTOR_LAYERS=${ACTOR_LAYERS:-2}
M=${M:-1.0}
MAX_STEPS=${MAX_STEPS:-12000}
WARMUP=${WARMUP:-1000}

echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID:-local} (array task ${TASK_ID})"
echo "Env:     $ENV_NAME"
echo "Config:  N=$NB_ARMS V=$BUDGET"
echo "Variant: deeptopj + gated_attention + adv_normalize"
echo "Steps:   $MAX_STEPS (warmup $WARMUP)"
echo "Seeds:   ${SEEDS[*]}"

cd "$ENV_DIR"

for SEED in "${SEEDS[@]}"; do
    LOG_CSV="$OUT_DIR/${ENV_NAME}_N${NB_ARMS}_V${BUDGET}_deeptopjreg_seed${SEED}.csv"
    echo "----"
    echo "[seed $SEED] -> $LOG_CSV"
    "$ENV_PY" -u main_DeepTOPJ.py \
        --nb_arms "$NB_ARMS" \
        --budget "$BUDGET" \
        --seed "$SEED" \
        --max_steps "$MAX_STEPS" \
        --warmup "$WARMUP" \
        --actor_hidden "$ACTOR_HIDDEN" \
        --actor_heads "$ACTOR_HEADS" \
        --actor_layers "$ACTOR_LAYERS" \
        --M "$M" \
        --actor_gated \
        --adv_normalize \
        --reward_log "$LOG_CSV"
done

echo "task ${TASK_ID} done"
