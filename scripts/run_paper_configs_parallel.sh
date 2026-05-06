#!/bin/bash
#SBATCH -A research
#SBATCH -J deeptop_c1_paper
#SBATCH -p u22
#SBATCH --nodelist=gnode091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6000M
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/paper_%j.out
#SBATCH --error=/home2/eshaan.sharma/Reinforcement-Learning-Project/slurm_logs/paper_%j.err
set -uo pipefail

# Joint-Summary-Augmented DeepTOP (C1) full paper-config benchmark.
#
# Single sbatch job (avoids QOSMaxSubmitJobPerUserLimit). Forks all 36
# runs as parallel background processes inside the job, throttled to
# PARALLEL at a time. Each per-run stdout/stderr goes to its own file
# so the main log stays clean.
#
# Layout = 2 envs x 3 (N,V) configs x 2 modes x 3 seeds = 36 runs.
# Configs match the paper README: (10,3), (20,5), (30,6).

REPO=/home2/eshaan.sharma/Reinforcement-Learning-Project
ENV_PY=/ssd_scratch/eshaan.sharma/conda_envs/rl-project/bin/python
OUT_DIR=$REPO/paper_out
RUN_LOG_DIR=$REPO/paper_run_logs
mkdir -p "$OUT_DIR" "$RUN_LOG_DIR" "$REPO/slurm_logs"

# Caches on /ssd_scratch (node-local, gnode091).
export HF_HOME=/ssd_scratch/eshaan.sharma/hf_cache
export PIP_CACHE_DIR=/ssd_scratch/eshaan.sharma/pip_cache
export TORCH_HOME=/ssd_scratch/eshaan.sharma/torch_cache
export XDG_CACHE_HOME=/ssd_scratch/eshaan.sharma/xdg_cache

# Each background run is single-threaded. The DeepTOP per-arm update
# loop is small enough that BLAS parallelism just adds overhead;
# single-threaded keeps CPU pinning clean.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Concurrency. gnode091 has 11.2 GB total; interactive job holds ~4 GB,
# leaving ~7 GB. Worst-case per run (N=30 augmented) is ~940 MB.
# PARALLEL=4 uses ~3.6 GB worst case -- comfortable 3 GB headroom.
# Each background run uses OMP=1, so 4 runs share the 8 reserved CPUs.
# Total wallclock = ceil(36/4) * ~40 min = ~6 hours.
PARALLEL=${PARALLEL:-4}

NB_ARMS_LIST=(10 20 30)
BUDGET_LIST=(3 5 6)
SEEDS=(1 2 3)
ENV_NAMES=(line recovering)
ENV_DIRS=("$REPO/RMAB" "$REPO/recovering_bandits_rmab/recovering_RMAB")

run_one() {
    local env_name="$1" env_dir="$2" nb_arms="$3" budget="$4" mode_tag="$5" seed="$6"
    local log_csv="$OUT_DIR/${env_name}_N${nb_arms}_V${budget}_${mode_tag}_seed${seed}.csv"
    local stdout="$RUN_LOG_DIR/${env_name}_N${nb_arms}_V${budget}_${mode_tag}_seed${seed}.out"

    local joint_args=()
    if [ "$mode_tag" = "aug" ]; then
        joint_args=(--joint-summary --summary_dim 100)
    fi

    if [ -s "$log_csv" ]; then
        # Already produced (e.g. on a re-run). Skip to save time.
        echo "[skip] $log_csv exists" > "$stdout"
        return 0
    fi

    {
        echo "=== $env_name N=$nb_arms V=$budget mode=$mode_tag seed=$seed ==="
        cd "$env_dir"
        "$ENV_PY" -u main_DeepTOP.py \
            --nb_arms "$nb_arms" \
            --budget "$budget" \
            --seed "$seed" \
            --max_steps 12000 \
            --warmup 1000 \
            --reward_log "$log_csv" \
            "${joint_args[@]}"
        echo "=== done: $env_name N=$nb_arms V=$budget mode=$mode_tag seed=$seed ==="
    } &> "$stdout"
}

echo "Host:     $(hostname)"
echo "Job:      ${SLURM_JOB_ID:-local}"
echo "Cpus:     ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Parallel: $PARALLEL"
echo "Out dir:  $OUT_DIR"
echo "Per-run logs: $RUN_LOG_DIR"

start_ts=$(date +%s)
running=0
total=0
for env_idx in 0 1; do
    env_name=${ENV_NAMES[$env_idx]}
    env_dir=${ENV_DIRS[$env_idx]}
    for cfg_idx in 0 1 2; do
        nb_arms=${NB_ARMS_LIST[$cfg_idx]}
        budget=${BUDGET_LIST[$cfg_idx]}
        for mode_tag in base aug; do
            for seed in "${SEEDS[@]}"; do
                total=$((total + 1))
                run_one "$env_name" "$env_dir" "$nb_arms" "$budget" "$mode_tag" "$seed" &
                running=$((running + 1))
                if [ "$running" -ge "$PARALLEL" ]; then
                    wait -n
                    running=$((running - 1))
                fi
            done
        done
    done
done

wait

end_ts=$(date +%s)
echo "All $total runs completed in $((end_ts - start_ts)) s."
echo "Aggregate with:"
echo "  $ENV_PY $REPO/plot_paper_configs.py"

# Print quick smoke summary of which CSVs ended up populated.
echo "---"
echo "CSV inventory:"
ls -1 "$OUT_DIR"/*.csv 2>/dev/null | wc -l
