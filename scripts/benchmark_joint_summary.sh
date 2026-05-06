#!/bin/bash
#SBATCH -A research
#SBATCH -J deeptop_c1
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/ssd_scratch/eshaan.sharma/logs/deeptop_c1_%j.out
#SBATCH --error=/ssd_scratch/eshaan.sharma/logs/deeptop_c1_%j.err
set -euo pipefail

# Joint-Summary-Augmented DeepTOP (C1) benchmark on lineEnv.
# Runs baseline DeepTOP-RMAB and the C1 variant (3 seeds each) and
# writes reward CSVs + a comparison plot to ./benchmark_out/.

# CPU-only run (project does not use GPUs).
mkdir -p /ssd_scratch/eshaan.sharma/logs

ENV_PY=/ssd_scratch/eshaan.sharma/conda_envs/rl-project/bin/python

# Caches on /ssd_scratch
export HF_HOME=/ssd_scratch/eshaan.sharma/hf_cache
export HUGGINGFACE_HUB_CACHE=/ssd_scratch/eshaan.sharma/hf_cache
export TRANSFORMERS_CACHE=/ssd_scratch/eshaan.sharma/hf_cache
export PIP_CACHE_DIR=/ssd_scratch/eshaan.sharma/pip_cache
export TORCH_HOME=/ssd_scratch/eshaan.sharma/torch_cache
export XDG_CACHE_HOME=/ssd_scratch/eshaan.sharma/xdg_cache

# Keep BLAS / OMP from oversubscribing the 8 reserved CPUs.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd /home2/eshaan.sharma/Reinforcement-Learning-Project

echo "Host:   $(hostname)"
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Python: $ENV_PY"

"$ENV_PY" -u benchmark_joint_summary.py
