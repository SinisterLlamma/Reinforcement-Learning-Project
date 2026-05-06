"""Small benchmark for Joint-Summary-Augmented DeepTOP (C1).

Runs both `--joint-summary False` (baseline DeepTOP-RMAB) and
`--joint-summary True` (C1) on the lineEnv N=10, V=3 setting for a
short budget of training steps. Uses 3 seeds per mode. Saves a CSV
of per-100-step average rewards and a comparison plot, and prints
final-100-step mean +/- std.

Run from repo root:
    python benchmark_joint_summary.py
"""

import os
import subprocess
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
RMAB_DIR = os.path.join(REPO, 'RMAB')
OUT_DIR = os.path.join(REPO, 'benchmark_out')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3]
MAX_STEPS = 12000   # one reset-cycle short of the original's 13000
NB_ARMS = 10
BUDGET = 3


def run_one(seed, joint_summary):
    tag = 'aug' if joint_summary else 'base'
    log_path = os.path.join(OUT_DIR, f'rewards_{tag}_seed{seed}.csv')
    cmd = [
        sys.executable, '-u', 'main_DeepTOP.py',
        '--nb_arms', str(NB_ARMS),
        '--budget', str(BUDGET),
        '--seed', str(seed),
        '--max_steps', str(MAX_STEPS),
        '--warmup', '1000',
        '--reward_log', log_path,
    ]
    if joint_summary:
        cmd.append('--joint-summary')
    print(f'=== running {tag} seed={seed} ===')
    proc = subprocess.run(cmd, cwd=RMAB_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f'run failed: tag={tag} seed={seed}')
    return log_path


def load_curve(path):
    steps, vals = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['global_step']))
            vals.append(float(row['avg_reward']))
    return np.array(steps), np.array(vals)


def main():
    curves = {'base': [], 'aug': []}
    for joint in [False, True]:
        tag = 'aug' if joint else 'base'
        for seed in SEEDS:
            path = run_one(seed, joint)
            steps, vals = load_curve(path)
            curves[tag].append((steps, vals))

    # Align by step grid (all runs use the same step schedule).
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = {}
    for tag, label, color in [('base', 'DeepTOP (baseline)', 'C0'),
                              ('aug', 'C1: Joint-Summary-Augmented', 'C1')]:
        all_vals = np.stack([v for _, v in curves[tag]])  # (n_seeds, n_points)
        steps = curves[tag][0][0]
        mean = all_vals.mean(0)
        std = all_vals.std(0)
        ax.plot(steps, mean, label=label, color=color)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        last100_per_seed = all_vals[:, -1]   # last logged 100-step bucket per seed
        summary[tag] = (last100_per_seed.mean(), last100_per_seed.std())

    ax.set_xlabel('training step')
    ax.set_ylabel('avg reward over 100 steps (sum across arms)')
    ax.set_title(f'lineEnv  N={NB_ARMS}  V={BUDGET}  ({len(SEEDS)} seeds)')
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'comparison.png')
    fig.savefig(plot_path, dpi=120)
    print(f'\nplot saved to {plot_path}')

    print('\nFinal logged 100-step bucket (mean +/- std across seeds):')
    for tag, label in [('base', 'baseline       '), ('aug', 'joint-summary  ')]:
        m, s = summary[tag]
        print(f'  {label}: {m:.3f} +/- {s:.3f}')


if __name__ == '__main__':
    main()
