"""Aggregate per-(mode, seed) reward CSVs from benchmark_out/ and
produce a comparison plot + final-bucket mean/std summary.

Run after the slurm array (scripts/run_benchmark_array.sh) has
finished writing all 6 CSVs.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, 'benchmark_out')
SEEDS = [1, 2, 3]


def load_curve(path):
    steps, vals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row['global_step']))
            vals.append(float(row['avg_reward']))
    return np.array(steps), np.array(vals)


def main():
    curves = {'base': [], 'aug': []}
    for tag in curves:
        for seed in SEEDS:
            path = os.path.join(OUT_DIR, f'rewards_{tag}_seed{seed}.csv')
            if not os.path.exists(path):
                raise FileNotFoundError(f'missing {path}; did the array finish?')
            curves[tag].append(load_curve(path))

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = {}
    for tag, label, color in [('base', 'DeepTOP (baseline)', 'C0'),
                              ('aug', 'C1: Joint-Summary-Augmented', 'C1')]:
        vals = np.stack([v for _, v in curves[tag]])
        steps = curves[tag][0][0]
        mean, std = vals.mean(0), vals.std(0)
        ax.plot(steps, mean, label=label, color=color)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        summary[tag] = (vals[:, -1].mean(), vals[:, -1].std())

    ax.set_xlabel('training step')
    ax.set_ylabel('avg reward over 100 steps (sum across arms)')
    ax.set_title('lineEnv  N=10  V=3  (3 seeds)')
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'comparison.png')
    fig.savefig(plot_path, dpi=120)
    print(f'plot saved to {plot_path}')

    print('\nFinal logged 100-step bucket (mean +/- std across seeds):')
    for tag, label in [('base', 'baseline       '), ('aug', 'joint-summary  ')]:
        m, s = summary[tag]
        print(f'  {label}: {m:.3f} +/- {s:.3f}')


if __name__ == '__main__':
    main()
