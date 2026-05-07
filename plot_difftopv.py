"""Aggregate the DiffTopV benchmark and plot it against the DeepTOP baseline.

Reads CSVs under paper_out/ produced by:
  - scripts/run_difftopv_array.sh   -> {env}_N{N}_V{V}_difftopv_seed{s}.csv
  - scripts/run_paper_configs_array.sh -> {env}_N{N}_V{V}_base_seed{s}.csv
                                       -> {env}_N{N}_V{V}_aug_seed{s}.csv

Produces:
  paper_out/difftopv_comparison.png : 2x3 grid (envs x (N,V) configs),
      mean +/- std reward curves for DeepTOP base, C1-aug, and DiffTopV.
  paper_out/difftopv_summary.csv    : per (env, N, V, mode) final-bucket
      mean and std across seeds.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, 'paper_out')
SEEDS = [1, 2, 3]
ENVS = ['line', 'recovering']
CONFIGS = [(10, 3), (20, 5), (30, 6)]

# Modes plotted side by side. Drop one if its CSVs aren't on disk.
MODES = ['base', 'aug', 'difftopv']
MODE_LABELS = {
    'base': 'DeepTOP (baseline)',
    'aug': 'C1: Joint-Summary-Augmented',
    'difftopv': 'DiffTopV (ours)',
}
MODE_COLORS = {'base': 'C0', 'aug': 'C1', 'difftopv': 'C2'}


def load_curve(path):
    steps, vals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row['global_step']))
            vals.append(float(row['avg_reward']))
    return np.array(steps), np.array(vals)


def csv_path(env, n, v, mode, seed):
    return os.path.join(OUT_DIR, f'{env}_N{n}_V{v}_{mode}_seed{seed}.csv')


def main():
    fig, axes = plt.subplots(len(ENVS), len(CONFIGS),
                             figsize=(15, 8), sharex='col')
    summary_rows = []
    missing = []

    for i, env in enumerate(ENVS):
        for j, (n, v) in enumerate(CONFIGS):
            ax = axes[i][j]
            for mode in MODES:
                curves = []
                for seed in SEEDS:
                    p = csv_path(env, n, v, mode, seed)
                    if not os.path.exists(p):
                        missing.append(p)
                        continue
                    curves.append(load_curve(p))
                if not curves:
                    continue
                min_len = min(c[1].shape[0] for c in curves)
                vals = np.stack([c[1][:min_len] for c in curves])
                steps = curves[0][0][:min_len]
                mean, std = vals.mean(0), vals.std(0)
                ax.plot(steps, mean, color=MODE_COLORS[mode],
                        label=MODE_LABELS[mode], linewidth=1.5)
                ax.fill_between(steps, mean - std, mean + std,
                                color=MODE_COLORS[mode], alpha=0.18)

                last = vals[:, -1]
                summary_rows.append((env, n, v, mode,
                                     float(last.mean()), float(last.std()),
                                     int(vals.shape[0])))

            ax.set_title(f'{env}  N={n}  V={v}')
            if i == len(ENVS) - 1:
                ax.set_xlabel('training step')
            if j == 0:
                ax.set_ylabel('avg reward (per 100 steps)')
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(loc='lower right', fontsize=9)

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'difftopv_comparison.png')
    fig.savefig(out_png, dpi=140)
    print(f'wrote {out_png}')

    out_csv = os.path.join(OUT_DIR, 'difftopv_summary.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['env', 'N', 'V', 'mode', 'final_mean', 'final_std', 'n_seeds'])
        for row in summary_rows:
            w.writerow(row)
    print(f'wrote {out_csv}')

    print()
    print(f'{"env":<11} {"N":>3} {"V":>3} {"mode":<10} {"mean":>8} {"std":>8} {"#seeds":>6}')
    for env, n, v, mode, m, s, k in summary_rows:
        print(f'{env:<11} {n:>3} {v:>3} {mode:<10} {m:>8.3f} {s:>8.3f} {k:>6}')

    if missing:
        print(f'\n[note] {len(missing)} CSV(s) missing — partial plot. First few:')
        for p in missing[:5]:
            print(f'  {p}')


if __name__ == '__main__':
    main()
