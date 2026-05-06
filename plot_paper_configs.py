"""Aggregate the paper-configs benchmark.

Reads every CSV under paper_out/ produced by
scripts/run_paper_configs_array.sh and produces:
  1. paper_out/comparison_grid.png : 2x3 grid (envs x (N,V) configs),
     each panel shows baseline vs augmented mean +/- std reward curves.
  2. paper_out/summary.csv : per (env, N, V, mode) final-bucket
     mean+/-std across seeds, plus the (aug - base) delta.
The summary table is also pretty-printed to stdout.
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
MODES = ['base', 'aug']
MODE_LABELS = {'base': 'DeepTOP (baseline)',
               'aug':  'C1: Joint-Summary-Augmented'}
MODE_COLORS = {'base': 'C0', 'aug': 'C1'}


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
    summary_rows = []  # (env, n, v, mode, mean, std)
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
                vals = np.stack([c[1] for c in curves])
                steps = curves[0][0]
                # Some seeds may have +/-1 logged points; truncate to min length
                min_len = min(c[1].shape[0] for c in curves)
                vals = np.stack([c[1][:min_len] for c in curves])
                steps = curves[0][0][:min_len]

                mean, std = vals.mean(0), vals.std(0)
                ax.plot(steps, mean, color=MODE_COLORS[mode],
                        label=MODE_LABELS[mode])
                ax.fill_between(steps, mean - std, mean + std,
                                color=MODE_COLORS[mode], alpha=0.2)

                last = vals[:, -1]
                summary_rows.append((env, n, v, mode, last.mean(), last.std()))

            ax.set_title(f'{env}  N={n}  V={v}')
            if i == len(ENVS) - 1:
                ax.set_xlabel('training step')
            if j == 0:
                ax.set_ylabel('avg reward / 100 steps')
            if i == 0 and j == 0:
                ax.legend(loc='lower right', fontsize=8)

    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'comparison_grid.png')
    fig.savefig(plot_path, dpi=120)
    print(f'plot saved to {plot_path}')

    if missing:
        print(f'\nWARNING: {len(missing)} CSVs missing (still running or failed):')
        for m in missing[:10]:
            print(f'  {m}')
        if len(missing) > 10:
            print(f'  ... and {len(missing) - 10} more')

    # Pretty-printed table + CSV.
    print('\nFinal logged 100-step bucket (mean +/- std across seeds):')
    print(f'  {"env":<11} {"N":>3} {"V":>3}  {"baseline":>20}  {"augmented":>20}  {"delta":>10}')
    by_key = {}
    for env, n, v, mode, m, s in summary_rows:
        by_key[(env, n, v, mode)] = (m, s)
    for env in ENVS:
        for (n, v) in CONFIGS:
            base = by_key.get((env, n, v, 'base'))
            aug  = by_key.get((env, n, v, 'aug'))
            base_str = f'{base[0]:.3f} +/- {base[1]:.3f}' if base else '   missing   '
            aug_str  = f'{aug[0]:.3f} +/- {aug[1]:.3f}'  if aug  else '   missing   '
            delta_str = f'{aug[0] - base[0]:+.3f}' if (base and aug) else '   --'
            print(f'  {env:<11} {n:>3} {v:>3}  {base_str:>20}  {aug_str:>20}  {delta_str:>10}')

    summary_csv = os.path.join(OUT_DIR, 'summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['env', 'N', 'V', 'mode', 'final_mean', 'final_std'])
        for row in summary_rows:
            w.writerow(row)
    print(f'\nsummary table -> {summary_csv}')


if __name__ == '__main__':
    main()
