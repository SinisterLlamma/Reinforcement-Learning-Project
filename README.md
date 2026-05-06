This code is for the NeurIPS 2022 paper:


Khaled Nakhleh, I-Hong Hou. DeepTOP: Deep Threshold-Optimal Policy for MDPs and RMABs. In Advances in Neural Information Processing Systems (NeurIPS 2022), volume 36, December 2022.


## Description
---


The algorithm was implemented for the Markov Decision Process (MDP), and the Restless Multi-Armed Bandits (RMABs) settings.
Corresponding settings are stored in separate directories (recovering bandits setting is under the "recovering_bandits_rmab" directory).

As an example to run an MDP algorithm, run from the MDP directory:
```
python3 -u main_DeepTOP_charging.py > output_DeepTOP_charging.txt &
python3 -u main_DeepTOP_inventory.py > output_DeepTOP_inventory.txt &
python3 -u main_DeepTOP_make_to_stock.py > output_DeepTOP_make_to_stock.txt &
```

Also for the RMAB algorithm run: 

```
python3 -u main_DeepTOP.py --nb_arms 10 --budget 3 > output_DeepTOP_A10B3.txt & 
python3 -u main_DeepTOP.py --nb_arms 20 --budget 5 > output_DeepTOP_A20B5.txt & 
python3 -u main_DeepTOP.py --nb_arms 30 --budget 6 > output_DeepTOP_A30B6.txt & 
```

## Joint-Summary-Augmented DeepTOP (C1 extension)
---

A variant of DeepTOP-RMAB where each arm's actor takes both its own state
`s_i` and the empirical state distribution across all arms `\hat{mu}_t`.
The critic remains per-arm and unchanged.

Toggle it via the `--joint-summary` flag (default off, baseline behaviour
preserved). The summary length is set by `--summary_dim` (defaults to 100,
which matches both `lineEnv` and the recovering-bandit env).

```
# baseline (original paper)
python3 -u main_DeepTOP.py --nb_arms 10 --budget 3

# C1: joint-summary-augmented actor
python3 -u main_DeepTOP.py --nb_arms 10 --budget 3 --joint-summary --summary_dim 100
```

The same flags are available in
`recovering_bandits_rmab/recovering_RMAB/main_DeepTOP.py`. To reproduce the
small comparison plot on lineEnv (3 seeds, 12000 steps, both modes), run
from the repo root:

```
python3 benchmark_joint_summary.py
```

Output (CSV reward curves and a comparison plot) is written to
`benchmark_out/`.

## Acknowledgment
---

The source code relies on classes and functions from other open-source repositories. 
Cited code includes recognition in its respective file.

The LPQL and WIBQL implementations under "tabular_methods/" and "recovering_bandits_rmab/recovering_tabular_methods/"
were taken from the repository: https://github.com/killian-34/MAIQL_and_LPQL 

