
import os
import torch
import random
import argparse
from copy import deepcopy
import numpy as np
import time
import sys
sys.path.insert(0, './venv/')
from recoveringEnv import recoveringBanditsEnv
from DeepTOPJ import DeepTOPJ_RMAB


THETAVALS = [[10., 0.2, 0.0], [8.5, 0.4, 0.0], [7., 0.6, 0.0], [5.5, 0.8, 0.0]]


def initializeEnv():
    global envs, state_dims, action_dims, state_sizes, action_sizes, nb_arms, args
    for i in range(nb_arms):
        envs.append(recoveringBanditsEnv(seed=(i * args.seed) + 2357,
                                         thetaVals=THETAVALS[i % 4],
                                         noiseVar=0.0, maxWait=100))
        state_dims.append(1)
        action_dims.append(1)
        state_sizes.append([100])
        action_sizes.append([2])


def resetEnvs():
    global states, envs
    states.clear()
    for i in range(len(envs)):
        states.append(envs[i].reset())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepTOP-J (Joint actor) for RMABs')

    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--rate', default=0.001, type=float)
    parser.add_argument('--prate', default=0.0001, type=float)
    parser.add_argument('--warmup', default=1000, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--bsize', default=64, type=int)
    parser.add_argument('--rmsize', default=60000, type=int)
    parser.add_argument('--window_length', default=1, type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--validate_episodes', default=20, type=int)
    parser.add_argument('--max_episode_length', default=500, type=int)
    parser.add_argument('--validate_steps', default=2000, type=int)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float)
    parser.add_argument('--train_iter', default=200000, type=int)
    parser.add_argument('--epsilon', default=500, type=int)
    parser.add_argument('--seed', default=87452, type=int)
    parser.add_argument('--resume', default='default', type=str)
    parser.add_argument('--nb_arms', default=0, type=int)
    parser.add_argument('--budget', default=0, type=int)
    parser.add_argument('--max_steps', default=260001, type=int)
    parser.add_argument('--reward_log', default=None, type=str)
    parser.add_argument('--actor_hidden', default=128, type=int)
    parser.add_argument('--actor_heads', default=4, type=int)
    parser.add_argument('--actor_layers', default=2, type=int)
    parser.add_argument('--M', default=1.0, type=float)
    parser.add_argument('--actor_gated', dest='actor_gated', action='store_true',
                        help='ReZero-style zero-init residual gates on each '
                             'attention block; biases the joint actor toward '
                             'per-arm scoring at init.')
    parser.add_argument('--adv_normalize', dest='adv_normalize', action='store_true',
                        help='Per-arm advantage normalisation by running std '
                             'before summing into the shared-phi gradient.')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    nb_arms = args.nb_arms
    budget = args.budget

    envs = []
    states = []
    state_dims = []
    action_dims = []
    state_sizes = []
    action_sizes = []
    initializeEnv()

    hidden = [128, 128]

    # Per-arm metadata: (theta0, theta1) of each arm.
    arm_metadata = np.asarray(
        [[THETAVALS[i % 4][0], THETAVALS[i % 4][1]] for i in range(nb_arms)],
        dtype=np.float32,
    )

    agent = DeepTOPJ_RMAB(nb_arms, budget, state_dims, action_dims,
                          state_sizes, action_sizes, hidden, args,
                          arm_metadata=arm_metadata)

    resetEnvs()
    agent.reset(states)

    cumulative_reward = 0
    reward_log_rows = []

    iteration = 0
    num_step = 0

    for t in range(args.max_steps):
        if t % 13000 == 0:
            iteration += 1
            num_step = 0
            print(f'iteration {iteration}')
            agent = DeepTOPJ_RMAB(nb_arms, budget, state_dims, action_dims,
                                  state_sizes, action_sizes, hidden, args,
                                  arm_metadata=arm_metadata)
            resetEnvs()
            agent.reset(states)

        agent.is_training = True
        num_step += 1

        if num_step <= args.warmup:
            action = agent.random_action()
        elif random.uniform(0, 1.0) < 0.05:
            action = agent.random_action()
        else:
            action = agent.select_action(states)

        next_state, reward, done, info = [], [], [], []
        for i in range(len(envs)):
            ns, r, d, inf = envs[i].step(action[i])
            next_state.append(ns)
            reward.append(r)
            done.append(d)
            info.append(inf)
        next_state = deepcopy(next_state)

        agent.observe(reward, next_state, done)
        if num_step > args.warmup:
            cumulative_reward += sum(reward)
            agent.update_policy()
            if (num_step - args.warmup) % 100 == 0:
                avg = cumulative_reward / 100
                print(f'{avg}')
                reward_log_rows.append((t, float(avg)))
                cumulative_reward = 0
        states = deepcopy(next_state)

    if args.reward_log is not None:
        with open(args.reward_log, 'w') as f:
            f.write('global_step,avg_reward\n')
            for step, avg in reward_log_rows:
                f.write(f'{step},{avg}\n')
