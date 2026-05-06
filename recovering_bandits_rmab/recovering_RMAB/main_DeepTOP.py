
import os
import torch
import random
import argparse
from copy import deepcopy
import itertools
import numpy as np
import operator
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import copy
from math import ceil
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0,'./venv/')
from recoveringEnv import recoveringBanditsEnv
from DeepTOP import DeepTOP_RMAB



THETAVALS = [[10., 0.2, 0.0], [8.5, 0.4, 0.0], [7., 0.6, 0.0], [5.5, 0.8, 0.0]]

def initializeEnv():
    global envs, state_dims, action_dims, state_sizes, action_sizes, nb_arms, args
    for i in range(nb_arms):
        envs.append(recoveringBanditsEnv(seed=(i*args.seed)+2357, thetaVals=THETAVALS[i%4], noiseVar=0.0, maxWait=100))
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

    parser = argparse.ArgumentParser(description='DeepTOP implementation in the RMAB setting')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=1000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=500, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=87452, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--nb_arms', default=0, type=int, help='Number of arms')
    parser.add_argument('--budget', default=0, type=int, help='Budget')
    # Joint-Summary-Augmented DeepTOP (C1) extension
    parser.add_argument('--joint-summary', dest='joint_summary', action='store_true',
                        help='Use Joint-Summary-Augmented DeepTOP: actor takes (s_i, joint summary)')
    parser.add_argument('--summary_dim', default=100, type=int,
                        help='Length of the joint-summary vector (= state-space size). 100 for recovering bandits.')
    parser.add_argument('--max_steps', default=260001, type=int,
                        help='Total training steps (default matches the original paper).')
    parser.add_argument('--reward_log', default=None, type=str,
                        help='Optional path to write per-100-step average rewards as CSV.')


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
    #initialize agent
    hidden = [128, 128]
    # Recovering-bandit state is in {1..maxWait}; map to bin index s-1 in [0, summary_dim).
    state_to_summary_idx = (lambda s: int(s[0]) - 1)
    agent = DeepTOP_RMAB(nb_arms, budget, state_dims, action_dims, state_sizes, action_sizes, hidden, args,
                         state_to_summary_idx=state_to_summary_idx)
    if args.joint_summary:
        total = sum(p.numel() for p in agent.actors[0].parameters())
        print(f'[C1] joint-summary actor parameter count (per arm): {total}')

    resetEnvs()
    agent.reset(states)

    cumulative_reward = 0
    reward_log_rows = []

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)

    iteration = 0
    num_step = 0

    for t in range(args.max_steps):
        if t % 13000 == 0:
            iteration = iteration + 1
            num_step = 0
            print(f'iteration {iteration}')
            agent = DeepTOP_RMAB(nb_arms, budget, state_dims, action_dims, state_sizes, action_sizes, hidden, args,
                                 state_to_summary_idx=state_to_summary_idx)

            resetEnvs()
            agent.reset(states)

        agent.is_training = True
        num_step = num_step + 1

        # agent pick action ...
        if num_step <= args.warmup:
            action = agent.random_action()
        elif random.uniform(0, 1.0) < 0.05:
            action = agent.random_action()
        else:
            action = agent.select_action(states)

        # env response with next_state, reward, terminate_info
        next_state = []
        reward = []
        done = []
        info = []
        for i in range(len(envs)):
            next_state_i, reward_i, done_i, info_i = envs[i].step(action[i])
            next_state.append(next_state_i)
            reward.append(reward_i)
            done.append(done_i)
            info.append(info_i)
        next_state = deepcopy(next_state)

        # agent observe and update policy
        agent.observe(reward, next_state, done)
        if num_step > args.warmup:
            cumulative_reward = cumulative_reward + sum(reward)
            agent.update_policy()
            if( (num_step-args.warmup)%100 == 0 ):
                avg = cumulative_reward/100
                print(f'{avg}')
                reward_log_rows.append((t, float(avg)))
                cumulative_reward = 0
        states = deepcopy(next_state)

    if args.reward_log is not None:
        with open(args.reward_log, 'w') as f:
            f.write('global_step,avg_reward\n')
            for step, avg in reward_log_rows:
                f.write(f'{step},{avg}\n')


