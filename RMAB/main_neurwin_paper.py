
import os
import torch
import random
import argparse
from copy import deepcopy
import itertools
import numpy as np
from model import Actor
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
from lineEnv import lineEnv
from neurwin_test import NeurWIN_Wrapper


def initializeEnv():
    global envs, state_dims, action_dims, state_sizes, action_sizes, nb_arms, p, q, args
    for i in range(nb_arms):
        envs.append(lineEnv(seed=(i*args.seed)+2357, N=100, OptX=99, p=p[i], q=q[i]))
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

    parser = argparse.ArgumentParser(description='NeurWIN paper-config eval (CSV-logging variant)')

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
    parser.add_argument('--validate_episodes', default=20, type=int)
    parser.add_argument('--max_episode_length', default=500, type=int)
    parser.add_argument('--validate_steps', default=2000, type=int)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float)
    parser.add_argument('--train_iter', default=200000, type=int)
    parser.add_argument('--epsilon', default=50000, type=int)
    parser.add_argument('--seed', default=87452, type=int)
    parser.add_argument('--resume', default='default', type=str)
    parser.add_argument('--nb_arms', default=0, type=int)
    parser.add_argument('--budget', default=0, type=int)
    parser.add_argument('--max_steps', default=12000, type=int,
                        help='Total eval steps (matches the DeepTOP/DiffTopV paper-config protocol).')
    parser.add_argument('--reward_log', default=None, type=str,
                        help='Path to write per-100-step average rewards as CSV.')


    args = parser.parse_args()

    directory = (f'neurwin_training_results/arms_{args.nb_arms}_activate_{args.budget}/')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    p = np.linspace(start=0.2, stop=0.8, num=args.nb_arms)
    q = p

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

    EPISODE_LENGTH = 300
    BATCH_SIZE = 5

    agent = NeurWIN_Wrapper(nb_arms, budget, state_dims, action_dims, state_sizes, action_sizes, hidden, args)
    resetEnvs()
    agent.get_agents(directory=directory, episode=0)

    cumulative_reward = 0
    reward_log_rows = []

    iteration = 1
    num_step = 0
    print(f'iteration {iteration}')

    for t in range(args.max_steps):
        agent.is_training = True
        num_step = num_step + 1

        if num_step <= args.warmup:
            action = agent.random_action()
        elif random.uniform(0, 1.0) < 0.05:
            action = agent.random_action()
        else:
            action = agent.select_action(states)

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

        if num_step > args.warmup:
            cumulative_reward = cumulative_reward + sum(reward)

            if ((num_step - args.warmup) % 100 == 0):
                avg = cumulative_reward / 100
                print(f'{avg}')
                reward_log_rows.append((t, float(avg)))
                cumulative_reward = 0

            if (int((num_step - args.warmup) % (EPISODE_LENGTH * BATCH_SIZE)) == 0):
                episode = int((num_step - args.warmup) / EPISODE_LENGTH)
                agent.get_agents(directory=directory, episode=episode)

        states = deepcopy(next_state)

    if args.reward_log is not None:
        with open(args.reward_log, 'w') as f:
            f.write('global_step,avg_reward\n')
            for step, avg in reward_log_rows:
                f.write(f'{step},{avg}\n')
