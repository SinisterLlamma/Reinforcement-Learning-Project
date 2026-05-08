

import math
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import Actor
from memory import SequentialMemory, sample_batch_indexes
from random_process import OrnsteinUhlenbeckProcess
from util import *


criterion = nn.MSELoss()


def _fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic_NoLambda(nn.Module):
    """Per-arm Q(s, a). Same architecture as DeepTOP's Critic but without the
    activation-cost λ input — DiffTopV doesn't use the Lagrangian framing."""
    def __init__(self, nb_state, nb_actions, hidden, init_w=5e-1):
        super().__init__()
        self.fc = nn.ModuleList()
        for layer in range(len(hidden) + 1):
            if layer == 0:
                self.fc.append(nn.Linear(nb_state, hidden[0]))
            elif layer == math.floor(len(hidden) / 2):
                self.layer_num_for_action = layer
                self.fc.append(nn.Linear(hidden[layer - 1] + nb_actions, hidden[layer]))
            elif layer == len(hidden):
                self.fc.append(nn.Linear(hidden[layer - 1], 1))
            else:
                self.fc.append(nn.Linear(hidden[layer - 1], hidden[layer]))
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in range(len(self.fc)):
            if layer == len(self.fc) - 1:
                self.fc[layer].weight.data.uniform_(-init_w, init_w)
            else:
                self.fc[layer].weight.data = _fanin_init(self.fc[layer].weight.data.size())

    def forward(self, xs):
        x, a = xs
        out = x
        for layer in range(len(self.fc)):
            if layer == self.layer_num_for_action:
                out = self.fc[layer](torch.cat([out, a], -1))
            else:
                out = self.fc[layer](out)
            if layer < len(self.fc) - 1:
                out = self.relu(out)
        return out


class DiffTopV_RMAB(object):
    def __init__(self, nb_arms, budget, state_dims, action_dims, state_sizes, action_sizes, hidden, args):
        self.nb_arms = nb_arms
        self.budget = budget
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.beta = float(getattr(args, 'beta', 5.0))
        self.bisection_iters = int(getattr(args, 'bisection_iters', 50))
        self.bisection_tol = float(getattr(args, 'bisection_tol', 1e-6))

        self.actors = []
        self.actor_optims = []
        self.critics = []
        self.critic_targets = []
        self.critic_optims = []
        self.memories = []
        self.random_processes = []
        self.s_t = []
        self.a_t = []

        for arm in range(nb_arms):
            self.actors.append(Actor(self.state_dims[arm], 1, hidden))
            self.actor_optims.append(Adam(self.actors[arm].parameters(), lr=args.prate))

            self.critics.append(Critic_NoLambda(self.state_dims[arm], 1, hidden))
            self.critic_targets.append(Critic_NoLambda(self.state_dims[arm], 1, hidden))
            self.critic_optims.append(Adam(self.critics[arm].parameters(), lr=args.rate))

            hard_update(self.critic_targets[arm], self.critics[arm])

            self.memories.append(SequentialMemory(limit=args.rmsize, window_length=args.window_length))
            self.random_processes.append(
                OrnsteinUhlenbeckProcess(size=action_dims[arm], theta=args.ou_theta, mu=args.ou_mu,
                                         sigma=args.ou_sigma))
            self.s_t.append(None)
            self.a_t.append(None)

        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.is_training = True

        if self.device == torch.device('cuda'):
            self.cuda()

    def _compute_implicit_threshold(self, scores, V, beta):
        """Differentiable soft top-V threshold via bisection + Newton refinement.

        Bisection runs under no_grad to get a numerical lam_num, then a single
        Newton step (differentiable in scores) gives the implicit gradient for free.
        """
        with torch.no_grad():
            lo = scores.min(dim=1).values - 10.0 / beta   # [B]
            hi = scores.max(dim=1).values + 10.0 / beta   # [B]
            for _ in range(self.bisection_iters):
                mid = (lo + hi) / 2.0
                residual = torch.sigmoid(beta * (scores - mid.unsqueeze(1))).sum(dim=1) - V
                lo = torch.where(residual > 0, mid, lo)
                hi = torch.where(residual > 0, hi, mid)
            lam_num = (lo + hi) / 2.0  # [B], detached from compute graph

        # Newton refinement: one step at the converged lam_num.
        # lam_num acts as a constant; gradients flow through scores only.
        sig = torch.sigmoid(beta * (scores - lam_num.unsqueeze(1)))        # [B, N]
        sig_prime = sig * (1.0 - sig)                                       # [B, N]
        denom = (beta * sig_prime.sum(dim=1)).clamp(min=1e-8)              # [B]
        lam = lam_num + (V - sig.sum(dim=1)) / denom                       # [B]
        return lam

    def _hard_top_v_mask(self, scores, V):
        """Return [B, N] binary mask with 1 at the top-V scoring arms per row."""
        B, N = scores.shape
        if V >= N:
            return torch.ones(B, N, device=scores.device)
        _, indices = torch.topk(scores, V, dim=1)
        mask = torch.zeros(B, N, device=scores.device)
        mask.scatter_(1, indices, 1.0)
        return mask

    def update_policy(self):
        # One set of indices shared across all arms (synchronized sampling)
        batch_idxs = sample_batch_indexes(0, self.memories[0].nb_entries - 1, size=self.batch_size)

        state_batches = []
        action_batches = []
        reward_batches = []
        next_state_batches = []

        for arm in range(self.nb_arms):
            s, a, r, s1, _ = self.memories[arm].sample_and_split(self.batch_size, batch_idxs=batch_idxs)
            state_batches.append(torch.FloatTensor(s).to(self.device))
            action_batches.append(torch.FloatTensor(a).to(self.device))
            reward_batches.append(torch.FloatTensor(r).to(self.device))
            next_state_batches.append(torch.FloatTensor(s1).to(self.device))

        ones = torch.ones(self.batch_size, 1, device=self.device)
        zeros = torch.zeros(self.batch_size, 1, device=self.device)

        # Joint next-action mask: same across arms, compute once.
        with torch.no_grad():
            next_scores = torch.cat(
                [self.actors[j](next_state_batches[j]) for j in range(self.nb_arms)], dim=1
            )  # [B, N]
            next_a_mask = self._hard_top_v_mask(next_scores, self.budget)  # [B, N]

        # Critic update: per-arm bootstrap reuses the shared next_a_mask
        for arm in range(self.nb_arms):
            with torch.no_grad():
                next_a_arm = next_a_mask[:, arm:arm + 1]                       # [B, 1]
                target_q = reward_batches[arm] + self.discount * \
                           self.critic_targets[arm]([next_state_batches[arm], next_a_arm])

            pred_q = self.critics[arm]([state_batches[arm], action_batches[arm]])
            critic_loss = criterion(pred_q, target_q)
            self.critics[arm].zero_grad()
            critic_loss.backward()
            self.critic_optims[arm].step()
            soft_update(self.critic_targets[arm], self.critics[arm], self.tau)

        # Actor update: joint objective coupled through soft top-V projection
        scores = torch.cat(
            [self.actors[i](state_batches[i]) for i in range(self.nb_arms)], dim=1
        )  # [B, N]
        lam = self._compute_implicit_threshold(scores, self.budget, self.beta)  # [B]
        probs = torch.sigmoid(self.beta * (scores - lam.unsqueeze(1)))          # [B, N]

        with torch.no_grad():
            advantages = torch.cat([
                self.critics[i]([state_batches[i], ones]) - self.critics[i]([state_batches[i], zeros])
                for i in range(self.nb_arms)
            ], dim=1)  # [B, N]

        actor_loss = -(probs * advantages).sum(dim=1).mean()
        for arm in range(self.nb_arms):
            self.actor_optims[arm].zero_grad()
        actor_loss.backward()
        for arm in range(self.nb_arms):
            self.actor_optims[arm].step()

    def eval(self):
        for arm in range(self.nb_arms):
            self.actors[arm].eval()
            self.critics[arm].eval()
            self.critic_targets[arm].eval()

    def cuda(self):
        torch.cuda.set_device(1)
        for arm in range(self.nb_arms):
            self.actors[arm].cuda()
            self.critics[arm].cuda()
            self.critic_targets[arm].cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            for arm in range(self.nb_arms):
                self.memories[arm].append(self.s_t[arm], self.a_t[arm], r_t[arm], done[arm])
                self.s_t[arm] = s_t1[arm]

    def random_action(self):
        indices = []
        for arm in range(self.nb_arms):
            indices.append(np.random.uniform(-10., 10.))
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)
        sort_indices.append(-2)  # sentinel for the budget = nb_arms edge case
        actions = []
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
                self.a_t[arm] = 1
            else:
                actions.append(0)
                self.a_t[arm] = 0
        return actions

    def select_action(self, s_t, decay_epsilon=True):
        indices = []
        for arm in range(self.nb_arms):
            s_tensor = torch.FloatTensor(self.s_t[arm]).to(self.device)
            idx_val = self.actors[arm].forward(s_tensor).cpu().detach().numpy()[0]
            indices.append(idx_val)
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)
        sort_indices.append(
            sort_indices[self.nb_arms - 1] - 2)  # sentinel always below all scores
        actions = []
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
                self.a_t[arm] = 1
            else:
                actions.append(0)
                self.a_t[arm] = 0
        if decay_epsilon:
            self.epsilon -= self.depsilon
        return actions

    def reset(self, obs):
        self.s_t = obs
        for arm in range(self.nb_arms):
            self.random_processes[arm].reset_states()
