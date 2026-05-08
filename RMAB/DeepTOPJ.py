"""DeepTOP-J: Joint-Actor DeepTOP for RMABs.

Replaces the N independent per-arm actors of DeepTOP-RMAB with a single
permutation-equivariant joint actor (Set Transformer) that takes the full
joint state as input and outputs N per-arm thresholds. Per-arm critics are
unchanged and identical to those used by DeepTOP_RMAB.

The actor gradient follows directly from applying the threshold-policy-
gradient theorem (Theorem 2 of Nakhleh & Hou, NeurIPS 2022) per-arm
conditional on the joint state, then summing over arms because a single phi
is shared:

    grad_phi K^J = E_s [ sum_i [Q_i(s_i, 1; lam=mu_i(s)) - Q_i(s_i, 0; lam=mu_i(s))]
                         * grad_phi mu_i(s) ]

Critic update is identical to DeepTOP-RMAB (alternative-problem net reward
r_i - lam * a_i + bootstrap from per-arm target net at sampled lam).
"""

from collections import deque
import random as pyrandom

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from model import Critic
from util import hard_update, soft_update, to_tensor


criterion = nn.MSELoss()


class JointActor(nn.Module):
    """Permutation-equivariant joint actor (Set Transformer-style).

    Input  : tokens of shape (B, N, token_dim) or (N, token_dim).
    Output : per-arm thresholds of shape (B, N) or (N,).

    Each token = (per-arm state, per-arm metadata). Self-attention couples
    arms together; the linear head projects each token down to a scalar
    threshold. Permutation equivariance is preserved because attention is
    symmetric over tokens.
    """

    def __init__(self, token_dim, hidden_dim=128, n_heads=4, n_layers=2,
                 gated=False):
        """If `gated=True`, each attention block has a learnable scalar
        alpha (init 0) gating its residual: x = LN(x + alpha * attn(x)).
        At init the actor is exactly a token-wise MLP (per-arm scoring);
        the joint pathway is opened only if gradients say it helps.
        """
        super().__init__()
        self.gated = bool(gated)
        self.embed = nn.Linear(token_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        self.gates = nn.ParameterList() if self.gated else None
        for _ in range(n_layers):
            blk = nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True),
                'ln1':  nn.LayerNorm(hidden_dim),
                'ff':   nn.Sequential(
                            nn.Linear(hidden_dim, 2 * hidden_dim),
                            nn.ReLU(),
                            nn.Linear(2 * hidden_dim, hidden_dim),
                        ),
                'ln2':  nn.LayerNorm(hidden_dim),
            })
            self.blocks.append(blk)
            if self.gated:
                self.gates.append(nn.Parameter(torch.zeros(1)))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, tokens):
        squeeze = (tokens.dim() == 2)
        if squeeze:
            tokens = tokens.unsqueeze(0)
        x = self.embed(tokens)
        for i, blk in enumerate(self.blocks):
            a, _ = blk['attn'](x, x, x, need_weights=False)
            if self.gated:
                a = self.gates[i] * a
            x = blk['ln1'](x + a)
            f = blk['ff'](x)
            x = blk['ln2'](x + f)
        out = self.head(x).squeeze(-1)
        return out.squeeze(0) if squeeze else out


class JointReplayBuffer:
    """Stores joint (state, action, reward, next_state) tuples per timestep.

    Sampling preserves joint correlations: each minibatch row is a single
    timestep with all N arms together.
    """

    def __init__(self, capacity, N, state_dim):
        self.buffer = deque(maxlen=capacity)
        self.N = N
        self.state_dim = state_dim

    def push(self, joint_state, joint_action, joint_reward, joint_next_state):
        self.buffer.append((
            np.asarray(joint_state, dtype=np.float32).copy(),
            np.asarray(joint_action, dtype=np.int64).copy(),
            np.asarray(joint_reward, dtype=np.float32).copy(),
            np.asarray(joint_next_state, dtype=np.float32).copy(),
        ))

    def sample(self, batch_size):
        batch = pyrandom.sample(self.buffer, batch_size)
        s, a, r, sn = zip(*batch)
        return (
            torch.from_numpy(np.stack(s)),    # (B, N, d)
            torch.from_numpy(np.stack(a)),    # (B, N)
            torch.from_numpy(np.stack(r)),    # (B, N)
            torch.from_numpy(np.stack(sn)),   # (B, N, d)
        )

    def __len__(self):
        return len(self.buffer)


class DeepTOPJ_RMAB(object):
    """Joint-actor DeepTOP. Mirrors the interface of DeepTOP_RMAB so the
    main loop need not change."""

    def __init__(self, nb_arms, budget, state_dims, action_dims,
                 state_sizes, action_sizes, hidden, args, arm_metadata=None):
        self.nb_arms = nb_arms
        self.budget = budget
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # All arms here have the same per-arm state dim (= state_dims[0]).
        self.state_dim = state_dims[0]

        # arm_metadata: shape (N, meta_dim). Required for heterogeneous arms
        # so the permutation-equivariant joint actor can distinguish them.
        if arm_metadata is None:
            arm_metadata = np.zeros((nb_arms, 1), dtype=np.float32)
        arm_metadata = np.asarray(arm_metadata, dtype=np.float32)
        if arm_metadata.ndim == 1:
            arm_metadata = arm_metadata.reshape(-1, 1)
        assert arm_metadata.shape[0] == nb_arms
        self.meta_dim = arm_metadata.shape[1]
        self.arm_metadata = torch.from_numpy(arm_metadata).to(self.device)

        token_dim = self.state_dim + self.meta_dim

        # Joint actor (single shared phi). With actor_gated=True the joint
        # pathway is closed at init (per-arm scoring) and only opens if it
        # earns gradient signal -- regularisation against gratuitous coupling.
        self.actor = JointActor(
            token_dim=token_dim,
            hidden_dim=int(getattr(args, 'actor_hidden', 128)),
            n_heads=int(getattr(args, 'actor_heads', 4)),
            n_layers=int(getattr(args, 'actor_layers', 2)),
            gated=bool(getattr(args, 'actor_gated', False)),
        ).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        # Per-arm advantage normalisation (running std) to balance gradient
        # magnitudes across heterogeneous arms when summed into shared phi.
        self.adv_normalize = bool(getattr(args, 'adv_normalize', False))
        if self.adv_normalize:
            self._adv_running_var = torch.ones(nb_arms, device=self.device)
            self._adv_running_decay = 0.99

        # Per-arm critics (identical to DeepTOP-RMAB).
        self.critics = []
        self.critic_targets = []
        self.critic_optims = []
        for arm in range(nb_arms):
            c = Critic(self.state_dims[arm] + 1, 1, hidden).to(self.device)
            ct = Critic(self.state_dims[arm] + 1, 1, hidden).to(self.device)
            hard_update(ct, c)
            self.critics.append(c)
            self.critic_targets.append(ct)
            self.critic_optims.append(Adam(c.parameters(), lr=args.rate))

        # Joint replay buffer
        self.buffer = JointReplayBuffer(args.rmsize, nb_arms, self.state_dim)

        # Hyperparameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.M = float(getattr(args, 'M', 1.0))

        self.epsilon = 1.0
        self.depsilon = 1.0 / args.epsilon
        self.is_training = True

        # Per-arm bookkeeping mirroring DeepTOP_RMAB so observe() works the same way.
        self.s_t = [None] * nb_arms
        self.a_t = [None] * nb_arms

    # ---------- helpers ----------

    def _stack_joint_state(self, states):
        """Per-arm states (list/np array of length-state_dim arrays) -> (N, d) tensor."""
        arr = np.asarray([np.asarray(s, dtype=np.float32).reshape(-1) for s in states],
                         dtype=np.float32)
        return torch.from_numpy(arr).to(self.device)

    def _make_tokens(self, joint_state):
        if joint_state.dim() == 2:
            return torch.cat([joint_state, self.arm_metadata], dim=-1)
        B = joint_state.shape[0]
        meta = self.arm_metadata.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([joint_state, meta], dim=-1)

    # ---------- main interface ----------

    def random_action(self):
        indices = [np.random.uniform(-1., 1.) for _ in range(self.nb_arms)]
        sort_indices = sorted(indices, reverse=True)
        sort_indices.append(-2)  # sentinel for budget == nb_arms
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
        with torch.no_grad():
            joint = self._stack_joint_state(self.s_t)
            tokens = self._make_tokens(joint)
            indices = self.actor(tokens).cpu().numpy().tolist()

        sort_indices = sorted(indices, reverse=True)
        # sentinel below the smallest score so budget == nb_arms case works
        sort_indices.append(sort_indices[-1] - 2)
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

    def observe(self, r_t, s_t1, done):
        if not self.is_training:
            return
        # Snapshot the joint transition into the buffer, then advance per-arm s_t.
        joint_s = np.asarray([np.asarray(s, dtype=np.float32).reshape(-1)
                              for s in self.s_t], dtype=np.float32)
        joint_sn = np.asarray([np.asarray(s, dtype=np.float32).reshape(-1)
                               for s in s_t1], dtype=np.float32)
        joint_a = np.asarray(self.a_t, dtype=np.int64)
        joint_r = np.asarray(r_t, dtype=np.float32)
        self.buffer.push(joint_s, joint_a, joint_r, joint_sn)
        for arm in range(self.nb_arms):
            self.s_t[arm] = s_t1[arm]

    def update_policy(self):
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, s_next = self.buffer.sample(self.batch_size)
        s = s.to(self.device)
        a = a.to(self.device).float()
        r = r.to(self.device)
        s_next = s_next.to(self.device)
        B = s.shape[0]

        # ----- per-arm critic update (same as DeepTOP-RMAB) -----
        for arm in range(self.nb_arms):
            si = s[:, arm]                             # (B, d)
            ai = a[:, arm].unsqueeze(-1)               # (B, 1)
            ri = r[:, arm].unsqueeze(-1)               # (B, 1)
            si_next = s_next[:, arm]                   # (B, d)

            # lam ~ Uniform(-M, M)
            lam = torch.empty(B, 1, device=self.device).uniform_(-self.M, self.M)

            with torch.no_grad():
                ones = torch.ones(B, 1, device=self.device)
                zeros = torch.zeros(B, 1, device=self.device)
                q_plus  = self.critic_targets[arm]([si_next, lam, ones])
                q_minus = self.critic_targets[arm]([si_next, lam, zeros])
                next_a = torch.clamp(torch.sign(q_plus - q_minus), min=0.0)
                next_q = self.critic_targets[arm]([si_next, lam, next_a])
                net_r = ri - lam * ai
                target_q = net_r + self.discount * next_q

            self.critics[arm].zero_grad()
            q_pred = self.critics[arm]([si, lam, ai])
            value_loss = criterion(q_pred, target_q)
            value_loss.backward()
            self.critic_optims[arm].step()

        # ----- joint actor update -----
        tokens = self._make_tokens(s)                 # (B, N, token_dim)
        thresholds = self.actor(tokens)               # (B, N)

        # Per-arm advantage Q(s_i,1;mu_i) - Q(s_i,0;mu_i) is treated as a scalar
        # multiplier on grad mu_i (detach from critic graph, exactly like DeepTOP_RMAB).
        ones = torch.ones(B, 1, device=self.device)
        zeros = torch.zeros(B, 1, device=self.device)
        advantages = []
        for arm in range(self.nb_arms):
            si = s[:, arm]
            mu_i = thresholds[:, arm:arm + 1]
            with torch.no_grad():
                q_plus  = self.critics[arm]([si, mu_i, ones])
                q_minus = self.critics[arm]([si, mu_i, zeros])
                advantages.append((q_plus - q_minus).detach())
        adv = torch.cat(advantages, dim=1)            # (B, N)

        # Optional: per-arm advantage normalisation by running std so high-
        # reward arms don't dominate the shared-phi gradient.
        if self.adv_normalize:
            with torch.no_grad():
                batch_var = adv.var(dim=0).clamp(min=1e-8)             # (N,)
                self._adv_running_var.mul_(self._adv_running_decay).add_(
                    (1.0 - self._adv_running_decay) * batch_var)
                scale = self._adv_running_var.sqrt().unsqueeze(0)      # (1, N)
            adv = adv / scale

        # K^J ascend -> minimize -sum_i adv_i * mu_i, mean over batch.
        actor_loss = -(adv * thresholds).sum(dim=1).mean()

        self.actor.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update of per-arm critic targets
        for arm in range(self.nb_arms):
            soft_update(self.critic_targets[arm], self.critics[arm], self.tau)

    def eval(self):
        self.actor.eval()
        for c in self.critics:
            c.eval()
        for ct in self.critic_targets:
            ct.eval()

    def cuda(self):
        # Already moved during __init__. Kept for interface parity.
        pass

    def reset(self, obs):
        self.s_t = list(obs)
