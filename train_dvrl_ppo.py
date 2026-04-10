"""dvrl_obelix.py — Deep Variational Reinforcement Learning for OBELIX.

Implements the paper:
    Igl et al. 2018 — "Deep Variational Reinforcement Learning for POMDPs"
    arXiv:1806.02426

Architecture (Algorithm 1 from the paper, adapted for OBELIX 18-dim obs):
    obs_t (18) + action_t-1 (5)
        ↓  phi_o / phi_a  (obs/action encoders)
        ↓
    Particle filter belief update  (K particles)
        For each particle k:
            1. Resample ancestor from p(u^k | w^1:K_{t-1})
            2. Sample z^k_t ~ q_phi(z | h^k_{t-1}, phi_o, phi_a)    [encoder]
            3. h^k_t = GRU(h^k_{t-1}, z^k_t, phi_o, phi_a)           [transition]
            4. w^k_t = p(z^k_t|h^k,a) * p(o_t|h^k,z^k,a)             [weight]
                       ─────────────────────────────────────
                       q_phi(z^k_t|h^k,obs,a)
        5. Aggregate: h_hat_t = AggGRU([(z^k,h^k,w^k) for k])
        ↓
    Actor π(a | h_hat_t)  +  Critic V(h_hat_t)
        ↓
    Loss = L_rl + lambda_E * L_elbo
         = PPO_loss + lambda_E * (- mean_t log (1/K * sum_k w^k_t))

Key design choices for OBELIX:
    - K=8 particles (paper uses 15-30; 8 is fast on 18-dim input)
    - GRU instead of LSTM (paper uses GRU throughout)
    - Bernoulli decoder for binary sonar bits (obs[0:16])
      Normal decoder for continuous-ish IR and stuck bits (obs[16:18])
    - lambda_E=1.0 (paper recommends this for low-dimensional observations)
    - Particle states (h,z,w) are maintained per-worker in the main process
    - Compatible with VecEnv and existing PPO rollout buffer pattern

Usage:
    python dvrl_obelix.py --obelix_py ./obelix.py --n_envs 8 --K 8
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli, Categorical, Normal
from tqdm import tqdm

from vec_env import VecEnv


# =============================================================================
# Constants
# =============================================================================
ACTIONS  = ["L45", "L22", "FW", "R22", "R45"]
N_ACT    = len(ACTIONS)
OBS_DIM  = 18
OBS_CONT = 2    # obs[16], obs[17] decoded with Normal
OBS_BIN  = 16   # obs[0:16] decoded with Bernoulli


def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda"); print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps");  print("[Device] Apple MPS")
    else:
        d = torch.device("cpu");  print("[Device] CPU")
    return d

DEVICE = get_device()


# =============================================================================
# Encoders  (phi_o, phi_a, phi_z — paper Section 3.3 and Appendix A)
# =============================================================================
class ObsEncoder(nn.Module):
    """phi_o: 18-dim obs → enc_o_dim feature vector."""
    def __init__(self, obs_dim: int = OBS_DIM, enc_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, enc_dim), nn.ReLU(),
        )
    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.net(o)

class ActionEncoder(nn.Module):
    """phi_a: one-hot action → enc_a_dim feature vector."""
    def __init__(self, n_actions: int = N_ACT, enc_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_actions, enc_dim), nn.ReLU(),
        )
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)

class ZEncoder(nn.Module):
    """phi_z: latent z → enc_z_dim feature vector."""
    def __init__(self, z_dim: int, enc_dim: int):
        super().__init__()
        self.net = nn.Linear(z_dim, enc_dim)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(z))


# =============================================================================
# DVRL Generative Model  (paper Equations 14-18)
# =============================================================================
class DVRLGenerativeModel(nn.Module):
    """
    Implements the per-particle computation from Algorithm 1:

        Prior:    p_theta(z_t | h_{t-1}, a_{t-1})   — stochastic transition
        Decoder:  p_theta(o_t | h_{t-1}, z_t, a_{t-1})
        Posterior: q_phi(z_t | h_{t-1}, o_t, a_{t-1})  — encoder / proposal
        Transition: h_t = GRU(h_{t-1}, z_t, o_t, a_{t-1})

    For OBELIX:
        - h_dim: GRU hidden state per particle
        - z_dim: stochastic latent variable per particle
        - obs decoded as Bernoulli (sonar bits) + Normal (IR, stuck)
    """

    def __init__(
        self,
        h_dim:    int = 64,
        z_dim:    int = 32,
        enc_o:    int = 32,
        enc_a:    int = 16,
        enc_z:    int = 32,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        # ── Proposal / encoder: q_phi(z_t | h_{t-1}, enc_o, enc_a) ────────
        # Outputs mean + log_var of Normal distribution over z
        self._q_net = nn.Sequential(
            nn.Linear(h_dim + enc_o + enc_a, 64), nn.ReLU(),
        )
        self._q_mean    = nn.Linear(64, z_dim)
        self._q_log_var = nn.Linear(64, z_dim)

        # ── Prior: p_theta(z_t | h_{t-1}, enc_a) ───────────────────────────
        self._p_net = nn.Sequential(
            nn.Linear(h_dim + enc_a, 64), nn.ReLU(),
        )
        self._p_mean    = nn.Linear(64, z_dim)
        self._p_log_var = nn.Linear(64, z_dim)

        # ── Transition: h_t = GRU(h_{t-1}, [z_t, enc_o, enc_a]) ─────────
        self._gru = nn.GRUCell(
            input_size=enc_z + enc_o + enc_a,
            hidden_size=h_dim,
        )

        # ── Decoder: p_theta(o_t | h_{t-1}, z_t, enc_a) ─────────────────
        # Separate heads: binary sonar (Bernoulli) + IR/stuck (Normal)
        self._dec_net = nn.Sequential(
            nn.Linear(h_dim + enc_z + enc_a, 64), nn.ReLU(),
        )
        self._dec_bin_logits = nn.Linear(64, OBS_BIN)   # Bernoulli logits
        self._dec_cont_mean  = nn.Linear(64, OBS_CONT)  # Normal mean
        self._dec_cont_lv    = nn.Linear(64, OBS_CONT)  # Normal log_var

    def forward(
        self,
        h_prev:  torch.Tensor,   # (batch, h_dim)
        enc_o:   torch.Tensor,   # (batch, enc_o)  — encoded current obs
        enc_a:   torch.Tensor,   # (batch, enc_a)  — encoded prev action
        obs_raw: torch.Tensor,   # (batch, OBS_DIM) — raw obs for decoder target
        phi_z:   nn.Module,      # ZEncoder module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One belief update step for a single particle batch.

        Returns
        -------
        h_new     : (batch, h_dim)   — updated GRU hidden state
        z         : (batch, z_dim)   — sampled stochastic latent
        log_weight: (batch,)         — log importance weight log(w_t)
                    = log p(z|h,a) + log p(o|h,z,a) - log q(z|h,o,a)
        """
        # ── 1. Sample z from proposal q_phi ─────────────────────────────
        q_in   = torch.cat([h_prev, enc_o, enc_a], dim=-1)
        q_feat = self._q_net(q_in)
        q_mean = self._q_mean(q_feat)
        q_lv   = self._q_log_var(q_feat).clamp(-4, 4)
        q_std  = (0.5 * q_lv).exp()

        # Reparameterisation trick (differentiable sample)
        # Reparameterisation trick — Eq.16: sample z differentiably
        eps    = torch.randn_like(q_mean)
        z      = q_mean + eps * q_std

        # ── 2. Evaluate prior p_theta(z | h, a) ─────────────────────────
        p_in   = torch.cat([h_prev, enc_a], dim=-1)
        p_feat = self._p_net(p_in)
        p_mean = self._p_mean(p_feat)
        p_lv   = self._p_log_var(p_feat).clamp(-4, 4)

        log_p_z = Normal(p_mean, (0.5 * p_lv).exp()).log_prob(z).sum(-1)
        log_q_z = Normal(q_mean, q_std).log_prob(z).sum(-1)

        # ── 3. Update h via GRU ──────────────────────────────────────────
        enc_z_feat = phi_z(z)
        gru_in     = torch.cat([enc_z_feat, enc_o, enc_a], dim=-1)
        h_new      = self._gru(gru_in, h_prev)

        # ── 4. Decode observation p_theta(o | h, z, a) ──────────────────
        dec_in    = torch.cat([h_prev, enc_z_feat, enc_a], dim=-1)
        dec_feat  = self._dec_net(dec_in)

        # Binary sonar bits
        bin_logits = self._dec_bin_logits(dec_feat)
        log_p_bin  = Bernoulli(logits=bin_logits).log_prob(
            obs_raw[:, :OBS_BIN]
        ).sum(-1)

        # Continuous bits (IR sensor, stuck flag — treated as soft Bernoulli)
        cont_mean = torch.sigmoid(self._dec_cont_mean(dec_feat))
        cont_lv   = self._dec_cont_lv(dec_feat).clamp(-4, 4)
        log_p_cont = Normal(cont_mean, (0.5 * cont_lv).exp()).log_prob(
            obs_raw[:, OBS_BIN:]
        ).sum(-1)

        log_p_o = log_p_bin + log_p_cont

        # ── 5. Importance weight: log w_t = log p(z) + log p(o) - log q(z)
        log_weight = log_p_z + log_p_o - log_q_z

        return h_new, z, log_weight


# =============================================================================
# DVRL Encoder  (full belief update — K particles, aggregator GRU)
# =============================================================================
class DVRLEncoder(nn.Module):
    """
    Full DVRL encoder: maintains K particles and produces a belief summary h_hat_t
    for the actor and critic.

    Particle state per worker: (h_k, z_k, w_k) for k=1..K
    This is stored OUTSIDE the network (in main()) and passed in each step.

    The belief summary h_hat_t is produced by a second GRU that sequentially
    processes all K particles (as in the paper, Section 3.3).
    """

    def __init__(
        self,
        K:        int = 8,       # number of particles
        h_dim:    int = 64,      # GRU hidden dim per particle
        z_dim:    int = 32,      # stochastic latent dim
        enc_o:    int = 32,      # obs encoding dim
        enc_a:    int = 16,      # action encoding dim
        enc_z:    int = 32,      # z encoding dim
        agg_dim:  int = 128,     # aggregator GRU output = belief summary dim
    ):
        super().__init__()
        self.K       = K
        self.h_dim   = h_dim
        self.z_dim   = z_dim
        self.agg_dim = agg_dim

        # Input encoders
        self.phi_o = ObsEncoder(OBS_DIM, enc_o)
        self.phi_a = ActionEncoder(N_ACT,  enc_a)
        self.phi_z = ZEncoder(z_dim, enc_z)

        # Per-particle generative model
        self.gen = DVRLGenerativeModel(h_dim, z_dim, enc_o, enc_a, enc_z)

        # Aggregator GRU: processes K particles sequentially → h_hat_t
        # Input per particle: (enc_z, h_k, w_k) = (enc_z + h_dim + 1)
        self.agg_gru = nn.GRUCell(
            input_size  = enc_z + h_dim + 1,
            hidden_size = agg_dim,
        )

    def init_particles(
        self, batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialise particle state for `batch` independent workers.

        Returns
        -------
        h_particles   : (batch, K, h_dim)  — GRU hidden states
        z_particles   : (batch, K, z_dim)  — stochastic latents
        log_w_particles: (batch, K)        — log importance weights (uniform)
        h_hat          : (batch, agg_dim)  — aggregated belief summary
        """
        h    = torch.zeros(batch, self.K, self.h_dim,   device=device)
        z    = torch.zeros(batch, self.K, self.z_dim,   device=device)
        lw   = torch.zeros(batch, self.K,               device=device)  # log uniform
        hhat = torch.zeros(batch, self.agg_dim,         device=device)
        return h, z, lw, hhat

    def forward(
        self,
        obs_raw:     torch.Tensor,    # (batch, OBS_DIM)
        prev_action: torch.Tensor,    # (batch,)  int64 action index at t-1
        h_particles: torch.Tensor,    # (batch, K, h_dim)
        z_particles: torch.Tensor,    # (batch, K, z_dim)
        log_w:       torch.Tensor,    # (batch, K)
        h_hat_prev:  torch.Tensor,    # (batch, agg_dim)
    ) -> Tuple[
        torch.Tensor,  # h_hat_new  (batch, agg_dim) — belief summary for actor/critic
        torch.Tensor,  # h_new      (batch, K, h_dim) — updated particle h
        torch.Tensor,  # z_new      (batch, K, z_dim) — updated particle z
        torch.Tensor,  # log_w_new  (batch, K)        — updated log weights
        torch.Tensor,  # elbo_term  (batch,)          — log(1/K * sum exp(log_w))
    ]:
        batch = obs_raw.shape[0]

        # Encode shared inputs (same for all particles)
        a_onehot = F.one_hot(prev_action, N_ACT).float()   # (batch, N_ACT)
        enc_o    = self.phi_o(obs_raw)                      # (batch, enc_o)
        enc_a    = self.phi_a(a_onehot)                     # (batch, enc_a)

        # ── Resample ancestors based on current weights (Eq. 15) ─────────
        # Normalise log weights → probabilities
        log_w_norm = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        w_norm     = log_w_norm.exp()                        # (batch, K)

        # Sample ancestor indices proportional to weights
        anc_idx = torch.multinomial(w_norm, self.K, replacement=True)  # (batch, K)

        # Gather ancestor particles
        anc_h = h_particles.gather(
            1, anc_idx.unsqueeze(-1).expand(-1, -1, self.h_dim)
        )  # (batch, K, h_dim)

        # ── Per-particle update (Eqs. 16-18) ─────────────────────────────
        h_new_list   = []
        z_new_list   = []
        log_w_list   = []

        # Expand encoded inputs for all particles
        enc_o_exp = enc_o.unsqueeze(1).expand(-1, self.K, -1).reshape(batch * self.K, -1)
        enc_a_exp = enc_a.unsqueeze(1).expand(-1, self.K, -1).reshape(batch * self.K, -1)
        obs_exp   = obs_raw.unsqueeze(1).expand(-1, self.K, -1).reshape(batch * self.K, -1)
        h_anc_exp = anc_h.reshape(batch * self.K, self.h_dim)

        h_new_flat, z_new_flat, log_w_flat = self.gen(
            h_anc_exp, enc_o_exp, enc_a_exp, obs_exp, self.phi_z
        )

        h_new   = h_new_flat.reshape(batch, self.K, self.h_dim)
        z_new   = z_new_flat.reshape(batch, self.K, self.z_dim)
        log_w_new = log_w_flat.reshape(batch, self.K)

        # ── ELBO term: log(1/K * sum_k exp(log_w_k)) (Eq. 19) ───────────
        # = logsumexp(log_w) - log(K)
        elbo_term = torch.logsumexp(log_w_new, dim=1) - np.log(self.K)  # (batch,)

        # ── Aggregate particles into belief summary h_hat (Section 3.3) ──
        # Process K particles sequentially through aggregator GRU
        agg_h = h_hat_prev  # start from previous belief summary
        w_norm_new = torch.softmax(log_w_new, dim=1)
        for k in range(self.K):
            enc_z_k = self.phi_z(z_new[:, k, :])              # (batch, enc_z)
            w_k     = w_norm_new[:, k:k+1]                    # (batch, 1)
            agg_in  = torch.cat([enc_z_k, h_new[:, k, :], w_k], dim=-1)
            agg_h   = self.agg_gru(agg_in, agg_h)

        return agg_h, h_new, z_new, log_w_new, elbo_term


# =============================================================================
# Actor-Critic (conditions on belief summary h_hat_t)
# =============================================================================
class ActorCritic(nn.Module):
    def __init__(self, belief_dim: int, n_actions: int = N_ACT):
        super().__init__()
        self.actor  = nn.Linear(belief_dim, n_actions)
        self.critic = nn.Linear(belief_dim, 1)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01); nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0);  nn.init.zeros_(self.critic.bias)

    def forward(self, h_hat: torch.Tensor):
        return self.actor(h_hat), self.critic(h_hat).squeeze(-1)

    def get_action(self, h_hat, deterministic=False):
        logits, value = self(h_hat)
        dist   = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# =============================================================================
# Full DVRL agent (encoder + actor-critic)
# =============================================================================
class DVRLAgent(nn.Module):
    def __init__(self, K=8, h_dim=64, z_dim=32, agg_dim=128):
        super().__init__()
        self.encoder = DVRLEncoder(K=K, h_dim=h_dim, z_dim=z_dim, agg_dim=agg_dim)
        self.ac       = ActorCritic(belief_dim=agg_dim)

    def init_belief(self, batch: int, device: torch.device):
        return self.encoder.init_particles(batch, device)

    def step(self, obs, prev_action, h_p, z_p, log_w, h_hat):
        """Single-step inference. Returns action + updated particle state."""
        h_hat_new, h_p_new, z_p_new, log_w_new, elbo = self.encoder(
            obs, prev_action, h_p, z_p, log_w, h_hat
        )
        action, log_prob, entropy, value = self.ac.get_action(h_hat_new)
        return action, log_prob, entropy, value, h_hat_new, h_p_new, z_p_new, log_w_new, elbo


# =============================================================================
# Rollout buffer (stores belief summary h_hat per step — no BPTT needed here
# because BPTT flows through the DVRL encoder's per-step computation graph)
# =============================================================================
class RolloutBuffer:
    def __init__(self, rollout_len, n_envs, belief_dim):
        self.T, self.N = rollout_len, n_envs
        self.belief_dim = belief_dim
        self.clear()

    def clear(self):
        self.obs      = []   # raw obs_t per step
        self.prev_actions = []
        self.actions  = []; self.log_probs = []
        self.rewards  = []; self.values    = []
        self.dones    = []
        self.init_state = None

    def store_init_state(self, h_p, z_p, log_w, h_hat):
        self.init_state = (
            h_p.detach().clone(),
            z_p.detach().clone(),
            log_w.detach().clone(),
            h_hat.detach().clone(),
        )

    def add(self, obs, prev_actions, actions, log_probs, rewards, values, dones):
        for i in range(self.N):
            self.obs      .append(obs[i])
            self.prev_actions.append(prev_actions[i])
            self.actions  .append(actions[i])
            self.log_probs.append(log_probs[i])
            self.rewards  .append(rewards[i])
            self.values   .append(values[i])
            self.dones    .append(dones[i])

    def _t(self): return len(self.rewards) // self.N

    def compute_gae(self, last_vals, gamma=0.99, lam=0.95):
        T, N = self._t(), self.N; n = T * N
        r = np.array(self.rewards[:n], np.float32).reshape(T, N)
        v = np.array(self.values [:n], np.float32).reshape(T, N)
        d = np.array(self.dones  [:n], np.float32).reshape(T, N)
        A = np.zeros_like(r); g = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = last_vals if t == T-1 else v[t+1]
            delta = r[t] + gamma * nv * (1-d[t]) - v[t]
            g = delta + gamma * lam * (1-d[t]) * g; A[t] = g
        return torch.from_numpy(A.reshape(-1)), torch.from_numpy((A+v).reshape(-1))

    def tensors(self, device):
        n = self._t() * self.N
        obs     = torch.tensor(np.array(self.obs[:n]), dtype=torch.float32, device=device)
        prev_a  = torch.tensor(np.array(self.prev_actions[:n]), dtype=torch.long, device=device)
        acts    = torch.tensor(np.array(self.actions[:n]), dtype=torch.long, device=device)
        oldlogp = torch.tensor(np.array(self.log_probs[:n]), dtype=torch.float32, device=device)
        return obs, prev_a, acts, oldlogp


# =============================================================================
# PPO + ELBO update  (Equation 20 from paper)
# =============================================================================
def ppo_elbo_update(
    agent, opt, buf, last_vals, device,
    gamma=0.99, lam=0.95, clip_eps=0.2,
    vf_coef=0.5, ent_coef=0.002, lambda_E=1.0,
    n_epochs=4, mini_batch=64, max_grad=0.5,
):
    """
    Joint PPO + ELBO loss from Equation 20:
        L_DVRL = L_A (policy) + lambda_H * L_H (entropy)
               + lambda_V * L_V (value)
               + lambda_E * L_ELBO

    The ELBO term L_ELBO = -mean_t [ log(1/K * sum_k w^k_t) ]
    is already computed during rollout collection as `elbos` in the buffer.
    We just need to include it in the loss here.
    """
    del mini_batch  # Full-sequence recomputation is required so encoder gradients flow through time.

    advantages, returns = buf.compute_gae(last_vals, gamma, lam)
    adv_std = advantages.std(unbiased=False)
    if not torch.isfinite(adv_std) or adv_std < 1e-8:
        adv_std = torch.tensor(1.0)
    advantages = ((advantages - advantages.mean()) / (adv_std + 1e-8)).to(device)
    returns    = returns.to(device)

    obs, prev_actions, actions, old_logp = buf.tensors(device)
    T = buf._t()
    N = buf.N
    obs_seq = obs.reshape(T, N, OBS_DIM)
    prev_action_seq = prev_actions.reshape(T, N)
    metrics = collections.defaultdict(list)
    init_h_p, init_z_p, init_log_w, init_h_hat = buf.init_state

    for _ in range(n_epochs):
        h_p   = init_h_p.to(device)
        z_p   = init_z_p.to(device)
        log_w = init_log_w.to(device)
        h_hat = init_h_hat.to(device)

        logits_seq = []
        values_seq = []
        elbo_seq   = []

        for t in range(T):
            h_hat, h_p, z_p, log_w, elbo = agent.encoder(
                obs_seq[t], prev_action_seq[t], h_p, z_p, log_w, h_hat
            )
            logits_t, values_t = agent.ac(h_hat)
            logits_seq.append(logits_t)
            values_seq.append(values_t)
            elbo_seq.append(elbo)

            if t < T - 1:
                mask = (1.0 - torch.tensor(np.asarray(buf.dones).reshape(T, N)[t], dtype=torch.float32, device=device)).view(N, 1)
                h_p = h_p * mask.unsqueeze(1)
                z_p = z_p * mask.unsqueeze(1)
                log_w = log_w * mask
                h_hat = h_hat * mask

        logits = torch.stack(logits_seq, dim=0).reshape(T * N, N_ACT)
        values = torch.stack(values_seq, dim=0).reshape(T * N)
        elbos  = torch.stack(elbo_seq, dim=0).reshape(T * N)

        dist   = Categorical(logits=logits)
        new_lp = dist.log_prob(actions)
        entropy = dist.entropy()

        ratio    = torch.exp(new_lp - old_logp)
        policy_l = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages).mean()
        value_l  = F.mse_loss(values, returns)
        elbo_l   = -elbos.mean()

        loss = policy_l + vf_coef * value_l - ent_coef * entropy.mean() + lambda_E * elbo_l

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad)
        opt.step()

        with torch.no_grad():
            kl = ((ratio - 1) - torch.log(ratio)).mean().item()

        metrics["policy_loss"].append(policy_l.item())
        metrics["value_loss"] .append(value_l.item())
        metrics["elbo_loss"]  .append(elbo_l.item())
        metrics["entropy"]    .append(entropy.mean().item())
        metrics["approx_kl"]  .append(kl)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# =============================================================================
# Env utilities
# =============================================================================
def import_obelix(path):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def make_env_fn(OBELIX, args, seed):
    def _make():
        env = OBELIX(
            scaling_factor=args.scaling_factor, arena_size=args.arena_size,
            max_steps=args.max_steps, wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty, box_speed=args.box_speed, seed=seed,
        )
        orig = env.step
        env.step = lambda a: orig(a, render=False)
        return env
    return _make


# =============================================================================
# Main training loop
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="DVRL trainer for OBELIX")
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights_dvrl.pth")
    ap.add_argument("--episodes",        type=int,   default=3000)
    ap.add_argument("--seed",            type=int,   default=0)
    ap.add_argument("--device",          type=str,   default=None)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--n_envs",          type=int,   default=8)
    # DVRL architecture
    ap.add_argument("--K",               type=int,   default=8,
                    help="Number of particles. Paper uses 15-30; 8 is good for OBELIX.")
    ap.add_argument("--h_dim",           type=int,   default=64,
                    help="Per-particle GRU hidden dim.")
    ap.add_argument("--z_dim",           type=int,   default=32,
                    help="Per-particle stochastic latent dim.")
    ap.add_argument("--agg_dim",         type=int,   default=128,
                    help="Aggregator GRU output dim = belief summary dim.")
    # Training
    ap.add_argument("--lr",              type=float, default=1e-4)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--gae_lam",         type=float, default=0.95)
    ap.add_argument("--clip_eps",        type=float, default=0.2)
    ap.add_argument("--vf_coef",         type=float, default=0.5)
    ap.add_argument("--ent_coef",        type=float, default=0.01)
    ap.add_argument("--lambda_E",        type=float, default=1.0,
                    help="Weight on ELBO loss. Paper: 0.1 for Atari, 1.0 for low-dim.")
    ap.add_argument("--n_epochs",        type=int,   default=4)
    ap.add_argument("--rollout_len",     type=int,   default=256)
    ap.add_argument("--mini_batch",      type=int,   default=64)
    ap.add_argument("--max_grad",        type=float, default=0.5)
    ap.add_argument("--reward_scale",    type=float, default=100.0)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)

    OBELIX   = import_obelix(args.obelix_py)
    make_fns = [make_env_fn(OBELIX, args, args.seed+i) for i in range(args.n_envs)]
    vec      = VecEnv(make_fns=make_fns, reward_shaping_fn=None)

    agent = DVRLAgent(K=args.K, h_dim=args.h_dim, z_dim=args.z_dim, agg_dim=args.agg_dim)
    agent = agent.to(device)
    opt   = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    buf = RolloutBuffer(args.rollout_len, args.n_envs, args.agg_dim)

    # Particle state: maintained per-worker in main process, passed to encoder each step
    h_p, z_p, log_w, h_hat = agent.init_belief(args.n_envs, device)
    prev_actions = torch.zeros(args.n_envs, dtype=torch.long, device=device)

    init_seeds   = [args.seed + i for i in range(args.n_envs)]
    obs_arr      = np.array(vec.reset(seeds=init_seeds), dtype=np.float32)

    ep_ret   = np.zeros(args.n_envs, np.float32)
    ep_steps = np.zeros(args.n_envs, np.int32)
    last_done = np.zeros(args.n_envs, bool)
    episodes_done = total_steps = update_count = success_count = 0
    best_return   = -float("inf")
    train_start   = time.time()
    window_returns: List[float] = []
    window_steps:   List[int]   = []
    recent_m = collections.defaultdict(lambda: collections.deque(maxlen=20))
    last_log = 0; LOG_EVERY = 10

    print(f"\n[DVRL] K={args.K} h={args.h_dim} z={args.z_dim} agg={args.agg_dim} "
          f"envs={args.n_envs} rollout={args.rollout_len} lambda_E={args.lambda_E}")
    print(f"[DVRL] difficulty={args.difficulty} wall={args.wall_obstacles} "
          f"reward_scale={args.reward_scale}\n")

    pbar = tqdm(total=args.episodes, desc="DVRL", unit="ep", ncols=110)

    while episodes_done < args.episodes:
        buf.clear()
        buf.store_init_state(h_p, z_p, log_w, h_hat)

        for _ in range(args.rollout_len):
            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, entropy, value, \
                h_hat, h_p, z_p, log_w, elbo = agent.step(
                    obs_t, prev_actions, h_p, z_p, log_w, h_hat
                )

            action_idx  = action.cpu().numpy()
            action_strs = [ACTIONS[a] for a in action_idx]

            results      = vec.step(action_strs)
            next_obs_arr = np.array([r[0] for r in results], dtype=np.float32)
            raw_rewards  = np.array([r[1] for r in results], dtype=np.float32)
            dones        = np.array([r[2] for r in results], dtype=bool)

            scaled = raw_rewards / args.reward_scale

            buf.add(
                obs      = obs_arr,
                prev_actions = prev_actions.detach().cpu().numpy(),
                actions  = action_idx,
                log_probs= log_prob.cpu().numpy(),
                rewards  = scaled,
                values   = value.cpu().numpy(),
                dones    = dones.astype(np.float32),
            )

            ep_ret   += scaled; ep_steps += 1
            total_steps += args.n_envs; last_done[:] = dones

            for i in range(args.n_envs):
                if not dones[i]: continue
                success = bool(raw_rewards[i] >= 100.0)
                success_count += int(success)
                window_returns.append(float(ep_ret[i]))
                window_steps.append(int(ep_steps[i]))
                if ep_ret[i] > best_return:
                    best_return = float(ep_ret[i])
                    torch.save(agent.cpu().state_dict(), args.out+".best")
                    agent.to(device)
                episodes_done += 1
                pbar.update(1)
                pbar.set_postfix({"ret": f"{ep_ret[i]:.1f}",
                                  "best": f"{best_return:.1f}", "suc": success_count})
                new_seed = args.seed + args.n_envs + episodes_done
                next_obs_arr[i] = np.array(vec.reset_one(i, seed=new_seed), np.float32)
                ep_ret[i] = ep_steps[i] = 0
                # Reset this worker's particle state
                h_p[i, :, :]  = 0.0; z_p[i, :, :]  = 0.0
                log_w[i, :]   = 0.0; h_hat[i, :]   = 0.0
                action_idx[i] = 0
                if episodes_done >= args.episodes: break

            prev_actions = torch.tensor(action_idx, dtype=torch.long, device=device)
            obs_arr = next_obs_arr
            if episodes_done >= args.episodes: break

        # Bootstrap last values
        with torch.no_grad():
            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
            _, _, _, lv, _, _, _, _, _ = agent.step(
                obs_t,
                prev_actions,
                h_p.clone(),
                z_p.clone(),
                log_w.clone(),
                h_hat.clone(),
            )
            last_vals = np.where(last_done, 0.0, lv.detach().cpu().numpy())

        metrics = ppo_elbo_update(
            agent, opt, buf, last_vals, device,
            gamma=args.gamma, lam=args.gae_lam, clip_eps=args.clip_eps,
            vf_coef=args.vf_coef, ent_coef=args.ent_coef, lambda_E=args.lambda_E,
            n_epochs=args.n_epochs, mini_batch=args.mini_batch, max_grad=args.max_grad,
        )

        update_count += 1
        for k, v in metrics.items(): recent_m[k].append(v)

        if (episodes_done // LOG_EVERY) > (last_log // LOG_EVERY) and window_returns:
            elapsed = time.time() - train_start
            tqdm.write(
                f"\n┌─ ep {episodes_done-len(window_returns)+1:>4d}–{episodes_done:<4d} "
                f"({elapsed:.0f}s) ─────────────────────────────────────────\n"
                f"│ Avg Return : {np.mean(window_returns):8.2f}   Best : {best_return:8.2f}   "
                f"Successes : {success_count}\n"
                f"│ Policy     : {np.mean(recent_m['policy_loss']):8.4f}   "
                f"Value : {np.mean(recent_m['value_loss']):8.4f}   "
                f"ELBO : {np.mean(recent_m['elbo_loss']):8.4f}\n"
                f"│ Entropy    : {np.mean(recent_m['entropy']):8.4f}   "
                f"KL : {np.mean(recent_m['approx_kl']):8.4f}   "
                f"Updates : {update_count}   Steps : {total_steps:,}\n"
                f"└{'─'*80}"
            )
            last_log = episodes_done; window_returns.clear(); window_steps.clear()

    pbar.close(); vec.close()
    torch.save(agent.cpu().state_dict(), args.out)
    elapsed = time.time() - train_start
    print(f"\nSaved: {args.out}  (.best also saved)")
    print(f"Time : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Stats: {episodes_done} ep | {total_steps:,} steps | "
          f"{success_count} successes ({100*success_count/max(1,episodes_done):.2f}%)")


if __name__ == "__main__":
    main()
