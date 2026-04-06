from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


OBS_DIM = 18
N_ACT = 5


class RunningMeanStd:
    """Simple running mean/variance tracker for reward normalization."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = int(x.size)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / np.sqrt(self.var + eps)

    def _update_from_moments(self, batch_mean: float, batch_var: float, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = max(new_var, 1e-8)
        self.count = total_count


class LatentBeliefActorCritic(nn.Module):
    """PPO policy/value model with learned recurrent latent belief.

    Inputs at time t:
      - raw observation obs_t (18 dims)
      - previous action a_{t-1} as one-hot
      - previous reward r_{t-1}
      - previous done flag d_{t-1}

    Components:
      - observation encoder: obs_t -> e_t
      - recurrent belief: [e_t, a_{t-1}, r_{t-1}] -> h_t
      - actor / critic on h_t
      - auxiliary heads:
          * next observation prediction from (h_t, a_t)
          * reward prediction from (h_t, a_t)
          * done prediction from (h_t, a_t)
          * inverse dynamics from (e_t, e_{t+1})
          * forward latent prediction from (h_t, a_t) -> e_{t+1}
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACT,
        obs_embed_dim: int = 64,
        hidden_dim: int = 128,
        aux_hidden: int = 128,
        use_latent_learning: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.use_latent_learning = use_latent_learning
        self.obs_embed_dim = obs_embed_dim if use_latent_learning else obs_dim
        self.hidden_dim = hidden_dim

        if use_latent_learning:
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, self.obs_embed_dim),
                nn.Tanh(),
            )
        else:
            self.obs_encoder = nn.Identity()

        self.belief = nn.LSTM(
            input_size=self.obs_embed_dim + n_actions + 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        pred_in = hidden_dim + n_actions
        self.obs_head = nn.Sequential(
            nn.Linear(pred_in, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(pred_in, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(pred_in, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, 1),
        )
        self.forward_head = nn.Sequential(
            nn.Linear(pred_in, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, self.obs_embed_dim),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(2 * self.obs_embed_dim, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, n_actions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = 0.01 if module in {self.actor} else np.sqrt(2)
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

        for name, param in self.belief.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_hidden(
        self, batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.hidden_dim, device=device)
        c = torch.zeros(1, batch, self.hidden_dim, device=device)
        return h, c

    def _action_one_hot(self, action_idx: torch.Tensor) -> torch.Tensor:
        oh = torch.zeros(action_idx.shape[0], self.n_actions, device=action_idx.device)
        valid = action_idx >= 0
        if valid.any():
            oh[valid, action_idx[valid]] = 1.0
        return oh

    def _build_step_input(
        self,
        obs: torch.Tensor,
        prev_action_idx: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_feat = self.obs_encoder(obs)
        prev_action_oh = self._action_one_hot(prev_action_idx)
        prev_reward = prev_reward.view(-1, 1)
        prev_done = prev_done.view(-1, 1)
        step_in = torch.cat([obs_feat, prev_action_oh, prev_reward, prev_done], dim=-1)
        return step_in, obs_feat

    def forward_step(
        self,
        obs: torch.Tensor,
        prev_action_idx: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        step_in, obs_feat = self._build_step_input(obs, prev_action_idx, prev_reward, prev_done)
        out, hidden = self.belief(step_in.unsqueeze(1), hidden)
        belief_t = out.squeeze(1)
        logits = self.actor(belief_t)
        value = self.critic(belief_t).squeeze(-1)
        return logits, value, hidden, belief_t, obs_feat

    def get_action(
        self,
        obs: torch.Tensor,
        prev_action_idx: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ):
        logits, value, hidden, belief_t, obs_feat = self.forward_step(
            obs, prev_action_idx, prev_reward, prev_done, hidden
        )
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, hidden, belief_t, obs_feat

    def intrinsic_reward(
        self,
        belief_t: torch.Tensor,
        action_idx: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        next_feat = self.obs_encoder(next_obs).detach()
        act_oh = self._action_one_hot(action_idx)
        pred_in = torch.cat([belief_t, act_oh], dim=-1)
        next_feat_pred = self.forward_head(pred_in)
        return 0.5 * torch.mean((next_feat_pred - next_feat) ** 2, dim=-1)

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,          # (N, T, obs_dim)
        prev_action_seq: torch.Tensor,  # (N, T)
        prev_reward_seq: torch.Tensor,  # (N, T)
        prev_done_seq: torch.Tensor,    # (N, T)
        actions_seq: torch.Tensor,      # (N, T)
        next_obs_seq: torch.Tensor,     # (N, T, obs_dim)
        ext_reward_seq: torch.Tensor,   # (N, T)
        dones_seq: torch.Tensor,        # (N, T)
        init_h: torch.Tensor,
        init_c: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        n_envs, T, _ = obs_seq.shape

        obs_flat = obs_seq.reshape(n_envs * T, self.obs_dim)
        next_obs_flat = next_obs_seq.reshape(n_envs * T, self.obs_dim)
        obs_feat_flat = self.obs_encoder(obs_flat)
        next_obs_feat_flat = self.obs_encoder(next_obs_flat).detach()
        obs_feat_seq = obs_feat_flat.reshape(n_envs, T, self.obs_embed_dim)
        next_obs_feat_seq = next_obs_feat_flat.reshape(n_envs, T, self.obs_embed_dim)

        prev_action_oh_seq = torch.zeros(
            n_envs, T, self.n_actions, device=obs_seq.device, dtype=torch.float32
        )
        valid_prev = prev_action_seq >= 0
        if valid_prev.any():
            prev_action_oh_seq[valid_prev] = torch.nn.functional.one_hot(
                prev_action_seq[valid_prev], num_classes=self.n_actions
            ).float()

        belief_inputs = torch.cat(
            [obs_feat_seq, prev_action_oh_seq, prev_reward_seq.unsqueeze(-1), prev_done_seq.unsqueeze(-1)], dim=-1
        )

        h, c = init_h, init_c
        belief_outputs = []
        for t in range(T):
            out, (h, c) = self.belief(belief_inputs[:, t : t + 1, :], (h, c))
            belief_t = out.squeeze(1)
            belief_outputs.append(belief_t)
            if t < T - 1:
                mask = (1.0 - dones_seq[:, t]).view(1, n_envs, 1)
                h = h * mask
                c = c * mask

        belief_seq = torch.stack(belief_outputs, dim=1)          # (N,T,H)
        logits_seq = self.actor(belief_seq)                      # (N,T,A)
        values_seq = self.critic(belief_seq).squeeze(-1)         # (N,T)

        action_oh_seq = torch.nn.functional.one_hot(
            actions_seq, num_classes=self.n_actions
        ).float()
        pred_in_seq = torch.cat([belief_seq, action_oh_seq], dim=-1)

        obs_pred_seq = self.obs_head(pred_in_seq)
        reward_pred_seq = self.reward_head(pred_in_seq).squeeze(-1)
        done_logit_seq = self.done_head(pred_in_seq).squeeze(-1)
        next_feat_pred_seq = self.forward_head(pred_in_seq)
        inverse_logits_seq = self.inverse_head(
            torch.cat([obs_feat_seq, next_obs_feat_seq], dim=-1)
        )

        logits_flat = logits_seq.permute(1, 0, 2).reshape(T * n_envs, self.n_actions)
        values_flat = values_seq.permute(1, 0).reshape(T * n_envs)
        obs_pred_flat = obs_pred_seq.permute(1, 0, 2).reshape(T * n_envs, self.obs_dim)
        reward_pred_flat = reward_pred_seq.permute(1, 0).reshape(T * n_envs)
        done_logit_flat = done_logit_seq.permute(1, 0).reshape(T * n_envs)
        next_feat_pred_flat = next_feat_pred_seq.permute(1, 0, 2).reshape(
            T * n_envs, self.obs_embed_dim
        )
        inverse_logits_flat = inverse_logits_seq.permute(1, 0, 2).reshape(
            T * n_envs, self.n_actions
        )
        next_obs_flat_ordered = next_obs_seq.permute(1, 0, 2).reshape(T * n_envs, self.obs_dim)
        ext_reward_flat = ext_reward_seq.permute(1, 0).reshape(T * n_envs)
        done_flat = dones_seq.permute(1, 0).reshape(T * n_envs)
        next_obs_feat_flat_ordered = next_obs_feat_seq.permute(1, 0, 2).reshape(
            T * n_envs, self.obs_embed_dim
        )

        dist = Categorical(logits=logits_flat)
        entropy_flat = dist.entropy()

        return {
            "logits_flat": logits_flat,
            "values_flat": values_flat,
            "entropy_flat": entropy_flat,
            "obs_pred_flat": obs_pred_flat,
            "reward_pred_flat": reward_pred_flat,
            "done_logit_flat": done_logit_flat,
            "inverse_logits_flat": inverse_logits_flat,
            "next_feat_pred_flat": next_feat_pred_flat,
            "next_feat_target_flat": next_obs_feat_flat_ordered,
            "next_obs_target_flat": next_obs_flat_ordered,
            "ext_reward_target_flat": ext_reward_flat,
            "done_target_flat": done_flat,
        }
