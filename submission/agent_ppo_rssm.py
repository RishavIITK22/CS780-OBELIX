from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


OBS_DIM = 18
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACT = len(ACTIONS)
DEVICE = torch.device("cpu")
MAX_EPISODE_STEPS = 1000


def one_hot(actions: torch.Tensor, n: int = N_ACT) -> torch.Tensor:
    return nn.functional.one_hot(actions.long(), num_classes=n).float()


class RSSMActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = N_ACT,
        obs_embed_dim: int = 64,
        h_dim: int = 128,
        z_dim: int = 32,
        hidden: int = 128,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.min_std = min_std
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_embed_dim),
            nn.ELU(),
            nn.Linear(obs_embed_dim, obs_embed_dim),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(z_dim + action_dim + 2, h_dim)
        self.prior = nn.Sequential(nn.Linear(h_dim, hidden), nn.ELU(), nn.Linear(hidden, 2 * z_dim))
        self.posterior = nn.Sequential(
            nn.Linear(h_dim + obs_embed_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2 * z_dim),
        )
        belief_dim = h_dim + z_dim
        self.actor = nn.Sequential(nn.Linear(belief_dim, hidden), nn.Tanh(), nn.Linear(hidden, action_dim))
        self.critic = nn.Sequential(nn.Linear(belief_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.obs_head = nn.Sequential(nn.Linear(belief_dim, hidden), nn.ELU(), nn.Linear(hidden, obs_dim))
        self.reward_head = nn.Sequential(nn.Linear(belief_dim, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.done_head = nn.Sequential(nn.Linear(belief_dim, hidden), nn.ELU(), nn.Linear(hidden, 1))

    def init_state(self, batch_size: int):
        h = torch.zeros(batch_size, self.h_dim, device=DEVICE)
        z = torch.zeros(batch_size, self.z_dim, device=DEVICE)
        return h, z

    def _stats(self, raw):
        mu, raw_std = torch.chunk(raw, 2, dim=-1)
        std = nn.functional.softplus(raw_std) + self.min_std
        return mu, std

    def observe_step(self, obs, prev_action_onehot, prev_reward, prev_done, h_prev, z_prev):
        reset = prev_done.view(-1, 1).bool()
        h_prev = torch.where(reset, torch.zeros_like(h_prev), h_prev)
        z_prev = torch.where(reset, torch.zeros_like(z_prev), z_prev)
        gru_in = torch.cat([z_prev, prev_action_onehot, prev_reward.view(-1, 1), prev_done.view(-1, 1)], dim=-1)
        h = self.gru(gru_in, h_prev)
        obs_embed = self.obs_encoder(obs)
        post_mu, _ = self._stats(self.posterior(torch.cat([h, obs_embed], dim=-1)))
        z = post_mu
        belief = torch.cat([h, z], dim=-1)
        return self.actor(belief), self.critic(belief).squeeze(-1), h, z


_here = os.path.dirname(os.path.abspath(__file__))
_weights_path = os.path.join(_here, "weights_ppo_rssm.pth")
_payload = torch.load(_weights_path, map_location=DEVICE, weights_only=False)
_config = _payload.get("config", {})
_net = RSSMActorCritic(
    obs_embed_dim=int(_config.get("obs_embed_dim", 64)),
    h_dim=int(_config.get("h_dim", 128)),
    z_dim=int(_config.get("z_dim", 32)),
    hidden=int(_config.get("hidden", 128)),
    min_std=float(_config.get("min_std", 0.1)),
).to(DEVICE)
_net.load_state_dict(_payload["model_state_dict"] if "model_state_dict" in _payload else _payload)
_net.eval()

_h, _z = _net.init_state(1)
_prev_action = torch.zeros(1, dtype=torch.long, device=DEVICE)
_prev_reward = torch.zeros(1, dtype=torch.float32, device=DEVICE)
_prev_done = torch.ones(1, dtype=torch.float32, device=DEVICE)
_step_count = 0


def reset_agent():
    global _h, _z, _prev_action, _prev_reward, _prev_done, _step_count
    _h, _z = _net.init_state(1)
    _prev_action = torch.zeros(1, dtype=torch.long, device=DEVICE)
    _prev_reward = torch.zeros(1, dtype=torch.float32, device=DEVICE)
    _prev_done = torch.ones(1, dtype=torch.float32, device=DEVICE)
    _step_count = 0


@torch.no_grad()
def policy(obs, rng=None):
    global _h, _z, _prev_action, _prev_reward, _prev_done, _step_count
    if _step_count >= MAX_EPISODE_STEPS:
        reset_agent()
    obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=DEVICE).view(1, -1)
    logits, _, _h, _z = _net.observe_step(
        obs_t,
        one_hot(_prev_action),
        _prev_reward,
        _prev_done,
        _h,
        _z,
    )
    action_idx = int(torch.argmax(logits, dim=-1).item())
    _prev_action = torch.tensor([action_idx], dtype=torch.long, device=DEVICE)
    _prev_reward.zero_()
    _prev_done.zero_()
    _step_count += 1
    return ACTIONS[action_idx]
