from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L22", "FW", "R22"]
ACTIONS_W = ["L45", "FW", "R45"]
FINDER_OBS_DIM = 18
PUSHER_OBS_DIM = 18
UNWEDGER_OBS_DIM = 18
GRU_HIDDEN = 64

PUSH_GRACE_STEPS = 7
UNWEDGE_GRACE_STEPS = 10
POST_UNWEDGE_COOLDOWN = 10
MAX_EPISODE_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_find_obs(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32)


def get_push_obs(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32)


def get_unwedge_obs(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, n_actions: int = len(ACTIONS), hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        f = self.backbone(x)
        return self.actor(f), self.critic(f).squeeze(-1)


class GRUActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = UNWEDGER_OBS_DIM,
        n_actions: int = len(ACTIONS_W),
        enc_hidden: int = 64,
        gru_hidden: int = GRU_HIDDEN,
    ):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, enc_hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRU(enc_hidden, gru_hidden, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(p, gain=math.sqrt(2))
            elif "weight_hh" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def zero_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.gru_hidden, device=device)

    def forward_step(self, obs: torch.Tensor, h: torch.Tensor):
        enc = self.encoder(obs).unsqueeze(1)
        out, h_new = self.gru(enc, h.to(enc.dtype))
        out = out.squeeze(1)
        return self.actor(out), self.critic(out).squeeze(-1), h_new


def _agent_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_state(name: str):
    path = os.path.join(_agent_dir(), f"weights_{name}.pth")
    payload = torch.load(path, map_location=device, weights_only=False)
    return payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload


_find_model = ActorCritic(FINDER_OBS_DIM, n_actions=len(ACTIONS), hidden=256).to(device)
_push_model = ActorCritic(PUSHER_OBS_DIM, n_actions=len(ACTIONS), hidden=128).to(device)
_unwedge_model = GRUActorCritic().to(device)

_find_model.load_state_dict(_load_state("find"), strict=True)
_push_model.load_state_dict(_load_state("push"), strict=True)
_unwedge_model.load_state_dict(_load_state("unwedge"), strict=True)

_find_model.eval()
_push_model.eval()
_unwedge_model.eval()

_gru_hidden = _unwedge_model.zero_hidden()
_push_grace = 0
_ir_streak = 0          # consecutive steps IR has been active; threshold=2 before push_grace fires
_unwedge_active = False
_unwedge_grace = 0
_post_unwedge_cooldown = 0
_steps = 0
_prev_obs: np.ndarray | None = None


def _reset_agent():
    global _gru_hidden, _push_grace, _ir_streak, _unwedge_active, _unwedge_grace, _post_unwedge_cooldown, _steps, _prev_obs
    _gru_hidden = _unwedge_model.zero_hidden()
    _push_grace = 0
    _ir_streak = 0
    _unwedge_active = False
    _unwedge_grace = 0
    _post_unwedge_cooldown = 0
    _steps = 0
    _prev_obs = None


def _maybe_reset(obs: np.ndarray):
    global _steps, _prev_obs
    if _prev_obs is None:
        _prev_obs = obs.copy()
        return

    # Only reset at episode boundary — never on blank obs (robot regularly
    # sees nothing mid-episode; np.all(obs==0) was firing spuriously).
    if _steps >= MAX_EPISODE_STEPS:
        _reset_agent()
        _prev_obs = obs.copy()


@torch.no_grad()
def policy(obs, rng=None):
    global _gru_hidden, _push_grace, _ir_streak, _unwedge_active, _unwedge_grace, _post_unwedge_cooldown, _steps, _prev_obs

    raw = np.asarray(obs, dtype=np.float32)
    _maybe_reset(raw)

    ir_on = bool(raw[16] == 1)
    stuck_on = bool(raw[17] == 1)

    if stuck_on:
        _unwedge_active = True
        _unwedge_grace = UNWEDGE_GRACE_STEPS
    elif _unwedge_grace > 0:
        _unwedge_grace -= 1
        if _unwedge_grace == 0:
            _unwedge_active = False
            _post_unwedge_cooldown = POST_UNWEDGE_COOLDOWN

    if _post_unwedge_cooldown > 0:
        _post_unwedge_cooldown -= 1

    # Require 2+ consecutive IR steps before entering push mode.
    # A single-step IR flash from a wall contact should not trigger push grace.
    if ir_on and _post_unwedge_cooldown == 0:
        _ir_streak += 1
        if _ir_streak >= 2:
            _push_grace = PUSH_GRACE_STEPS
    else:
        _ir_streak = 0
        if _push_grace > 0:
            _push_grace -= 1

    if _unwedge_active:
        mode = "unwedge"
    elif _push_grace > 0:
        mode = "push"
    else:
        mode = "find"

    # Hard rule: IR active and not stuck → always move forward.
    # This guarantees attachment in ≤2 steps regardless of which network
    # mode is active, matching the training reward probe behaviour.
    if ir_on and not stuck_on:
        _steps += 1
        _prev_obs = raw.copy()
        return "FW"

    if mode == "find":
        x = torch.from_numpy(get_find_obs(raw)).to(device).unsqueeze(0)
        logits, _ = _find_model(x)
        action_idx = int(torch.argmax(logits, dim=-1).item())
        action = ACTIONS[action_idx]
    elif mode == "push":
        x = torch.from_numpy(get_push_obs(raw)).to(device).unsqueeze(0)
        logits, _ = _push_model(x)
        action_idx = int(torch.argmax(logits, dim=-1).item())
        action = ACTIONS[action_idx]
    else:
        x = torch.from_numpy(get_unwedge_obs(raw)).to(device).unsqueeze(0)
        logits, _, h_new = _unwedge_model.forward_step(x, _gru_hidden)
        _gru_hidden = h_new
        action_idx = int(torch.argmax(logits, dim=-1).item())
        action = ACTIONS_W[action_idx]

    _steps += 1
    _prev_obs = raw.copy()
    return action
