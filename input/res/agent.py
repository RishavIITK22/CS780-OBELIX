from __future__ import annotations

import collections
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

# Constants
ACTIONS:            Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_N_ACTIONS:         int           = len(ACTIONS)
_OBS_DIM:           int           = 18
_MAX_EPISODE_STEPS: int           = 1000


# Actor Critic Network
class _ActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, _N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


#Frames stacker
class _FrameStack:
    def __init__(self, k: int):
        self.k      = k
        self.frames = collections.deque(maxlen=k)

    @property
    def ready(self) -> bool:
        return len(self.frames) == self.k

    def reset(self, obs: np.ndarray) -> None:
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs.copy())

    def step(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs.copy())
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)



_model:      Optional[_ActorCritic] = None
_stacker:    Optional[_FrameStack]  = None
_step_count: int                    = 0


def reset() -> None:
    """Reset episode state. Call at the start of each new episode."""
    global _step_count
    if _stacker is not None:
        _stacker.frames.clear()
    _step_count = 0


#Weights loader
def _load_once() -> None:
    global _model, _stacker

    if _model is not None:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("weights_ppo_fs.pth", "weights_ppo.pth", "weights.pth"):
        p = os.path.join(here, name)
        if os.path.exists(p):
            wpath = p
            break
    else:
        raise FileNotFoundError(
            "No weights file found next to agent_fs.py. "
            "Expected 'weights_ppo_fs.pth', 'weights_ppo.pth', or 'weights.pth'."
        )

    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Infer k and hidden from checkpoint: trunk.0.weight shape = (hidden, k*18)
    w0     = sd["trunk.0.weight"]
    hidden = w0.shape[0]
    in_dim = w0.shape[1]

    if in_dim % _OBS_DIM != 0:
        raise ValueError(
            f"Checkpoint in_dim={in_dim} is not divisible by _OBS_DIM={_OBS_DIM}."
        )
    k = in_dim // _OBS_DIM

    model = _ActorCritic(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(sd, strict=True)
    model.eval()

    _model   = model
    _stacker = _FrameStack(k=k)


#Policy
@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _step_count

    _load_once()

    # Auto-reset at episode boundary
    if not _stacker.ready or _step_count >= _MAX_EPISODE_STEPS:
        _stacker.reset(obs)
        _step_count = 0

    stacked    = _stacker.step(obs)
    _step_count += 1

    x          = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
    logits, _  = _model(x)
    action_idx = int(logits.squeeze(0).argmax().item())

    return ACTIONS[action_idx]
