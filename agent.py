"""Submission agent — PPO + frame stacking.

The evaluator imports this file and calls policy(obs, rng) once per step.
Action space (strings): 'L45', 'L22', 'FW', 'R22', 'R45'
Observation: numpy array shape (18,), values 0/1.

Architecture (mirrors train_ppo.py):
  - FrameStack : k frames × 18 bits  (k inferred from checkpoint)
  - ActorCritic: trunk = Linear(in,h)+Tanh → Linear(h,h)+Tanh
                 actor head = Linear(h, 5)   (critic loaded but unused)
  - Inference  : deterministic — argmax over actor logits (no sampling)

Hyperparameters k (stack depth) and h (hidden size) are inferred from the
checkpoint's weight shapes so this file never needs editing after retraining
with different --stack or --hidden values.

Episode boundary handling:
  The Codabench evaluator imports this module ONCE and calls policy_fn for
  ALL episodes without calling reset() between them.  The stacker is full
  at the start of episode 2+ and would carry stale frames from the previous
  episode.  We fix this with a step counter: after _MAX_EPISODE_STEPS steps
  without an explicit reset() the stacker is cleared automatically.
  _MAX_EPISODE_STEPS matches the evaluator's max_steps=1000.  For episodes
  that end early (success), the first ~k steps of the next episode will use
  a slightly stale stack, but this affects at most k/1000 < 1% of steps.
  Calling reset() explicitly avoids even this — harnesses that support it
  should call it at each env.reset().
"""

from __future__ import annotations

import collections
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

# ── Fixed constants ────────────────────────────────────────────────────────────
ACTIONS:            Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_OBS_DIM:           int           = 18    # raw obs length — always 18 for OBELIX
_N_ACTIONS:         int           = len(ACTIONS)
_MAX_EPISODE_STEPS: int           = 1000  # matches evaluator's max_steps


# ── Network (must mirror ActorCritic in train_ppo.py) ─────────────────────────
class ActorCritic(nn.Module):
    """Shared-trunk MLP Actor-Critic.  Only the actor head is used at inference.
    The critic head is present so load_state_dict(strict=True) succeeds without
    manually filtering checkpoint keys.
    """

    def __init__(self, in_dim: int, hidden: int, n_actions: int = _N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ── Frame stacker ──────────────────────────────────────────────────────────────
class _FrameStack:
    def __init__(self, k: int, obs_dim: int = _OBS_DIM):
        self.k      = k
        self.obs_dim = obs_dim
        self.frames  = collections.deque(maxlen=k)

    @property
    def ready(self) -> bool:
        return len(self.frames) == self.k

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Clear and warm-start: fill all k slots with obs."""
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs.copy())
        return self._get()

    def step(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs.copy())
        return self._get()

    def _get(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


# ── Module-level persistent state ─────────────────────────────────────────────
_model:       Optional[ActorCritic] = None
_stacker:     Optional[_FrameStack] = None   # created in _load_once()
_push_active: bool                  = False   # sticky IR-contact latch
_step_count:  int                   = 0       # steps since last episode reset


def reset() -> None:
    """Reset episode state. Call this at the start of each new episode.

    The Codabench evaluator does NOT call this between episodes, so
    policy() uses _step_count to auto-reset at episode boundaries instead.
    Harnesses that do call reset() get perfectly clean episode starts.
    """
    global _push_active, _step_count
    if _stacker is not None:
        _stacker.frames.clear()
    _push_active = False
    _step_count  = 0


# ── Weight loader ──────────────────────────────────────────────────────────────
def _load_once() -> None:
    """Load weights and build model + stacker with hyperparams from checkpoint.

    FIX (original bug): the old code hardcoded _STACK_K=8 and _HIDDEN=128,
    causing a size mismatch when the checkpoint was trained with different
    values (e.g. --stack 10 gives trunk.0.weight shape (128, 180) not (128, 144)).

    Fix: read in_dim and hidden directly from trunk.0.weight in the state dict:
        trunk.0.weight shape = (hidden, in_dim)
        k_stack = in_dim / _OBS_DIM
    This works for any --stack and --hidden combination without code changes.
    """
    global _model, _stacker

    if _model is not None:
        return

    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights_ppo.pth")
    if not os.path.exists(wpath):
        wpath_fallback = os.path.join(here, "weights.pth")
        if os.path.exists(wpath_fallback):
            wpath = wpath_fallback
        else:
            raise FileNotFoundError(
                "Neither 'weights_ppo.pth' nor 'weights.pth' found next to agent.py. "
                "Train with train_ppo.py and place the weights file in the same directory."
            )

    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Infer architecture from checkpoint weight shapes — no hardcoding needed.
    # trunk.0.weight : (hidden, in_dim)  — first Linear layer of the trunk
    trunk0  = sd["trunk.0.weight"]     # shape: (hidden, in_dim)
    hidden  = trunk0.shape[0]          # out_features = hidden size
    in_dim  = trunk0.shape[1]          # in_features  = stack_k * obs_dim

    if in_dim % _OBS_DIM != 0:
        raise ValueError(
            f"Checkpoint in_dim={in_dim} is not divisible by _OBS_DIM={_OBS_DIM}. "
            "Checkpoint may be from a different environment."
        )
    k_stack = in_dim // _OBS_DIM

    m = ActorCritic(in_dim=in_dim, hidden=hidden)
    m.load_state_dict(sd, strict=True)
    m.eval()

    _model   = m
    _stacker = _FrameStack(k=k_stack, obs_dim=_OBS_DIM)


# ── Policy ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return a deterministic action for the given 18-dim observation.

    Frame stacking: the raw 18-bit obs is pushed onto the deque and the
    full (k*18)-dim stacked vector is forwarded through the network.

    Episode boundary auto-reset: after _MAX_EPISODE_STEPS steps the stacker
    and push latch are cleared, matching the start-of-episode state from
    training.  This handles the Codabench evaluator which does not call
    reset() between episodes.
    """
    global _push_active, _step_count

    _load_once()   # no-op on every call after the first

    # ── Auto-reset at episode boundary ────────────────────────────────────────
    # Triggered when: (a) stacker is empty (first call after import/reset()),
    # or (b) step counter reached _MAX_EPISODE_STEPS (new episode starting).
    if not _stacker.ready or _step_count >= _MAX_EPISODE_STEPS:
        _stacker.reset(obs)
        _push_active = False
        _step_count  = 0

    stacked = _stacker.step(obs)
    _step_count += 1

    # ── Sticky push latch (mirrors env.enable_push) ───────────────────────────
    # obs[16] is the IR contact sensor. env.enable_push latches True on first
    # contact and stays True for the rest of the episode; we mirror that here.
    if obs[16]:
        _push_active = True

    # ── Deterministic inference ───────────────────────────────────────────────
    x          = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)  # (1, k*18)
    logits, _  = _model(x)                                                 # (1, 5)
    action_idx = int(logits.squeeze(0).argmax().item())

    return ACTIONS[action_idx]