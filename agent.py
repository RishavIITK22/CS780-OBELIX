"""Submission agent — PPO + LSTM with BeliefStateEncoder (frame stacking).

The evaluator imports this file and calls policy(obs, rng) once per step.
Action space (strings): 'L45', 'L22', 'FW', 'R22', 'R45'
Observation: numpy array shape (18,), values 0/1.

Architecture (mirrors train_ppo_lstm.py):
  - BeliefStateEncoder(stack_k) : 18-dim obs -> (stack_k * 49 + 5)-dim
  - ActorCriticLSTM: encoder(in_dim -> 128 -> 64) + LSTM(64, H) + actor/critic
  - Inference: deterministic argmax over actor logits

stack_k and hidden_dim are inferred from checkpoint weight shapes.

Episode boundary:
  Codabench calls policy() for ALL episodes without calling reset() between
  them. We auto-reset LSTM hidden state and encoder after _MAX_EPISODE_STEPS.
  Harnesses that do call reset() explicitly get perfectly clean episode starts.
"""

from __future__ import annotations

import collections
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIONS:            Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_N_ACTIONS:         int           = len(ACTIONS)
_CORE_DIM:          int           = 49   # BeliefStateEncoder.CORE_DIM
_MAX_EPISODE_STEPS: int           = 1000


# ── BeliefStateEncoder (inline, mirrors state_encoder.py) ─────────────────────
class _BeliefStateEncoder:
    def __init__(
        self,
        stack_k: int = 4,
        max_steps_since_seen: int = 30,
        max_stuck_steps: int = 20,
        max_seen_streak: int = 30,
        max_lost_streak: int = 30,
    ):
        self.stack_k = stack_k
        self.max_steps_since_seen = max_steps_since_seen
        self.max_stuck_steps = max_stuck_steps
        self.max_seen_streak = max_seen_streak
        self.max_lost_streak = max_lost_streak
        self.output_dim = stack_k * _CORE_DIM + _N_ACTIONS
        self.reset()

    def reset(self) -> None:
        self._prev_ir = 0.0
        self._prev_stuck = 0.0
        self._prev_visible = 0.0
        self._steps_since_seen = self.max_steps_since_seen
        self._stuck_steps = 0
        self._seen_streak = 0
        self._lost_streak = 0
        self._prev_dir_summary = np.zeros(3, dtype=np.float32)
        self._prev_total_strength = 0.0
        self._prev_lr_balance = 0.0
        self._frames: collections.deque = collections.deque(
            [np.zeros(_CORE_DIM, dtype=np.float32) for _ in range(self.stack_k)],
            maxlen=self.stack_k,
        )

    def encode(self, obs: np.ndarray, prev_action_idx: Optional[int] = None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        far, near = obs[0:16:2], obs[1:16:2]
        ir, stuck = float(obs[16]), float(obs[17])
        strengths = 2.0 * near + far

        total_strength = float(np.sum(strengths))
        visible = 1.0 if (total_strength > 0.0 or ir > 0.0) else 0.0

        if visible > 0.0:
            self._steps_since_seen = 0
            self._seen_streak = min(self._seen_streak + 1, self.max_seen_streak)
            self._lost_streak = 0
        else:
            self._steps_since_seen = min(self._steps_since_seen + 1, self.max_steps_since_seen)
            self._lost_streak = min(self._lost_streak + 1, self.max_lost_streak)
            self._seen_streak = 0
        if stuck > 0:
            self._stuck_steps = min(self._stuck_steps + 1, self.max_stuck_steps)
        else:
            self._stuck_steps = 0

        dir_summary = np.array([
            float(strengths[0] + strengths[1]),
            float(strengths[2] + strengths[3] + strengths[4] + strengths[5]),
            float(strengths[6] + strengths[7]),
        ], dtype=np.float32)
        near_count = float(np.sum(near))
        far_count = float(np.sum(far))
        strongest_strength = float(np.max(strengths)) if strengths.size else 0.0
        sector_mass = np.sum(strengths)
        if sector_mass > 0:
            sector_positions = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
            sector_centroid = float(np.dot(strengths, sector_positions) / sector_mass)
        else:
            sector_centroid = 0.0

        left, front, right = float(dir_summary[0]), float(dir_summary[1]), float(dir_summary[2])
        front_ratio = front / max(total_strength, 1.0)
        left_ratio = left / max(total_strength, 1.0)
        right_ratio = right / max(total_strength, 1.0)
        lr_balance = (right - left) / max(total_strength, 1.0)
        just_lost_ir = 1.0 if (self._prev_ir == 1.0 and ir == 0.0) else 0.0

        geometry = np.array([
            total_strength / 24.0,
            near_count / 8.0,
            far_count / 8.0,
            strongest_strength / 3.0,
            sector_centroid,
            front_ratio,
            left_ratio,
            right_ratio,
        ], dtype=np.float32)

        temporal = np.array([
            self._steps_since_seen / self.max_steps_since_seen,
            self._seen_streak / self.max_seen_streak,
            self._lost_streak / self.max_lost_streak,
            self._stuck_steps / self.max_stuck_steps,
            visible,
            1.0 if (self._prev_ir == 0.0 and ir == 1.0) else 0.0,
            just_lost_ir,
            1.0 if (self._prev_stuck == 1.0 and stuck == 0.0) else 0.0,
        ], dtype=np.float32)

        transition = np.array([
            (total_strength - self._prev_total_strength) / 24.0,
            (front - float(self._prev_dir_summary[1])) / 12.0,
            lr_balance - self._prev_lr_balance,
            visible - self._prev_visible,
        ], dtype=np.float32)

        self._prev_ir = ir
        self._prev_stuck = stuck
        self._prev_visible = visible
        self._prev_dir_summary = dir_summary.copy()
        self._prev_total_strength = total_strength
        self._prev_lr_balance = lr_balance

        prev_act_oh = np.zeros(_N_ACTIONS, dtype=np.float32)
        if prev_action_idx is not None:
            prev_act_oh[prev_action_idx] = 1.0

        core = np.concatenate([obs, strengths, dir_summary, geometry, temporal, transition])
        self._frames.append(core)
        return np.concatenate([*self._frames, prev_act_oh])


# ── Network (must mirror ActorCriticLSTM in train_ppo_lstm.py) ────────────────
class _ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
        )
        self.lstm   = nn.LSTM(input_size=64, hidden_size=hidden_dim,
                              num_layers=1, batch_first=True)
        self.actor  = nn.Linear(hidden_dim, _N_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        enc = self.encoder(x).unsqueeze(1)          # (1, 1, 64)
        out, new_hidden = self.lstm(enc, hidden)     # (1, 1, H)
        logits = self.actor(out.squeeze(1))          # (1, N_ACTIONS)
        return logits, new_hidden


# ── Module-level persistent state ─────────────────────────────────────────────
_model:           Optional[_ActorCriticLSTM]    = None
_encoder:         Optional[_BeliefStateEncoder] = None
_hidden_dim:      int                           = 0
_h:               Optional[torch.Tensor]        = None
_c:               Optional[torch.Tensor]        = None
_step_count:      int                           = 0
_prev_action_idx: int                           = 2   # default FW


def reset() -> None:
    global _h, _c, _step_count, _prev_action_idx
    if _encoder is not None:
        _encoder.reset()
    if _model is not None:
        _h = torch.zeros(1, 1, _hidden_dim)
        _c = torch.zeros(1, 1, _hidden_dim)
    _step_count      = 0
    _prev_action_idx = 2


# ── Weight loader ─────────────────────────────────────────────────────────────
def _load_once() -> None:
    global _model, _encoder, _hidden_dim, _h, _c

    if _model is not None:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("weights_ppo_lstm.pth", "weights.pth"):
        p = os.path.join(here, name)
        if os.path.exists(p):
            wpath = p
            break
    else:
        raise FileNotFoundError(
            "No weights file found next to agent.py. "
            "Expected 'weights_ppo_lstm.pth' or 'weights.pth'."
        )

    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # encoder.0.weight shape = (128, input_dim) where input_dim = stack_k*49 + 5
    input_dim  = sd["encoder.0.weight"].shape[1]
    hidden_dim = sd["lstm.weight_ih_l0"].shape[0] // 4

    if (input_dim - _N_ACTIONS) % _CORE_DIM != 0:
        raise ValueError(
            f"Checkpoint input_dim={input_dim} inconsistent with CORE_DIM={_CORE_DIM}."
        )
    stack_k = (input_dim - _N_ACTIONS) // _CORE_DIM

    model = _ActorCriticLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(sd, strict=True)
    model.eval()

    _model      = model
    _encoder    = _BeliefStateEncoder(stack_k=stack_k)
    _hidden_dim = hidden_dim
    _h          = torch.zeros(1, 1, hidden_dim)
    _c          = torch.zeros(1, 1, hidden_dim)


# ── Policy ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _h, _c, _step_count, _prev_action_idx

    _load_once()

    if _step_count >= _MAX_EPISODE_STEPS:
        reset()

    feat = _encoder.encode(obs, _prev_action_idx)
    x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

    logits, (_h, _c) = _model(x, (_h, _c))
    action_idx       = int(logits.squeeze(0).argmax().item())

    _step_count      += 1
    _prev_action_idx  = action_idx
    return ACTIONS[action_idx]
