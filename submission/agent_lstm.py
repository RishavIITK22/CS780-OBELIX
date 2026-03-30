"""Submission agent — PPO + LSTM with BeliefStateEncoder.

The evaluator imports this file and calls policy(obs, rng) once per step.
Action space (strings): 'L45', 'L22', 'FW', 'R22', 'R45'
Observation: numpy array shape (18,), values 0/1.

Architecture (mirrors train_ppo_lstm.py):
  - BeliefStateEncoder : 18-dim obs  ->  38-dim encoded feature
  - ActorCriticLSTM    : encoder(38 -> 128 -> 64) + LSTM(64, H) + actor/critic
  - Inference          : deterministic argmax over actor logits

Episode boundary:
  Codabench calls policy() for ALL episodes without calling reset() between
  them. We auto-reset LSTM hidden state and encoder after _MAX_EPISODE_STEPS.
  Harnesses that call reset() explicitly get perfectly clean episode starts.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIONS:            Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_N_ACTIONS:         int           = len(ACTIONS)
_MAX_EPISODE_STEPS: int           = 1000


# ── BeliefStateEncoder (must mirror state_encoder.py) ─────────────────────────
class _BeliefStateEncoder:
    """Inline copy of BeliefStateEncoder so the agent file is self-contained."""

    def __init__(self, max_steps_since_seen: int = 30, max_stuck_steps: int = 20):
        self.max_steps_since_seen = max_steps_since_seen
        self.max_stuck_steps      = max_stuck_steps
        self.reset()

    def reset(self) -> None:
        self._prev_ir          = 0.0
        self._prev_stuck       = 0.0
        self._steps_since_seen = self.max_steps_since_seen
        self._stuck_steps      = 0

    def encode(self, obs: np.ndarray, prev_action_idx: Optional[int] = None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        far   = obs[0:16:2]
        near  = obs[1:16:2]
        ir    = float(obs[16])
        stuck = float(obs[17])

        strengths   = 2.0 * near + far
        left        = float(strengths[0] + strengths[1])
        front       = float(strengths[2] + strengths[3] + strengths[4] + strengths[5])
        right       = float(strengths[6] + strengths[7])
        dir_summary = np.array([left, front, right], dtype=np.float32)

        if float(np.sum(strengths)) > 0 or ir > 0:
            self._steps_since_seen = 0
        else:
            self._steps_since_seen = min(
                self._steps_since_seen + 1, self.max_steps_since_seen
            )

        if stuck > 0:
            self._stuck_steps = min(self._stuck_steps + 1, self.max_stuck_steps)
        else:
            self._stuck_steps = 0

        just_got_ir    = 1.0 if (self._prev_ir == 0.0 and ir == 1.0) else 0.0
        just_recovered = 1.0 if (self._prev_stuck == 1.0 and stuck == 0.0) else 0.0

        temporal = np.array([
            self._steps_since_seen / self.max_steps_since_seen,
            self._stuck_steps      / self.max_stuck_steps,
            just_got_ir,
            just_recovered,
        ], dtype=np.float32)

        prev_act_oh = np.zeros(_N_ACTIONS, dtype=np.float32)
        if prev_action_idx is not None:
            prev_act_oh[prev_action_idx] = 1.0

        self._prev_ir    = ir
        self._prev_stuck = stuck

        return np.concatenate([obs, strengths, dir_summary, temporal, prev_act_oh])


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
        enc = self.encoder(x).unsqueeze(1)
        out, new_hidden = self.lstm(enc, hidden)
        logits = self.actor(out.squeeze(1))
        return logits, new_hidden


# ── Module-level persistent state ─────────────────────────────────────────────
_model:           Optional[_ActorCriticLSTM] = None
_hidden_dim:      int                        = 0
_h:               Optional[torch.Tensor]     = None
_c:               Optional[torch.Tensor]     = None
_encoder:         _BeliefStateEncoder        = _BeliefStateEncoder()
_step_count:      int                        = 0
_prev_action_idx: int                        = 2   # default FW


def reset() -> None:
    """Reset episode state. Call at the start of each new episode."""
    global _h, _c, _step_count, _prev_action_idx
    if _model is not None:
        _h = torch.zeros(1, 1, _hidden_dim)
        _c = torch.zeros(1, 1, _hidden_dim)
    _encoder.reset()
    _step_count      = 0
    _prev_action_idx = 2


# ── Weight loader ─────────────────────────────────────────────────────────────
def _load_once() -> None:
    global _model, _hidden_dim, _h, _c

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

    input_dim  = sd["encoder.0.weight"].shape[1]
    hidden_dim = sd["lstm.weight_ih_l0"].shape[0] // 4

    model = _ActorCriticLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(sd, strict=True)
    model.eval()

    _model      = model
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
