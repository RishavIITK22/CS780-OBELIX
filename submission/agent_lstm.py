"""Submission agent — PPO + LSTM.

The evaluator imports this file and calls policy(obs, rng) once per step.
Action space (strings): 'L45', 'L22', 'FW', 'R22', 'R45'
Observation: numpy array shape (18,), values 0/1.

Architecture (mirrors train_ppo_lstm.py exactly):
  - Encoder : Linear(18, 64) + Tanh
  - LSTM    : input=64, hidden=h (inferred from checkpoint)
  - Actor   : Linear(h, 5)       (critic loaded but unused at inference)
  - Inference: deterministic — argmax over actor logits

Hidden size h is inferred from checkpoint weight shapes so this file
never needs editing after retraining with a different --hidden value.

Episode boundary handling:
  The Codabench evaluator imports this module ONCE and calls policy_fn for
  ALL episodes without calling reset() between them.  The LSTM hidden state
  (h, c) would carry stale context across episode boundaries.
  Fix: a step counter auto-resets h, c, and the push latch after
  _MAX_EPISODE_STEPS steps, matching the evaluator's max_steps=1000.
  For episodes that end early (success), the first few steps of the next
  episode start with a slightly stale hidden state, but this is minor.
  Harnesses that do call reset() get perfectly clean episode starts.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Fixed constants ────────────────────────────────────────────────────────────
ACTIONS:            Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_OBS_DIM:           int           = 18    # raw obs length — always 18 for OBELIX
_ENC_DIM:           int           = 64    # encoder output size (hardcoded in trainer)
_N_ACTIONS:         int           = len(ACTIONS)
_MAX_EPISODE_STEPS: int           = 1000  # matches evaluator's max_steps


# ── Network (must mirror ActorCriticLSTM in train_ppo_lstm.py) ────────────────
class ActorCriticLSTM(nn.Module):
    """Encoder → LSTM → Actor / Critic.

    The critic head is included so load_state_dict(strict=True) succeeds
    without manually filtering checkpoint keys.
    """

    def __init__(self, hidden_dim: int, n_actions: int = _N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(_OBS_DIM, _ENC_DIM),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size=_ENC_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.actor  = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs:    torch.Tensor,                       # (1, 18) — single step
        hidden: Tuple[torch.Tensor, torch.Tensor],  # ((1,1,H), (1,1,H))
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        enc = self.encoder(obs).unsqueeze(1)        # (1, 1, 64)
        out, new_hidden = self.lstm(enc, hidden)    # out: (1, 1, H)
        out = out.squeeze(1)                        # (1, H)
        return self.actor(out), self.critic(out).squeeze(-1), new_hidden


# ── Module-level persistent state ─────────────────────────────────────────────
_model:       Optional[ActorCriticLSTM]          = None
_hidden_dim:  int                                = 0
_h:           Optional[torch.Tensor]             = None   # (1, 1, H)
_c:           Optional[torch.Tensor]             = None   # (1, 1, H)
_push_active: bool                               = False  # sticky IR-contact latch
_step_count:  int                                = 0      # steps since last reset


def reset() -> None:
    """Reset episode state. Call at the start of each new episode.

    Zeroes the LSTM hidden state and push latch so episode N's memory does
    not bleed into episode N+1.  If the evaluator does not call this,
    policy() auto-resets via _step_count after _MAX_EPISODE_STEPS steps.
    """
    global _push_active, _step_count, _h, _c
    if _model is not None:
        _h = torch.zeros(1, 1, _hidden_dim)
        _c = torch.zeros(1, 1, _hidden_dim)
    _push_active = False
    _step_count  = 0


# ── Weight loader ──────────────────────────────────────────────────────────────
def _load_once() -> None:
    """Load checkpoint and build model with hidden_dim inferred from weights.

    hidden_dim is inferred from lstm.weight_ih_l0, which has shape
    (4 * hidden_dim, enc_dim) — the factor of 4 comes from the four LSTM
    gates (input, forget, cell, output) concatenated along dim 0.
    This makes the agent work with any --hidden value without code changes.
    """
    global _model, _hidden_dim, _h, _c

    if _model is not None:
        return

    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights_ppo_lstm.pth")
    if not os.path.exists(wpath):
        wpath_fallback = os.path.join(here, "weights.pth")
        if os.path.exists(wpath_fallback):
            wpath = wpath_fallback
        else:
            raise FileNotFoundError(
                "Neither 'weights_ppo_lstm.pth' nor 'weights.pth' found next to agent.py. "
                "Train with train_ppo_lstm.py and place the weights file in the same directory."
            )

    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # lstm.weight_ih_l0 shape: (4 * hidden_dim, enc_dim)
    # Divide axis-0 by 4 to recover hidden_dim.
    ih = sd["lstm.weight_ih_l0"]     # shape: (4*H, enc_dim)
    if ih.shape[0] % 4 != 0:
        raise ValueError(
            f"lstm.weight_ih_l0 has {ih.shape[0]} rows, expected a multiple of 4. "
            "Checkpoint may be from a different architecture."
        )
    hidden_dim = ih.shape[0] // 4

    m = ActorCriticLSTM(hidden_dim=hidden_dim)
    m.load_state_dict(sd, strict=True)
    m.eval()

    _model      = m
    _hidden_dim = hidden_dim
    _h          = torch.zeros(1, 1, hidden_dim)
    _c          = torch.zeros(1, 1, hidden_dim)


# ── Policy ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return a deterministic action for the given 18-dim observation.

    The LSTM hidden state (h, c) is carried across calls so the network has
    temporal context over the episode, matching training behaviour.

    Episode auto-reset: after _MAX_EPISODE_STEPS steps, h and c are zeroed
    and the push latch is cleared — handling the Codabench evaluator which
    does not call reset() between episodes.
    """
    global _push_active, _step_count, _h, _c

    _load_once()   # no-op after the first call

    # ── Auto-reset at episode boundary ────────────────────────────────────────
    if _step_count >= _MAX_EPISODE_STEPS:
        _h           = torch.zeros(1, 1, _hidden_dim)
        _c           = torch.zeros(1, 1, _hidden_dim)
        _push_active = False
        _step_count  = 0

    # ── Single-step LSTM forward ──────────────────────────────────────────────
    x              = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 18)
    logits, _, (_h, _c) = _model(x, (_h, _c))                            # logits: (1, 5)
    _step_count   += 1

    # ── Sticky push latch (mirrors env.enable_push) ───────────────────────────
    if obs[16]:
        _push_active = True

    # ── Deterministic inference ───────────────────────────────────────────────
    action_idx = int(logits.squeeze(0).argmax().item())
    return ACTIONS[action_idx]
