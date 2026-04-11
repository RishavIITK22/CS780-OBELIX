"""agent_epb.py — Inference agent for the Explicit Phase Belief (EPB) policy.

Self-contained: all classes are inlined so this file can be submitted as-is.

Architecture:
  Input  : [raw obs (18)] + [compact belief (16)] + [fsm suggestion (5)] = 39 dims
  Network: MLP trunk → dual-head actor (find / push) + soft gate + critic
  Memory : CompactBeliefState — explicit, non-decaying belief variables.
           Unlike GRU hidden states, the last-known box direction NEVER decays
           on blank (no-sensor) steps.

The gate uses belief.attached (hard binary) at inference time, since we don't
have access to the reward signal that supervised the soft gate during training.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn

# ── constants ─────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cpu")
ACTIONS    = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS  = len(ACTIONS)
BELIEF_DIM = 16
IN_DIM     = 18 + BELIEF_DIM + N_ACTIONS   # 39

_MAX_INFER_STEPS = 1000   # auto-reset after this many steps (1 episode)


# ── observation parser ────────────────────────────────────────────────────────

def _parse_obs(obs):
    obs = np.asarray(obs, dtype=int)
    lf  = int(obs[0] or obs[2])
    ln  = int(obs[1] or obs[3])
    ff  = int(obs[4] or obs[6] or obs[8] or obs[10])
    fn  = int(obs[5] or obs[7] or obs[9] or obs[11])
    rf  = int(obs[12] or obs[14])
    rn  = int(obs[13] or obs[15])
    ir  = int(obs[16])
    stk = int(obs[17])
    any_s = int(any(obs[:17]))
    return lf, ln, ff, fn, rf, rn, ir, stk, any_s


# ── compact belief state ──────────────────────────────────────────────────────

class CompactBeliefState:
    """Explicit non-decaying belief state.  See compact_belief.py for details."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.attached            = False
        self.ir_on_steps         = 0    # consecutive steps IR has been active
        self.last_visible_left   = 0
        self.last_visible_front  = 0
        self.last_visible_right  = 0
        self.last_visible_ir     = 0
        self.steps_since_visible = 500
        self.consecutive_stuck   = 0
        self.just_recovered      = False
        self._prev_stuck         = False
        self.spin_dir            = random.choice([0, 1])
        self.blind_steps         = 0
        self.step_count          = 0
        self.push_stuck_count    = 0
        self.steps_in_phase      = 0
        self.last_action         = "FW"

    def update(self, obs, action: str) -> None:
        lf, ln, ff, fn, rf, rn, ir, stk, any_s = _parse_obs(obs)
        self.step_count += 1

        if not self.attached:
            if ir:
                # Count consecutive steps IR is active — regardless of action.
                # The old heuristic (ir AND last_action==FW) was too strict:
                # when coming from a spin (last_action=L22), the first IR step
                # never counted and the counter kept resetting on moving boxes.
                self.ir_on_steps += 1
                if self.ir_on_steps >= 2:
                    self.attached        = True
                    self.steps_in_phase  = 0
                    self.push_stuck_count = 0
            else:
                self.ir_on_steps = 0

            # Backup: stuck while last observation had IR → likely pushed into wall
            # with attached box (attachment detection was late).
            if stk and self.last_visible_ir and not self.attached:
                self.attached        = True
                self.steps_in_phase  = 0
                self.push_stuck_count = 0

        if any_s:
            self.last_visible_left  = int(bool(lf or ln))
            self.last_visible_front = int(bool(ff or fn or ir))
            self.last_visible_right = int(bool(rf or rn))
            self.last_visible_ir    = ir
            self.steps_since_visible = 0
            self.blind_steps        = 0
        else:
            self.steps_since_visible = min(self.steps_since_visible + 1, 500)
            if not self.attached:
                self.blind_steps += 1
                if self.blind_steps > 0 and self.blind_steps % 100 == 0:
                    self.spin_dir = 1 - self.spin_dir

        self.just_recovered = bool(self._prev_stuck and not stk)
        self._prev_stuck    = bool(stk)
        if stk:
            self.consecutive_stuck += 1
        else:
            self.consecutive_stuck = 0

        if self.attached:
            self.push_stuck_count = (self.push_stuck_count + 1) if stk else 0

        self.steps_in_phase += 1
        self.last_action     = action

    def to_vector(self) -> np.ndarray:
        return np.array([
            float(self.attached),
            float(self.last_visible_left),
            float(self.last_visible_front),
            float(self.last_visible_right),
            float(self.last_visible_ir),
            min(self.steps_since_visible / 200.0, 1.0),
            min(self.consecutive_stuck / 10.0,    1.0),
            float(self.just_recovered),
            float(self.spin_dir),
            min(self.step_count / 1000.0,         1.0),
            min(self.blind_steps / 100.0,         1.0),
            min(self.push_stuck_count / 5.0,      1.0),
            min(self.steps_in_phase / 200.0,      1.0),
            float(self._prev_stuck),
            min(self.ir_on_steps / 2.0,           1.0),   # attachment confidence
            float(self.steps_since_visible == 0),
        ], dtype=np.float32)

    def fsm_suggest(self, obs) -> str:
        lf, ln, ff, fn, rf, rn, ir, stk, _ = _parse_obs(obs)

        if self.attached:
            if stk:
                return "L45" if (self.push_stuck_count % 4) < 2 else "R45"
            return "FW"

        if stk:
            return "L45" if (self.consecutive_stuck % 4) < 2 else "R45"
        if ir:   return "FW"
        if fn:   return "FW"
        if ff:
            if rn and not ln:  return "L22"
            if ln and not rn:  return "R22"
            return "FW"
        if ln and not rn:  return "L22"
        if rn and not ln:  return "R22"
        if lf and not rf:  return "L22"
        if rf and not lf:  return "R22"
        return "L22" if self.spin_dir else "R22"


def _build_input(obs, belief: CompactBeliefState) -> np.ndarray:
    obs_arr   = np.asarray(obs, dtype=np.float32)
    fsm_act   = belief.fsm_suggest(obs)
    fsm_oh    = np.zeros(N_ACTIONS, dtype=np.float32)
    fsm_oh[ACTIONS.index(fsm_act)] = 1.0
    return np.concatenate([obs_arr, belief.to_vector(), fsm_oh])


# ── network ───────────────────────────────────────────────────────────────────

class DualHeadActorCritic(nn.Module):
    def __init__(self, in_dim: int = IN_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.find_actor = nn.Linear(hidden, N_ACTIONS)
        self.push_actor = nn.Linear(hidden, N_ACTIONS)
        self.gate_fc    = nn.Linear(hidden, 1)
        self.critic     = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h    = self.trunk(x)
        gate = torch.sigmoid(self.gate_fc(h))
        logits = (1.0 - gate) * self.find_actor(h) + gate * self.push_actor(h)
        return logits, self.critic(h).squeeze(-1), gate.squeeze(-1)

    def find_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.find_actor(self.trunk(x))

    def push_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.push_actor(self.trunk(x))


# ── weight loading ────────────────────────────────────────────────────────────

_here          = os.path.dirname(os.path.abspath(__file__))
_weights_path  = os.path.join(_here, "weights_epb.pth")
_payload       = torch.load(_weights_path, map_location=DEVICE, weights_only=False)
_cfg           = _payload.get("config", {})
_net           = DualHeadActorCritic(
    in_dim = int(_cfg.get("in_dim",  IN_DIM)),
    hidden = int(_cfg.get("hidden",  128)),
).to(DEVICE)
_net.load_state_dict(
    _payload["model_state_dict"] if "model_state_dict" in _payload else _payload
)
_net.eval()


# ── episode state ─────────────────────────────────────────────────────────────

_belief     = CompactBeliefState()
_step_count = 0


def reset_agent() -> None:
    """Call between episodes to reset belief state and step counter."""
    global _belief, _step_count
    _belief.reset()
    _step_count = 0


# ── policy ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def policy(obs, rng=None) -> str:
    """Map observation vector to one of: 'L45', 'L22', 'FW', 'R22', 'R45'."""
    global _step_count

    # Auto-reset at episode boundary (evaluate.py doesn't call reset_agent)
    if _step_count >= _MAX_INFER_STEPS:
        reset_agent()

    obs_arr = np.asarray(obs, dtype=int)
    ir  = int(obs_arr[16])
    stk = int(obs_arr[17])

    # ── Hard rules (no network needed) ────────────────────────────────
    # Rule 1: IR active + not stuck → always move forward.
    #   This is the correct action whether the box is about to be attached
    #   or is already attached.  The FSM bias during training was doing this
    #   softly; making it hard removes all ambiguity at inference.
    if ir and not stk:
        action = "FW"
        _belief.update(obs, action)
        _step_count += 1
        return action

    # Rule 2: attached + not stuck → push forward unconditionally.
    #   The push_actor wasn't given enough training signal to be reliable.
    #   Hard FW is the universally correct push action when the path is clear.
    if _belief.attached and not stk:
        action = "FW"
        _belief.update(obs, action)
        _step_count += 1
        return action

    # Rule 3: stuck → alternating turns to unwedge.
    if stk:
        cnt    = _belief.push_stuck_count if _belief.attached else _belief.consecutive_stuck
        action = "L45" if (cnt % 4) < 2 else "R45"
        _belief.update(obs, action)
        _step_count += 1
        return action

    # ── Network: find phase only (no IR, not attached, not stuck) ─────
    x_np   = _build_input(obs, _belief)
    x      = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits = _net.find_logits(x)
    action = ACTIONS[int(torch.argmax(logits, dim=-1).item())]

    _belief.update(obs, action)
    _step_count += 1
    return action
