from __future__ import annotations

import collections
from typing import Optional
import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACT   = len(ACTIONS)
OBS_DIM = 18

# Dimension of one encoded frame (without prev_action_oh).
# Stacking operates on this core; prev_action is appended once at the end.
_CORE_DIM = 33   # 18 + 8 + 3 + 4


class BeliefStateEncoder:
    """Belief encoder with built-in frame stacking for OBELIX (POMDP).

    Each raw 18-bit obs is first enriched into a 38-dim single frame:
        raw obs           (18)  — sonar bits, IR, stuck
        sector strengths   (8)  — 2*near + far per sector
        direction summary  (3)  — left / front / right aggregates
        temporal memory    (4)  — steps_since_seen, stuck_steps (normed),
                                   just_got_ir, just_recovered
        prev action OH     (5)  — current-step action context

    The 33-dim core (all but prev_action_oh) is pushed onto a deque of depth k.
    The network input is the stacked cores concatenated with the current
    prev_action_oh appended once:

        output = [core_t-(k-1), ..., core_t-1, core_t, prev_action_oh_t]
        output_dim = k * 33 + 5

    Why stack the core and not raw obs:
      - Stacking raw 18-bit obs repeats redundant zeros during box-invisible
        phases — the network can't distinguish "was visible 3 steps ago" from
        "hasn't been visible for 20 steps".
      - Stacking the core gives the network access to the direction summary,
        sector strengths, and temporal counters at each past timestep — a true
        history of *what the agent perceived and how long things lasted*.

    Why prev_action_oh only once (not stacked):
      - The network already has the stacked core; stacking prev_action would
        duplicate information already implicit in the temporal features.
      - One prev_action_oh is enough for the actor to condition its next move.

    The reward shaper always reads the current single-frame 38-dim vector
    via encode_single(), which uses fixed indices and is unaffected by k.

    Parameters
    ----------
    stack_k : int
        Number of frames to stack. k=1 reduces to the original flat encoder.
    max_steps_since_seen : int
    max_stuck_steps : int
    """

    CORE_DIM = _CORE_DIM  # 33, exposed for external use

    def __init__(
        self,
        stack_k:              int = 4,
        max_steps_since_seen: int = 30,
        max_stuck_steps:      int = 20,
    ):
        self.stack_k              = stack_k
        self.max_steps_since_seen = max_steps_since_seen
        self.max_stuck_steps      = max_stuck_steps
        self.output_dim           = stack_k * _CORE_DIM + N_ACT  # e.g. 4*33+5 = 137
        self.reset()

    def reset(self) -> None:
        self._prev_ir          = 0.0
        self._prev_stuck       = 0.0
        self._steps_since_seen = self.max_steps_since_seen
        self._stuck_steps      = 0
        # Pre-fill deque with zeros so first encode() is well-defined
        self._frames: collections.deque = collections.deque(
            [np.zeros(_CORE_DIM, dtype=np.float32)] * self.stack_k,
            maxlen=self.stack_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, obs: np.ndarray, prev_action_idx: Optional[int] = None) -> np.ndarray:
        """Return stacked encoded vector for the network.

        Shape: (output_dim,) = (stack_k * 33 + 5,)
        """
        core, prev_act_oh = self._encode_frame(obs, prev_action_idx)
        self._frames.append(core)
        # [oldest ... newest, prev_action_oh]
        return np.concatenate([*self._frames, prev_act_oh])

    def encode_single(self, obs: np.ndarray, prev_action_idx: Optional[int] = None) -> np.ndarray:
        """Return the current single-frame 38-dim vector (for reward shaping).

        Does NOT update internal state — call encode() for that.
        Uses the temporal state already updated by the most recent encode() call.

        Shape: (38,)  layout identical to the old flat encoder.
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        far, near, ir, stuck = self._unpack(obs)
        strengths   = 2.0 * near + far
        dir_summary = self._dir_summary(strengths)
        temporal    = np.array([
            self._steps_since_seen / self.max_steps_since_seen,
            self._stuck_steps      / self.max_stuck_steps,
            1.0 if (self._prev_ir == 0.0 and ir == 1.0) else 0.0,
            1.0 if (self._prev_stuck == 1.0 and stuck == 0.0) else 0.0,
        ], dtype=np.float32)
        prev_act_oh = np.zeros(N_ACT, dtype=np.float32)
        if prev_action_idx is not None:
            prev_act_oh[prev_action_idx] = 1.0
        return np.concatenate([obs, strengths, dir_summary, temporal, prev_act_oh])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_frame(
        self, obs: np.ndarray, prev_action_idx: Optional[int]
    ):
        """Compute core (33,) and prev_action_oh (5,), update temporal state."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        far, near, ir, stuck = self._unpack(obs)
        strengths   = 2.0 * near + far
        dir_summary = self._dir_summary(strengths)

        # Update temporal counters
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

        self._prev_ir    = ir
        self._prev_stuck = stuck

        prev_act_oh = np.zeros(N_ACT, dtype=np.float32)
        if prev_action_idx is not None:
            prev_act_oh[prev_action_idx] = 1.0

        core = np.concatenate([obs, strengths, dir_summary, temporal])  # (33,)
        return core, prev_act_oh

    @staticmethod
    def _unpack(obs: np.ndarray):
        far   = obs[0:16:2]
        near  = obs[1:16:2]
        ir    = float(obs[16])
        stuck = float(obs[17])
        return far, near, ir, stuck

    @staticmethod
    def _dir_summary(strengths: np.ndarray) -> np.ndarray:
        left  = float(strengths[0] + strengths[1])
        front = float(strengths[2] + strengths[3] + strengths[4] + strengths[5])
        right = float(strengths[6] + strengths[7])
        return np.array([left, front, right], dtype=np.float32)
