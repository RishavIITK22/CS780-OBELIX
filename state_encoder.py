from __future__ import annotations

import collections
from typing import Optional

import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACT = len(ACTIONS)
OBS_DIM = 18

# Legacy single-frame layout kept for reward shaper compatibility:
#   raw obs (18) + strengths (8) + dir_summary (3) + temporal (4) + prev_action (5)
_LEGACY_SINGLE_DIM = 38

# Richer core stacked for policy input:
#   raw obs (18)
#   sector strengths (8)
#   direction summary (3)
#   geometry / occupancy summary (8)
#   temporal memory (8)
#   transition cues (4)
_CORE_DIM = 49


class BeliefStateEncoder:
    """Richer belief-state encoder with built-in frame stacking for OBELIX.

    Design goals
    ------------
    - Preserve the old public API (`encode`, `encode_single`, `output_dim`).
    - Keep `encode_single()` backward-compatible at 38 dims for the existing
      reward shaper.
    - Enrich the stacked policy input with more geometry, persistence, and
      transition information that is helpful in a POMDP.

    Single-frame rich core (49 dims)
    --------------------------------
    1. Raw observation                 (18)
    2. Sector strengths                (8)  = 2 * near + far
    3. Direction summary               (3)  = left / front / right
    4. Geometry / occupancy summary    (8)
    5. Temporal memory                 (8)
    6. Transition cues                 (4)

    Network input
    -------------
    The 49-dim core is stacked for k timesteps and the current previous-action
    one-hot (5 dims) is appended once:

        output = [core_t-(k-1), ..., core_t, prev_action_oh_t]
        output_dim = k * 49 + 5

    Legacy compatibility
    --------------------
    `encode_single()` returns the old 38-dim layout so existing reward shaping
    code does not need to change.
    """

    CORE_DIM = _CORE_DIM
    LEGACY_SINGLE_DIM = _LEGACY_SINGLE_DIM

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
        self.output_dim = stack_k * _CORE_DIM + N_ACT
        self.reset()

    def reset(self) -> None:
        self._prev_ir = 0.0
        self._prev_stuck = 0.0
        self._prev_visible = 0.0
        self._steps_since_seen = self.max_steps_since_seen
        self._stuck_steps = 0
        self._seen_streak = 0
        self._lost_streak = 0
        self._prev_strengths = np.zeros(8, dtype=np.float32)
        self._prev_dir_summary = np.zeros(3, dtype=np.float32)
        self._prev_total_strength = 0.0
        self._prev_lr_balance = 0.0
        self._last_obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._last_legacy_temporal = np.zeros(4, dtype=np.float32)
        self._last_prev_action_oh = np.zeros(N_ACT, dtype=np.float32)

        # Pre-fill so the first encode() is well-defined.
        self._frames: collections.deque = collections.deque(
            [np.zeros(_CORE_DIM, dtype=np.float32) for _ in range(self.stack_k)],
            maxlen=self.stack_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, obs: np.ndarray, prev_action_idx: Optional[int] = None) -> np.ndarray:
        """Return the stacked rich belief vector for the policy network."""
        core, prev_act_oh, legacy_temporal = self._encode_frame(obs, prev_action_idx)
        self._frames.append(core)
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        self._last_prev_action_oh = prev_act_oh.copy()
        self._last_legacy_temporal = legacy_temporal.copy()
        return np.concatenate([*self._frames, prev_act_oh])

    def encode_single(self, obs: np.ndarray, prev_action_idx: Optional[int] = None) -> np.ndarray:
        """Return the legacy 38-dim single-frame vector.

        This stays compatible with the old reward shaper layout:
            [raw obs(18), strengths(8), dir_summary(3), temporal(4), prev_action(5)]
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        far, near, _, _ = self._unpack(obs)
        strengths = 2.0 * near + far
        dir_summary = self._dir_summary(strengths)

        # If called right after encode() on the same observation, use the exact
        # cached temporal edges from that transition. Otherwise fall back to a
        # non-mutating estimate from current internal state.
        if np.array_equal(obs, self._last_obs):
            temporal = self._last_legacy_temporal
            prev_act_oh = self._last_prev_action_oh if prev_action_idx is None else self._prev_action_oh(prev_action_idx)
        else:
            temporal = self._peek_legacy_temporal(obs)
            prev_act_oh = self._prev_action_oh(prev_action_idx)

        return np.concatenate([obs, strengths, dir_summary, temporal, prev_act_oh])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_frame(self, obs: np.ndarray, prev_action_idx: Optional[int]):
        """Compute rich core and update internal episode state."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        far, near, ir, stuck = self._unpack(obs)
        strengths = 2.0 * near + far
        dir_summary = self._dir_summary(strengths)

        total_strength = float(np.sum(strengths))
        left_strength = float(dir_summary[0])
        front_strength = float(dir_summary[1])
        right_strength = float(dir_summary[2])
        visible = 1.0 if (total_strength > 0.0 or ir > 0.0) else 0.0

        near_count = float(np.sum(near))
        far_count = float(np.sum(far))
        strongest_strength = float(np.max(strengths)) if strengths.size else 0.0
        strongest_sector = int(np.argmax(strengths)) if strengths.size else 0
        sector_mass = np.sum(strengths)
        if sector_mass > 0:
            sector_positions = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
            sector_centroid = float(np.dot(strengths, sector_positions) / sector_mass)
        else:
            sector_centroid = 0.0

        front_ratio = front_strength / max(total_strength, 1.0)
        left_ratio = left_strength / max(total_strength, 1.0)
        right_ratio = right_strength / max(total_strength, 1.0)
        lr_balance = (right_strength - left_strength) / max(total_strength, 1.0)

        just_got_ir = 1.0 if (self._prev_ir == 0.0 and ir == 1.0) else 0.0
        just_lost_ir = 1.0 if (self._prev_ir == 1.0 and ir == 0.0) else 0.0
        just_got_stuck = 1.0 if (self._prev_stuck == 0.0 and stuck == 1.0) else 0.0
        just_recovered = 1.0 if (self._prev_stuck == 1.0 and stuck == 0.0) else 0.0

        if visible > 0.0:
            self._steps_since_seen = 0
            self._seen_streak = min(self._seen_streak + 1, self.max_seen_streak)
            self._lost_streak = 0
        else:
            self._steps_since_seen = min(
                self._steps_since_seen + 1, self.max_steps_since_seen
            )
            self._lost_streak = min(self._lost_streak + 1, self.max_lost_streak)
            self._seen_streak = 0

        if stuck > 0.0:
            self._stuck_steps = min(self._stuck_steps + 1, self.max_stuck_steps)
        else:
            self._stuck_steps = 0

        legacy_temporal = np.array(
            [
                self._steps_since_seen / self.max_steps_since_seen,
                self._stuck_steps / self.max_stuck_steps,
                just_got_ir,
                just_recovered,
            ],
            dtype=np.float32,
        )

        geometry = np.array(
            [
                total_strength / 24.0,
                near_count / 8.0,
                far_count / 8.0,
                strongest_strength / 3.0,
                sector_centroid,
                front_ratio,
                left_ratio,
                right_ratio,
            ],
            dtype=np.float32,
        )

        temporal = np.array(
            [
                self._steps_since_seen / self.max_steps_since_seen,
                self._seen_streak / self.max_seen_streak,
                self._lost_streak / self.max_lost_streak,
                self._stuck_steps / self.max_stuck_steps,
                visible,
                just_got_ir,
                just_lost_ir,
                just_recovered,
            ],
            dtype=np.float32,
        )

        transition = np.array(
            [
                (total_strength - self._prev_total_strength) / 24.0,
                (front_strength - float(self._prev_dir_summary[1])) / 12.0,
                lr_balance - self._prev_lr_balance,
                visible - self._prev_visible,
            ],
            dtype=np.float32,
        )

        core = np.concatenate(
            [obs, strengths, dir_summary, geometry, temporal, transition]
        ).astype(np.float32)
        prev_act_oh = self._prev_action_oh(prev_action_idx)

        self._prev_ir = ir
        self._prev_stuck = stuck
        self._prev_visible = visible
        self._prev_strengths = strengths.copy()
        self._prev_dir_summary = dir_summary.copy()
        self._prev_total_strength = total_strength
        self._prev_lr_balance = lr_balance

        return core, prev_act_oh, legacy_temporal

    def _peek_legacy_temporal(self, obs: np.ndarray) -> np.ndarray:
        """Non-mutating temporal estimate for compatibility callers."""
        _, _, ir, stuck = self._unpack(obs)
        return np.array(
            [
                self._steps_since_seen / self.max_steps_since_seen,
                self._stuck_steps / self.max_stuck_steps,
                1.0 if (self._prev_ir == 0.0 and ir == 1.0) else 0.0,
                1.0 if (self._prev_stuck == 1.0 and stuck == 0.0) else 0.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _prev_action_oh(prev_action_idx: Optional[int]) -> np.ndarray:
        prev_act_oh = np.zeros(N_ACT, dtype=np.float32)
        if prev_action_idx is not None:
            prev_act_oh[prev_action_idx] = 1.0
        return prev_act_oh

    @staticmethod
    def _unpack(obs: np.ndarray):
        far = obs[0:16:2]
        near = obs[1:16:2]
        ir = float(obs[16])
        stuck = float(obs[17])
        return far, near, ir, stuck

    @staticmethod
    def _dir_summary(strengths: np.ndarray) -> np.ndarray:
        left = float(strengths[0] + strengths[1])
        front = float(strengths[2] + strengths[3] + strengths[4] + strengths[5])
        right = float(strengths[6] + strengths[7])
        return np.array([left, front, right], dtype=np.float32)
