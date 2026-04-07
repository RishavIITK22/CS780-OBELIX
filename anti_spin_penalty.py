from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


LEFT_ACTIONS = {0, 1}
RIGHT_ACTIONS = {3, 4}


@dataclass
class AntiSpinPenaltyConfig:
    same_turn_threshold: int = 4
    same_turn_penalty: float = 0.01
    alternating_window: int = 6
    alternating_penalty: float = 0.015
    turn_ratio_window: int = 8
    turn_ratio_threshold: float = 0.75
    turn_ratio_penalty: float = 0.01
    low_progress_obs_delta: float = 0.05


class AntiSpinPenaltyTracker:
    """Per-environment anti-spin penalty tracker.

    The tracker is intentionally lightweight and observation-based:
    - repeated same-direction turning
    - alternating left/right turn loops
    - high recent turn ratio with very little observation change

    Penalties are only meant to bias the `find` policy away from easy wall-spin
    attractors, so callers should typically apply them only during `find`.
    """

    def __init__(self, config: Optional[AntiSpinPenaltyConfig] = None):
        self.config = config or AntiSpinPenaltyConfig()
        self.reset()

    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        self._same_turn_streak = 0
        self._last_turn_dir = 0
        self._recent_dirs: Deque[int] = collections.deque(
            maxlen=self.config.turn_ratio_window
        )
        self._recent_progress: Deque[float] = collections.deque(
            maxlen=self.config.turn_ratio_window
        )
        self._recent_obs_delta: Deque[float] = collections.deque(
            maxlen=self.config.turn_ratio_window
        )
        self._last_obs = None if obs is None else np.asarray(obs, dtype=np.float32).reshape(-1)

    def step(
        self,
        behavior: str,
        obs: np.ndarray,
        action_idx: int,
        next_obs: np.ndarray,
        raw_reward: float,
        attach_reward_threshold: float,
    ) -> float:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        turn_dir = self._turn_dir(action_idx)
        if turn_dir != 0:
            if turn_dir == self._last_turn_dir:
                self._same_turn_streak += 1
            else:
                self._same_turn_streak = 1
            self._last_turn_dir = turn_dir
        else:
            self._same_turn_streak = 0
            self._last_turn_dir = 0

        obs_delta = float(np.mean(np.abs(next_obs - obs)))
        made_progress = float(
            raw_reward >= attach_reward_threshold
            or bool(obs[16])
            or bool(next_obs[16])
        )

        self._recent_dirs.append(turn_dir)
        self._recent_progress.append(made_progress)
        self._recent_obs_delta.append(obs_delta)
        self._last_obs = next_obs

        if behavior != "find" or made_progress:
            return 0.0

        penalty = 0.0
        if self._same_turn_streak >= self.config.same_turn_threshold:
            penalty -= self.config.same_turn_penalty * (
                self._same_turn_streak - self.config.same_turn_threshold + 1
            )

        if self._is_alternating_turn_loop():
            penalty -= self.config.alternating_penalty

        if self._has_high_turn_ratio_low_progress_loop():
            penalty -= self.config.turn_ratio_penalty

        return penalty

    @staticmethod
    def _turn_dir(action_idx: int) -> int:
        if action_idx in LEFT_ACTIONS:
            return -1
        if action_idx in RIGHT_ACTIONS:
            return 1
        return 0

    def _is_alternating_turn_loop(self) -> bool:
        window = self.config.alternating_window
        if len(self._recent_dirs) < window:
            return False

        dirs = list(self._recent_dirs)[-window:]
        if any(d == 0 for d in dirs):
            return False
        if any(self._recent_progress):
            return False

        return all(dirs[i] == -dirs[i - 1] for i in range(1, len(dirs)))

    def _has_high_turn_ratio_low_progress_loop(self) -> bool:
        window = self.config.turn_ratio_window
        if len(self._recent_dirs) < window:
            return False

        dirs = np.asarray(list(self._recent_dirs)[-window:], dtype=np.int32)
        progress = np.asarray(list(self._recent_progress)[-window:], dtype=np.float32)
        obs_delta = np.asarray(list(self._recent_obs_delta)[-window:], dtype=np.float32)

        turn_ratio = float(np.mean(dirs != 0))
        mean_progress = float(np.mean(progress))
        mean_obs_delta = float(np.mean(obs_delta))

        return (
            turn_ratio >= self.config.turn_ratio_threshold
            and mean_progress <= 0.0
            and mean_obs_delta <= self.config.low_progress_obs_delta
        )
