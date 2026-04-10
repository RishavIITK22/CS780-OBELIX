from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FindShapeConfig:
    far_sensor_bonus: float = 0.002
    near_sensor_bonus: float = 0.005
    stuck_penalty_base: float = 0.01
    stuck_penalty_growth: float = 0.005
    max_stuck_penalty: float = 0.08


@dataclass
class PushShapeConfig:
    contact_bonus: float = 0.01
    forward_bonus: float = 0.005
    stuck_penalty_base: float = 0.01
    stuck_penalty_growth: float = 0.005
    max_stuck_penalty: float = 0.08


@dataclass
class UnwedgeShapeConfig:
    recover_bonus: float = 0.05
    repeat_turn_penalty: float = 0.01
    stuck_penalty_base: float = 0.01
    stuck_penalty_growth: float = 0.005
    max_stuck_penalty: float = 0.08


class NullRewardHook:
    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        del obs

    def shape(
        self,
        obs: np.ndarray,
        action_idx: int,
        next_obs: np.ndarray,
        raw_reward: float,
        done: bool,
    ) -> float:
        del obs, action_idx, next_obs, raw_reward, done
        return 0.0


class FindRewardHook:
    def __init__(self, config: Optional[FindShapeConfig] = None):
        self.config = config or FindShapeConfig()
        self._stuck_streak = 0

    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        obs = None if obs is None else np.asarray(obs).reshape(-1)
        self._stuck_streak = 1 if obs is not None and bool(obs[17]) else 0

    def shape(
        self,
        obs: np.ndarray,
        action_idx: int,
        next_obs: np.ndarray,
        raw_reward: float,
        done: bool,
    ) -> float:
        del obs, action_idx, raw_reward, done
        next_obs = np.asarray(next_obs).reshape(-1)
        far_count = float(np.sum(next_obs[0:16:2] > 0.5))
        near_count = float(np.sum(next_obs[1:16:2] > 0.5))

        reward = (
            self.config.far_sensor_bonus * far_count
            + self.config.near_sensor_bonus * near_count
        )

        if bool(next_obs[17]):
            self._stuck_streak += 1
            penalty = self.config.stuck_penalty_base + self.config.stuck_penalty_growth * (self._stuck_streak - 1)
            reward -= min(penalty, self.config.max_stuck_penalty)
        else:
            self._stuck_streak = 0

        return float(reward)


class PushRewardHook:
    def __init__(self, config: Optional[PushShapeConfig] = None):
        self.config = config or PushShapeConfig()
        self._stuck_streak = 0

    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        obs = None if obs is None else np.asarray(obs).reshape(-1)
        self._stuck_streak = 1 if obs is not None and bool(obs[17]) else 0

    def shape(
        self,
        obs: np.ndarray,
        action_idx: int,
        next_obs: np.ndarray,
        raw_reward: float,
        done: bool,
    ) -> float:
        del obs, raw_reward, done
        next_obs = np.asarray(next_obs).reshape(-1)
        reward = 0.0
        if bool(next_obs[16]):
            reward += self.config.contact_bonus
            if action_idx == 2 and not bool(next_obs[17]):
                reward += self.config.forward_bonus
        if bool(next_obs[17]):
            self._stuck_streak += 1
            penalty = self.config.stuck_penalty_base + self.config.stuck_penalty_growth * (self._stuck_streak - 1)
            reward -= min(penalty, self.config.max_stuck_penalty)
        else:
            self._stuck_streak = 0
        return reward


class UnwedgeRewardHook:
    def __init__(self, config: Optional[UnwedgeShapeConfig] = None):
        self.config = config or UnwedgeShapeConfig()
        self._last_turn: int = 0
        self._stuck_streak: int = 0

    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        obs = None if obs is None else np.asarray(obs).reshape(-1)
        self._last_turn = 0
        self._stuck_streak = 1 if obs is not None and bool(obs[17]) else 0

    def shape(
        self,
        obs: np.ndarray,
        action_idx: int,
        next_obs: np.ndarray,
        raw_reward: float,
        done: bool,
    ) -> float:
        del raw_reward, done
        obs = np.asarray(obs).reshape(-1)
        next_obs = np.asarray(next_obs).reshape(-1)
        reward = 0.0

        if bool(obs[17]) and not bool(next_obs[17]):
            reward += self.config.recover_bonus

        turn_dir = 0
        if action_idx in (0, 1):
            turn_dir = -1
        elif action_idx in (3, 4):
            turn_dir = 1

        if bool(next_obs[17]) and turn_dir != 0 and turn_dir == self._last_turn:
            reward -= self.config.repeat_turn_penalty

        if bool(next_obs[17]):
            self._stuck_streak += 1
            penalty = self.config.stuck_penalty_base + self.config.stuck_penalty_growth * (self._stuck_streak - 1)
            reward -= min(penalty, self.config.max_stuck_penalty)
        else:
            self._stuck_streak = 0

        self._last_turn = turn_dir
        return reward
