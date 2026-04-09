from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PushShapeConfig:
    contact_bonus: float = 0.01
    forward_bonus: float = 0.005
    stuck_penalty: float = 0.01


@dataclass
class UnwedgeShapeConfig:
    recover_bonus: float = 0.05
    repeat_turn_penalty: float = 0.01


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


class PushRewardHook:
    def __init__(self, config: Optional[PushShapeConfig] = None):
        self.config = config or PushShapeConfig()

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
        del obs, raw_reward, done
        next_obs = np.asarray(next_obs).reshape(-1)
        reward = 0.0
        if bool(next_obs[16]):
            reward += self.config.contact_bonus
            if action_idx == 2 and not bool(next_obs[17]):
                reward += self.config.forward_bonus
        if bool(next_obs[17]):
            reward -= self.config.stuck_penalty
        return reward


class UnwedgeRewardHook:
    def __init__(self, config: Optional[UnwedgeShapeConfig] = None):
        self.config = config or UnwedgeShapeConfig()
        self._last_turn: int = 0

    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        del obs
        self._last_turn = 0

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

        self._last_turn = turn_dir
        return reward

