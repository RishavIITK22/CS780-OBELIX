from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


FIND = "find"
PUSH = "push"
UNWEDGE = "unwedge"
BEHAVIORS = (FIND, PUSH, UNWEDGE)


@dataclass
class BehaviorManagerConfig:
    push_linger_steps: int = 5
    unwedge_linger_steps: int = 5
    attach_reward_threshold: float = 90.0
    sticky_push: bool = True
    activate_push_on_ir: bool = False


class BehaviorManager:
    """Paper-style arbitration for OBELIX.

    Priority order:
        unwedge > push > find

    Notes
    -----
    - In the simulator, attachment is most reliably detected from the reward
      spike on first contact, so that is used to activate push.
    - The original paper keeps behaviors active for a few extra timesteps after
      the applicability predicate stops being true. We mirror that with linger
      timers for push and unwedge.
    - Because the simulator latches box attachment internally until success,
      `sticky_push=True` keeps push active until reset once attachment happens.
    """

    def __init__(self, config: Optional[BehaviorManagerConfig] = None):
        self.config = config or BehaviorManagerConfig()
        self.reset()

    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        self._push_active = False
        self._push_timer = 0
        self._unwedge_timer = 0
        if obs is not None and self._is_stuck(obs):
            self._unwedge_timer = self.config.unwedge_linger_steps

    def current_behavior(self, obs: np.ndarray) -> str:
        if self._is_stuck(obs) or self._unwedge_timer > 0:
            return UNWEDGE
        if self._push_active or self._push_timer > 0:
            return PUSH
        return FIND

    def update(self, obs: np.ndarray, raw_reward: float, done: bool) -> None:
        if done:
            return

        stuck = self._is_stuck(obs)
        ir_on = self._has_ir(obs)

        if (
            raw_reward >= self.config.attach_reward_threshold
            or (self.config.activate_push_on_ir and ir_on)
        ):
            self._push_active = True
            self._push_timer = self.config.push_linger_steps
        elif self._push_active and not self.config.sticky_push:
            if ir_on:
                self._push_timer = self.config.push_linger_steps
            elif self._push_timer > 0:
                self._push_timer -= 1
            else:
                self._push_active = False
        elif self._push_timer > 0:
            self._push_timer -= 1

        if stuck:
            self._unwedge_timer = self.config.unwedge_linger_steps
        elif self._unwedge_timer > 0:
            self._unwedge_timer -= 1

    @staticmethod
    def _has_ir(obs: np.ndarray) -> bool:
        obs = np.asarray(obs).reshape(-1)
        return bool(obs[16])

    @staticmethod
    def _is_stuck(obs: np.ndarray) -> bool:
        obs = np.asarray(obs).reshape(-1)
        return bool(obs[17])
