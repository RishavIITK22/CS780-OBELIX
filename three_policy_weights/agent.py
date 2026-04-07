"""Inference agent for three-policy OBELIX PPO.

Expected files next to this agent:
    three_policy_weights/weights_find.pth
    three_policy_weights/weights_push.pth
    three_policy_weights/weights_unwedge.pth
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_N_ACTIONS = len(ACTIONS)
_FIND = "find"
_PUSH = "push"
_UNWEDGE = "unwedge"
BEHAVIORS = (_FIND, _PUSH, _UNWEDGE)


class BehaviorManagerConfig:
    def __init__(
        self,
        push_linger_steps: int = 5,
        unwedge_linger_steps: int = 5,
        attach_reward_threshold: float = 90.0,
        sticky_push: bool = True,
        activate_push_on_ir: bool = False,
    ):
        self.push_linger_steps = push_linger_steps
        self.unwedge_linger_steps = unwedge_linger_steps
        self.attach_reward_threshold = attach_reward_threshold
        self.sticky_push = sticky_push
        self.activate_push_on_ir = activate_push_on_ir


class BehaviorManager:
    def __init__(self, config: Optional[BehaviorManagerConfig] = None):
        self.config = config or BehaviorManagerConfig()
        self.reset()

    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        self._push_active = False
        self._push_timer = 0
        self._unwedge_timer = 0
        self._prev_ir = False
        if obs is not None:
            self._prev_ir = self._has_ir(obs)
            if self._is_stuck(obs):
                self._unwedge_timer = self.config.unwedge_linger_steps

    def current_behavior(self, obs: np.ndarray) -> str:
        if self._is_stuck(obs) or self._unwedge_timer > 0:
            return _UNWEDGE
        if self._push_active or self._push_timer > 0:
            return _PUSH
        return _FIND

    def update(self, obs: np.ndarray, raw_reward: float, done: bool) -> None:
        if done:
            self.reset()
            return
        stuck = self._is_stuck(obs)
        ir_on = self._has_ir(obs)
        ir_rising = ir_on and not self._prev_ir
        if raw_reward >= self.config.attach_reward_threshold or (
            self.config.activate_push_on_ir and ir_rising
        ):
            self._push_active = True
            if not self.config.sticky_push:
                self._push_timer = self.config.push_linger_steps
        elif self._push_active and not self.config.sticky_push:
            if ir_on:
                pass
            elif self._prev_ir and not ir_on:
                self._push_timer = self.config.push_linger_steps
            elif self._push_timer > 0:
                self._push_timer -= 1
                if self._push_timer == 0:
                    self._push_active = False
            else:
                self._push_active = False

        if stuck:
            self._unwedge_timer = self.config.unwedge_linger_steps
        elif self._unwedge_timer > 0:
            self._unwedge_timer -= 1

        self._prev_ir = ir_on

    @staticmethod
    def _has_ir(obs: np.ndarray) -> bool:
        return bool(np.asarray(obs).reshape(-1)[16])

    @staticmethod
    def _is_stuck(obs: np.ndarray) -> bool:
        return bool(np.asarray(obs).reshape(-1)[17])


class _ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, _N_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


_models: Optional[Dict[str, _ActorCritic]] = None
_manager: Optional[BehaviorManager] = None
_step_count: int = 0
_max_episode_steps: int = 1000


def reset() -> None:
    global _step_count
    if _manager is not None:
        _manager.reset()
    _step_count = 0


def _load_once() -> None:
    global _models, _manager
    if _models is not None:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(here, "three_policy_weights")

    paths = {
        behavior: os.path.join(weights_dir, f"weights_{behavior}.pth")
        for behavior in BEHAVIORS
    }
    missing = [behavior for behavior, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing three-policy checkpoints for: {missing}. "
            f"Expected them under {weights_dir}."
        )

    state_dicts = {
        behavior: torch.load(path, map_location="cpu")
        for behavior, path in paths.items()
    }

    input_dim = state_dicts[_FIND]["trunk.0.weight"].shape[1]
    hidden_dim = state_dicts[_FIND]["trunk.0.weight"].shape[0]

    _models = {}
    for behavior in BEHAVIORS:
        model = _ActorCritic(input_dim=input_dim, hidden_dim=hidden_dim)
        model.load_state_dict(state_dicts[behavior], strict=True)
        model.eval()
        _models[behavior] = model

    _manager = BehaviorManager(
        BehaviorManagerConfig(
            push_linger_steps=5,
            unwedge_linger_steps=5,
            attach_reward_threshold=90.0,
            sticky_push=True,
            activate_push_on_ir=True,
        )
    )


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    del rng
    global _step_count

    _load_once()

    if _step_count >= _max_episode_steps:
        reset()

    behavior = _manager.current_behavior(obs)
    feat = np.asarray(obs, dtype=np.float32).reshape(-1)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

    logits, _ = _models[behavior](x)
    action_idx = int(logits.squeeze(0).argmax().item())

    _step_count += 1
    _manager.update(obs, raw_reward=0.0, done=False)

    return ACTIONS[action_idx]
