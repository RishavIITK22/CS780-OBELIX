"""Inference agent for three-policy OBELIX PPO.

Expected files next to this agent:
    weights_find.pth
    weights_push.pth
    weights_unwedge.pth
"""

from __future__ import annotations

import collections
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_N_ACTIONS = len(ACTIONS)
_CORE_DIM = 49
_FIND = "find"
_PUSH = "push"
_UNWEDGE = "unwedge"
_BEHAVIORS = (_FIND, _PUSH, _UNWEDGE)


class _BehaviorManagerConfig:
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


class _BehaviorManager:
    def __init__(self, config: Optional[_BehaviorManagerConfig] = None):
        self.config = config or _BehaviorManagerConfig()
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

        if (
            raw_reward >= self.config.attach_reward_threshold
            or (self.config.activate_push_on_ir and ir_rising)
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
        obs = np.asarray(obs).reshape(-1)
        return bool(obs[16])

    @staticmethod
    def _is_stuck(obs: np.ndarray) -> bool:
        obs = np.asarray(obs).reshape(-1)
        return bool(obs[17])


class _BeliefStateEncoder:
    def __init__(
        self,
        stack_k: int = 8,
        max_steps_since_seen: int = 30,
        max_stuck_steps: int = 20,
    ):
        self.stack_k = stack_k
        self.max_steps_since_seen = max_steps_since_seen
        self.max_stuck_steps = max_stuck_steps
        self.max_seen_streak = 30
        self.max_lost_streak = 30
        self.output_dim = stack_k * _CORE_DIM + _N_ACTIONS
        self.reset()

    def reset(self) -> None:
        self._prev_ir = 0.0
        self._prev_stuck = 0.0
        self._prev_visible = 0.0
        self._steps_since_seen = self.max_steps_since_seen
        self._stuck_steps = 0
        self._seen_streak = 0
        self._lost_streak = 0
        self._prev_dir_summary = np.zeros(3, dtype=np.float32)
        self._prev_total_strength = 0.0
        self._prev_lr_balance = 0.0
        self._frames: collections.deque = collections.deque(
            [np.zeros(_CORE_DIM, dtype=np.float32) for _ in range(self.stack_k)],
            maxlen=self.stack_k,
        )

    def encode(self, obs: np.ndarray, prev_action_idx: Optional[int] = None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        far, near = obs[0:16:2], obs[1:16:2]
        ir, stuck = float(obs[16]), float(obs[17])
        strengths = 2.0 * near + far

        total_strength = float(np.sum(strengths))
        visible = 1.0 if (total_strength > 0.0 or ir > 0.0) else 0.0

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
        if stuck > 0:
            self._stuck_steps = min(self._stuck_steps + 1, self.max_stuck_steps)
        else:
            self._stuck_steps = 0

        dir_summary = np.array(
            [
                float(strengths[0] + strengths[1]),
                float(strengths[2] + strengths[3] + strengths[4] + strengths[5]),
                float(strengths[6] + strengths[7]),
            ],
            dtype=np.float32,
        )
        near_count = float(np.sum(near))
        far_count = float(np.sum(far))
        strongest_strength = float(np.max(strengths)) if strengths.size else 0.0
        sector_mass = np.sum(strengths)
        if sector_mass > 0:
            sector_positions = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
            sector_centroid = float(np.dot(strengths, sector_positions) / sector_mass)
        else:
            sector_centroid = 0.0

        left = float(dir_summary[0])
        front = float(dir_summary[1])
        right = float(dir_summary[2])
        front_ratio = front / max(total_strength, 1.0)
        left_ratio = left / max(total_strength, 1.0)
        right_ratio = right / max(total_strength, 1.0)
        lr_balance = (right - left) / max(total_strength, 1.0)
        just_lost_ir = 1.0 if (self._prev_ir == 1.0 and ir == 0.0) else 0.0

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
                1.0 if (self._prev_ir == 0.0 and ir == 1.0) else 0.0,
                just_lost_ir,
                1.0 if (self._prev_stuck == 1.0 and stuck == 0.0) else 0.0,
            ],
            dtype=np.float32,
        )

        transition = np.array(
            [
                (total_strength - self._prev_total_strength) / 24.0,
                (front - float(self._prev_dir_summary[1])) / 12.0,
                lr_balance - self._prev_lr_balance,
                visible - self._prev_visible,
            ],
            dtype=np.float32,
        )

        self._prev_ir = ir
        self._prev_stuck = stuck
        self._prev_visible = visible
        self._prev_dir_summary = dir_summary.copy()
        self._prev_total_strength = total_strength
        self._prev_lr_balance = lr_balance

        prev_act_oh = np.zeros(_N_ACTIONS, dtype=np.float32)
        if prev_action_idx is not None:
            prev_act_oh[prev_action_idx] = 1.0

        core = np.concatenate([obs, strengths, dir_summary, geometry, temporal, transition])
        self._frames.append(core)
        return np.concatenate([*self._frames, prev_act_oh])


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
_encoder: Optional[_BeliefStateEncoder] = None
_manager: Optional[_BehaviorManager] = None
_prev_action_idx: Optional[int] = None
_step_count: int = 0
_max_episode_steps: int = 1000


def reset() -> None:
    global _prev_action_idx, _step_count
    if _encoder is not None:
        _encoder.reset()
    if _manager is not None:
        _manager.reset()
    _prev_action_idx = None
    _step_count = 0


def _load_once() -> None:
    global _models, _encoder, _manager
    if _models is not None:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    paths = {
        behavior: os.path.join(here, f"weights_{behavior}.pth")
        for behavior in _BEHAVIORS
    }
    missing = [behavior for behavior, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing three-policy checkpoints for: {missing}. "
            f"Expected them next to {os.path.basename(__file__)}."
        )

    state_dicts = {
        behavior: torch.load(path, map_location="cpu")
        for behavior, path in paths.items()
    }

    input_dim = state_dicts[_FIND]["trunk.0.weight"].shape[1]
    hidden_dim = state_dicts[_FIND]["trunk.0.weight"].shape[0]
    stack_k = (input_dim - _N_ACTIONS) // _CORE_DIM

    _models = {}
    for behavior in _BEHAVIORS:
        model = _ActorCritic(input_dim=input_dim, hidden_dim=hidden_dim)
        model.load_state_dict(state_dicts[behavior], strict=True)
        model.eval()
        _models[behavior] = model

    _encoder = _BeliefStateEncoder(stack_k=stack_k)
    _manager = _BehaviorManager(
        _BehaviorManagerConfig(
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
    global _prev_action_idx, _step_count

    _load_once()

    if _step_count >= _max_episode_steps:
        reset()

    behavior = _manager.current_behavior(obs)
    feat = _encoder.encode(obs, _prev_action_idx)
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

    logits, _ = _models[behavior](x)
    action_idx = int(logits.squeeze(0).argmax().item())

    _prev_action_idx = action_idx
    _step_count += 1
    _manager.update(obs, raw_reward=0.0, done=False)

    return ACTIONS[action_idx]
