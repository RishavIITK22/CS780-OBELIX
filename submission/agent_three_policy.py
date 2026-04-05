"""Inference agent for bundled three-policy OBELIX PPO.

Expected file next to this agent:
    weights.pth
"""

from __future__ import annotations

import collections
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from behavior_manager import BEHAVIORS, BehaviorManager, BehaviorManagerConfig


ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")
_N_ACTIONS = len(ACTIONS)
_CORE_DIM = 33
_FIND = "find"


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
        self.output_dim = stack_k * _CORE_DIM + _N_ACTIONS
        self.reset()

    def reset(self) -> None:
        self._prev_ir = 0.0
        self._prev_stuck = 0.0
        self._steps_since_seen = self.max_steps_since_seen
        self._stuck_steps = 0
        self._frames: collections.deque = collections.deque(
            [np.zeros(_CORE_DIM, dtype=np.float32)] * self.stack_k,
            maxlen=self.stack_k,
        )

    def encode(self, obs: np.ndarray, prev_action_idx: Optional[int] = None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        far, near = obs[0:16:2], obs[1:16:2]
        ir, stuck = float(obs[16]), float(obs[17])
        strengths = 2.0 * near + far

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

        dir_summary = np.array(
            [
                float(strengths[0] + strengths[1]),
                float(strengths[2] + strengths[3] + strengths[4] + strengths[5]),
                float(strengths[6] + strengths[7]),
            ],
            dtype=np.float32,
        )
        temporal = np.array(
            [
                self._steps_since_seen / self.max_steps_since_seen,
                self._stuck_steps / self.max_stuck_steps,
                1.0 if (self._prev_ir == 0.0 and ir == 1.0) else 0.0,
                1.0 if (self._prev_stuck == 1.0 and stuck == 0.0) else 0.0,
            ],
            dtype=np.float32,
        )

        self._prev_ir = ir
        self._prev_stuck = stuck

        prev_act_oh = np.zeros(_N_ACTIONS, dtype=np.float32)
        if prev_action_idx is not None:
            prev_act_oh[prev_action_idx] = 1.0

        core = np.concatenate([obs, strengths, dir_summary, temporal])
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
_manager: Optional[BehaviorManager] = None
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
    weight_path = os.path.join(here, "weights.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            "Missing bundled checkpoint 'weights.pth' next to agent_three_policy.py."
        )

    bundle = torch.load(weight_path, map_location="cpu")
    meta = bundle.get("meta", {})

    for behavior in BEHAVIORS:
        if behavior not in bundle:
            raise KeyError(f"Bundled checkpoint is missing '{behavior}' weights.")

    input_dim = bundle[_FIND]["trunk.0.weight"].shape[1]
    hidden_dim = int(meta.get("hidden", bundle[_FIND]["trunk.0.weight"].shape[0]))
    stack_k = int(meta.get("stack", (input_dim - _N_ACTIONS) // _CORE_DIM))

    _models = {}
    for behavior in BEHAVIORS:
        model = _ActorCritic(input_dim=input_dim, hidden_dim=hidden_dim)
        model.load_state_dict(bundle[behavior], strict=True)
        model.eval()
        _models[behavior] = model

    _encoder = _BeliefStateEncoder(stack_k=stack_k)
    _manager = BehaviorManager(
        BehaviorManagerConfig(
            push_linger_steps=int(meta.get("push_linger_steps", 5)),
            unwedge_linger_steps=int(meta.get("unwedge_linger_steps", 5)),
            attach_reward_threshold=float(meta.get("attach_reward_threshold", 90.0)),
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
