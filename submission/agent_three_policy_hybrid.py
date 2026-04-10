"""Inference agent for the hybrid three-policy OBELIX pipeline.

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
_OBS_DIM = 18
_N_ACTIONS = len(ACTIONS)
_FIND = "find"
_PUSH = "push"
_UNWEDGE = "unwedge"
_BEHAVIORS = (_FIND, _PUSH, _UNWEDGE)
_MAX_EPISODE_STEPS = 1000


class _BehaviorManagerConfig:
    def __init__(
        self,
        push_linger_steps: int = 5,
        unwedge_linger_steps: int = 5,
        attach_reward_threshold: float = 90.0,
        sticky_push: bool = True,
        activate_push_on_ir: bool = True,
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
        return bool(np.asarray(obs).reshape(-1)[16])

    @staticmethod
    def _is_stuck(obs: np.ndarray) -> bool:
        return bool(np.asarray(obs).reshape(-1)[17])


def _prev_action_one_hot(prev_action_idx: int) -> np.ndarray:
    oh = np.zeros(_N_ACTIONS, dtype=np.float32)
    if prev_action_idx >= 0:
        oh[int(prev_action_idx)] = 1.0
    return oh


class _IdentityAdapter:
    def __init__(self, include_prev_action: bool, include_prev_reward: bool, include_prev_done: bool):
        self.include_prev_action = include_prev_action
        self.include_prev_reward = include_prev_reward
        self.include_prev_done = include_prev_done

    def reset(self) -> None:
        return

    def transform_step(
        self,
        obs: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
        prev_done: float,
    ) -> np.ndarray:
        parts = [np.asarray(obs, dtype=np.float32).reshape(-1)]
        if self.include_prev_action:
            parts.append(_prev_action_one_hot(prev_action_idx))
        if self.include_prev_reward:
            parts.append(np.asarray([prev_reward], dtype=np.float32))
        if self.include_prev_done:
            parts.append(np.asarray([prev_done], dtype=np.float32))
        return np.concatenate(parts, dtype=np.float32)


class _FrameStackAdapter:
    def __init__(self, stack_k: int, include_prev_action: bool, include_prev_reward: bool, include_prev_done: bool):
        self.stack_k = stack_k
        self.include_prev_action = include_prev_action
        self.include_prev_reward = include_prev_reward
        self.include_prev_done = include_prev_done
        self.reset()

    def reset(self) -> None:
        self._frames = collections.deque(
            [np.zeros(_OBS_DIM, dtype=np.float32) for _ in range(self.stack_k)],
            maxlen=self.stack_k,
        )

    def transform_step(
        self,
        obs: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
        prev_done: float,
    ) -> np.ndarray:
        if prev_done > 0.5:
            self.reset()
        self._frames.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        parts = [*self._frames]
        if self.include_prev_action:
            parts.append(_prev_action_one_hot(prev_action_idx))
        if self.include_prev_reward:
            parts.append(np.asarray([prev_reward], dtype=np.float32))
        if self.include_prev_done:
            parts.append(np.asarray([prev_done], dtype=np.float32))
        return np.concatenate(parts, dtype=np.float32)


class _MLPActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, _N_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, batch, 1, device=device),
            torch.zeros(1, batch, 1, device=device),
        )

    def forward_step(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        del hidden
        z = self.trunk(x)
        return self.actor(z), self.critic(z).squeeze(-1), self.init_hidden(x.shape[0], x.device)


class _RecurrentActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, kind: str = "lstm"):
        super().__init__()
        self.kind = kind
        self.hidden_dim = hidden_dim
        if kind == "lstm":
            self.core = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif kind == "gru":
            self.core = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif kind == "rnn":
            self.core = nn.RNN(input_dim, hidden_dim, nonlinearity="tanh", batch_first=True)
        else:
            raise ValueError(f"Unsupported architecture: {kind}")
        self.actor = nn.Linear(hidden_dim, _N_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, batch, self.hidden_dim, device=device),
            torch.zeros(1, batch, self.hidden_dim, device=device),
        )

    def forward_step(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        if self.kind == "lstm":
            out, (h, c) = self.core(x.unsqueeze(1), hidden)
        else:
            out, h = self.core(x.unsqueeze(1), hidden[0])
            c = hidden[1]
        z = out.squeeze(1)
        return self.actor(z), self.critic(z).squeeze(-1), (h, c)


class _DirectionalDuelingQNet(nn.Module):
    def __init__(self, history_len: int = 8, hidden_dim: int = 128, history_kind: str = "gru"):
        super().__init__()
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.history_kind = history_kind
        if history_kind == "gru":
            self.encoder = nn.GRU(_OBS_DIM, hidden_dim, batch_first=True)
        elif history_kind == "lstm":
            self.encoder = nn.LSTM(_OBS_DIM, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported history encoder: {history_kind}")
        self.dir_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, _N_ACTIONS),
        )

    def forward(self, history: torch.Tensor):
        out, _ = self.encoder(history)
        z = out[:, -1, :]
        dir_logits = self.dir_head(z)
        value = self.value_head(z)
        adv = self.adv_head(z)
        q = value + adv - adv.mean(dim=-1, keepdim=True)
        return q, dir_logits


_models: Optional[Dict[str, nn.Module]] = None
_adapters: Optional[Dict[str, object]] = None
_hidden: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
_manager: Optional[_BehaviorManager] = None
_find_history_len: int = 8
_find_history: Optional[collections.deque] = None
_prev_action_idx: int = -1
_prev_reward: float = 0.0
_prev_done: float = 1.0
_step_count: int = 0


def reset() -> None:
    global _hidden, _find_history, _prev_action_idx, _prev_reward, _prev_done, _step_count
    if _models is not None:
        _hidden = {
            behavior: _models[behavior].init_hidden(1, torch.device("cpu"))
            for behavior in (_PUSH, _UNWEDGE)
        }
    if _adapters is not None:
        for behavior in (_PUSH, _UNWEDGE):
            _adapters[behavior].reset()
    if _manager is not None:
        _manager.reset()
    _find_history = collections.deque(
        [np.zeros(_OBS_DIM, dtype=np.float32) for _ in range(_find_history_len)],
        maxlen=_find_history_len,
    )
    _prev_action_idx = -1
    _prev_reward = 0.0
    _prev_done = 1.0
    _step_count = 0


def _load_payload(path: str):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("config", {})
    return payload, {}


def _infer_architecture(config: Dict[str, object], state_dict: Dict[str, torch.Tensor]) -> str:
    arch = config.get("architecture")
    if arch is not None:
        return str(arch)
    if "core.weight_ih_l0" in state_dict:
        return "lstm"
    in_dim = int(state_dict["trunk.0.weight"].shape[1])
    return "stack_mlp" if in_dim > (_OBS_DIM + _N_ACTIONS) else "mlp"


def _build_ppo_from_config(config: Dict[str, object], state_dict: Dict[str, torch.Tensor]):
    architecture = _infer_architecture(config, state_dict)
    hidden = int(config.get("hidden", state_dict["actor.weight"].shape[1]))
    stack_k = int(config.get("stack_k", 4))

    if architecture == "stack_mlp":
        adapter = _FrameStackAdapter(stack_k, True, False, False)
        input_dim = int(state_dict["trunk.0.weight"].shape[1])
        model = _MLPActorCritic(input_dim=input_dim, hidden_dim=hidden)
    elif architecture == "mlp":
        adapter = _IdentityAdapter(True, False, False)
        input_dim = int(state_dict["trunk.0.weight"].shape[1])
        model = _MLPActorCritic(input_dim=input_dim, hidden_dim=hidden)
    elif architecture in {"lstm", "gru", "rnn"}:
        adapter = _IdentityAdapter(True, True, True)
        input_dim = int(state_dict["core.weight_ih_l0"].shape[1])
        model = _RecurrentActorCritic(input_dim=input_dim, hidden_dim=hidden, kind=architecture)
    else:
        raise ValueError(f"Unsupported behavior architecture: {architecture}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, adapter


def _build_find_from_config(config: Dict[str, object], state_dict: Dict[str, torch.Tensor]):
    global _find_history_len
    hidden = int(config.get("hidden", 128))
    history_len = int(config.get("history_len", 8))
    history_encoder = str(config.get("history_encoder", "gru"))
    model = _DirectionalDuelingQNet(history_len=history_len, hidden_dim=hidden, history_kind=history_encoder)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _find_history_len = history_len
    return model


def _load_once() -> None:
    global _models, _adapters, _hidden, _manager
    if _models is not None:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    paths = {behavior: os.path.join(here, f"weights_{behavior}.pth") for behavior in _BEHAVIORS}
    missing = [behavior for behavior, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing hybrid three-policy checkpoints for: {missing}. "
            f"Expected them next to {os.path.basename(__file__)}."
        )

    _models = {}
    _adapters = {}

    find_state, find_config = _load_payload(paths[_FIND])
    _models[_FIND] = _build_find_from_config(find_config, find_state)

    for behavior in (_PUSH, _UNWEDGE):
        state_dict, config = _load_payload(paths[behavior])
        model, adapter = _build_ppo_from_config(config, state_dict)
        _models[behavior] = model
        _adapters[behavior] = adapter

    _hidden = {
        behavior: _models[behavior].init_hidden(1, torch.device("cpu"))
        for behavior in (_PUSH, _UNWEDGE)
    }
    _manager = _BehaviorManager(_BehaviorManagerConfig())
    reset()


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    del rng
    global _prev_action_idx, _prev_reward, _prev_done, _step_count

    _load_once()
    if _step_count >= _MAX_EPISODE_STEPS:
        reset()

    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if _prev_done > 0.5:
        _find_history.clear()
        for _ in range(_find_history_len):
            _find_history.append(np.zeros(_OBS_DIM, dtype=np.float32))
    _find_history.append(obs)

    behavior = _manager.current_behavior(obs)
    if behavior == _FIND:
        hist = np.stack(_find_history, axis=0)
        x = torch.tensor(hist, dtype=torch.float32).unsqueeze(0)
        q, _ = _models[_FIND](x)
        action_idx = int(q.squeeze(0).argmax().item())
    else:
        feat = _adapters[behavior].transform_step(obs, _prev_action_idx, _prev_reward, _prev_done)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        logits, _, hidden_new = _models[behavior].forward_step(x, _hidden[behavior])
        _hidden[behavior] = hidden_new
        action_idx = int(logits.squeeze(0).argmax().item())

    _prev_action_idx = action_idx
    _prev_reward = 0.0
    _prev_done = 0.0
    _step_count += 1
    _manager.update(obs, raw_reward=0.0, done=False)
    return ACTIONS[action_idx]
