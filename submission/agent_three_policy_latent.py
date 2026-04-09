"""Inference agent for latent three-policy OBELIX PPO.

Expected files next to this agent:
    weights_find.pth
    weights_push.pth
    weights_unwedge.pth
"""

from __future__ import annotations

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


class _LatentBeliefActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = _OBS_DIM,
        n_actions: int = _N_ACTIONS,
        obs_embed_dim: int = 64,
        hidden_dim: int = 256,
        aux_hidden: int = 128,
        use_latent_learning: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.use_latent_learning = use_latent_learning
        self.obs_embed_dim = obs_embed_dim if use_latent_learning else obs_dim
        self.hidden_dim = hidden_dim

        if use_latent_learning:
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, self.obs_embed_dim),
                nn.Tanh(),
            )
        else:
            self.obs_encoder = nn.Identity()

        self.belief = nn.LSTM(
            input_size=self.obs_embed_dim + n_actions + 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        pred_in = hidden_dim + n_actions
        self.obs_head = nn.Sequential(
            nn.Linear(pred_in, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(pred_in, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(pred_in, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, 1),
        )
        self.forward_head = nn.Sequential(
            nn.Linear(pred_in, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, self.obs_embed_dim),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(2 * self.obs_embed_dim, aux_hidden),
            nn.Tanh(),
            nn.Linear(aux_hidden, n_actions),
        )

    def init_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.hidden_dim, device=device)
        c = torch.zeros(1, batch, self.hidden_dim, device=device)
        return h, c

    def _action_one_hot(self, action_idx: torch.Tensor) -> torch.Tensor:
        oh = torch.zeros(action_idx.shape[0], self.n_actions, device=action_idx.device)
        valid = action_idx >= 0
        if valid.any():
            oh[valid, action_idx[valid]] = 1.0
        return oh

    def forward_step(
        self,
        obs: torch.Tensor,
        prev_action_idx: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        obs_feat = self.obs_encoder(obs)
        prev_action_oh = self._action_one_hot(prev_action_idx)
        step_in = torch.cat(
            [obs_feat, prev_action_oh, prev_reward.view(-1, 1), prev_done.view(-1, 1)],
            dim=-1,
        )
        out, hidden = self.belief(step_in.unsqueeze(1), hidden)
        belief_t = out.squeeze(1)
        logits = self.actor(belief_t)
        value = self.critic(belief_t).squeeze(-1)
        return logits, value, hidden


_models: Optional[Dict[str, _LatentBeliefActorCritic]] = None
_hidden: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
_manager: Optional[_BehaviorManager] = None
_prev_action_idx: int = -1
_prev_reward: float = 0.0
_prev_done: float = 1.0
_step_count: int = 0


def reset() -> None:
    global _hidden, _manager, _prev_action_idx, _prev_reward, _prev_done, _step_count
    if _models is not None:
        _hidden = {
            behavior: _models[behavior].init_hidden(1, torch.device("cpu"))
            for behavior in _BEHAVIORS
        }
    if _manager is not None:
        _manager.reset()
    _prev_action_idx = -1
    _prev_reward = 0.0
    _prev_done = 1.0
    _step_count = 0


def _load_payload(path: str):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("config", {})
    return payload, {}


def _load_once() -> None:
    global _models, _hidden, _manager
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
            f"Missing latent three-policy checkpoints for: {missing}. "
            f"Expected them next to {os.path.basename(__file__)}."
        )

    state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
    configs: Dict[str, Dict[str, object]] = {}
    for behavior, path in paths.items():
        state_dicts[behavior], configs[behavior] = _load_payload(path)

    cfg = configs[_FIND]
    hidden_dim = int(cfg.get("hidden", state_dicts[_FIND]["actor.weight"].shape[1]))
    use_latent_learning = bool(
        cfg.get("latent_learning", any(k.startswith("obs_encoder.0.") for k in state_dicts[_FIND]))
    )
    if use_latent_learning:
        obs_embed_dim = int(cfg.get("obs_embed_dim", state_dicts[_FIND]["obs_encoder.2.weight"].shape[0]))
    else:
        obs_embed_dim = _OBS_DIM

    _models = {}
    for behavior in _BEHAVIORS:
        model = _LatentBeliefActorCritic(
            obs_dim=_OBS_DIM,
            n_actions=_N_ACTIONS,
            obs_embed_dim=obs_embed_dim,
            hidden_dim=hidden_dim,
            use_latent_learning=use_latent_learning,
        )
        model.load_state_dict(state_dicts[behavior], strict=True)
        model.eval()
        _models[behavior] = model

    _hidden = {
        behavior: _models[behavior].init_hidden(1, torch.device("cpu"))
        for behavior in _BEHAVIORS
    }
    _manager = _BehaviorManager(_BehaviorManagerConfig())


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    del rng
    global _hidden, _prev_action_idx, _prev_reward, _prev_done, _step_count

    _load_once()
    if _step_count >= _MAX_EPISODE_STEPS:
        reset()

    behavior = _manager.current_behavior(obs)
    obs_t = torch.tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
    prev_action_t = torch.tensor([_prev_action_idx], dtype=torch.long)
    prev_reward_t = torch.tensor([_prev_reward], dtype=torch.float32)
    prev_done_t = torch.tensor([_prev_done], dtype=torch.float32)

    logits, _, hidden_new = _models[behavior].forward_step(
        obs=obs_t,
        prev_action_idx=prev_action_t,
        prev_reward=prev_reward_t,
        prev_done=prev_done_t,
        hidden=_hidden[behavior],
    )
    _hidden[behavior] = hidden_new

    action_idx = int(logits.squeeze(0).argmax().item())
    _prev_action_idx = action_idx
    _prev_reward = 0.0
    _prev_done = 0.0
    _step_count += 1
    _manager.update(obs, raw_reward=0.0, done=False)
    return ACTIONS[action_idx]
