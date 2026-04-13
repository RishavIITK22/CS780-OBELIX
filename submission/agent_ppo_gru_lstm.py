from __future__ import annotations

import math
import os
import glob

import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L22", "FW", "R22"]
ACTIONS_W = ["L45", "FW", "R45"]

FINDER_OBS_DIM = 18
PUSHER_OBS_DIM = 18
UNWEDGER_OBS_DIM = 18

FINDER_GRU_HIDDEN = 256
PUSHER_GRU_HIDDEN = 128
UNWEDGER_LSTM_HIDDEN = 64

PUSH_GRACE_STEPS = 7
UNWEDGE_GRACE_STEPS = 10
POST_UNWEDGE_COOLDOWN = 10
MAX_EPISODE_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_find_obs(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32)


def get_push_obs(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32)


def get_unwedge_obs(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32)


class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        enc_hidden: int,
        rnn_hidden: int,
        core_type: str,
    ):
        super().__init__()
        if core_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported recurrent core: {core_type}")
        self.rnn_hidden = rnn_hidden
        self.core_type = core_type

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, enc_hidden),
            nn.Tanh(),
        )
        if core_type == "gru":
            self.rnn = nn.GRU(enc_hidden, rnn_hidden, batch_first=True)
        else:
            self.rnn = nn.LSTM(enc_hidden, rnn_hidden, batch_first=True)

        self.actor = nn.Sequential(
            nn.Linear(rnn_hidden, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(rnn_hidden, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(p, gain=math.sqrt(2))
            elif "weight_hh" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def zero_hidden(self):
        h = torch.zeros(1, 1, self.rnn_hidden, device=device)
        if self.core_type == "lstm":
            return h, torch.zeros_like(h)
        return h

    def forward_step(self, obs: torch.Tensor, h):
        enc = self.encoder(obs).unsqueeze(1)
        if self.core_type == "lstm":
            h0, c0 = h
            out, h_new = self.rnn(enc, (h0.to(enc.dtype), c0.to(enc.dtype)))
        else:
            out, h_new = self.rnn(enc, h.to(enc.dtype))
        out = out.squeeze(1)
        return self.actor(out), self.critic(out).squeeze(-1), h_new


def _agent_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_weight_paths(name: str, fallback: str | None = None):
    here = _agent_dir()
    candidates = [os.path.join(here, f"weights_{name}.pth")]
    candidates.extend(sorted(glob.glob(os.path.join(here, f"*_{name}.pth"))))
    if fallback is not None:
        candidates.append(os.path.join(here, f"weights_{fallback}.pth"))
        candidates.extend(sorted(glob.glob(os.path.join(here, f"*_{fallback}.pth"))))

    seen = set()
    paths = []
    for path in candidates:
        if path not in seen and os.path.exists(path):
            seen.add(path)
            paths.append(path)
    return paths


def _load_model(model: nn.Module, name: str, fallback: str | None = None):
    tried = []
    for path in _candidate_weight_paths(name, fallback):
        tried.append(os.path.basename(path))
        payload = torch.load(path, map_location=device, weights_only=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        try:
            model.load_state_dict(state, strict=True)
            return
        except RuntimeError:
            continue

    raise FileNotFoundError(
        f"Missing compatible weights for {name}. Tried: {', '.join(tried) or 'none'}"
    )


def _sample_from_logits(logits: torch.Tensor, rng=None) -> int:
    probs = torch.softmax(logits.squeeze(0), dim=-1).detach().cpu().numpy()
    probs = probs / probs.sum()
    if rng is not None:
        return int(rng.choice(len(probs), p=probs))
    return int(np.random.choice(len(probs), p=probs))


_find_model = RecurrentActorCritic(
    FINDER_OBS_DIM, len(ACTIONS), 64, FINDER_GRU_HIDDEN, "gru"
).to(device)
_push_model = RecurrentActorCritic(
    PUSHER_OBS_DIM, len(ACTIONS), 64, PUSHER_GRU_HIDDEN, "gru"
).to(device)
_unwedge_model = RecurrentActorCritic(
    UNWEDGER_OBS_DIM, len(ACTIONS_W), 64, UNWEDGER_LSTM_HIDDEN, "lstm"
).to(device)

_load_model(_find_model, "finder", fallback="find")
_load_model(_push_model, "pusher", fallback="push")
_load_model(_unwedge_model, "unwedger", fallback="unwedge")

_find_model.eval()
_push_model.eval()
_unwedge_model.eval()

_find_hidden = _find_model.zero_hidden()
_push_hidden = _push_model.zero_hidden()
_unwedge_hidden = _unwedge_model.zero_hidden()

_push_grace = 0
_unwedge_active = False
_unwedge_grace = 0
_post_unwedge_cooldown = 0
_steps = 0
_prev_obs: np.ndarray | None = None


def reset() -> None:
    _reset_agent()


def _reset_agent():
    global _find_hidden, _push_hidden, _unwedge_hidden
    global _push_grace, _unwedge_active, _unwedge_grace, _post_unwedge_cooldown
    global _steps, _prev_obs

    _find_hidden = _find_model.zero_hidden()
    _push_hidden = _push_model.zero_hidden()
    _unwedge_hidden = _unwedge_model.zero_hidden()
    _push_grace = 0
    _unwedge_active = False
    _unwedge_grace = 0
    _post_unwedge_cooldown = 0
    _steps = 0
    _prev_obs = None


def _maybe_reset(obs: np.ndarray):
    global _steps, _prev_obs
    if _prev_obs is None:
        _prev_obs = obs.copy()
        return

    if _steps >= MAX_EPISODE_STEPS:
        _reset_agent()
        _prev_obs = obs.copy()


@torch.no_grad()
def policy(obs, rng=None):
    global _find_hidden, _push_hidden, _unwedge_hidden
    global _push_grace, _unwedge_active, _unwedge_grace, _post_unwedge_cooldown
    global _steps, _prev_obs

    raw = np.asarray(obs, dtype=np.float32)
    _maybe_reset(raw)

    ir_on = bool(raw[16] == 1)
    stuck_on = bool(raw[17] == 1)

    if stuck_on:
        _unwedge_active = True
        _unwedge_grace = UNWEDGE_GRACE_STEPS
    elif _unwedge_grace > 0:
        _unwedge_grace -= 1
        if _unwedge_grace == 0:
            _unwedge_active = False
            _post_unwedge_cooldown = POST_UNWEDGE_COOLDOWN

    if _post_unwedge_cooldown > 0:
        _post_unwedge_cooldown -= 1

    if ir_on and _post_unwedge_cooldown == 0:
        _push_grace = PUSH_GRACE_STEPS
    elif _push_grace > 0:
        _push_grace -= 1

    if _unwedge_active:
        mode = "unwedge"
    elif _push_grace > 0:
        mode = "push"
    else:
        mode = "find"

    if mode == "find":
        x = torch.from_numpy(get_find_obs(raw)).to(device).unsqueeze(0)
        logits, _, h_new = _find_model.forward_step(x, _find_hidden)
        _find_hidden = h_new
        action_idx = _sample_from_logits(logits, rng)
        action = ACTIONS[action_idx]
    elif mode == "push":
        x = torch.from_numpy(get_push_obs(raw)).to(device).unsqueeze(0)
        logits, _, h_new = _push_model.forward_step(x, _push_hidden)
        _push_hidden = h_new
        action_idx = _sample_from_logits(logits, rng)
        action = ACTIONS[action_idx]
    else:
        x = torch.from_numpy(get_unwedge_obs(raw)).to(device).unsqueeze(0)
        logits, _, h_new = _unwedge_model.forward_step(x, _unwedge_hidden)
        _unwedge_hidden = h_new
        action_idx = _sample_from_logits(logits, rng)
        action = ACTIONS_W[action_idx]

    _steps += 1
    _prev_obs = raw.copy()
    return action
