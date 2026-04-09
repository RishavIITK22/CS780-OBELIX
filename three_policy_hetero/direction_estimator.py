from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


OBS_DIM = 18
N_ACT = 5
DIR_UNKNOWN = 0
DIR_LEFT = 1
DIR_FRONT = 2
DIR_RIGHT = 3
N_DIR = 4


def obs_direction_label(obs: np.ndarray, visible_threshold: float = 0.5) -> int:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    far = obs[0:16:2]
    near = obs[1:16:2]
    strengths = 2.0 * near + far
    left = float(strengths[0] + strengths[1])
    front = float(strengths[2] + strengths[3] + strengths[4] + strengths[5])
    right = float(strengths[6] + strengths[7])
    total = left + front + right
    if total < visible_threshold and float(obs[16]) == 0.0:
        return DIR_UNKNOWN
    if front >= left and front >= right:
        return DIR_FRONT
    if left >= right:
        return DIR_LEFT
    return DIR_RIGHT


def exploration_bias_from_direction_probs(
    dir_probs: np.ndarray,
    confidence: float,
) -> np.ndarray:
    left_prior = np.array([0.42, 0.32, 0.12, 0.08, 0.06], dtype=np.float32)
    front_prior = np.array([0.06, 0.12, 0.64, 0.12, 0.06], dtype=np.float32)
    right_prior = np.array([0.06, 0.08, 0.12, 0.32, 0.42], dtype=np.float32)
    unknown_prior = np.array([0.18, 0.16, 0.32, 0.16, 0.18], dtype=np.float32)

    priors = np.stack(
        [unknown_prior, left_prior, front_prior, right_prior],
        axis=0,
    )
    mix = dir_probs @ priors
    uniform = np.full(N_ACT, 1.0 / N_ACT, dtype=np.float32)
    alpha = float(np.clip(confidence, 0.0, 1.0))
    biased = (1.0 - alpha) * uniform + alpha * mix
    biased = np.clip(biased, 1e-6, None)
    return biased / np.sum(biased)


class DirectionalDuelingQNet(nn.Module):
    def __init__(
        self,
        history_len: int = 8,
        hidden_dim: int = 128,
        history_kind: str = "gru",
        n_actions: int = N_ACT,
    ):
        super().__init__()
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.history_kind = history_kind

        if history_kind == "gru":
            self.encoder = nn.GRU(OBS_DIM, hidden_dim, batch_first=True)
        elif history_kind == "lstm":
            self.encoder = nn.LSTM(OBS_DIM, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported history encoder: {history_kind}")

        self.dir_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, N_DIR),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for name, param in self.encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        if self.history_kind == "lstm":
            out, _ = self.encoder(history)
        else:
            out, _ = self.encoder(history)
        return out[:, -1, :]

    def forward(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(history)
        dir_logits = self.dir_head(z)
        value = self.value_head(z)
        adv = self.adv_head(z)
        q = value + adv - adv.mean(dim=-1, keepdim=True)
        return q, dir_logits

