from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


N_ACT = 5


class MLPActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_actions: int = N_ACT):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def init_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, 1, device=device)
        c = torch.zeros(1, batch, 1, device=device)
        return h, c

    def forward_step(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        del hidden
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1), self.init_hidden(x.shape[0], x.device)

    def get_action(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ):
        logits, value, hidden = self.forward_step(x, hidden)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, hidden

    def evaluate_sequence(
        self,
        x_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        init_hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        del dones_seq, init_hidden
        n_envs, T, in_dim = x_seq.shape
        flat = x_seq.reshape(n_envs * T, in_dim)
        h = self.trunk(flat)
        logits_flat = self.actor(h)
        values_flat = self.critic(h).squeeze(-1)
        return logits_flat, values_flat


class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_actions: int = N_ACT,
        kind: str = "lstm",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.kind = kind

        if kind == "lstm":
            self.core = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif kind == "gru":
            self.core = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif kind == "rnn":
            self.core = nn.RNN(input_dim, hidden_dim, nonlinearity="tanh", batch_first=True)
        else:
            raise ValueError(f"Unsupported recurrent kind: {kind}")

        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = 0.01 if module is self.actor else np.sqrt(2)
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        for name, param in self.core.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_hidden(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.hidden_dim, device=device)
        c = torch.zeros(1, batch, self.hidden_dim, device=device)
        return h, c

    def forward_step(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        if self.kind == "lstm":
            out, (h, c) = self.core(x.unsqueeze(1), hidden)
        else:
            out, h = self.core(x.unsqueeze(1), hidden[0])
            c = hidden[1]
        z = out.squeeze(1)
        return self.actor(z), self.critic(z).squeeze(-1), (h, c)

    def get_action(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ):
        logits, value, hidden = self.forward_step(x, hidden)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, hidden

    def evaluate_sequence(
        self,
        x_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        init_hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        n_envs, T, _ = x_seq.shape
        h, c = init_hidden
        outputs = []
        for t in range(T):
            if self.kind == "lstm":
                out, (h_new, c_new) = self.core(x_seq[:, t : t + 1, :], (h, c))
            else:
                out, h_new = self.core(x_seq[:, t : t + 1, :], h)
                c_new = c
            h, c = h_new, c_new
            outputs.append(out.squeeze(1))
            if t < T - 1:
                mask = (1.0 - dones_seq[:, t]).view(1, n_envs, 1)
                h = h * mask
                c = c * mask

        z_seq = torch.stack(outputs, dim=1)
        logits_flat = self.actor(z_seq).permute(1, 0, 2).reshape(T * n_envs, self.n_actions)
        values_flat = self.critic(z_seq).squeeze(-1).permute(1, 0).reshape(T * n_envs)
        return logits_flat, values_flat

