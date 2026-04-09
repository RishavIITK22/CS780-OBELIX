from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from three_policy_hetero.direction_estimator import (
    DIR_FRONT,
    DirectionalDuelingQNet,
    exploration_bias_from_direction_probs,
)


@dataclass
class FindRewardShapeConfig:
    confidence_gain_coef: float = 0.05
    front_progress_bonus: float = 0.05
    front_conf_bonus_coef: float = 0.05
    stuck_penalty: float = 0.05


class PrioritizedHistoryReplay:
    def __init__(
        self,
        capacity: int,
        history_len: int,
        obs_dim: int = 18,
        alpha: float = 0.6,
        eps: float = 1e-5,
    ):
        self.capacity = capacity
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.eps = eps

        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0

        self.hist = np.zeros((capacity, history_len, obs_dim), dtype=np.float32)
        self.next_hist = np.zeros((capacity, history_len, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.dir_target = np.zeros(capacity, dtype=np.int64)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        hist: np.ndarray,
        action: int,
        reward: float,
        next_hist: np.ndarray,
        done: bool,
        dir_target: int,
    ) -> None:
        i = self.ptr
        self.hist[i] = hist
        self.next_hist[i] = next_hist
        self.action[i] = action
        self.reward[i] = reward
        self.done[i] = float(done)
        self.dir_target[i] = dir_target
        self.priorities[i] = self.max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float, device: torch.device) -> Dict[str, torch.Tensor]:
        probs = self.priorities[: self.size] ** self.alpha
        probs /= probs.sum()
        idx = np.random.choice(self.size, size=batch_size, p=probs)
        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()
        return {
            "idx": idx,
            "hist": torch.tensor(self.hist[idx], dtype=torch.float32, device=device),
            "next_hist": torch.tensor(self.next_hist[idx], dtype=torch.float32, device=device),
            "action": torch.tensor(self.action[idx], dtype=torch.long, device=device),
            "reward": torch.tensor(self.reward[idx], dtype=torch.float32, device=device),
            "done": torch.tensor(self.done[idx], dtype=torch.float32, device=device),
            "dir_target": torch.tensor(self.dir_target[idx], dtype=torch.long, device=device),
            "weights": torch.tensor(weights, dtype=torch.float32, device=device),
        }

    def update_priorities(self, idx: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float32)
        self.priorities[idx] = np.maximum(priorities + self.eps, self.eps)
        self.max_priority = max(self.max_priority, float(np.max(self.priorities[idx])))

    def __len__(self) -> int:
        return self.size


def select_action(
    net: DirectionalDuelingQNet,
    history: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[int, float, np.ndarray]:
    hist_t = torch.tensor(history[None, ...], dtype=torch.float32, device=device)
    with torch.no_grad():
        q, dir_logits = net(hist_t)
        q_np = q.squeeze(0).cpu().numpy()
        dir_probs = torch.softmax(dir_logits, dim=-1).squeeze(0).cpu().numpy()
    confidence = float(np.max(dir_probs))
    if rng.random() < epsilon:
        probs = exploration_bias_from_direction_probs(dir_probs, confidence)
        action = int(rng.choice(np.arange(q_np.shape[0]), p=probs))
    else:
        action = int(np.argmax(q_np))
    return action, confidence, dir_probs


def shape_find_reward(
    cfg: FindRewardShapeConfig,
    cur_dir_probs: np.ndarray,
    next_dir_probs: np.ndarray,
    next_obs: np.ndarray,
) -> float:
    cur_conf = float(np.max(cur_dir_probs))
    next_conf = float(np.max(next_dir_probs))
    cur_dir = int(np.argmax(cur_dir_probs))
    next_dir = int(np.argmax(next_dir_probs))

    reward = 0.0
    if next_conf > cur_conf:
        reward += cfg.confidence_gain_coef * (next_conf - cur_conf)
    if cur_dir != DIR_FRONT and next_dir == DIR_FRONT and next_conf > 0.35:
        reward += cfg.front_progress_bonus
    if next_dir == DIR_FRONT and next_conf > 0.35:
        reward += cfg.front_conf_bonus_coef * next_conf
    if bool(np.asarray(next_obs).reshape(-1)[17]):
        reward -= cfg.stuck_penalty
    return float(reward)


def d3qn_update(
    online_net: DirectionalDuelingQNet,
    target_net: DirectionalDuelingQNet,
    optimizer: optim.Optimizer,
    replay: PrioritizedHistoryReplay,
    batch_size: int,
    beta: float,
    gamma: float,
    dir_coef: float,
    device: torch.device,
    max_grad: float,
) -> Dict[str, float]:
    batch = replay.sample(batch_size, beta, device)
    q, dir_logits = online_net(batch["hist"])
    q_sa = q.gather(1, batch["action"].unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_online, _ = online_net(batch["next_hist"])
        next_actions = next_q_online.argmax(dim=1)
        next_q_target, next_dir_logits = target_net(batch["next_hist"])
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = batch["reward"] + gamma * (1.0 - batch["done"]) * next_q

    td_error = q_sa - target
    td_loss = (batch["weights"] * td_error.pow(2)).mean()
    dir_loss = nn.functional.cross_entropy(dir_logits, batch["dir_target"])
    loss = td_loss + dir_coef * dir_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), max_grad)
    optimizer.step()

    replay.update_priorities(batch["idx"], np.abs(td_error.detach().cpu().numpy()))

    with torch.no_grad():
        next_dir_probs = torch.softmax(next_dir_logits, dim=-1)
        confidence = float(next_dir_probs.max(dim=1).values.mean().item())

    return {
        "find_td_loss": float(td_loss.item()),
        "find_dir_loss": float(dir_loss.item()),
        "find_conf": confidence,
        "find_q": float(q_sa.mean().item()),
    }

