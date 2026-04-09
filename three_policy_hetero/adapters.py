from __future__ import annotations

import collections
from typing import List, Optional

import numpy as np


OBS_DIM = 18
N_ACT = 5


def prev_action_one_hot(prev_action_idx: int) -> np.ndarray:
    oh = np.zeros(N_ACT, dtype=np.float32)
    if prev_action_idx is not None and prev_action_idx >= 0:
        oh[int(prev_action_idx)] = 1.0
    return oh


class IdentityAdapter:
    def __init__(
        self,
        n_envs: int,
        include_prev_action: bool = True,
        include_prev_reward: bool = True,
        include_prev_done: bool = True,
    ):
        self.n_envs = n_envs
        self.include_prev_action = include_prev_action
        self.include_prev_reward = include_prev_reward
        self.include_prev_done = include_prev_done

        self.output_dim = OBS_DIM
        if include_prev_action:
            self.output_dim += N_ACT
        if include_prev_reward:
            self.output_dim += 1
        if include_prev_done:
            self.output_dim += 1

    def reset_env(self, env_idx: int) -> None:
        del env_idx

    def transform_step(
        self,
        env_idx: int,
        obs: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
        prev_done: float,
    ) -> np.ndarray:
        del env_idx
        return self._build_feature(obs, prev_action_idx, prev_reward, prev_done)

    def peek_step(
        self,
        env_idx: int,
        obs: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
        prev_done: float,
    ) -> np.ndarray:
        return self.transform_step(env_idx, obs, prev_action_idx, prev_reward, prev_done)

    def transform_batch(
        self,
        obs_seq: np.ndarray,
        prev_action_seq: np.ndarray,
        prev_reward_seq: np.ndarray,
        prev_done_seq: np.ndarray,
    ) -> np.ndarray:
        n_envs, T, _ = obs_seq.shape
        out = np.zeros((n_envs, T, self.output_dim), dtype=np.float32)
        for i in range(n_envs):
            for t in range(T):
                out[i, t] = self._build_feature(
                    obs_seq[i, t],
                    int(prev_action_seq[i, t]),
                    float(prev_reward_seq[i, t]),
                    float(prev_done_seq[i, t]),
                )
        return out

    def _build_feature(
        self,
        obs: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
        prev_done: float,
    ) -> np.ndarray:
        parts: List[np.ndarray] = [np.asarray(obs, dtype=np.float32).reshape(-1)]
        if self.include_prev_action:
            parts.append(prev_action_one_hot(prev_action_idx))
        if self.include_prev_reward:
            parts.append(np.asarray([prev_reward], dtype=np.float32))
        if self.include_prev_done:
            parts.append(np.asarray([prev_done], dtype=np.float32))
        return np.concatenate(parts, dtype=np.float32)


class FrameStackAdapter:
    def __init__(
        self,
        n_envs: int,
        stack_k: int = 4,
        include_prev_action: bool = True,
        include_prev_reward: bool = False,
        include_prev_done: bool = False,
    ):
        self.n_envs = n_envs
        self.stack_k = stack_k
        self.include_prev_action = include_prev_action
        self.include_prev_reward = include_prev_reward
        self.include_prev_done = include_prev_done
        self.output_dim = stack_k * OBS_DIM
        if include_prev_action:
            self.output_dim += N_ACT
        if include_prev_reward:
            self.output_dim += 1
        if include_prev_done:
            self.output_dim += 1

        self._frames = [
            collections.deque(
                [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(stack_k)],
                maxlen=stack_k,
            )
            for _ in range(n_envs)
        ]

    def reset_env(self, env_idx: int) -> None:
        self._frames[env_idx].clear()
        self._frames[env_idx].extend(
            [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(self.stack_k)]
        )

    def transform_step(
        self,
        env_idx: int,
        obs: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
        prev_done: float,
    ) -> np.ndarray:
        if prev_done > 0.5:
            self.reset_env(env_idx)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._frames[env_idx].append(obs)
        return self._concat(list(self._frames[env_idx]), prev_action_idx, prev_reward, prev_done)

    def peek_step(
        self,
        env_idx: int,
        obs: np.ndarray,
        prev_action_idx: int,
        prev_reward: float,
        prev_done: float,
    ) -> np.ndarray:
        frames = collections.deque(self._frames[env_idx], maxlen=self.stack_k)
        if prev_done > 0.5:
            frames.clear()
            frames.extend(
                [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(self.stack_k)]
            )
        frames.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        return self._concat(list(frames), prev_action_idx, prev_reward, prev_done)

    def transform_batch(
        self,
        obs_seq: np.ndarray,
        prev_action_seq: np.ndarray,
        prev_reward_seq: np.ndarray,
        prev_done_seq: np.ndarray,
    ) -> np.ndarray:
        n_envs, T, _ = obs_seq.shape
        out = np.zeros((n_envs, T, self.output_dim), dtype=np.float32)
        for i in range(n_envs):
            frames = collections.deque(
                [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(self.stack_k)],
                maxlen=self.stack_k,
            )
            for t in range(T):
                if prev_done_seq[i, t] > 0.5:
                    frames.clear()
                    frames.extend(
                        [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(self.stack_k)]
                    )
                frames.append(np.asarray(obs_seq[i, t], dtype=np.float32).reshape(-1))
                out[i, t] = self._concat(
                    list(frames),
                    int(prev_action_seq[i, t]),
                    float(prev_reward_seq[i, t]),
                    float(prev_done_seq[i, t]),
                )
        return out

    def _concat(
        self,
        frames: List[np.ndarray],
        prev_action_idx: int,
        prev_reward: float,
        prev_done: float,
    ) -> np.ndarray:
        parts: List[np.ndarray] = [*frames]
        if self.include_prev_action:
            parts.append(prev_action_one_hot(prev_action_idx))
        if self.include_prev_reward:
            parts.append(np.asarray([prev_reward], dtype=np.float32))
        if self.include_prev_done:
            parts.append(np.asarray([prev_done], dtype=np.float32))
        return np.concatenate(parts, dtype=np.float32)
