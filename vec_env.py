"""vec_env.py — generic vectorized environment using multiprocessing.

Runs N independent environment instances in separate subprocesses and
steps them in parallel via multiprocessing.Pipe.

Works with any environment that follows this protocol:
    env.reset(seed=int)  -> obs
    env.step(action)     -> (obs, reward, done)   # or (obs, reward, done, info)

The worker optionally accepts a reward-shaping callable so shaping logic
stays out of both the env and the trainer.

Usage
-----
    from vec_env import VecEnv

    # 1. Define a factory function for one env instance
    def make_env(seed):
        return lambda: MyEnv(seed=seed)

    # 2. Optional reward shaping (runs inside the worker subprocess)
    def shape(reward, done, env):
        failed = done and reward < -50
        r = reward / 100.0 - 0.005
        return r, failed          # must return (shaped_reward, failed_flag)

    # 3. Build the vectorised env
    vec = VecEnv(
        make_fns=[make_env(i) for i in range(8)],
        reward_shaping_fn=shape,   # omit for raw rewards
    )

    # 4. Use it
    obs_list = vec.reset(seeds=[0,1,2,3,4,5,6,7])

    results = vec.step(actions)
    # results[i] = (obs, reward, done)          if no shaping
    # results[i] = (obs, reward, done, failed)  if shaping provided

    vec.close()

Notes
-----
- Workers are daemon processes: they die automatically if the main process exits.
- reset_one() lets you reset a single worker mid-training without touching others.
- The shaping callable receives (raw_reward, done, env) so it has access to
  env internals (e.g. env.active_state) if needed.
"""

from __future__ import annotations
from multiprocessing import Process, Pipe
from typing import Any, Callable, List, Optional

import numpy as np


# ── Worker (runs in subprocess) ───────────────────────────────────────────────

def _worker(
    make_fn: Callable,
    conn,
    reward_shaping_fn: Optional[Callable],
) -> None:
    env = make_fn()
    while True:
        cmd, data = conn.recv()

        if cmd == "reset":
            obs = env.reset(seed=data)
            conn.send(obs)

        elif cmd == "step":
            result = env.step(data)

            # Support both 3-tuple (obs, reward, done)
            # and 4-tuple (obs, reward, done, info) envs.
            if len(result) == 3:
                obs, reward, done = result
            else:
                obs, reward, done = result[0], result[1], result[2]

            if reward_shaping_fn is not None:
                shaped_reward, failed = reward_shaping_fn(float(reward), bool(done), env)
                conn.send((obs, shaped_reward, bool(done), bool(failed)))
            else:
                conn.send((obs, float(reward), bool(done)))

        elif cmd == "close":
            conn.close()
            break


# ── VecEnv ────────────────────────────────────────────────────────────────────

class VecEnv:
    """Vectorized environment: N envs in N subprocesses, stepped in parallel.

    Parameters
    ----------
    make_fns : list of callables
        Each element is a zero-argument callable that constructs one env
        instance, e.g. ``lambda: MyEnv(seed=i)``.
    reward_shaping_fn : callable, optional
        If provided, called inside each worker after every step:
            shaped_reward, failed = reward_shaping_fn(raw_reward, done, env)
        When omitted, raw (reward, done) are returned unchanged.
    """

    def __init__(
        self,
        make_fns: List[Callable],
        reward_shaping_fn: Optional[Callable] = None,
    ):
        self.n = len(make_fns)
        self._reward_shaping_fn = reward_shaping_fn

        pairs = [Pipe() for _ in make_fns]
        self._parents = [p for p, _ in pairs]
        workers       = [w for _, w in pairs]

        self._procs = [
            Process(
                target=_worker,
                args=(fn, w, reward_shaping_fn),
                daemon=True,
            )
            for fn, w in zip(make_fns, workers)
        ]
        for p in self._procs:
            p.start()

    # ------------------------------------------------------------------
    def reset(self, seeds: List[int]) -> List[np.ndarray]:
        """Reset all envs and return their initial observations."""
        for pipe, seed in zip(self._parents, seeds):
            pipe.send(("reset", seed))
        return [pipe.recv() for pipe in self._parents]

    def reset_one(self, idx: int, seed: int) -> np.ndarray:
        """Reset a single env by index without touching the others."""
        self._parents[idx].send(("reset", seed))
        return self._parents[idx].recv()

    # ------------------------------------------------------------------
    def step(self, actions: List[Any]) -> List[tuple]:
        """Send actions to all envs simultaneously and collect results.

        Returns
        -------
        list of tuples
            (obs, reward, done)          when no reward_shaping_fn
            (obs, shaped_reward, done, failed)  when reward_shaping_fn provided
        """
        for pipe, action in zip(self._parents, actions):
            pipe.send(("step", action))
        return [pipe.recv() for pipe in self._parents]

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Shut down all worker processes cleanly."""
        for pipe in self._parents:
            pipe.send(("close", None))
        for proc in self._procs:
            proc.join()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        shaping = self._reward_shaping_fn.__name__ if self._reward_shaping_fn else "none"
        return f"VecEnv(n={self.n}, reward_shaping={shaping})"