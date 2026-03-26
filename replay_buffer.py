"""replay_buffer.py — generic circular replay buffer on pinned CPU memory.

Supports an arbitrary set of named fields so the same class works across
DQN, Double DQN, DDPG, SAC, TD3, and any other off-policy algorithm.

Usage
-----
    from replay_buffer import ReplayBuffer

    # Define fields: name -> (dtype, shape_suffix)
    # shape_suffix is appended to (capacity,) to get the full tensor shape.
    # Use () for scalars, (obs_dim,) for vectors, (h, w, c) for images.
    buf = ReplayBuffer(
        capacity=100_000,
        fields={
            "obs":        (torch.float32, (obs_dim,)),
            "action":     (torch.long,    ()),
            "reward":     (torch.float32, ()),
            "next_obs":   (torch.float32, (obs_dim,)),
            "failed":     (torch.float32, ()),
        },
    )

    buf.add(obs=s, action=a, reward=r, next_obs=s2, failed=False)

    batch = buf.sample(256, device=torch.device("cuda"))
    # batch is a dict: {"obs": Tensor, "action": Tensor, ...}
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import torch


class ReplayBuffer:
    """Circular replay buffer stored in pinned CPU memory for fast GPU transfer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    fields : dict[str, (dtype, shape_suffix)]
        Specifies the tensors to allocate.
        Each value is a (torch.dtype, tuple) pair where the tuple is appended
        to (capacity,) to produce the full tensor shape.
        Examples:
            "obs":    (torch.float32, (18,))   → shape (capacity, 18)
            "action": (torch.long,    ())       → shape (capacity,)
            "reward": (torch.float32, ())       → shape (capacity,)
    pin_memory : bool
        Whether to use pinned (page-locked) CPU memory. Speeds up CPU→GPU
        transfers when training on CUDA. Set False for CPU-only workflows.
    """

    def __init__(
        self,
        capacity: int,
        fields: Dict[str, Tuple[torch.dtype, tuple]],
        pin_memory: bool = True,
    ):
        self.capacity  = capacity
        self.ptr       = 0
        self.size      = 0
        self._fields   = list(fields.keys())
        self._buffers: Dict[str, torch.Tensor] = {}

        for name, (dtype, shape_suffix) in fields.items():
            shape = (capacity,) + shape_suffix
            t = torch.zeros(shape, dtype=dtype)
            self._buffers[name] = t.pin_memory() if pin_memory else t

    # ------------------------------------------------------------------
    def add(self, **kwargs: Any) -> None:
        """Store one transition.

        Pass one keyword argument per field defined at construction.

        Example
        -------
            buf.add(obs=s, action=a, reward=r, next_obs=s2, failed=False)
        """
        for name in self._fields:
            val = kwargs[name]
            self._buffers[name][self.ptr] = torch.as_tensor(
                val, dtype=self._buffers[name].dtype
            )
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample a random batch and move it to device.

        Returns
        -------
        dict[str, torch.Tensor]
            One tensor per field, each of shape (batch_size, *shape_suffix)
            or (batch_size,) for scalar fields.
        """
        idx = torch.randint(0, self.size, (batch_size,))
        kw  = dict(device=device, non_blocking=True)
        return {name: self._buffers[name][idx].to(**kw) for name in self._fields}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        fields_str = ", ".join(
            f"{n}:{tuple(self._buffers[n].shape[1:]) or '()'}"
            for n in self._fields
        )
        return (
            f"ReplayBuffer(capacity={self.capacity}, size={self.size}, "
            f"fields=[{fields_str}])"
        )