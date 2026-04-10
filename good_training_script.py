"""Single-file OBELIX three-policy PPO trainer inspired by ppo_gru_un.py.

This script keeps the core implementation ideas from ppo_gru_un.py:
    - finder: MLP actor-critic
    - pusher: MLP actor-critic
    - unwedger: GRU actor-critic with truncated BPTT
    - ppo_gru_un-style mode switching (no BehaviorManager)
    - VecEnv-based parallel rollout collection

It keeps the more structured logging / checkpoint style used in the newer
repo trainers while preserving the original ppo_gru_un observations and
reward shaping.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.distributions import Categorical

from vec_env import VecEnv


ACTIONS = ["L22", "FW", "R22"]
UNWEDGE_ACTIONS = ["L45", "FW", "R45"]
OBS_DIM = 18
GRU_HIDDEN = 64
GRU_CHUNK_LEN = 16
FIND = "find"
PUSH = "push"
UNWEDGE = "unwedger"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        d = torch.device("cpu")
        print("[Device] CPU")
    return d


DEVICE = get_device()


def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def make_env_fn(OBELIX, args, seed: int):
    def _make():
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=seed,
        )
        orig = env.step
        env.step = lambda a: orig(a, render=False)
        return env

    return _make


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, 64), nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def get_action(self, x: torch.Tensor, logit_bias: Optional[torch.Tensor] = None):
        logits, value = self(x)
        if logit_bias is not None:
            logits = logits + logit_bias
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class GRUActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = len(UNWEDGE_ACTIONS),
        enc_hidden: int = 64,
        gru_hidden: int = GRU_HIDDEN,
    ):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.encoder = nn.Sequential(nn.Linear(obs_dim, enc_hidden), nn.Tanh())
        self.gru = nn.GRU(enc_hidden, gru_hidden, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(gru_hidden, 32), nn.Tanh(),
            nn.Linear(32, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(gru_hidden, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(p, gain=math.sqrt(2))
            elif "weight_hh" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def zero_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.gru_hidden, device=DEVICE)

    def forward_step(self, obs: torch.Tensor, h: torch.Tensor):
        enc = self.encoder(obs).unsqueeze(1)
        out, h_new = self.gru(enc, h.to(enc.dtype))
        out = out.squeeze(1)
        return self.actor(out), self.critic(out).squeeze(-1), h_new

    def forward_sequence(self, obs_seq: torch.Tensor, h0: torch.Tensor):
        bsz, steps, _ = obs_seq.shape
        enc = self.encoder(obs_seq.view(bsz * steps, -1)).view(bsz, steps, -1)
        out, _ = self.gru(enc, h0.to(enc.dtype))
        out = out.contiguous().view(bsz * steps, -1)
        return self.actor(out), self.critic(out).squeeze(-1)

    def get_action(self, obs: torch.Tensor, h: torch.Tensor, logit_bias: Optional[torch.Tensor] = None):
        logits, value, h_new = self.forward_step(obs, h)
        if logit_bias is not None:
            logits = logits + logit_bias
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, h_new


class RolloutBuffer:
    def __init__(self, horizon: int, obs_dim: int, gamma: float, lam: float):
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lam = lam
        self._pin = DEVICE.type == "cuda"
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.horizon, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.horizon, dtype=np.int64)
        self.rewards = np.zeros(self.horizon, dtype=np.float32)
        self.dones = np.zeros(self.horizon, dtype=np.float32)
        self.logprobs = np.zeros(self.horizon, dtype=np.float32)
        self.values = np.zeros(self.horizon, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, logprob, value):
        if self.ptr >= self.horizon:
            return
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(self, last_value: float):
        adv = np.zeros(self.ptr, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            next_v = last_value if t == self.ptr - 1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_v * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * last_gae
            adv[t] = last_gae
        return adv, adv + self.values[: self.ptr]

    def get_batches(self, last_value: float, batch_size: int):
        if self.ptr < 2:
            return
        adv, ret = self.compute_gae(last_value)
        adv_std = adv.std()
        if not np.isfinite(adv_std) or adv_std < 1e-8:
            adv_std = 1.0
        adv = (adv - adv.mean()) / (adv_std + 1e-8)

        def _gpu(arr, dtype=torch.float32):
            t = torch.from_numpy(arr)
            if self._pin:
                t = t.pin_memory()
            return t.to(DEVICE, non_blocking=True).to(dtype)

        obs_t = _gpu(self.obs[: self.ptr])
        act_t = _gpu(self.actions[: self.ptr], dtype=torch.int64)
        lp_t = _gpu(self.logprobs[: self.ptr])
        adv_t = _gpu(adv)
        ret_t = _gpu(ret)

        idx = torch.randperm(self.ptr, device=DEVICE)
        for start in range(0, self.ptr, batch_size):
            mb = idx[start : start + batch_size]
            yield obs_t[mb], act_t[mb], lp_t[mb], adv_t[mb], ret_t[mb]


class GRURolloutBuffer:
    def __init__(self, horizon: int, obs_dim: int, gru_hidden: int, gamma: float, lam: float, chunk_len: int = GRU_CHUNK_LEN):
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.gru_hidden = gru_hidden
        self.gamma = gamma
        self.lam = lam
        self.chunk_len = chunk_len
        self._pin = DEVICE.type == "cuda"
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.horizon, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.horizon, dtype=np.int64)
        self.rewards = np.zeros(self.horizon, dtype=np.float32)
        self.dones = np.zeros(self.horizon, dtype=np.float32)
        self.logprobs = np.zeros(self.horizon, dtype=np.float32)
        self.values = np.zeros(self.horizon, dtype=np.float32)
        self.hiddens = np.zeros((self.horizon, self.gru_hidden), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, logprob, value, h: torch.Tensor):
        if self.ptr >= self.horizon:
            return
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr] = value
        self.hiddens[self.ptr] = h.squeeze().detach().cpu().numpy()
        self.ptr += 1

    def compute_gae(self, last_value: float):
        adv = np.zeros(self.ptr, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            next_v = last_value if t == self.ptr - 1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_v * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * last_gae
            adv[t] = last_gae
        return adv, adv + self.values[: self.ptr]

    def get_chunks(self, last_value: float):
        if self.ptr < 2:
            return
        adv, ret = self.compute_gae(last_value)
        adv_std = adv.std()
        if not np.isfinite(adv_std) or adv_std < 1e-8:
            adv_std = 1.0
        adv = (adv - adv.mean()) / (adv_std + 1e-8)

        def _gpu(arr, dtype=torch.float32):
            t = torch.from_numpy(arr)
            if self._pin:
                t = t.pin_memory()
            return t.to(DEVICE, non_blocking=True).to(dtype)

        for start in range(0, self.ptr, self.chunk_len):
            end = min(start + self.chunk_len, self.ptr)
            sl = slice(start, end)
            yield (
                _gpu(self.obs[sl]).unsqueeze(0),
                _gpu(self.actions[sl], dtype=torch.int64),
                _gpu(self.logprobs[sl]),
                _gpu(adv[sl]),
                _gpu(ret[sl]),
                _gpu(self.hiddens[start]).view(1, 1, -1),
            )


def ppo_update(net, opt, scaler, buffer, last_value, epochs, batch_size, clip_eps, vf_coef, ent_coef, max_grad_norm, use_amp):
    total_pg = total_vf = total_ent = n = 0.0
    for _ in range(epochs):
        for sb, ab, old_lp, adv, ret in buffer.get_batches(last_value, batch_size):
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                new_lp, entropy, values = net.evaluate(sb, ab)
                ratio = torch.exp(new_lp - old_lp)
                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps),
                ).mean()
                vf_loss = 0.5 * (values - ret).pow(2).mean()
                ent_loss = -entropy.mean()
                loss = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()

            total_pg += pg_loss.item()
            total_vf += vf_loss.item()
            total_ent += (-ent_loss.item())
            n += 1
    return (total_pg / n, total_vf / n, total_ent / n) if n > 0 else (0.0, 0.0, 0.0)


def ppo_update_gru(net, opt, scaler, buffer, last_value, epochs, clip_eps, vf_coef, ent_coef, max_grad_norm, use_amp):
    chunks = list(buffer.get_chunks(last_value))
    if not chunks:
        return 0.0, 0.0, 0.0
    total_pg = total_vf = total_ent = n = 0.0
    for _ in range(epochs):
        random.shuffle(chunks)
        for obs_c, act_c, old_lp_c, adv_c, ret_c, h0_c in chunks:
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                logits, values = net.forward_sequence(obs_c, h0_c)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(act_c)
                entropy = dist.entropy()
                ratio = torch.exp(new_lp - old_lp_c)
                pg_loss = torch.max(
                    -adv_c * ratio,
                    -adv_c * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps),
                ).mean()
                vf_loss = 0.5 * (values - ret_c).pow(2).mean()
                ent_loss = -entropy.mean()
                loss = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()

            total_pg += pg_loss.item()
            total_vf += vf_loss.item()
            total_ent += (-ent_loss.item())
            n += 1
    return (total_pg / n, total_vf / n, total_ent / n) if n > 0 else (0.0, 0.0, 0.0)


def _raw(net):
    return net._orig_mod if hasattr(net, "_orig_mod") else net


def save_checkpoint(path: str, agents: Dict[str, dict], episode: int, steps: int):
    data = {"episode": episode, "steps": steps}
    for name, ag in agents.items():
        data[f"{name}_net"] = _raw(ag["net"]).state_dict()
        data[f"{name}_opt"] = ag["opt"].state_dict()
        data[f"{name}_scaler"] = ag["scaler"].state_dict()
    torch.save(data, path)


def load_checkpoint(path: str, agents: Dict[str, dict]):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    for name, ag in agents.items():
        _raw(ag["net"]).load_state_dict(ck[f"{name}_net"])
        ag["opt"].load_state_dict(ck[f"{name}_opt"])
        if f"{name}_scaler" in ck:
            ag["scaler"].load_state_dict(ck[f"{name}_scaler"])
    print(f"[resume] Loaded {path}")
    return int(ck["steps"]), int(ck["episode"])


def load_weights_dir(load_dir: str, agents: Dict[str, dict]):
    name_to_file = {
        "finder": "weights_finder.pth",
        "pusher": "weights_pusher.pth",
        "unwedger": "weights_unwedger.pth",
    }
    missing = []
    for name, fname in name_to_file.items():
        path = os.path.join(load_dir, fname)
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError("Missing warm-start weights:\n" + "\n".join(missing))

    for name, fname in name_to_file.items():
        path = os.path.join(load_dir, fname)
        payload = torch.load(path, map_location=DEVICE, weights_only=False)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        _raw(agents[name]["net"]).load_state_dict(state_dict, strict=True)
    print(f"[warm-start] Loaded weights from {load_dir}")


def make_obs(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw, dtype=np.float32).reshape(-1)


def build_agent(kind: str, obs_dim: int, hidden: int, lr: float, gamma: float, lam: float, horizon: int, use_amp: bool):
    if kind == "gru":
        net = GRUActorCritic(obs_dim=obs_dim, n_actions=len(UNWEDGE_ACTIONS), enc_hidden=64, gru_hidden=GRU_HIDDEN).to(DEVICE)
        opt = optim.Adam(_raw(net).parameters(), lr=lr, eps=1e-5)
        scaler = GradScaler("cuda", enabled=use_amp)
        return {
            "net": net,
            "opt": opt,
            "scaler": scaler,
            "is_gru": True,
            "make_buffer": lambda: GRURolloutBuffer(horizon, obs_dim, GRU_HIDDEN, gamma, lam),
            "make_hidden": lambda: _raw(net).zero_hidden(),
        }

    n_actions = len(ACTIONS)
    net = ActorCritic(obs_dim, hidden=hidden, n_actions=n_actions).to(DEVICE)
    opt = optim.Adam(_raw(net).parameters(), lr=lr, eps=1e-5)
    scaler = GradScaler("cuda", enabled=use_amp)
    return {
        "net": net,
        "opt": opt,
        "scaler": scaler,
        "is_gru": False,
        "make_buffer": lambda: RolloutBuffer(horizon, obs_dim, gamma, lam),
    }


def main():
    ap = argparse.ArgumentParser(description="Single-file three-policy PPO trainer inspired by ppo_gru_un.py")
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="good_training_weights")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.97)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--load_dir", type=str, default=None)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--find_hidden", type=int, default=256)
    ap.add_argument("--push_hidden", type=int, default=128)
    ap.add_argument("--push_contact_bonus", type=float, default=3.0)
    ap.add_argument("--push_else_penalty", type=float, default=3.0)
    ap.add_argument("--unwedge_stuck_penalty", type=float, default=1.0)
    ap.add_argument("--unwedge_stuck_penalty_cap", type=float, default=5.0)
    ap.add_argument("--unwedge_escape_bonus", type=float, default=20.0)
    ap.add_argument("--push_grace_steps", type=int, default=7)
    ap.add_argument("--unwedge_grace_steps", type=int, default=10)
    ap.add_argument("--post_unwedge_cooldown", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    use_amp = (DEVICE.type == "cuda") and (not args.no_amp)
    os.makedirs(args.out_dir, exist_ok=True)
    make_fns = [make_env_fn(OBELIX, args, args.seed + i) for i in range(args.n_envs)]
    vec = VecEnv(make_fns=make_fns, reward_shaping_fn=None)

    agents = {
        "finder": build_agent("mlp", OBS_DIM, args.find_hidden, args.lr, args.gamma, args.lam, args.horizon, use_amp),
        "pusher": build_agent("mlp", OBS_DIM, args.push_hidden, args.lr, args.gamma, args.lam, args.horizon, use_amp),
        "unwedger": build_agent("gru", OBS_DIM, 0, args.lr, args.gamma, args.lam, args.horizon, use_amp),
    }

    steps = 0
    start_ep = 0
    if args.resume:
        steps, start_ep = load_checkpoint(args.resume, agents)
    elif args.load_dir:
        load_weights_dir(args.load_dir, agents)

    print(f"[opt] AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"[agents] finder=MLP({args.find_hidden}) | pusher=MLP({args.push_hidden}) | unwedger=GRU(h={GRU_HIDDEN}, chunk={GRU_CHUNK_LEN}) | envs={args.n_envs}")

    obs_arr = np.asarray(vec.reset(seeds=[args.seed + i for i in range(args.n_envs)]), dtype=np.float32)

    for ag in agents.values():
        ag["buffers"] = [ag["make_buffer"]() for _ in range(args.n_envs)]
        if ag["is_gru"]:
            ag["h_list"] = [ag["make_hidden"]() for _ in range(args.n_envs)]

    ep = start_ep
    ep_reward = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps = np.zeros(args.n_envs, dtype=np.int32)
    behavior_counts = [{FIND: 0, PUSH: 0, UNWEDGE: 0} for _ in range(args.n_envs)]
    box_attached = np.zeros(args.n_envs, dtype=bool)
    push_grace = np.zeros(args.n_envs, dtype=np.int32)
    unwedge_steps = np.zeros(args.n_envs, dtype=np.int32)
    unwedge_active = np.zeros(args.n_envs, dtype=bool)
    unwedge_grace = np.zeros(args.n_envs, dtype=np.int32)
    post_unwedge_cooldown = np.zeros(args.n_envs, dtype=np.int32)
    success_count = 0
    best_return = -float("inf")
    rewards_history: List[float] = []
    recent_successes = collections.deque(maxlen=200)
    recent_update_metrics = collections.defaultdict(lambda: collections.deque(maxlen=20))
    train_start = time.time()
    last_log_ep = 0
    log_every = 20
    window_returns: List[float] = []
    window_steps: List[int] = []

    try:
        while ep < args.episodes:
            for ag in agents.values():
                for buf in ag["buffers"]:
                    buf.reset()

            horizon_steps = 0
            while horizon_steps < args.horizon and ep < args.episodes:
                behavior_ids = np.empty(args.n_envs, dtype=object)
                action_idx = np.zeros(args.n_envs, dtype=np.int64)
                log_probs = np.zeros(args.n_envs, dtype=np.float32)
                values = np.zeros(args.n_envs, dtype=np.float32)
                h_before_list: List[Optional[torch.Tensor]] = [None] * args.n_envs

                for i in range(args.n_envs):
                    raw = obs_arr[i]
                    ir_on = bool(raw[16] == 1)
                    stuck_on = bool(raw[17] == 1)

                    if stuck_on:
                        unwedge_active[i] = True
                        unwedge_grace[i] = args.unwedge_grace_steps
                    elif unwedge_grace[i] > 0:
                        unwedge_grace[i] -= 1
                        if unwedge_grace[i] == 0:
                            unwedge_active[i] = False
                            post_unwedge_cooldown[i] = args.post_unwedge_cooldown

                    if post_unwedge_cooldown[i] > 0:
                        post_unwedge_cooldown[i] -= 1

                    if ir_on and post_unwedge_cooldown[i] == 0:
                        push_grace[i] = args.push_grace_steps
                    elif push_grace[i] > 0:
                        push_grace[i] -= 1

                    if unwedge_active[i]:
                        behavior = UNWEDGE
                    elif push_grace[i] > 0:
                        behavior = PUSH
                    else:
                        behavior = FIND
                    behavior_ids[i] = behavior

                    ag = agents["finder" if behavior == FIND else "pusher" if behavior == PUSH else "unwedger"]
                    obs = make_obs(raw)
                    behavior_counts[i][behavior] += 1
                    st = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        with autocast(device_type=DEVICE.type, enabled=use_amp):
                            if behavior == UNWEDGE:
                                h_before = ag["h_list"][i]
                                action, log_prob, _, value, h_new = ag["net"].get_action(st, h_before)
                                ag["h_list"][i] = h_new
                                h_before_list[i] = h_before
                            else:
                                action, log_prob, _, value = ag["net"].get_action(st)
                    action_idx[i] = int(action.item())
                    log_probs[i] = float(log_prob.item())
                    values[i] = float(value.item())

                act_names = [UNWEDGE_ACTIONS[a] if behavior_ids[i] == UNWEDGE else ACTIONS[a] for i, a in enumerate(action_idx)]
                results = vec.step(act_names)
                next_obs_arr = np.asarray([r[0] for r in results], dtype=np.float32)
                env_rewards = np.asarray([r[1] for r in results], dtype=np.float32)
                dones = np.asarray([r[2] for r in results], dtype=bool)

                for i, behavior in enumerate(behavior_ids):
                    ag = agents["finder" if behavior == FIND else "pusher" if behavior == PUSH else "unwedger"]
                    obs = make_obs(obs_arr[i])
                    raw2 = next_obs_arr[i]
                    env_r = float(env_rewards[i])
                    done_i = bool(dones[i])
                    reward = env_r

                    if behavior == FIND:
                        if env_r >= 90.0:
                            box_attached[i] = True
                            push_grace[i] = args.push_grace_steps
                    elif behavior == PUSH:
                        if action_idx[i] == 1 and obs_arr[i][16] == 1 and raw2[17] == 0 and box_attached[i]:
                            reward += args.push_contact_bonus
                        else:
                            reward -= args.push_else_penalty
                        reward = float(np.clip(reward, -10.0, 10.0))
                    else:
                        unwedge_steps[i] += 1
                        if raw2[17] == 1:
                            reward -= min(args.unwedge_stuck_penalty * unwedge_steps[i], args.unwedge_stuck_penalty_cap)
                        elif raw2[17] == 0 and raw2[16] == 0:
                            reward += max(args.unwedge_escape_bonus - unwedge_steps[i] * 1.5, 5.0)
                            unwedge_steps[i] = 0
                        reward = float(np.clip(reward, -20.0, 20.0))

                    buf = ag["buffers"][i]
                    if behavior == UNWEDGE:
                        buf.add(obs, action_idx[i], reward, float(done_i), log_probs[i], values[i], h=h_before_list[i])
                    else:
                        buf.add(obs, action_idx[i], reward, float(done_i), log_probs[i], values[i])

                    ep_reward[i] += reward
                    ep_steps[i] += 1

                obs_arr = next_obs_arr
                steps += args.n_envs
                horizon_steps += args.n_envs

                for i in range(args.n_envs):
                    if not dones[i]:
                        continue
                    success = bool(env_rewards[i] >= 100.0)
                    success_count += int(success)
                    recent_successes.append(int(success))
                    rewards_history.append(float(ep_reward[i]))
                    window_returns.append(float(ep_reward[i]))
                    window_steps.append(int(ep_steps[i]))
                    best_return = max(best_return, float(ep_reward[i]))
                    ep += 1
                    if ep >= args.episodes:
                        break
                    reset_obs = np.asarray(vec.reset_one(i, seed=args.seed * 100003 + ep), dtype=np.float32)
                    obs_arr[i] = reset_obs
                    agents["unwedger"]["h_list"][i] = agents["unwedger"]["make_hidden"]()
                    ep_reward[i] = 0.0
                    ep_steps[i] = 0
                    behavior_counts[i] = {FIND: 0, PUSH: 0, UNWEDGE: 0}
                    box_attached[i] = False
                    push_grace[i] = 0
                    unwedge_steps[i] = 0
                    unwedge_active[i] = False
                    unwedge_grace[i] = 0
                    post_unwedge_cooldown[i] = 0

            with torch.no_grad():
                for name, ag in agents.items():
                    ag["last_values"] = np.zeros(args.n_envs, dtype=np.float32)
                    for i in range(args.n_envs):
                        buf = ag["buffers"][i]
                        if buf.ptr < 2:
                            continue
                        st_lv = torch.tensor(make_obs(obs_arr[i]), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        if ag["is_gru"]:
                            _, lv_tensor, _ = ag["net"].forward_step(st_lv, ag["h_list"][i])
                        else:
                            _, lv_tensor = ag["net"](st_lv)
                        ag["last_values"][i] = float(lv_tensor.item())

            for name, ag in agents.items():
                pg_vals = []
                vf_vals = []
                ent_vals = []
                for i, buf in enumerate(ag["buffers"]):
                    if buf.ptr < 2:
                        continue
                    if ag["is_gru"]:
                        pg, vf, ent = ppo_update_gru(
                            ag["net"], ag["opt"], ag["scaler"], buf, ag["last_values"][i],
                            args.epochs, args.clip_eps, args.vf_coef, args.ent_coef,
                            args.max_grad_norm, use_amp,
                        )
                    else:
                        if buf.ptr < args.batch:
                            continue
                        pg, vf, ent = ppo_update(
                            ag["net"], ag["opt"], ag["scaler"], buf, ag["last_values"][i],
                            args.epochs, args.batch, args.clip_eps, args.vf_coef,
                            args.ent_coef, args.max_grad_norm, use_amp,
                        )
                    pg_vals.append(pg); vf_vals.append(vf); ent_vals.append(ent)
                if pg_vals:
                    recent_update_metrics[f"{name}_pg"].append(float(np.mean(pg_vals)))
                    recent_update_metrics[f"{name}_vf"].append(float(np.mean(vf_vals)))
                    recent_update_metrics[f"{name}_ent"].append(float(np.mean(ent_vals)))

            if ep > 0 and (ep // log_every) > (last_log_ep // log_every) and window_returns:
                elapsed = time.time() - train_start
                avg_return = float(np.mean(window_returns))
                avg_steps = float(np.mean(window_steps))
                rolling_success = 100.0 * np.mean(recent_successes) if recent_successes else 0.0
                print(
                    f"\n┌─ ep {ep-len(window_returns)+1:>4d}–{ep:<4d} ({elapsed:.0f}s) "
                    f"────────────────────────────────────────────────────────"
                )
                print(
                    f"│ Avg Return : {avg_return:8.2f}   Best Return : {best_return:8.2f}   "
                    f"Rolling Success(200 ep): {rolling_success:6.2f}%"
                )
                print(
                    f"│ Avg Steps  : {avg_steps:8.1f}   Total steps : {steps:,}   Successes : {success_count}"
                )
                for key in ("finder", "pusher", "unwedger"):
                    pg = np.mean(recent_update_metrics[f"{key}_pg"]) if recent_update_metrics[f"{key}_pg"] else 0.0
                    vf = np.mean(recent_update_metrics[f"{key}_vf"]) if recent_update_metrics[f"{key}_vf"] else 0.0
                    ent = np.mean(recent_update_metrics[f"{key}_ent"]) if recent_update_metrics[f"{key}_ent"] else 0.0
                    print(f"│ {key:>8s} : PG={pg:8.4f}   VF={vf:8.4f}   Ent={ent:8.4f}")
                print(f"└{'─'*98}")
                last_log_ep = ep
                window_returns.clear()
                window_steps.clear()

            if ep > 0 and ep % 50 == 0:
                save_checkpoint(os.path.join(args.out_dir, "checkpoint.pth"), agents, ep, steps)

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint.")
        save_checkpoint(os.path.join(args.out_dir, "checkpoint_interrupt.pth"), agents, ep, steps)

    vec.close()
    for name, ag in agents.items():
        path = os.path.join(args.out_dir, f"weights_{name}.pth")
        torch.save(_raw(ag["net"]).state_dict(), path)
        print(f"Saved {name} -> {path}")


if __name__ == "__main__":
    main()
