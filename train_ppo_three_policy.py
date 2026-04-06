"""Three-policy PPO trainer for OBELIX using frame stacking or raw observations.

Implements the paper-style behavior decomposition:
    unwedge > push > find

Each behavior owns a separate PPO policy and value head. A fixed behavior
manager decides which policy acts at each step. Training uses:
    - VecEnv from vec_env.py
    - optional BeliefStateEncoder from state_encoder.py
    - raw environment rewards with simple scaling only

This file does not modify any existing training scripts.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from behavior_manager import (
    BEHAVIORS,
    FIND,
    PUSH,
    UNWEDGE,
    BehaviorManager,
    BehaviorManagerConfig,
)
from state_encoder import BeliefStateEncoder
from vec_env import VecEnv


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
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACT = len(ACTIONS)


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_actions: int = N_ACT):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor):
        logits, values = self(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


@dataclass
class Segment:
    obs: List[np.ndarray]
    actions: List[int]
    log_probs: List[float]
    rewards: List[float]
    values: List[float]


class BehaviorRolloutBuffer:
    """Stores truncated on-policy segments for one behavior."""

    def __init__(self, n_envs: int, gamma: float, gae_lam: float):
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.clear()

    def clear(self) -> None:
        self._segments = [
            Segment(obs=[], actions=[], log_probs=[], rewards=[], values=[])
            for _ in range(self.n_envs)
        ]
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []

    def add_step(
        self,
        env_idx: int,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
    ) -> None:
        seg = self._segments[env_idx]
        seg.obs.append(np.asarray(obs, dtype=np.float32))
        seg.actions.append(int(action))
        seg.log_probs.append(float(log_prob))
        seg.rewards.append(float(reward))
        seg.values.append(float(value))

    def has_open_segment(self, env_idx: int) -> bool:
        return len(self._segments[env_idx].rewards) > 0

    def close_segment(self, env_idx: int, bootstrap_value: float = 0.0) -> None:
        seg = self._segments[env_idx]
        if not seg.rewards:
            return

        rewards = np.asarray(seg.rewards, dtype=np.float32)
        values = np.asarray(seg.values, dtype=np.float32)
        advantages = np.zeros_like(rewards)

        last_gae = 0.0
        next_value = float(bootstrap_value)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lam * last_gae
            advantages[t] = last_gae
            next_value = values[t]

        returns = advantages + values

        self.obs.extend(seg.obs)
        self.actions.extend(seg.actions)
        self.log_probs.extend(seg.log_probs)
        self.advantages.extend(advantages.tolist())
        self.returns.extend(returns.tolist())

        self._segments[env_idx] = Segment(
            obs=[], actions=[], log_probs=[], rewards=[], values=[]
        )

    def size(self) -> int:
        return len(self.actions)

    def tensors(self, device: torch.device):
        obs = torch.tensor(np.asarray(self.obs), dtype=torch.float32, device=device)
        actions = torch.tensor(np.asarray(self.actions), dtype=torch.long, device=device)
        old_log_probs = torch.tensor(
            np.asarray(self.log_probs), dtype=torch.float32, device=device
        )
        advantages = torch.tensor(
            np.asarray(self.advantages), dtype=torch.float32, device=device
        )
        returns = torch.tensor(
            np.asarray(self.returns), dtype=torch.float32, device=device
        )
        return obs, actions, old_log_probs, advantages, returns


def ppo_update(
    net: ActorCritic,
    opt: optim.Optimizer,
    buf: BehaviorRolloutBuffer,
    device: torch.device,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    n_epochs: int,
    mini_batch: int,
    max_grad: float,
) -> Dict[str, float]:
    obs, actions, old_log_probs, advantages, returns = buf.tensors(device)
    if obs.shape[0] == 0:
        return {}

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    total = obs.shape[0]

    metrics = collections.defaultdict(list)

    for _ in range(n_epochs):
        idx = torch.randperm(total, device=device)
        for start in range(0, total, mini_batch):
            mb = idx[start : start + mini_batch]

            new_log_p, entropy, values = net.evaluate(obs[mb], actions[mb])
            ratio = torch.exp(new_log_p - old_log_probs[mb])

            adv_mb = advantages[mb]
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, returns[mb])
            entropy_loss = -entropy.mean()
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad)
            opt.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
            metrics["approx_kl"].append(approx_kl)

    metrics["n_samples"] = [float(total)]
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def make_env_fn(OBELIX, args, worker_seed: int):
    def _make():
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=worker_seed,
        )
        original_step = env.step
        env.step = lambda action: original_step(action, render=False)
        return env

    return _make


def save_checkpoint_bundle(
    out_dir: str,
    nets: Dict[str, ActorCritic],
    device: torch.device,
    suffix: str = "",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for behavior, net in nets.items():
        path = os.path.join(out_dir, f"weights_{behavior}{suffix}.pth")
        torch.save(net.cpu().state_dict(), path)
        net.to(device)


def main():
    ap = argparse.ArgumentParser(
        description="Three-policy PPO trainer for OBELIX using frame stacking or raw observations"
    )
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="three_policy_weights")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--n_envs", type=int, default=8)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lam", type=float, default=0.95)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.02)
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--rollout_len", type=int, default=512)
    ap.add_argument("--mini_batch", type=int, default=64)
    ap.add_argument("--max_grad", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--stack", type=int, default=8)
    ap.add_argument(
        "--no_state_encoder",
        action="store_true",
        help="Use raw 18-d observations instead of the handcrafted belief-state encoder.",
    )

    ap.add_argument("--reward_scale", type=float, default=20.0)
    ap.add_argument("--min_samples_per_behavior", type=int, default=100)

    ap.add_argument("--push_linger_steps", type=int, default=5)
    ap.add_argument("--unwedge_linger_steps", type=int, default=5)
    ap.add_argument("--attach_reward_threshold", type=float, default=90.0)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else DEVICE

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    make_fns = [
        make_env_fn(OBELIX, args, worker_seed=args.seed + i)
        for i in range(args.n_envs)
    ]
    vec = VecEnv(make_fns=make_fns, reward_shaping_fn=None)

    use_state_encoder = not args.no_state_encoder
    encoders = (
        [BeliefStateEncoder(stack_k=args.stack) for _ in range(args.n_envs)]
        if use_state_encoder
        else []
    )
    manager_cfg = BehaviorManagerConfig(
        push_linger_steps=args.push_linger_steps,
        unwedge_linger_steps=args.unwedge_linger_steps,
        attach_reward_threshold=args.attach_reward_threshold,
        sticky_push=True,
        activate_push_on_ir=True,
    )
    managers = [BehaviorManager(manager_cfg) for _ in range(args.n_envs)]

    obs_dim = encoders[0].output_dim if use_state_encoder else 18
    nets = {
        behavior: ActorCritic(in_dim=obs_dim, hidden=args.hidden).to(device)
        for behavior in BEHAVIORS
    }
    opts = {
        behavior: optim.Adam(nets[behavior].parameters(), lr=args.lr, eps=1e-5)
        for behavior in BEHAVIORS
    }
    schedulers = {
        behavior: optim.lr_scheduler.ConstantLR(opts[behavior], factor=1.0, total_iters=999999)
        for behavior in BEHAVIORS
    }

    buffers = {
        behavior: BehaviorRolloutBuffer(
            n_envs=args.n_envs, gamma=args.gamma, gae_lam=args.gae_lam
        )
        for behavior in BEHAVIORS
    }

    init_seeds = [args.seed + i for i in range(args.n_envs)]
    raw_obs_list = vec.reset(seeds=init_seeds)
    for i in range(args.n_envs):
        managers[i].reset(raw_obs_list[i])
        if use_state_encoder:
            encoders[i].reset()
    if use_state_encoder:
        obs_arr = np.asarray(
            [encoders[i].encode(raw_obs_list[i]) for i in range(args.n_envs)],
            dtype=np.float32,
        )
    else:
        obs_arr = np.asarray(raw_obs_list, dtype=np.float32)

    ep_ret = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps = np.zeros(args.n_envs, dtype=np.int32)
    behavior_steps = {behavior: 0 for behavior in BEHAVIORS}
    episodes_done = 0
    total_steps = 0
    update_count = 0
    best_avg_return = -float("inf")
    window_returns: List[float] = []
    window_steps: List[int] = []
    success_count = 0
    recent_metrics = {
        behavior: collections.defaultdict(lambda: collections.deque(maxlen=20))
        for behavior in BEHAVIORS
    }
    train_start = time.time()
    last_log_ep = 0
    log_every = 10

    print(
        f"\n[Three-Policy PPO] envs={args.n_envs} stack={args.stack} obs_dim={obs_dim} "
        f"rollout={args.rollout_len} reward_scale={args.reward_scale}"
    )
    print(
        f"[Three-Policy PPO] policies={', '.join(BEHAVIORS)} "
        f"difficulty={args.difficulty} wall={args.wall_obstacles} "
        f"repr={'state-encoder' if use_state_encoder else 'raw'}\n"
    )

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=120)

    while episodes_done < args.episodes:
        for behavior in BEHAVIORS:
            buffers[behavior].clear()

        for _ in range(args.rollout_len):
            current_behaviors = [
                managers[i].current_behavior(raw_obs_list[i]) for i in range(args.n_envs)
            ]

            action_indices = np.zeros(args.n_envs, dtype=np.int64)
            log_probs = np.zeros(args.n_envs, dtype=np.float32)
            values = np.zeros(args.n_envs, dtype=np.float32)

            for behavior in BEHAVIORS:
                idx = np.asarray(
                    [i for i, b in enumerate(current_behaviors) if b == behavior],
                    dtype=np.int64,
                )
                if idx.size == 0:
                    continue

                behavior_steps[behavior] += int(idx.size)
                obs_t = torch.tensor(obs_arr[idx], dtype=torch.float32, device=device)
                with torch.no_grad():
                    actions_t, logp_t, _, values_t = nets[behavior].get_action(obs_t)

                action_indices[idx] = actions_t.cpu().numpy()
                log_probs[idx] = logp_t.cpu().numpy()
                values[idx] = values_t.cpu().numpy()

            action_strs = [ACTIONS[a] for a in action_indices]
            results = vec.step(action_strs)
            next_raw_obs_list = [r[0] for r in results]
            raw_rewards = np.asarray([r[1] for r in results], dtype=np.float32)
            dones = np.asarray([r[2] for r in results], dtype=bool)

            if use_state_encoder:
                next_obs_arr = np.empty_like(obs_arr)
                for i in range(args.n_envs):
                    next_obs_arr[i] = encoders[i].encode(next_raw_obs_list[i], action_indices[i])
            else:
                next_obs_arr = np.asarray(next_raw_obs_list, dtype=np.float32)

            scaled_rewards = raw_rewards / args.reward_scale

            for i, behavior in enumerate(current_behaviors):
                buffers[behavior].add_step(
                    env_idx=i,
                    obs=obs_arr[i],
                    action=action_indices[i],
                    log_prob=log_probs[i],
                    reward=float(scaled_rewards[i]),
                    value=float(values[i]),
                )

            ep_ret += scaled_rewards
            ep_steps += 1
            total_steps += args.n_envs

            for i, behavior in enumerate(current_behaviors):
                if dones[i]:
                    buffers[behavior].close_segment(i, bootstrap_value=0.0)

                    if raw_rewards[i] >= 100.0:
                        success_count += 1

                    window_returns.append(float(ep_ret[i]))
                    window_steps.append(int(ep_steps[i]))
                    episodes_done += 1

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "ret": f"{ep_ret[i]:.1f}",
                            "succ": success_count,
                            "find": behavior_steps[FIND],
                            "push": behavior_steps[PUSH],
                            "unw": behavior_steps[UNWEDGE],
                        }
                    )

                    new_seed = args.seed + args.n_envs + episodes_done
                    reset_obs = vec.reset_one(i, seed=new_seed)
                    raw_obs_list[i] = reset_obs
                    managers[i].reset(reset_obs)
                    if use_state_encoder:
                        encoders[i].reset()
                        next_obs_arr[i] = encoders[i].encode(reset_obs)
                    else:
                        next_obs_arr[i] = np.asarray(reset_obs, dtype=np.float32)
                    ep_ret[i] = 0.0
                    ep_steps[i] = 0

                    if episodes_done >= args.episodes:
                        break
                else:
                    managers[i].update(next_raw_obs_list[i], float(raw_rewards[i]), False)
                    raw_obs_list[i] = next_raw_obs_list[i]
                    next_behavior = managers[i].current_behavior(raw_obs_list[i])
                    if next_behavior != behavior:
                        buffers[behavior].close_segment(i, bootstrap_value=0.0)

            obs_arr = next_obs_arr

            if episodes_done >= args.episodes:
                break

        # Truncate any still-open segments with value bootstrap from the current state.
        post_behaviors = [
            managers[i].current_behavior(raw_obs_list[i]) for i in range(args.n_envs)
        ]
        for behavior in BEHAVIORS:
            idx = np.asarray(
                [
                    i
                    for i, b in enumerate(post_behaviors)
                    if b == behavior and buffers[behavior].has_open_segment(i)
                ],
                dtype=np.int64,
            )
            if idx.size == 0:
                continue

            obs_t = torch.tensor(obs_arr[idx], dtype=torch.float32, device=device)
            with torch.no_grad():
                _, last_values = nets[behavior](obs_t)
            for env_idx, bootstrap in zip(idx.tolist(), last_values.cpu().tolist()):
                buffers[behavior].close_segment(env_idx, bootstrap_value=float(bootstrap))

        update_metrics = {}
        for behavior in BEHAVIORS:
            if buffers[behavior].size() < args.min_samples_per_behavior:
                continue

            metrics = ppo_update(
                net=nets[behavior],
                opt=opts[behavior],
                buf=buffers[behavior],
                device=device,
                clip_eps=args.clip_eps,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                n_epochs=args.n_epochs,
                mini_batch=args.mini_batch,
                max_grad=args.max_grad,
            )
            if metrics:
                update_metrics[behavior] = metrics
                for key, value in metrics.items():
                    recent_metrics[behavior][key].append(value)
                schedulers[behavior].step()

        update_count += 1

        if (episodes_done // log_every) > (last_log_ep // log_every) and window_returns:
            avg_return = float(np.mean(window_returns))
            elapsed = time.time() - train_start

            if avg_return > best_avg_return:
                best_avg_return = avg_return
                save_checkpoint_bundle(args.out_dir, nets, device=device, suffix=".best")

            tqdm.write(
                f"\n┌─ ep {episodes_done - len(window_returns) + 1:>4d}–{episodes_done:<4d} "
                f"({elapsed:.0f}s) ─────────────────────────────────────────────"
            )
            tqdm.write(
                f"│ Avg Return : {avg_return:8.2f}   Avg Steps : {np.mean(window_steps):6.1f}   "
                f"Successes : {success_count}"
            )
            tqdm.write(
                f"│ Behavior steps : find={behavior_steps[FIND]:,} push={behavior_steps[PUSH]:,} "
                f"unwedge={behavior_steps[UNWEDGE]:,}   Updates : {update_count}"
            )
            for behavior in BEHAVIORS:
                if not recent_metrics[behavior]:
                    continue
                tqdm.write(
                    f"│ {behavior:>7s} : "
                    f"samples={np.mean(recent_metrics[behavior]['n_samples']):7.1f} "
                    f"policy={np.mean(recent_metrics[behavior]['policy_loss']):8.4f} "
                    f"value={np.mean(recent_metrics[behavior]['value_loss']):8.4f} "
                    f"entropy={np.mean(recent_metrics[behavior]['entropy']):8.4f}"
                )
            tqdm.write(f"└{'─' * 86}")

            last_log_ep = episodes_done
            window_returns.clear()
            window_steps.clear()

    pbar.close()
    vec.close()
    save_checkpoint_bundle(args.out_dir, nets, device=device)

    elapsed = time.time() - train_start
    print(f"\nSaved final checkpoints to: {args.out_dir}")
    print("Best rolling-average checkpoint suffix: .best")
    print(
        f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min) | "
        f"Episodes: {episodes_done} | Total steps: {total_steps:,}"
    )

if __name__ == "__main__":
    main()
