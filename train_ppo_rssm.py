"""PPO + RSSM belief-state trainer for OBELIX.

This is a compact Dreamer/RSSM-inspired trainer, but it stays PPO-based:
  - raw 18-bit observations are encoded into a recurrent stochastic belief
  - actor/critic act from belief = [deterministic h_t, stochastic z_t]
  - auxiliary losses train the belief model to predict obs/reward/done
  - no imagination rollouts yet; this keeps the implementation robust/simple
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from reward_shaper import RewardShaper
from state_encoder import BeliefStateEncoder
from vec_env import VecEnv


OBS_DIM = 18
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACT = len(ACTIONS)


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


def one_hot(actions: torch.Tensor, n: int = N_ACT) -> torch.Tensor:
    return nn.functional.one_hot(actions.long(), num_classes=n).float()


class RSSMActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = N_ACT,
        obs_embed_dim: int = 64,
        h_dim: int = 128,
        z_dim: int = 32,
        hidden: int = 128,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_embed_dim = obs_embed_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.min_std = min_std

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_embed_dim),
            nn.ELU(),
            nn.Linear(obs_embed_dim, obs_embed_dim),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(z_dim + action_dim + 2, h_dim)
        self.prior = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2 * z_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(h_dim + obs_embed_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2 * z_dim),
        )

        belief_dim = h_dim + z_dim
        self.actor = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.obs_head = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def _stats(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, raw_std = torch.chunk(raw, 2, dim=-1)
        std = nn.functional.softplus(raw_std) + self.min_std
        return mu, std

    @staticmethod
    def _sample(mu: torch.Tensor, std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        return mu + std * torch.randn_like(std)

    @staticmethod
    def _kl_normal(mu_q, std_q, mu_p, std_p):
        var_q = std_q.pow(2)
        var_p = std_p.pow(2)
        kl = torch.log(std_p / std_q) + (var_q + (mu_q - mu_p).pow(2)) / (2.0 * var_p) - 0.5
        return kl.sum(dim=-1)

    def observe_step(
        self,
        obs: torch.Tensor,
        prev_action_onehot: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        deterministic_z: bool = False,
    ) -> Dict[str, torch.Tensor]:
        reset = prev_done.view(-1, 1).bool()
        h_prev = torch.where(reset, torch.zeros_like(h_prev), h_prev)
        z_prev = torch.where(reset, torch.zeros_like(z_prev), z_prev)

        gru_in = torch.cat(
            [z_prev, prev_action_onehot, prev_reward.view(-1, 1), prev_done.view(-1, 1)],
            dim=-1,
        )
        h = self.gru(gru_in, h_prev)

        prior_mu, prior_std = self._stats(self.prior(h))
        obs_embed = self.obs_encoder(obs)
        post_mu, post_std = self._stats(self.posterior(torch.cat([h, obs_embed], dim=-1)))
        z = self._sample(post_mu, post_std, deterministic=deterministic_z)

        belief = torch.cat([h, z], dim=-1)
        logits = self.actor(belief)
        value = self.critic(belief).squeeze(-1)
        obs_logits = self.obs_head(belief)
        reward_pred = self.reward_head(belief).squeeze(-1)
        done_logits = self.done_head(belief).squeeze(-1)
        kl = self._kl_normal(post_mu, post_std, prior_mu, prior_std)

        return {
            "h": h,
            "z": z,
            "belief": belief,
            "logits": logits,
            "value": value,
            "obs_logits": obs_logits,
            "reward_pred": reward_pred,
            "done_logits": done_logits,
            "kl": kl,
            "prior_mu": prior_mu,
            "prior_std": prior_std,
            "post_mu": post_mu,
            "post_std": post_std,
        }

    def observe_sequence(
        self,
        obs: torch.Tensor,  # [T, B, obs_dim]
        prev_actions: torch.Tensor,  # [T, B]
        prev_rewards: torch.Tensor,  # [T, B]
        prev_dones: torch.Tensor,  # [T, B]
        h0: torch.Tensor,
        z0: torch.Tensor,
        deterministic_z: bool = False,
    ) -> Dict[str, torch.Tensor]:
        T, B, _ = obs.shape
        h, z = h0, z0
        outs = collections.defaultdict(list)
        for t in range(T):
            out = self.observe_step(
                obs[t],
                one_hot(prev_actions[t], self.action_dim),
                prev_rewards[t],
                prev_dones[t],
                h,
                z,
                deterministic_z=deterministic_z,
            )
            h, z = out["h"], out["z"]
            for key, value in out.items():
                outs[key].append(value)
        return {key: torch.stack(values, dim=0) for key, values in outs.items()}


class RolloutBuffer:
    def __init__(self, rollout_len: int, n_envs: int):
        self.rollout_len = rollout_len
        self.n_envs = n_envs
        self.clear()

    def clear(self):
        T, N = self.rollout_len, self.n_envs
        self.obs = np.zeros((T, N, OBS_DIM), dtype=np.float32)
        self.prev_actions = np.zeros((T, N), dtype=np.int64)
        self.prev_rewards = np.zeros((T, N), dtype=np.float32)
        self.prev_dones = np.ones((T, N), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.raw_rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, prev_actions, prev_rewards, prev_dones, actions, log_probs, rewards, raw_rewards, dones, values):
        t = self.ptr
        self.obs[t] = obs
        self.prev_actions[t] = prev_actions
        self.prev_rewards[t] = prev_rewards
        self.prev_dones[t] = prev_dones
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.raw_rewards[t] = raw_rewards
        self.dones[t] = dones
        self.values[t] = values
        self.ptr += 1

    def compute_advantages(self, last_values: np.ndarray, gamma: float, gae_lam: float):
        T, N = self.ptr, self.n_envs
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)
        next_values = last_values.astype(np.float32)
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lam * nonterminal * last_gae
            adv[t] = last_gae
            next_values = self.values[t]
        returns = adv + self.values[:T]
        return adv, returns


def ppo_rssm_update(
    net: RSSMActorCritic,
    opt: optim.Optimizer,
    buf: RolloutBuffer,
    h0: torch.Tensor,
    z0: torch.Tensor,
    last_values: np.ndarray,
    args,
    device: torch.device,
) -> Dict[str, float]:
    T, N = buf.ptr, buf.n_envs
    adv, returns = buf.compute_advantages(last_values, args.gamma, args.gae_lam)
    adv_flat = adv.reshape(-1)
    adv = (adv - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    obs = torch.tensor(buf.obs[:T], dtype=torch.float32, device=device)
    prev_actions = torch.tensor(buf.prev_actions[:T], dtype=torch.long, device=device)
    prev_rewards = torch.tensor(buf.prev_rewards[:T], dtype=torch.float32, device=device)
    prev_dones = torch.tensor(buf.prev_dones[:T], dtype=torch.float32, device=device)
    actions = torch.tensor(buf.actions[:T], dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buf.log_probs[:T], dtype=torch.float32, device=device)
    rewards = torch.tensor(buf.rewards[:T], dtype=torch.float32, device=device)
    dones = torch.tensor(buf.dones[:T], dtype=torch.float32, device=device)
    advantages = torch.tensor(adv, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    metrics = collections.defaultdict(list)
    for _ in range(args.n_epochs):
        env_perm = torch.randperm(N, device=device)
        for start in range(0, N, args.mini_batch_envs):
            env_idx = env_perm[start : start + args.mini_batch_envs]
            out = net.observe_sequence(
                obs[:, env_idx],
                prev_actions[:, env_idx],
                prev_rewards[:, env_idx],
                prev_dones[:, env_idx],
                h0[env_idx].detach(),
                z0[env_idx].detach(),
                deterministic_z=False,
            )

            logits = out["logits"]
            values = out["value"]
            dist = Categorical(logits=logits.reshape(-1, N_ACT))
            new_log_probs = dist.log_prob(actions[:, env_idx].reshape(-1)).view(T, -1)
            entropy = dist.entropy().view(T, -1)

            ratio = torch.exp(new_log_probs - old_log_probs[:, env_idx])
            adv_mb = advantages[:, env_idx]
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, returns_t[:, env_idx])
            entropy_loss = -entropy.mean()

            obs_loss = nn.functional.binary_cross_entropy_with_logits(out["obs_logits"], obs[:, env_idx])
            reward_loss = nn.functional.mse_loss(out["reward_pred"], rewards[:, env_idx])
            done_loss = nn.functional.binary_cross_entropy_with_logits(out["done_logits"], dones[:, env_idx])
            kl_loss = out["kl"].mean()

            loss = (
                policy_loss
                + args.vf_coef * value_loss
                + args.ent_coef * entropy_loss
                + args.obs_coef * obs_loss
                + args.reward_coef * reward_loss
                + args.done_coef * done_loss
                + args.kl_coef * kl_loss
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.max_grad)
            opt.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append((-entropy_loss).item())
            metrics["obs_loss"].append(obs_loss.item())
            metrics["reward_loss"].append(reward_loss.item())
            metrics["done_loss"].append(done_loss.item())
            metrics["kl_loss"].append(kl_loss.item())
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
            metrics["approx_kl"].append(approx_kl)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def save_checkpoint(path: str, net: RSSMActorCritic, opt: optim.Optimizer, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "config": {
                "obs_embed_dim": args.obs_embed_dim,
                "h_dim": args.h_dim,
                "z_dim": args.z_dim,
                "hidden": args.hidden,
                "min_std": args.min_std,
            },
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser(description="PPO + RSSM belief-state trainer for OBELIX")
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_ppo_rssm.pth")
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--obs_embed_dim", type=int, default=64)
    ap.add_argument("--h_dim", type=int, default=128)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--min_std", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae_lam", type=float, default=0.97)
    ap.add_argument("--clip_eps", type=float, default=0.15)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.003)
    ap.add_argument("--obs_coef", type=float, default=0.1)
    ap.add_argument("--reward_coef", type=float, default=0.05)
    ap.add_argument("--done_coef", type=float, default=0.05)
    ap.add_argument("--kl_coef", type=float, default=0.01)
    ap.add_argument("--rollout_len", type=int, default=128)
    ap.add_argument("--n_epochs", type=int, default=3)
    ap.add_argument("--mini_batch_envs", type=int, default=4)
    ap.add_argument("--max_grad", type=float, default=0.5)
    ap.add_argument("--reward_scale", type=float, default=50.0)
    ap.add_argument("--attach_reward_threshold", type=float, default=90.0)
    ap.add_argument(
        "--no_reward_shaper",
        action="store_true",
        help="Disable RewardShaper and train on scaled raw rewards only.",
    )
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument(
        "--log_updates",
        action="store_true",
        help="Print a compact line after every PPO/RSSM update, even if no episode finished.",
    )
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    vec = VecEnv(
        make_fns=[make_env_fn(OBELIX, args, args.seed + i) for i in range(args.n_envs)],
        reward_shaping_fn=None,
    )

    net = RSSMActorCritic(
        obs_embed_dim=args.obs_embed_dim,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        hidden=args.hidden,
        min_std=args.min_std,
    ).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    if args.load:
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[load] {args.load}")

    raw_obs = np.asarray(vec.reset([args.seed + i for i in range(args.n_envs)]), dtype=np.float32)
    reward_shapers = [RewardShaper(reward_scale=args.reward_scale) for _ in range(args.n_envs)]
    shaper_encoders = [BeliefStateEncoder(stack_k=1) for _ in range(args.n_envs)]
    enable_push = np.zeros(args.n_envs, dtype=bool)
    for i in range(args.n_envs):
        shaper_encoders[i].reset()
        shaper_encoders[i].encode(raw_obs[i], prev_action_idx=0)
    h, z = net.init_state(args.n_envs, device)
    prev_actions = np.zeros(args.n_envs, dtype=np.int64)
    prev_rewards = np.zeros(args.n_envs, dtype=np.float32)
    prev_dones = np.ones(args.n_envs, dtype=np.float32)

    ep_returns = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps = np.zeros(args.n_envs, dtype=np.int32)
    window_returns: List[float] = []
    window_steps: List[int] = []
    window_shaping: List[float] = []
    recent_success = collections.deque(maxlen=200)
    episodes_done = 0
    last_log_ep = 0
    total_steps = 0
    updates = 0
    best_return = -float("inf")
    train_start = time.time()

    print(
        f"\n[PPO+RSSM] envs={args.n_envs} h={args.h_dim} z={args.z_dim} "
        f"rollout={args.rollout_len} reward_scale={args.reward_scale}"
    )
    print(
        f"[PPO+RSSM] difficulty={args.difficulty} wall={args.wall_obstacles} "
        f"aux=(obs={args.obs_coef}, reward={args.reward_coef}, done={args.done_coef}, kl={args.kl_coef}) "
        f"reward_shaper={'OFF' if args.no_reward_shaper else 'ON'}\n"
    )

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=120)

    while episodes_done < args.episodes:
        buf = RolloutBuffer(args.rollout_len, args.n_envs)
        h0 = h.detach().clone()
        z0 = z.detach().clone()

        for _ in range(args.rollout_len):
            obs_t = torch.tensor(raw_obs, dtype=torch.float32, device=device)
            prev_actions_t = torch.tensor(prev_actions, dtype=torch.long, device=device)
            prev_rewards_t = torch.tensor(prev_rewards, dtype=torch.float32, device=device)
            prev_dones_t = torch.tensor(prev_dones, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = net.observe_step(
                    obs_t,
                    one_hot(prev_actions_t, N_ACT),
                    prev_rewards_t,
                    prev_dones_t,
                    h,
                    z,
                    deterministic_z=True,
                )
                logits, values = out["logits"], out["value"]
                dist = Categorical(logits=logits)
                actions_t = dist.sample()
                log_probs_t = dist.log_prob(actions_t)
                h, z = out["h"], out["z"]

            actions_np = actions_t.cpu().numpy()
            results = vec.step([ACTIONS[a] for a in actions_np])
            next_obs = np.asarray([r[0] for r in results], dtype=np.float32)
            raw_rewards = np.asarray([r[1] for r in results], dtype=np.float32)
            dones = np.asarray([r[2] for r in results], dtype=bool)
            rewards = raw_rewards / args.reward_scale
            if not args.no_reward_shaper:
                shaped_rewards = np.empty_like(rewards)
                for i in range(args.n_envs):
                    if raw_rewards[i] >= args.attach_reward_threshold:
                        enable_push[i] = True
                    shaper_encoders[i].encode(next_obs[i], prev_action_idx=int(actions_np[i]))
                    encoded_for_shaper = shaper_encoders[i].encode_single(
                        next_obs[i], prev_action_idx=int(actions_np[i])
                    )
                    shaped_rewards[i] = reward_shapers[i].shape(
                        raw_reward=float(rewards[i]),
                        encoded_obs=encoded_for_shaper,
                        done=bool(dones[i]),
                        enable_push=bool(enable_push[i]),
                        action=ACTIONS[int(actions_np[i])],
                    )
                window_shaping.extend((shaped_rewards - rewards).tolist())
                rewards = shaped_rewards

            buf.add(
                obs=raw_obs,
                prev_actions=prev_actions,
                prev_rewards=prev_rewards,
                prev_dones=prev_dones,
                actions=actions_np,
                log_probs=log_probs_t.cpu().numpy(),
                rewards=rewards,
                raw_rewards=raw_rewards,
                dones=dones.astype(np.float32),
                values=values.cpu().numpy(),
            )

            ep_returns += raw_rewards
            ep_steps += 1
            total_steps += args.n_envs
            prev_actions = actions_np
            prev_rewards = rewards.astype(np.float32)
            prev_dones = dones.astype(np.float32)
            raw_obs = next_obs

            for i in range(args.n_envs):
                if dones[i]:
                    terminal_reward = raw_rewards[i]
                    window_returns.append(float(ep_returns[i]))
                    window_steps.append(int(ep_steps[i]))
                    recent_success.append(float(terminal_reward >= 1000.0))
                    best_return = max(best_return, float(ep_returns[i]))
                    episodes_done += 1
                    pbar.update(1)

                    raw_obs[i] = vec.reset_one(i, seed=args.seed + args.n_envs + episodes_done + i)
                    reward_shapers[i].reset()
                    shaper_encoders[i].reset()
                    shaper_encoders[i].encode(raw_obs[i], prev_action_idx=0)
                    enable_push[i] = False
                    h[i] = 0.0
                    z[i] = 0.0
                    prev_actions[i] = 0
                    prev_rewards[i] = 0.0
                    prev_dones[i] = 1.0
                    ep_returns[i] = 0.0
                    ep_steps[i] = 0
                    if episodes_done >= args.episodes:
                        break

            if episodes_done >= args.episodes:
                break

        with torch.no_grad():
            obs_t = torch.tensor(raw_obs, dtype=torch.float32, device=device)
            prev_actions_t = torch.tensor(prev_actions, dtype=torch.long, device=device)
            prev_rewards_t = torch.tensor(prev_rewards, dtype=torch.float32, device=device)
            prev_dones_t = torch.tensor(prev_dones, dtype=torch.float32, device=device)
            out = net.observe_step(
                obs_t,
                one_hot(prev_actions_t, N_ACT),
                prev_rewards_t,
                prev_dones_t,
                h,
                z,
                deterministic_z=True,
            )
            last_values = out["value"].cpu().numpy() * (1.0 - prev_dones)

        metrics = ppo_rssm_update(net, opt, buf, h0, z0, last_values, args, device)
        updates += 1

        if window_returns:
            recent_win = min(max(1, len(window_returns)), args.log_every)
            recent_ret = float(np.mean(window_returns[-recent_win:]))
            succ = 100.0 * float(np.mean(recent_success)) if recent_success else 0.0
            pbar.set_postfix(
                ret=f"{recent_ret:.1f}",
                best=f"{best_return:.1f}",
                succ=f"{succ:.0f}",
                upd=updates,
            )
        else:
            pbar.set_postfix(
                live_ret=f"{float(np.mean(ep_returns)):.1f}",
                upd=updates,
                steps=total_steps,
            )

        if args.log_updates:
            tqdm.write(
                f"[update {updates}] episodes={episodes_done}/{args.episodes} "
                f"total_steps={total_steps} live_return_mean={float(np.mean(ep_returns)):.2f} "
                f"policy={metrics.get('policy_loss', 0.0):.4f} "
                f"value={metrics.get('value_loss', 0.0):.4f} "
                f"entropy={metrics.get('entropy', 0.0):.4f} "
                f"rssm_kl={metrics.get('kl_loss', 0.0):.4f}"
            )

        if (
            episodes_done > 0
            and (episodes_done - last_log_ep) >= args.log_every
            and window_returns
        ):
            win = min(args.log_every, len(window_returns))
            avg_return = float(np.mean(window_returns[-win:]))
            avg_steps = float(np.mean(window_steps[-win:]))
            succ = 100.0 * float(np.mean(recent_success)) if recent_success else 0.0
            speed = episodes_done / max(time.time() - train_start, 1e-6)
            tqdm.write(
                f"\n[ep {max(1, episodes_done-win+1)}-{episodes_done}] "
                f"avg_return={avg_return:.2f} best={best_return:.2f} "
                f"avg_steps={avg_steps:.1f} rolling_success(200)={succ:.2f}%"
            )
            tqdm.write(
                f"policy={metrics.get('policy_loss', 0.0):.4f} "
                f"value={metrics.get('value_loss', 0.0):.4f} "
                f"entropy={metrics.get('entropy', 0.0):.4f} "
                f"kl_ppo={metrics.get('approx_kl', 0.0):.4f} "
                f"rssm_kl={metrics.get('kl_loss', 0.0):.4f} "
                f"obs={metrics.get('obs_loss', 0.0):.4f} "
                f"rew={metrics.get('reward_loss', 0.0):.4f} "
                f"done={metrics.get('done_loss', 0.0):.4f} "
                f"updates={updates} speed={speed:.2f} ep/s"
            )
            if window_shaping:
                tqdm.write(f"reward_shaper_delta/step={float(np.mean(window_shaping)):.4f}")
                window_shaping.clear()
            last_log_ep = episodes_done

    vec.close()
    save_checkpoint(args.out, net, opt, args)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
