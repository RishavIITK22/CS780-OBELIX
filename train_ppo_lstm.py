from __future__ import annotations

import argparse
import collections
import importlib.util
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from learned_latent_curiosity import (
    LatentBeliefActorCritic,
    N_ACT,
    OBS_DIM,
    RunningMeanStd,
)
from vec_env import VecEnv


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


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


def import_class_from_path(path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("dynamic_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def import_obelix(path: str):
    return import_class_from_path(path, "OBELIX")


@dataclass
class CurriculumStage:
    name: str
    episodes: int
    difficulty: int
    wall_obstacles: bool
    box_speed: int
    max_steps: int


def _episode_split(total_episodes: int, ratios: List[float]) -> List[int]:
    counts = [int(total_episodes * r) for r in ratios]
    remainder = total_episodes - sum(counts)
    counts[-1] += remainder
    return counts


def build_curriculum(args) -> List[CurriculumStage]:
    if args.no_curriculum:
        return [
            CurriculumStage(
                name="single-stage",
                episodes=args.episodes,
                difficulty=args.difficulty,
                wall_obstacles=args.wall_obstacles,
                box_speed=args.box_speed,
                max_steps=args.max_steps,
            )
        ]

    if args.curriculum == "gentle":
        ratios = [0.20, 0.20, 0.25, 0.35]
        counts = _episode_split(args.episodes, ratios)
        return [
            CurriculumStage("static_open", counts[0], 0, False, 0, min(args.max_steps, 600)),
            CurriculumStage("static_wall", counts[1], 0, True, 0, min(args.max_steps, 700)),
            CurriculumStage("blinking_wall", counts[2], 2, True, 0, min(args.max_steps, 900)),
            CurriculumStage("moving_wall", counts[3], 3, True, args.box_speed, args.max_steps),
        ]

    # default / focused
    ratios = [0.15, 0.20, 0.25, 0.40]
    counts = _episode_split(args.episodes, ratios)
    return [
        CurriculumStage("static_open", counts[0], 0, False, 0, min(args.max_steps, 600)),
        CurriculumStage("static_wall", counts[1], 0, True, 0, min(args.max_steps, 750)),
        CurriculumStage("blinking_wall", counts[2], 2, True, 0, min(args.max_steps, 900)),
        CurriculumStage("moving_wall", counts[3], 3, True, args.box_speed, args.max_steps),
    ]


class RolloutBuffer:
    def __init__(self, rollout_len: int, n_envs: int):
        self.T = rollout_len
        self.N = n_envs
        self._init_h: Optional[torch.Tensor] = None
        self._init_c: Optional[torch.Tensor] = None
        self.clear()

    def clear(self) -> None:
        self.obs = []
        self.prev_actions = []
        self.prev_rewards = []
        self.prev_dones = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.ext_rewards = []
        self.values = []
        self.dones = []
        self.next_obs = []

    def store_init_hidden(self, h: torch.Tensor, c: torch.Tensor) -> None:
        self._init_h = h.detach().clone()
        self._init_c = c.detach().clone()

    def get_init_hidden(self, device: torch.device):
        assert self._init_h is not None and self._init_c is not None
        return self._init_h.to(device), self._init_c.to(device)

    def add_batch(
        self,
        obs: np.ndarray,
        prev_actions: np.ndarray,
        prev_rewards: np.ndarray,
        prev_dones: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        ext_rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        for i in range(self.N):
            self.obs.append(obs[i])
            self.prev_actions.append(prev_actions[i])
            self.prev_rewards.append(prev_rewards[i])
            self.prev_dones.append(prev_dones[i])
            self.actions.append(actions[i])
            self.log_probs.append(log_probs[i])
            self.rewards.append(rewards[i])
            self.ext_rewards.append(ext_rewards[i])
            self.values.append(values[i])
            self.dones.append(dones[i])
            self.next_obs.append(next_obs[i])

    def _t_actual(self) -> int:
        return len(self.rewards) // self.N

    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lam: float,
    ):
        T_actual = self._t_actual()
        n_trim = T_actual * self.N

        rewards = np.array(self.rewards[:n_trim], dtype=np.float32).reshape(T_actual, self.N)
        values = np.array(self.values[:n_trim], dtype=np.float32).reshape(T_actual, self.N)
        dones = np.array(self.dones[:n_trim], dtype=np.float32).reshape(T_actual, self.N)

        advantages = np.zeros((T_actual, self.N), dtype=np.float32)
        last_gae = np.zeros(self.N, dtype=np.float32)

        for t in reversed(range(T_actual)):
            next_val = last_values if t == T_actual - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * not_done - values[t]
            last_gae = delta + gamma * gae_lam * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return (
            torch.from_numpy(advantages.reshape(-1)),
            torch.from_numpy(returns.reshape(-1)),
        )

    def tensors(self, device: torch.device):
        T_actual = self._t_actual()
        n_trim = T_actual * self.N

        obs_flat = torch.tensor(np.array(self.obs[:n_trim]), dtype=torch.float32, device=device)
        next_obs_flat = torch.tensor(np.array(self.next_obs[:n_trim]), dtype=torch.float32, device=device)
        prev_actions_flat = torch.tensor(np.array(self.prev_actions[:n_trim]), dtype=torch.long, device=device)
        prev_rewards_flat = torch.tensor(np.array(self.prev_rewards[:n_trim]), dtype=torch.float32, device=device)
        prev_dones_flat = torch.tensor(np.array(self.prev_dones[:n_trim]), dtype=torch.float32, device=device)
        actions_flat = torch.tensor(np.array(self.actions[:n_trim]), dtype=torch.long, device=device)
        old_log_probs = torch.tensor(np.array(self.log_probs[:n_trim]), dtype=torch.float32, device=device)
        ext_rewards_flat = torch.tensor(np.array(self.ext_rewards[:n_trim]), dtype=torch.float32, device=device)
        dones_flat = torch.tensor(np.array(self.dones[:n_trim]), dtype=torch.float32, device=device)

        obs_seq = obs_flat.reshape(T_actual, self.N, OBS_DIM).permute(1, 0, 2).contiguous()
        next_obs_seq = next_obs_flat.reshape(T_actual, self.N, OBS_DIM).permute(1, 0, 2).contiguous()
        prev_action_seq = prev_actions_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        prev_reward_seq = prev_rewards_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        prev_done_seq = prev_dones_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        actions_seq = actions_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        ext_reward_seq = ext_rewards_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        dones_seq = dones_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()

        return {
            "obs_seq": obs_seq,
            "next_obs_seq": next_obs_seq,
            "prev_action_seq": prev_action_seq,
            "prev_reward_seq": prev_reward_seq,
            "prev_done_seq": prev_done_seq,
            "actions_seq": actions_seq,
            "ext_reward_seq": ext_reward_seq,
            "dones_seq": dones_seq,
            "actions_flat": actions_flat,
            "old_log_probs": old_log_probs,
        }


def ppo_update(
    net: LatentBeliefActorCritic,
    opt: optim.Optimizer,
    buf: RolloutBuffer,
    last_values: np.ndarray,
    device: torch.device,
    gamma: float,
    gae_lam: float,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    n_epochs: int,
    n_mini_batches: int,
    max_grad: float,
    target_kl: float,
    obs_coef: float,
    reward_coef: float,
    done_coef: float,
    inverse_coef: float,
    forward_coef: float,
):
    advantages, returns = buf.compute_gae(last_values, gamma, gae_lam)
    advantages = advantages.to(device)
    returns = returns.to(device)
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False)
    if not torch.isfinite(adv_std) or adv_std < 1e-8:
        adv_std = torch.tensor(1.0, device=device)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    tensors = buf.tensors(device)
    init_h, init_c = buf.get_init_hidden(device)
    T_actual = buf._t_actual()
    N = buf.N

    metrics = collections.defaultdict(list)
    early_stop = False

    for _ in range(n_epochs):
        worker_indices = torch.randperm(N, device=device)
        mb_size = max(1, N // n_mini_batches)

        for mb_start in range(0, N, mb_size):
            wb = worker_indices[mb_start : mb_start + mb_size]
            N_mb = wb.shape[0]

            outputs = net.evaluate_sequence(
                obs_seq=tensors["obs_seq"][wb],
                prev_action_seq=tensors["prev_action_seq"][wb],
                prev_reward_seq=tensors["prev_reward_seq"][wb],
                prev_done_seq=tensors["prev_done_seq"][wb],
                actions_seq=tensors["actions_seq"][wb],
                next_obs_seq=tensors["next_obs_seq"][wb],
                ext_reward_seq=tensors["ext_reward_seq"][wb],
                dones_seq=tensors["dones_seq"][wb],
                init_h=init_h[:, wb, :].contiguous(),
                init_c=init_c[:, wb, :].contiguous(),
            )

            t_idx = torch.arange(T_actual, device=device)
            global_idx = (t_idx.unsqueeze(0) * N + wb.unsqueeze(1))
            flat_idx = global_idx.permute(1, 0).reshape(-1)

            act_mb = tensors["actions_flat"][flat_idx]
            old_logp_mb = tensors["old_log_probs"][flat_idx]
            adv_mb = advantages[flat_idx]
            ret_mb = returns[flat_idx]

            dist = Categorical(logits=outputs["logits_flat"])
            new_logp = dist.log_prob(act_mb)
            entropy = outputs["entropy_flat"].mean()

            ratio = torch.exp(new_logp - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(outputs["values_flat"], ret_mb)

            obs_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["obs_pred_flat"], outputs["next_obs_target_flat"]
            )
            reward_loss = nn.functional.mse_loss(
                outputs["reward_pred_flat"], outputs["ext_reward_target_flat"]
            )
            done_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["done_logit_flat"], outputs["done_target_flat"]
            )
            inverse_loss = nn.functional.cross_entropy(
                outputs["inverse_logits_flat"], act_mb
            )
            forward_loss = 0.5 * torch.mean(
                (outputs["next_feat_pred_flat"] - outputs["next_feat_target_flat"]) ** 2
            )

            loss = (
                policy_loss
                + vf_coef * value_loss
                - ent_coef * entropy
                + obs_coef * obs_loss
                + reward_coef * reward_loss
                + done_coef * done_loss
                + inverse_coef * inverse_loss
                + forward_coef * forward_loss
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad)
            opt.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["obs_loss"].append(obs_loss.item())
            metrics["reward_loss"].append(reward_loss.item())
            metrics["done_loss"].append(done_loss.item())
            metrics["inverse_loss"].append(inverse_loss.item())
            metrics["forward_loss"].append(forward_loss.item())
            metrics["approx_kl"].append(approx_kl)
            metrics["clip_frac"].append(clip_frac)

            if approx_kl > 1.5 * target_kl:
                early_stop = True
                break

        if early_stop:
            break

    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    metrics["early_stop"] = float(early_stop)
    return metrics


def make_env_fn(OBELIX, args, stage: CurriculumStage, worker_seed: int):
    def _make():
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=stage.max_steps,
            wall_obstacles=stage.wall_obstacles,
            difficulty=stage.difficulty,
            box_speed=stage.box_speed,
            seed=worker_seed,
        )
        original_step = env.step
        env.step = lambda action: original_step(action, render=False)
        return env

    return _make


def save_checkpoint(path: str, net: LatentBeliefActorCritic, args) -> None:
    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": {
                "obs_embed_dim": args.obs_embed_dim,
                "hidden": args.hidden,
                "reward_scale": args.reward_scale,
                "intrinsic_coef": args.intrinsic_coef,
                "curriculum": args.curriculum,
                "use_curriculum": not args.no_curriculum,
            },
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser(
        description="PPO + learned recurrent latent belief + ICM-style intrinsic reward for OBELIX"
    )
    ap.add_argument("--obelix_py", type=str, required=True)

    ap.add_argument("--out", type=str, default="weights_ppo_lstm_latent.pth")
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--n_envs", type=int, default=16)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae_lam", type=float, default=0.97)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.002)
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--rollout_len", type=int, default=128)
    ap.add_argument("--n_mini_batches", type=int, default=4)
    ap.add_argument("--max_grad", type=float, default=0.5)

    ap.add_argument("--obs_embed_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)

    ap.add_argument("--reward_scale", type=float, default=20.0)
    ap.add_argument("--intrinsic_coef", type=float, default=0.1)
    ap.add_argument("--intrinsic_clip", type=float, default=5.0)

    ap.add_argument("--obs_coef", type=float, default=0.02)
    ap.add_argument("--reward_coef", type=float, default=0.01)
    ap.add_argument("--done_coef", type=float, default=0.03)
    ap.add_argument("--inverse_coef", type=float, default=0.20)
    ap.add_argument("--forward_coef", type=float, default=0.20)

    ap.add_argument("--curriculum", type=str, default="default", choices=["default", "gentle"])
    ap.add_argument("--no_curriculum", action="store_true")

    args = ap.parse_args()
    device = torch.device(args.device) if args.device else DEVICE

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    net = LatentBeliefActorCritic(
        obs_dim=OBS_DIM,
        n_actions=N_ACT,
        obs_embed_dim=args.obs_embed_dim,
        hidden_dim=args.hidden,
    ).to(device)

    if args.load is not None:
        sd = torch.load(args.load, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict(sd, strict=False)
        print(f"[warm-start] Loaded weights from {args.load}")

    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    total_updates = max(
        (args.episodes * args.max_steps) // (args.rollout_len * args.n_envs), 1
    )
    scheduler = optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.1, total_iters=total_updates
    )

    intrinsic_rms = RunningMeanStd()
    curriculum = build_curriculum(args)
    buf = RolloutBuffer(args.rollout_len, args.n_envs)

    best_return = -float("inf")
    best_success_rate = -1.0
    episodes_done = 0
    total_steps = 0
    update_count = 0
    success_count = 0
    train_start = time.time()

    recent_successes = collections.deque(maxlen=200)
    recent_metrics = collections.defaultdict(lambda: collections.deque(maxlen=20))
    last_log_ep = 0
    LOG_EVERY = 20

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=130)

    for stage_idx, stage in enumerate(curriculum):
        if stage.episodes <= 0:
            continue

        print(
            f"\n[Stage {stage_idx + 1}/{len(curriculum)}] {stage.name} | "
            f"episodes={stage.episodes} difficulty={stage.difficulty} "
            f"wall={stage.wall_obstacles} box_speed={stage.box_speed} max_steps={stage.max_steps}"
        )

        make_fns = [
            make_env_fn(OBELIX, args, stage=stage, worker_seed=args.seed + stage_idx * 10_000 + i)
            for i in range(args.n_envs)
        ]
        vec = VecEnv(make_fns=make_fns, reward_shaping_fn=None)

        h, c = net.init_hidden(args.n_envs, device)
        init_seeds = [args.seed + stage_idx * 10_000 + i for i in range(args.n_envs)]
        obs_arr = np.array(vec.reset(seeds=init_seeds), dtype=np.float32)
        prev_action_arr = np.full(args.n_envs, -1, dtype=np.int64)
        prev_reward_arr = np.zeros(args.n_envs, dtype=np.float32)
        prev_done_arr = np.zeros(args.n_envs, dtype=np.float32)
        ep_ret = np.zeros(args.n_envs, dtype=np.float32)
        ep_steps = np.zeros(args.n_envs, dtype=np.int32)
        last_done = np.zeros(args.n_envs, dtype=bool)
        stage_episodes_done = 0
        window_returns: List[float] = []
        window_steps: List[int] = []

        while stage_episodes_done < stage.episodes:
            buf.clear()
            buf.store_init_hidden(h, c)

            for _ in range(args.rollout_len):
                obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
                prev_action_t = torch.tensor(prev_action_arr, dtype=torch.long, device=device)
                prev_reward_t = torch.tensor(prev_reward_arr, dtype=torch.float32, device=device)
                prev_done_t = torch.tensor(prev_done_arr, dtype=torch.float32, device=device)

                with torch.no_grad():
                    (
                        actions_t,
                        logp_t,
                        _,
                        values_t,
                        (h, c),
                        belief_t,
                        _,
                    ) = net.get_action(obs_t, prev_action_t, prev_reward_t, prev_done_t, (h, c))

                action_idx = actions_t.cpu().numpy()
                action_strs = [ACTIONS[a] for a in action_idx]

                results = vec.step(action_strs)
                next_obs_arr = np.array([r[0] for r in results], dtype=np.float32)
                raw_rewards = np.array([r[1] for r in results], dtype=np.float32)
                dones = np.array([r[2] for r in results], dtype=bool)

                with torch.no_grad():
                    intrinsic_raw = net.intrinsic_reward(
                        belief_t=belief_t,
                        action_idx=actions_t,
                        next_obs=torch.tensor(next_obs_arr, dtype=torch.float32, device=device),
                    ).cpu().numpy()

                intrinsic_rms.update(intrinsic_raw)
                intrinsic_norm = intrinsic_rms.normalize(intrinsic_raw)
                intrinsic_norm = np.clip(intrinsic_norm, -args.intrinsic_clip, args.intrinsic_clip)

                ext_rewards = raw_rewards / args.reward_scale
                total_rewards = ext_rewards + args.intrinsic_coef * intrinsic_norm.astype(np.float32)

                buf.add_batch(
                    obs=obs_arr,
                    prev_actions=prev_action_arr,
                    prev_rewards=prev_reward_arr,
                    prev_dones=prev_done_arr,
                    actions=action_idx,
                    log_probs=logp_t.cpu().numpy(),
                    rewards=total_rewards,
                    ext_rewards=ext_rewards,
                    values=values_t.cpu().numpy(),
                    dones=dones.astype(np.float32),
                    next_obs=next_obs_arr,
                )

                ep_ret += total_rewards
                ep_steps += 1
                total_steps += args.n_envs
                last_done[:] = dones

                next_prev_action_arr = action_idx.astype(np.int64)
                next_prev_reward_arr = total_rewards.astype(np.float32)
                next_prev_done_arr = dones.astype(np.float32)

                for i in range(args.n_envs):
                    if not dones[i]:
                        continue

                    success = bool(raw_rewards[i] >= 100.0)
                    success_count += int(success)
                    recent_successes.append(int(success))

                    window_returns.append(float(ep_ret[i]))
                    window_steps.append(int(ep_steps[i]))

                    if ep_ret[i] > best_return:
                        best_return = float(ep_ret[i])
                        save_checkpoint(args.out + ".best_return", net.cpu(), args)
                        net.to(device)

                    episodes_done += 1
                    stage_episodes_done += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "stage": stage.name,
                            "ret": f"{ep_ret[i]:.1f}",
                            "best": f"{best_return:.1f}",
                            "succ": success_count,
                        }
                    )

                    new_seed = args.seed + stage_idx * 10_000 + args.n_envs + episodes_done
                    reset_obs = vec.reset_one(i, seed=new_seed)
                    next_obs_arr[i] = np.array(reset_obs, dtype=np.float32)
                    next_prev_action_arr[i] = -1
                    next_prev_reward_arr[i] = 0.0
                    next_prev_done_arr[i] = 1.0

                    h[:, i, :] = 0.0
                    c[:, i, :] = 0.0

                    ep_ret[i] = 0.0
                    ep_steps[i] = 0

                    if stage_episodes_done >= stage.episodes or episodes_done >= args.episodes:
                        break

                obs_arr = next_obs_arr
                prev_action_arr = next_prev_action_arr
                prev_reward_arr = next_prev_reward_arr
                prev_done_arr = next_prev_done_arr

                if stage_episodes_done >= stage.episodes or episodes_done >= args.episodes:
                    break

            with torch.no_grad():
                obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
                prev_action_t = torch.tensor(prev_action_arr, dtype=torch.long, device=device)
                prev_reward_t = torch.tensor(prev_reward_arr, dtype=torch.float32, device=device)
                prev_done_t = torch.tensor(prev_done_arr, dtype=torch.float32, device=device)
                _, last_v, _, _, _ = net.forward_step(obs_t, prev_action_t, prev_reward_t, prev_done_t, (h, c))
                last_vals = last_v.cpu().numpy()
            last_vals = np.where(last_done, 0.0, last_vals)

            metrics = ppo_update(
                net=net,
                opt=opt,
                buf=buf,
                last_values=last_vals,
                device=device,
                gamma=args.gamma,
                gae_lam=args.gae_lam,
                clip_eps=args.clip_eps,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                n_epochs=args.n_epochs,
                n_mini_batches=args.n_mini_batches,
                max_grad=args.max_grad,
                target_kl=args.target_kl,
                obs_coef=args.obs_coef,
                reward_coef=args.reward_coef,
                done_coef=args.done_coef,
                inverse_coef=args.inverse_coef,
                forward_coef=args.forward_coef,
            )

            scheduler.step()
            update_count += 1

            for k, v in metrics.items():
                recent_metrics[k].append(v)

            if len(recent_successes) >= 50:
                rolling_success = 100.0 * np.mean(recent_successes)
                if rolling_success > best_success_rate:
                    best_success_rate = rolling_success
                    save_checkpoint(args.out + ".best_success", net.cpu(), args)
                    net.to(device)

            if (episodes_done // LOG_EVERY) > (last_log_ep // LOG_EVERY) and window_returns:
                elapsed = time.time() - train_start
                rolling_success = 100.0 * np.mean(recent_successes) if recent_successes else 0.0
                tqdm.write(
                    f"\n┌─ ep {episodes_done - len(window_returns) + 1:>4d}–{episodes_done:<4d} "
                    f"({elapsed:.0f}s) | stage={stage.name} "
                    f"────────────────────────────────────"
                )
                tqdm.write(
                    f"│ Avg Return    : {np.mean(window_returns):8.2f}   "
                    f"Best Return : {best_return:8.2f}"
                )
                tqdm.write(
                    f"│ Avg Steps     : {np.mean(window_steps):8.1f}   "
                    f"Rolling Success(200 ep): {rolling_success:6.2f}%"
                )
                tqdm.write(
                    f"│ Policy Loss   : {np.mean(recent_metrics['policy_loss']):8.4f}   "
                    f"Value Loss : {np.mean(recent_metrics['value_loss']):8.4f}"
                )
                tqdm.write(
                    f"│ Forward Loss  : {np.mean(recent_metrics['forward_loss']):8.4f}   "
                    f"Inverse Loss : {np.mean(recent_metrics['inverse_loss']):8.4f}"
                )
                tqdm.write(
                    f"│ Obs/Reward/Done: "
                    f"{np.mean(recent_metrics['obs_loss']):.4f} / "
                    f"{np.mean(recent_metrics['reward_loss']):.4f} / "
                    f"{np.mean(recent_metrics['done_loss']):.4f}"
                )
                tqdm.write(
                    f"│ Entropy       : {np.mean(recent_metrics['entropy']):8.4f}   "
                    f"KL : {np.mean(recent_metrics['approx_kl']):8.4f}   "
                    f"ClipFrac : {np.mean(recent_metrics['clip_frac']):6.3f}"
                )
                tqdm.write(
                    f"│ LR            : {scheduler.get_last_lr()[0]:.2e}   "
                    f"Updates : {update_count}   Total steps : {total_steps:,}"
                )
                tqdm.write(f"└{'─' * 94}")

                last_log_ep = episodes_done
                window_returns.clear()
                window_steps.clear()

            if episodes_done >= args.episodes:
                break

        vec.close()
        if episodes_done >= args.episodes:
            break

    pbar.close()
    save_checkpoint(args.out, net.cpu(), args)
    elapsed = time.time() - train_start
    print(f"\nSaved : {args.out}")
    print(f"Time  : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(
        f"Stats : {episodes_done} episodes | {total_steps:,} steps | "
        f"{success_count} successes ({100 * success_count / max(1, episodes_done):.1f}%)"
    )


if __name__ == "__main__":
    main()
