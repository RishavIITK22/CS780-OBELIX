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

from anti_spin_penalty import AntiSpinPenaltyConfig, AntiSpinPenaltyTracker
from behavior_manager import (
    BEHAVIORS,
    FIND,
    PUSH,
    UNWEDGE,
    BehaviorManager,
    BehaviorManagerConfig,
)
from three_policy_hetero.adapters import FrameStackAdapter, IdentityAdapter
from three_policy_hetero.models import MLPActorCritic, RecurrentActorCritic
from three_policy_hetero.reward_hooks import (
    FindRewardHook,
    FindShapeConfig,
    NullRewardHook,
    PushRewardHook,
    PushShapeConfig,
    UnwedgeRewardHook,
    UnwedgeShapeConfig,
)
from vec_env import VecEnv


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM = 18


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
class BehaviorSpec:
    architecture: str
    hidden_dim: int
    stack_k: int = 1
    shape_reward: bool = False


class GlobalRolloutBuffer:
    def __init__(self, rollout_len: int, n_envs: int):
        self.T = rollout_len
        self.N = n_envs
        self.init_hidden: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.clear()

    def clear(self) -> None:
        self.obs = []
        self.prev_actions = []
        self.prev_rewards = []
        self.prev_dones = []
        self.actions = []
        self.behavior_ids = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.ext_rewards = []
        self.dones = []
        self.next_obs = []

    def store_init_hidden(self, hidden_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.init_hidden = {
            behavior: (h.detach().clone(), c.detach().clone())
            for behavior, (h, c) in hidden_dict.items()
        }

    def add_batch(
        self,
        obs: np.ndarray,
        prev_actions: np.ndarray,
        prev_rewards: np.ndarray,
        prev_dones: np.ndarray,
        actions: np.ndarray,
        behavior_ids: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        ext_rewards: np.ndarray,
        dones: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        for i in range(self.N):
            self.obs.append(obs[i])
            self.prev_actions.append(prev_actions[i])
            self.prev_rewards.append(prev_rewards[i])
            self.prev_dones.append(prev_dones[i])
            self.actions.append(actions[i])
            self.behavior_ids.append(behavior_ids[i])
            self.log_probs.append(log_probs[i])
            self.values.append(values[i])
            self.rewards.append(rewards[i])
            self.ext_rewards.append(ext_rewards[i])
            self.dones.append(dones[i])
            self.next_obs.append(next_obs[i])

    def _t_actual(self) -> int:
        return len(self.rewards) // self.N

    def tensors(self, device: torch.device):
        T_actual = self._t_actual()
        n_trim = T_actual * self.N

        to_np = lambda xs: np.asarray(xs[:n_trim])
        obs_flat = torch.tensor(to_np(self.obs), dtype=torch.float32, device=device)
        next_obs_flat = torch.tensor(to_np(self.next_obs), dtype=torch.float32, device=device)
        prev_actions_flat = torch.tensor(to_np(self.prev_actions), dtype=torch.long, device=device)
        prev_rewards_flat = torch.tensor(to_np(self.prev_rewards), dtype=torch.float32, device=device)
        prev_dones_flat = torch.tensor(to_np(self.prev_dones), dtype=torch.float32, device=device)
        actions_flat = torch.tensor(to_np(self.actions), dtype=torch.long, device=device)
        behavior_flat = np.asarray(self.behavior_ids[:n_trim], dtype=object)
        log_probs_flat = torch.tensor(to_np(self.log_probs), dtype=torch.float32, device=device)
        values_flat = torch.tensor(to_np(self.values), dtype=torch.float32, device=device)
        rewards_flat = torch.tensor(to_np(self.rewards), dtype=torch.float32, device=device)
        ext_rewards_flat = torch.tensor(to_np(self.ext_rewards), dtype=torch.float32, device=device)
        dones_flat = torch.tensor(to_np(self.dones), dtype=torch.float32, device=device)

        obs_seq = obs_flat.reshape(T_actual, self.N, OBS_DIM).permute(1, 0, 2).contiguous()
        next_obs_seq = next_obs_flat.reshape(T_actual, self.N, OBS_DIM).permute(1, 0, 2).contiguous()
        prev_action_seq = prev_actions_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        prev_reward_seq = prev_rewards_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        prev_done_seq = prev_dones_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        actions_seq = actions_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        rewards_seq = rewards_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        ext_rewards_seq = ext_rewards_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        dones_seq = dones_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        values_seq = values_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        log_probs_seq = log_probs_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        behavior_seq = behavior_flat.reshape(T_actual, self.N).T

        return {
            "obs_seq": obs_seq,
            "next_obs_seq": next_obs_seq,
            "prev_action_seq": prev_action_seq,
            "prev_reward_seq": prev_reward_seq,
            "prev_done_seq": prev_done_seq,
            "actions_seq": actions_seq,
            "rewards_seq": rewards_seq,
            "ext_rewards_seq": ext_rewards_seq,
            "dones_seq": dones_seq,
            "values_seq": values_seq,
            "log_probs_seq": log_probs_seq,
            "behavior_seq": behavior_seq,
            "T_actual": T_actual,
        }


def compute_behavior_advantages(
    rewards_seq: np.ndarray,
    values_seq: np.ndarray,
    behavior_seq: np.ndarray,
    dones_seq: np.ndarray,
    current_behavior: str,
    last_values: np.ndarray,
    last_behavior_seq: np.ndarray,
    last_done: np.ndarray,
    gamma: float,
    gae_lam: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, T = rewards_seq.shape
    advantages = np.zeros((N, T), dtype=np.float32)
    returns = np.zeros((N, T), dtype=np.float32)
    active_mask = behavior_seq == current_behavior

    for i in range(N):
        last_gae = 0.0
        for t in reversed(range(T)):
            if not active_mask[i, t]:
                last_gae = 0.0
                continue

            if t == T - 1:
                next_same = (last_behavior_seq[i] == current_behavior) and (not last_done[i])
                next_value = last_values[i] if next_same else 0.0
            else:
                next_same = active_mask[i, t + 1] and (not bool(dones_seq[i, t]))
                next_value = values_seq[i, t + 1] if next_same else 0.0

            next_nonterminal = 1.0 if next_same else 0.0
            delta = rewards_seq[i, t] + gamma * next_value * next_nonterminal - values_seq[i, t]
            last_gae = delta + gamma * gae_lam * next_nonterminal * last_gae
            advantages[i, t] = last_gae
            returns[i, t] = advantages[i, t] + values_seq[i, t]

    return advantages, returns, active_mask


def ppo_update_behavior(
    behavior: str,
    model: nn.Module,
    adapter,
    opt: optim.Optimizer,
    buf: GlobalRolloutBuffer,
    tensors: Dict[str, torch.Tensor],
    last_values: np.ndarray,
    last_behavior_seq: np.ndarray,
    last_done: np.ndarray,
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
) -> Dict[str, float]:
    rewards_seq = tensors["rewards_seq"].cpu().numpy()
    values_seq = tensors["values_seq"].cpu().numpy()
    behavior_seq = tensors["behavior_seq"]
    dones_seq_np = tensors["dones_seq"].cpu().numpy()

    advantages_np, returns_np, active_mask_np = compute_behavior_advantages(
        rewards_seq=rewards_seq,
        values_seq=values_seq,
        behavior_seq=behavior_seq,
        dones_seq=dones_seq_np,
        current_behavior=behavior,
        last_values=last_values,
        last_behavior_seq=last_behavior_seq,
        last_done=last_done,
        gamma=gamma,
        gae_lam=gae_lam,
    )

    active_mask_flat = torch.tensor(active_mask_np.T.reshape(-1), dtype=torch.bool, device=device)
    if int(active_mask_flat.sum().item()) == 0:
        return {}

    advantages = torch.tensor(advantages_np.T.reshape(-1), dtype=torch.float32, device=device)
    returns = torch.tensor(returns_np.T.reshape(-1), dtype=torch.float32, device=device)
    adv_active = advantages[active_mask_flat]
    adv_mean = adv_active.mean()
    adv_std = adv_active.std(unbiased=False)
    if not torch.isfinite(adv_std) or adv_std < 1e-8:
        adv_std = torch.tensor(1.0, device=device)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    init_h, init_c = buf.init_hidden[behavior]
    init_h = init_h.to(device)
    init_c = init_c.to(device)
    N = buf.N
    T_actual = tensors["T_actual"]
    metrics = collections.defaultdict(list)
    early_stop = False

    for _ in range(n_epochs):
        worker_indices = torch.randperm(N, device=device)
        mb_size = max(1, N // n_mini_batches)

        for mb_start in range(0, N, mb_size):
            wb = worker_indices[mb_start : mb_start + mb_size]
            obs_seq_np = tensors["obs_seq"][wb].cpu().numpy()
            prev_action_seq_np = tensors["prev_action_seq"][wb].cpu().numpy()
            prev_reward_seq_np = tensors["prev_reward_seq"][wb].cpu().numpy()
            prev_done_seq_np = tensors["prev_done_seq"][wb].cpu().numpy()
            x_seq_np = adapter.transform_batch(
                obs_seq_np, prev_action_seq_np, prev_reward_seq_np, prev_done_seq_np
            )
            x_seq = torch.tensor(x_seq_np, dtype=torch.float32, device=device)
            dones_seq = tensors["dones_seq"][wb]
            logits_flat, values_flat = model.evaluate_sequence(
                x_seq=x_seq,
                dones_seq=dones_seq,
                init_hidden=(init_h[:, wb, :].contiguous(), init_c[:, wb, :].contiguous()),
            )

            t_idx = torch.arange(T_actual, device=device)
            global_idx = (t_idx.unsqueeze(0) * N + wb.unsqueeze(1))
            flat_idx = global_idx.permute(1, 0).reshape(-1)
            mb_mask = active_mask_flat[flat_idx]
            if not bool(mb_mask.any()):
                continue

            act_all = tensors["actions_seq"][wb].permute(1, 0).reshape(-1)
            old_logp_all = tensors["log_probs_seq"][wb].permute(1, 0).reshape(-1)
            act_mb = act_all[mb_mask]
            old_logp_mb = old_logp_all[mb_mask]
            adv_mb = advantages[flat_idx][mb_mask]
            ret_mb = returns[flat_idx][mb_mask]

            dist = Categorical(logits=logits_flat)
            new_logp_all = dist.log_prob(act_all)
            new_logp = new_logp_all[mb_mask]
            entropy = dist.entropy()[mb_mask].mean()

            ratio = torch.exp(new_logp - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values_flat[mb_mask], ret_mb)
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            opt.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["approx_kl"].append(approx_kl)
            metrics["n_active"].append(float(mb_mask.sum().item()))

            if approx_kl > 1.5 * target_kl:
                early_stop = True
                break
        if early_stop:
            break

    if not metrics:
        return {}
    out = {k: float(np.mean(v)) for k, v in metrics.items()}
    out["early_stop"] = float(early_stop)
    return out


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


def build_behavior_specs(args) -> Dict[str, BehaviorSpec]:
    return {
        FIND: BehaviorSpec(args.find_arch, args.find_hidden, stack_k=args.find_stack, shape_reward=not args.no_find_shaping),
        PUSH: BehaviorSpec(args.push_arch, args.push_hidden, stack_k=args.push_stack, shape_reward=not args.no_push_shaping),
        UNWEDGE: BehaviorSpec(args.unwedge_arch, args.unwedge_hidden, stack_k=args.unwedge_stack, shape_reward=not args.no_unwedge_shaping),
    }


def build_behavior_components(args, specs: Dict[str, BehaviorSpec], n_envs: int, device: torch.device):
    models = {}
    adapters = {}
    hooks = {}

    for behavior, spec in specs.items():
        if spec.architecture == "stack_mlp":
            adapter = FrameStackAdapter(
                n_envs=n_envs,
                stack_k=spec.stack_k,
                include_prev_action=True,
                include_prev_reward=False,
                include_prev_done=False,
            )
            model = MLPActorCritic(adapter.output_dim, hidden_dim=spec.hidden_dim)
        elif spec.architecture == "mlp":
            adapter = IdentityAdapter(
                n_envs=n_envs,
                include_prev_action=True,
                include_prev_reward=False,
                include_prev_done=False,
            )
            model = MLPActorCritic(adapter.output_dim, hidden_dim=spec.hidden_dim)
        elif spec.architecture in {"lstm", "gru", "rnn"}:
            adapter = IdentityAdapter(
                n_envs=n_envs,
                include_prev_action=True,
                include_prev_reward=True,
                include_prev_done=True,
            )
            model = RecurrentActorCritic(
                input_dim=adapter.output_dim,
                hidden_dim=spec.hidden_dim,
                kind=spec.architecture,
            )
        else:
            raise ValueError(f"Unsupported architecture for {behavior}: {spec.architecture}")

        if behavior == FIND and spec.shape_reward:
            hook_factory = lambda: FindRewardHook(
                FindShapeConfig(
                    far_sensor_bonus=args.find_far_sensor_bonus,
                    near_sensor_bonus=args.find_near_sensor_bonus,
                    stuck_penalty_base=args.find_stuck_penalty_base,
                    stuck_penalty_growth=args.find_stuck_penalty_growth,
                    max_stuck_penalty=args.find_max_stuck_penalty,
                )
            )
        elif behavior == PUSH and spec.shape_reward:
            hook_factory = lambda: PushRewardHook(
                PushShapeConfig(
                    contact_bonus=args.push_contact_bonus,
                    forward_bonus=args.push_forward_bonus,
                    stuck_penalty_base=args.push_stuck_penalty_base,
                    stuck_penalty_growth=args.push_stuck_penalty_growth,
                    max_stuck_penalty=args.push_max_stuck_penalty,
                )
            )
        elif behavior == UNWEDGE and spec.shape_reward:
            hook_factory = lambda: UnwedgeRewardHook(
                UnwedgeShapeConfig(
                    recover_bonus=args.unwedge_recover_bonus,
                    repeat_turn_penalty=args.unwedge_repeat_turn_penalty,
                    stuck_penalty_base=args.unwedge_stuck_penalty_base,
                    stuck_penalty_growth=args.unwedge_stuck_penalty_growth,
                    max_stuck_penalty=args.unwedge_max_stuck_penalty,
                )
            )
        else:
            hook_factory = NullRewardHook

        models[behavior] = model.to(device)
        adapters[behavior] = adapter
        hooks[behavior] = [hook_factory() for _ in range(n_envs)]

    return models, adapters, hooks


def save_checkpoints(out_dir: str, models, specs, args, suffix: str = "") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for behavior, model in models.items():
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "behavior": behavior,
                    "architecture": specs[behavior].architecture,
                    "hidden": specs[behavior].hidden_dim,
                    "stack_k": specs[behavior].stack_k,
                    "reward_scale": args.reward_scale,
                },
            },
            os.path.join(out_dir, f"weights_{behavior}{suffix}.pth"),
        )


def load_checkpoints(load_dir: str, models, device: torch.device) -> None:
    missing = []
    for behavior in BEHAVIORS:
        path = os.path.join(load_dir, f"weights_{behavior}.pth")
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError("Missing warm-start checkpoints:\n" + "\n".join(missing))
    for behavior in BEHAVIORS:
        payload = torch.load(os.path.join(load_dir, f"weights_{behavior}.pth"), map_location=device)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        models[behavior].load_state_dict(state_dict, strict=True)
    print(f"[warm-start] Loaded checkpoints from: {load_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Heterogeneous three-policy PPO for OBELIX using raw observations and behavior-specific architectures"
    )
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="three_policy_hetero_weights")
    ap.add_argument("--load_dir", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--n_envs", type=int, default=16)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae_lam", type=float, default=0.97)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.005)
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--rollout_len", type=int, default=256)
    ap.add_argument("--n_mini_batches", type=int, default=4)
    ap.add_argument("--max_grad", type=float, default=0.5)

    ap.add_argument("--find_arch", type=str, default="lstm", choices=["mlp", "stack_mlp", "lstm", "gru", "rnn"])
    ap.add_argument("--push_arch", type=str, default="stack_mlp", choices=["mlp", "stack_mlp", "lstm", "gru", "rnn"])
    ap.add_argument("--unwedge_arch", type=str, default="gru", choices=["mlp", "stack_mlp", "lstm", "gru", "rnn"])
    ap.add_argument("--find_hidden", type=int, default=256)
    ap.add_argument("--push_hidden", type=int, default=256)
    ap.add_argument("--unwedge_hidden", type=int, default=128)
    ap.add_argument("--find_stack", type=int, default=1)
    ap.add_argument("--push_stack", type=int, default=4)
    ap.add_argument("--unwedge_stack", type=int, default=1)

    ap.add_argument("--reward_scale", type=float, default=20.0)
    ap.add_argument("--min_samples_per_behavior", type=int, default=64)
    ap.add_argument("--no_anti_spin", action="store_true")
    ap.add_argument("--same_turn_threshold", type=int, default=4)
    ap.add_argument("--same_turn_penalty", type=float, default=0.01)
    ap.add_argument("--alternating_window", type=int, default=6)
    ap.add_argument("--alternating_penalty", type=float, default=0.015)
    ap.add_argument("--turn_ratio_window", type=int, default=8)
    ap.add_argument("--turn_ratio_threshold", type=float, default=0.75)
    ap.add_argument("--turn_ratio_penalty", type=float, default=0.01)
    ap.add_argument("--low_progress_obs_delta", type=float, default=0.05)

    ap.add_argument("--no_find_shaping", action="store_true")
    ap.add_argument("--no_push_shaping", action="store_true")
    ap.add_argument("--no_unwedge_shaping", action="store_true")
    ap.add_argument("--find_far_sensor_bonus", type=float, default=0.002)
    ap.add_argument("--find_near_sensor_bonus", type=float, default=0.005)
    ap.add_argument("--find_stuck_penalty_base", type=float, default=0.01)
    ap.add_argument("--find_stuck_penalty_growth", type=float, default=0.005)
    ap.add_argument("--find_max_stuck_penalty", type=float, default=0.08)
    ap.add_argument("--push_contact_bonus", type=float, default=0.01)
    ap.add_argument("--push_forward_bonus", type=float, default=0.005)
    ap.add_argument("--push_stuck_penalty_base", type=float, default=0.01)
    ap.add_argument("--push_stuck_penalty_growth", type=float, default=0.005)
    ap.add_argument("--push_max_stuck_penalty", type=float, default=0.08)
    ap.add_argument("--unwedge_recover_bonus", type=float, default=0.05)
    ap.add_argument("--unwedge_repeat_turn_penalty", type=float, default=0.01)
    ap.add_argument("--unwedge_stuck_penalty_base", type=float, default=0.01)
    ap.add_argument("--unwedge_stuck_penalty_growth", type=float, default=0.005)
    ap.add_argument("--unwedge_max_stuck_penalty", type=float, default=0.08)

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
    make_fns = [make_env_fn(OBELIX, args, worker_seed=args.seed + i) for i in range(args.n_envs)]
    vec = VecEnv(make_fns=make_fns, reward_shaping_fn=None)

    specs = build_behavior_specs(args)
    models, adapters, hooks = build_behavior_components(args, specs, args.n_envs, device)
    if args.load_dir:
        load_checkpoints(args.load_dir, models, device)

    opts = {b: optim.Adam(models[b].parameters(), lr=args.lr, eps=1e-5) for b in BEHAVIORS}
    schedulers = {
        b: optim.lr_scheduler.LinearLR(
            opts[b], start_factor=1.0, end_factor=0.1,
            total_iters=max((args.episodes * args.max_steps) // (args.rollout_len * args.n_envs), 1),
        )
        for b in BEHAVIORS
    }

    manager_cfg = BehaviorManagerConfig(
        push_linger_steps=args.push_linger_steps,
        unwedge_linger_steps=args.unwedge_linger_steps,
        attach_reward_threshold=args.attach_reward_threshold,
        sticky_push=True,
        activate_push_on_ir=False,
    )
    managers = [BehaviorManager(manager_cfg) for _ in range(args.n_envs)]
    anti_spin_cfg = AntiSpinPenaltyConfig(
        same_turn_threshold=args.same_turn_threshold,
        same_turn_penalty=args.same_turn_penalty,
        alternating_window=args.alternating_window,
        alternating_penalty=args.alternating_penalty,
        turn_ratio_window=args.turn_ratio_window,
        turn_ratio_threshold=args.turn_ratio_threshold,
        turn_ratio_penalty=args.turn_ratio_penalty,
        low_progress_obs_delta=args.low_progress_obs_delta,
    )
    anti_spin_trackers = [AntiSpinPenaltyTracker(anti_spin_cfg) for _ in range(args.n_envs)]
    buf = GlobalRolloutBuffer(args.rollout_len, args.n_envs)

    hidden = {b: models[b].init_hidden(args.n_envs, device) for b in BEHAVIORS}
    behavior_steps = {b: 0 for b in BEHAVIORS}

    init_seeds = [args.seed + i for i in range(args.n_envs)]
    obs_arr = np.asarray(vec.reset(seeds=init_seeds), dtype=np.float32)
    prev_action_arr = np.full(args.n_envs, -1, dtype=np.int64)
    prev_reward_arr = np.zeros(args.n_envs, dtype=np.float32)
    prev_done_arr = np.zeros(args.n_envs, dtype=np.float32)
    ep_ret = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps = np.zeros(args.n_envs, dtype=np.int32)
    for i in range(args.n_envs):
        managers[i].reset(obs_arr[i])
        anti_spin_trackers[i].reset(obs_arr[i])
        for behavior in BEHAVIORS:
            adapters[behavior].reset_env(i)
            hooks[behavior][i].reset(obs_arr[i])

    episodes_done = 0
    total_steps = 0
    update_count = 0
    best_return = -float("inf")
    success_count = 0
    train_start = time.time()
    recent_successes = collections.deque(maxlen=200)
    recent_metrics = {b: collections.defaultdict(lambda: collections.deque(maxlen=20)) for b in BEHAVIORS}
    window_returns: List[float] = []
    window_steps: List[int] = []
    window_spin: List[float] = []
    log_every = 20
    last_log_ep = 0

    print(
        f"\n[Three-Policy Hetero PPO] difficulty={args.difficulty} wall={args.wall_obstacles} "
        f"find={specs[FIND].architecture}/{specs[FIND].hidden_dim} "
        f"push={specs[PUSH].architecture}/{specs[PUSH].hidden_dim} "
        f"unwedge={specs[UNWEDGE].architecture}/{specs[UNWEDGE].hidden_dim}"
    )
    print(
        f"[Three-Policy Hetero PPO] find_shaping={'OFF' if args.no_find_shaping else 'ON'} "
        f"push_shaping={'OFF' if args.no_push_shaping else 'ON'} "
        f"unwedge_shaping={'OFF' if args.no_unwedge_shaping else 'ON'} "
        f"anti_spin={'OFF' if args.no_anti_spin else 'FIND-only'} "
        f"push_activation=reward-spike"
    )

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=140)

    while episodes_done < args.episodes:
        buf.clear()
        buf.store_init_hidden(hidden)

        for _ in range(args.rollout_len):
            feature_cache = {
                behavior: np.stack(
                    [
                        adapters[behavior].transform_step(
                            i,
                            obs_arr[i],
                            int(prev_action_arr[i]),
                            float(prev_reward_arr[i]),
                            float(prev_done_arr[i]),
                        )
                        for i in range(args.n_envs)
                    ],
                    axis=0,
                )
                for behavior in BEHAVIORS
            }

            behavior_ids = np.array([managers[i].current_behavior(obs_arr[i]) for i in range(args.n_envs)], dtype=object)
            action_idx = np.zeros(args.n_envs, dtype=np.int64)
            log_probs = np.zeros(args.n_envs, dtype=np.float32)
            values = np.zeros(args.n_envs, dtype=np.float32)

            for behavior in BEHAVIORS:
                idx = np.where(behavior_ids == behavior)[0]
                if idx.size == 0:
                    continue
                behavior_steps[behavior] += int(idx.size)
                x = torch.tensor(feature_cache[behavior][idx], dtype=torch.float32, device=device)
                idx_t = torch.tensor(idx, dtype=torch.long, device=device)
                h_b, c_b = hidden[behavior]
                with torch.no_grad():
                    a_t, logp_t, _, v_t, (new_h, new_c) = models[behavior].get_action(
                        x,
                        (h_b[:, idx_t, :].contiguous(), c_b[:, idx_t, :].contiguous()),
                    )
                action_idx[idx] = a_t.cpu().numpy()
                log_probs[idx] = logp_t.cpu().numpy()
                values[idx] = v_t.cpu().numpy()
                h_b[:, idx_t, :] = new_h
                c_b[:, idx_t, :] = new_c
                hidden[behavior] = (h_b, c_b)

            results = vec.step([ACTIONS[a] for a in action_idx])
            next_obs_arr = np.asarray([r[0] for r in results], dtype=np.float32)
            raw_rewards = np.asarray([r[1] for r in results], dtype=np.float32)
            dones = np.asarray([r[2] for r in results], dtype=bool)

            ext_rewards = raw_rewards / args.reward_scale
            anti_spin_penalties = np.zeros(args.n_envs, dtype=np.float32)
            if not args.no_anti_spin:
                for i in range(args.n_envs):
                    anti_spin_penalties[i] = anti_spin_trackers[i].step(
                        behavior=str(behavior_ids[i]),
                        obs=obs_arr[i],
                        action_idx=int(action_idx[i]),
                        next_obs=next_obs_arr[i],
                        raw_reward=float(raw_rewards[i]),
                        attach_reward_threshold=args.attach_reward_threshold,
                    )

            shaped_rewards = np.zeros(args.n_envs, dtype=np.float32)
            for i, behavior in enumerate(behavior_ids):
                shaped_rewards[i] = hooks[str(behavior)][i].shape(
                    obs=obs_arr[i],
                    action_idx=int(action_idx[i]),
                    next_obs=next_obs_arr[i],
                    raw_reward=float(raw_rewards[i]),
                    done=bool(dones[i]),
                )

            total_rewards = ext_rewards + anti_spin_penalties + shaped_rewards

            buf.add_batch(
                obs=obs_arr,
                prev_actions=prev_action_arr,
                prev_rewards=prev_reward_arr,
                prev_dones=prev_done_arr,
                actions=action_idx,
                behavior_ids=behavior_ids,
                log_probs=log_probs,
                values=values,
                rewards=total_rewards,
                ext_rewards=ext_rewards,
                dones=dones.astype(np.float32),
                next_obs=next_obs_arr,
            )

            ep_ret += total_rewards
            ep_steps += 1
            window_spin.extend(anti_spin_penalties.tolist())
            total_steps += args.n_envs
            last_done = dones.copy()

            next_prev_action_arr = action_idx.astype(np.int64)
            next_prev_reward_arr = total_rewards.astype(np.float32)
            next_prev_done_arr = dones.astype(np.float32)

            for i, behavior in enumerate(behavior_ids):
                if not dones[i]:
                    managers[i].update(next_obs_arr[i], float(raw_rewards[i]), False)
                    obs_arr[i] = next_obs_arr[i]
                    continue

                success = bool(raw_rewards[i] >= 100.0)
                success_count += int(success)
                recent_successes.append(int(success))
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
                obs_arr[i] = np.asarray(reset_obs, dtype=np.float32)
                managers[i].reset(obs_arr[i])
                anti_spin_trackers[i].reset(obs_arr[i])
                for b in BEHAVIORS:
                    adapters[b].reset_env(i)
                    hooks[b][i].reset(obs_arr[i])
                    hidden[b][0][:, i, :] = 0.0
                    hidden[b][1][:, i, :] = 0.0
                next_prev_action_arr[i] = -1
                next_prev_reward_arr[i] = 0.0
                next_prev_done_arr[i] = 1.0
                ep_ret[i] = 0.0
                ep_steps[i] = 0
                if episodes_done >= args.episodes:
                    break

            prev_action_arr = next_prev_action_arr
            prev_reward_arr = next_prev_reward_arr
            prev_done_arr = next_prev_done_arr

            if episodes_done >= args.episodes:
                break

        tensors = buf.tensors(device)
        last_behavior_seq = np.array([managers[i].current_behavior(obs_arr[i]) for i in range(args.n_envs)], dtype=object)

        for behavior in BEHAVIORS:
            last_values = np.zeros(args.n_envs, dtype=np.float32)
            idx = np.where(last_behavior_seq == behavior)[0]
            if idx.size > 0:
                feats = np.stack(
                    [
                        adapters[behavior].peek_step(
                            i,
                            obs_arr[i],
                            int(prev_action_arr[i]),
                            float(prev_reward_arr[i]),
                            float(prev_done_arr[i]),
                        )
                        for i in idx
                    ],
                    axis=0,
                )
                x = torch.tensor(feats, dtype=torch.float32, device=device)
                idx_t = torch.tensor(idx, dtype=torch.long, device=device)
                h_b, c_b = hidden[behavior]
                with torch.no_grad():
                    _, v_last, _ = models[behavior].forward_step(
                        x, (h_b[:, idx_t, :].contiguous(), c_b[:, idx_t, :].contiguous())
                    )
                last_values[idx] = v_last.cpu().numpy()
            last_values = np.where(last_done, 0.0, last_values)

            if int(np.sum(tensors["behavior_seq"] == behavior)) < args.min_samples_per_behavior:
                continue

            metrics = ppo_update_behavior(
                behavior=behavior,
                model=models[behavior],
                adapter=adapters[behavior],
                opt=opts[behavior],
                buf=buf,
                tensors=tensors,
                last_values=last_values,
                last_behavior_seq=last_behavior_seq,
                last_done=last_done,
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
            )
            if metrics:
                for k, v in metrics.items():
                    recent_metrics[behavior][k].append(v)
                schedulers[behavior].step()

        update_count += 1

        if window_returns and (episodes_done // log_every) > (last_log_ep // log_every):
            elapsed = time.time() - train_start
            avg_return = float(np.mean(window_returns))
            if avg_return > best_return:
                best_return = avg_return
                save_checkpoints(args.out_dir + ".best", models, specs, args)

            rolling_success = 100.0 * np.mean(recent_successes) if recent_successes else 0.0
            tqdm.write(
                f"\n┌─ ep {episodes_done - len(window_returns) + 1:>4d}–{episodes_done:<4d} ({elapsed:.0f}s) "
                f"────────────────────────────────────────────────────────"
            )
            tqdm.write(
                f"│ Avg Return : {avg_return:8.2f}   Avg Steps : {np.mean(window_steps):6.1f}   "
                f"Rolling Success(200 ep): {rolling_success:6.2f}%"
            )
            if window_spin:
                tqdm.write(f"│ Avg anti-spin penalty/step : {np.mean(window_spin):8.4f}")
            tqdm.write(
                f"│ Behavior steps : find={behavior_steps[FIND]:,} push={behavior_steps[PUSH]:,} "
                f"unwedge={behavior_steps[UNWEDGE]:,}   Updates : {update_count}"
            )
            for behavior in BEHAVIORS:
                if not recent_metrics[behavior]:
                    continue
                tqdm.write(
                    f"│ {behavior:>7s} : arch={specs[behavior].architecture:>8s} "
                    f"active={np.mean(recent_metrics[behavior]['n_active']):7.1f} "
                    f"policy={np.mean(recent_metrics[behavior]['policy_loss']):8.4f} "
                    f"value={np.mean(recent_metrics[behavior]['value_loss']):8.4f} "
                    f"entropy={np.mean(recent_metrics[behavior]['entropy']):8.4f}"
                )
            tqdm.write(f"└{'─' * 98}")
            last_log_ep = episodes_done
            window_returns.clear()
            window_steps.clear()
            window_spin.clear()

    pbar.close()
    vec.close()
    save_checkpoints(args.out_dir, models, specs, args)
    elapsed = time.time() - train_start
    print(f"\nSaved final checkpoints to: {args.out_dir}")
    print(
        f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min) | Episodes: {episodes_done} | Total steps: {total_steps:,}"
    )


if __name__ == "__main__":
    main()
