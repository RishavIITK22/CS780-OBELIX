from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from behavior_manager import BEHAVIORS, FIND, PUSH, UNWEDGE, BehaviorManager, BehaviorManagerConfig
from learned_latent_curiosity import LatentBeliefActorCritic, N_ACT, OBS_DIM, RunningMeanStd
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
    counts[-1] += total_episodes - sum(counts)
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
    else:
        ratios = [0.15, 0.20, 0.25, 0.40]

    counts = _episode_split(args.episodes, ratios)
    return [
        CurriculumStage("static_open", counts[0], 0, False, 0, min(args.max_steps, 600)),
        CurriculumStage("static_wall", counts[1], 0, True, 0, min(args.max_steps, 750)),
        CurriculumStage("blinking_wall", counts[2], 2, True, 0, min(args.max_steps, 900)),
        CurriculumStage("moving_wall", counts[3], 3, True, args.box_speed, args.max_steps),
    ]


class GlobalRolloutBuffer:
    def __init__(self, rollout_len: int, n_envs: int):
        self.T = rollout_len
        self.N = n_envs
        self.clear()
        self.init_hidden: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

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

        obs_flat = torch.tensor(np.array(self.obs[:n_trim]), dtype=torch.float32, device=device)
        next_obs_flat = torch.tensor(np.array(self.next_obs[:n_trim]), dtype=torch.float32, device=device)
        prev_actions_flat = torch.tensor(np.array(self.prev_actions[:n_trim]), dtype=torch.long, device=device)
        prev_rewards_flat = torch.tensor(np.array(self.prev_rewards[:n_trim]), dtype=torch.float32, device=device)
        prev_dones_flat = torch.tensor(np.array(self.prev_dones[:n_trim]), dtype=torch.float32, device=device)
        actions_flat = torch.tensor(np.array(self.actions[:n_trim]), dtype=torch.long, device=device)
        behavior_flat = np.array(self.behavior_ids[:n_trim])
        log_probs_flat = torch.tensor(np.array(self.log_probs[:n_trim]), dtype=torch.float32, device=device)
        values_flat = torch.tensor(np.array(self.values[:n_trim]), dtype=torch.float32, device=device)
        rewards_flat = torch.tensor(np.array(self.rewards[:n_trim]), dtype=torch.float32, device=device)
        ext_rewards_flat = torch.tensor(np.array(self.ext_rewards[:n_trim]), dtype=torch.float32, device=device)
        dones_flat = torch.tensor(np.array(self.dones[:n_trim]), dtype=torch.float32, device=device)

        obs_seq = obs_flat.reshape(T_actual, self.N, OBS_DIM).permute(1, 0, 2).contiguous()
        next_obs_seq = next_obs_flat.reshape(T_actual, self.N, OBS_DIM).permute(1, 0, 2).contiguous()
        prev_action_seq = prev_actions_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        prev_reward_seq = prev_rewards_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        prev_done_seq = prev_dones_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        actions_seq = actions_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        dones_seq = dones_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        rewards_seq = rewards_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        ext_rewards_seq = ext_rewards_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        behavior_seq = behavior_flat.reshape(T_actual, self.N).T
        values_seq = values_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()
        log_probs_seq = log_probs_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()

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
            "behavior_seq": behavior_seq,
            "values_seq": values_seq,
            "log_probs_seq": log_probs_seq,
            "T_actual": T_actual,
        }


def evaluate_behavior_sequence(
    net: LatentBeliefActorCritic,
    obs_seq: torch.Tensor,
    prev_action_seq: torch.Tensor,
    prev_reward_seq: torch.Tensor,
    prev_done_seq: torch.Tensor,
    actions_seq: torch.Tensor,
    next_obs_seq: torch.Tensor,
    ext_reward_seq: torch.Tensor,
    dones_seq: torch.Tensor,
    active_mask_seq: torch.Tensor,
    init_h: torch.Tensor,
    init_c: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    n_envs, T, _ = obs_seq.shape

    obs_flat = obs_seq.reshape(n_envs * T, OBS_DIM)
    next_obs_flat = next_obs_seq.reshape(n_envs * T, OBS_DIM)
    obs_feat_flat = net.obs_encoder(obs_flat)
    next_feat_flat = net.obs_encoder(next_obs_flat).detach()
    obs_feat_seq = obs_feat_flat.reshape(n_envs, T, net.obs_embed_dim)
    next_feat_seq = next_feat_flat.reshape(n_envs, T, net.obs_embed_dim)

    prev_action_oh_seq = torch.zeros(
        n_envs, T, net.n_actions, device=obs_seq.device, dtype=torch.float32
    )
    valid_prev = prev_action_seq >= 0
    if valid_prev.any():
        prev_action_oh_seq[valid_prev] = torch.nn.functional.one_hot(
            prev_action_seq[valid_prev], num_classes=net.n_actions
        ).float()

    belief_inputs = torch.cat(
        [obs_feat_seq, prev_action_oh_seq, prev_reward_seq.unsqueeze(-1), prev_done_seq.unsqueeze(-1)], dim=-1
    )

    h, c = init_h, init_c
    belief_outputs = []
    for t in range(T):
        out, (h_new, c_new) = net.belief(belief_inputs[:, t : t + 1, :], (h, c))
        active = active_mask_seq[:, t].view(1, n_envs, 1)
        h = active * h_new + (1.0 - active) * h
        c = active * c_new + (1.0 - active) * c
        belief_outputs.append(h.squeeze(0))

        if t < T - 1:
            not_done = (1.0 - dones_seq[:, t]).view(1, n_envs, 1)
            h = h * not_done
            c = c * not_done

    belief_seq = torch.stack(belief_outputs, dim=1)
    logits_seq = net.actor(belief_seq)
    values_seq = net.critic(belief_seq).squeeze(-1)

    action_oh_seq = torch.nn.functional.one_hot(actions_seq, num_classes=net.n_actions).float()
    pred_in_seq = torch.cat([belief_seq, action_oh_seq], dim=-1)

    obs_pred_seq = net.obs_head(pred_in_seq)
    reward_pred_seq = net.reward_head(pred_in_seq).squeeze(-1)
    done_logit_seq = net.done_head(pred_in_seq).squeeze(-1)
    next_feat_pred_seq = net.forward_head(pred_in_seq)
    inverse_logits_seq = net.inverse_head(torch.cat([obs_feat_seq, next_feat_seq], dim=-1))

    logits_flat = logits_seq.permute(1, 0, 2).reshape(T * n_envs, net.n_actions)
    values_flat = values_seq.permute(1, 0).reshape(T * n_envs)
    obs_pred_flat = obs_pred_seq.permute(1, 0, 2).reshape(T * n_envs, OBS_DIM)
    reward_pred_flat = reward_pred_seq.permute(1, 0).reshape(T * n_envs)
    done_logit_flat = done_logit_seq.permute(1, 0).reshape(T * n_envs)
    next_feat_pred_flat = next_feat_pred_seq.permute(1, 0, 2).reshape(T * n_envs, net.obs_embed_dim)
    inverse_logits_flat = inverse_logits_seq.permute(1, 0, 2).reshape(T * n_envs, net.n_actions)

    dist = Categorical(logits=logits_flat)
    entropy_flat = dist.entropy()

    return {
        "logits_flat": logits_flat,
        "values_flat": values_flat,
        "entropy_flat": entropy_flat,
        "obs_pred_flat": obs_pred_flat,
        "reward_pred_flat": reward_pred_flat,
        "done_logit_flat": done_logit_flat,
        "next_feat_pred_flat": next_feat_pred_flat,
        "inverse_logits_flat": inverse_logits_flat,
        "next_feat_target_flat": next_feat_seq.permute(1, 0, 2).reshape(T * n_envs, net.obs_embed_dim),
        "next_obs_target_flat": next_obs_seq.permute(1, 0, 2).reshape(T * n_envs, OBS_DIM),
        "ext_reward_target_flat": ext_reward_seq.permute(1, 0).reshape(T * n_envs),
        "done_target_flat": dones_seq.permute(1, 0).reshape(T * n_envs),
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
    net: LatentBeliefActorCritic,
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
    obs_coef: float,
    reward_coef: float,
    done_coef: float,
    inverse_coef: float,
    forward_coef: float,
) -> Dict[str, float]:
    rewards_seq = tensors["rewards_seq"].cpu().numpy()
    values_seq = tensors["values_seq"].cpu().numpy()
    behavior_seq = tensors["behavior_seq"]
    dones_seq = tensors["dones_seq"].cpu().numpy()

    advantages_np, returns_np, active_mask_np = compute_behavior_advantages(
        rewards_seq=rewards_seq,
        values_seq=values_seq,
        behavior_seq=behavior_seq,
        dones_seq=dones_seq,
        current_behavior=behavior,
        last_values=last_values,
        last_behavior_seq=last_behavior_seq,
        last_done=last_done,
        gamma=gamma,
        gae_lam=gae_lam,
    )

    active_mask_flat = torch.tensor(
        active_mask_np.T.reshape(-1), dtype=torch.bool, device=device
    )
    n_active = int(active_mask_flat.sum().item())
    if n_active == 0:
        return {}

    advantages = torch.tensor(advantages_np.T.reshape(-1), dtype=torch.float32, device=device)
    returns = torch.tensor(returns_np.T.reshape(-1), dtype=torch.float32, device=device)
    advantages_active = advantages[active_mask_flat]
    adv_mean = advantages_active.mean()
    adv_std = advantages_active.std(unbiased=False)
    if not torch.isfinite(adv_std) or adv_std < 1e-8:
        adv_std = torch.tensor(1.0, device=device)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    init_h, init_c = buf.init_hidden[behavior]
    init_h = init_h.to(device)
    init_c = init_c.to(device)

    T_actual = tensors["T_actual"]
    N = buf.N
    metrics = collections.defaultdict(list)
    early_stop = False

    for _ in range(n_epochs):
        worker_indices = torch.randperm(N, device=device)
        mb_size = max(1, N // n_mini_batches)

        for mb_start in range(0, N, mb_size):
            wb = worker_indices[mb_start : mb_start + mb_size]
            outputs = evaluate_behavior_sequence(
                net=net,
                obs_seq=tensors["obs_seq"][wb],
                prev_action_seq=tensors["prev_action_seq"][wb],
                prev_reward_seq=tensors["prev_reward_seq"][wb],
                prev_done_seq=tensors["prev_done_seq"][wb],
                actions_seq=tensors["actions_seq"][wb],
                next_obs_seq=tensors["next_obs_seq"][wb],
                ext_reward_seq=tensors["ext_rewards_seq"][wb],
                dones_seq=tensors["dones_seq"][wb],
                active_mask_seq=torch.tensor(
                    active_mask_np[wb.cpu().numpy()], dtype=torch.float32, device=device
                ),
                init_h=init_h[:, wb, :].contiguous(),
                init_c=init_c[:, wb, :].contiguous(),
            )
            if not torch.isfinite(outputs["logits_flat"]).all():
                metrics["skipped_nan_batch"].append(1.0)
                continue

            t_idx = torch.arange(T_actual, device=device)
            global_idx = (t_idx.unsqueeze(0) * N + wb.unsqueeze(1))
            flat_idx = global_idx.permute(1, 0).reshape(-1)
            mb_mask = active_mask_flat[flat_idx]
            if not bool(mb_mask.any()):
                continue

            act_mb = tensors["actions_seq"][wb].permute(1, 0).reshape(-1)[mb_mask]
            old_logp_mb = tensors["log_probs_seq"][wb].permute(1, 0).reshape(-1)[mb_mask]
            adv_mb = advantages[flat_idx][mb_mask]
            ret_mb = returns[flat_idx][mb_mask]

            new_logp_all = Categorical(logits=outputs["logits_flat"]).log_prob(
                tensors["actions_seq"][wb].permute(1, 0).reshape(-1)
            )
            new_logp = new_logp_all[mb_mask]
            entropy = outputs["entropy_flat"][mb_mask].mean()

            ratio = torch.exp(new_logp - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(outputs["values_flat"][mb_mask], ret_mb)

            obs_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["obs_pred_flat"][mb_mask], outputs["next_obs_target_flat"][mb_mask]
            )
            reward_loss = nn.functional.mse_loss(
                outputs["reward_pred_flat"][mb_mask], outputs["ext_reward_target_flat"][mb_mask]
            )
            done_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["done_logit_flat"][mb_mask], outputs["done_target_flat"][mb_mask]
            )
            inverse_loss = nn.functional.cross_entropy(
                outputs["inverse_logits_flat"][mb_mask], act_mb
            )
            forward_loss = 0.5 * torch.mean(
                (outputs["next_feat_pred_flat"][mb_mask] - outputs["next_feat_target_flat"][mb_mask]) ** 2
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
            metrics["n_active"].append(float(n_active))

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


def save_checkpoints(out_dir: str, nets: Dict[str, LatentBeliefActorCritic], args, suffix: str = "") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for behavior, net in nets.items():
        torch.save(
            {
                "state_dict": net.state_dict(),
                "config": {
                    "obs_embed_dim": args.obs_embed_dim,
                    "hidden": args.hidden,
                    "behavior": behavior,
                    "latent_learning": args.latent_learning,
                    "use_curiosity": not args.no_curiosity and behavior == FIND,
                    "curriculum": args.curriculum,
                    "use_curriculum": not args.no_curriculum,
                },
            },
            os.path.join(out_dir, f"weights_{behavior}{suffix}.pth"),
        )


def main():
    ap = argparse.ArgumentParser(
        description="Three-policy PPO with learned latent belief, curriculum, and curiosity on find-phase only"
    )
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="three_policy_latent_weights")
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
    ap.add_argument(
        "--latent_learning",
        action="store_true",
        help="Use a learned observation latent before the recurrent belief. If unset, the policy acts on raw observations.",
    )

    ap.add_argument("--reward_scale", type=float, default=20.0)
    ap.add_argument("--intrinsic_coef", type=float, default=0.05)
    ap.add_argument("--intrinsic_clip", type=float, default=5.0)
    ap.add_argument("--no_curiosity", action="store_true")

    ap.add_argument("--obs_coef", type=float, default=0.02)
    ap.add_argument("--reward_coef", type=float, default=0.01)
    ap.add_argument("--done_coef", type=float, default=0.03)
    ap.add_argument("--inverse_coef", type=float, default=0.20)
    ap.add_argument("--forward_coef", type=float, default=0.20)

    ap.add_argument("--curriculum", type=str, default="default", choices=["default", "gentle"])
    ap.add_argument("--no_curriculum", action="store_true")

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
    nets = {
        behavior: LatentBeliefActorCritic(
            obs_dim=OBS_DIM,
            n_actions=N_ACT,
            obs_embed_dim=args.obs_embed_dim,
            hidden_dim=args.hidden,
            use_latent_learning=args.latent_learning,
        ).to(device)
        for behavior in BEHAVIORS
    }
    opts = {
        behavior: optim.Adam(nets[behavior].parameters(), lr=args.lr, eps=1e-5)
        for behavior in BEHAVIORS
    }
    schedulers = {
        behavior: optim.lr_scheduler.LinearLR(
            opts[behavior], start_factor=1.0, end_factor=0.1,
            total_iters=max((args.episodes * args.max_steps) // (args.rollout_len * args.n_envs), 1)
        )
        for behavior in BEHAVIORS
    }

    curriculum = build_curriculum(args)
    curiosity_rms = RunningMeanStd()
    buf = GlobalRolloutBuffer(args.rollout_len, args.n_envs)

    best_return = -float("inf")
    episodes_done = 0
    total_steps = 0
    update_count = 0
    success_count = 0
    train_start = time.time()

    recent_successes = collections.deque(maxlen=200)
    recent_metrics = {
        behavior: collections.defaultdict(lambda: collections.deque(maxlen=20))
        for behavior in BEHAVIORS
    }
    last_log_ep = 0
    LOG_EVERY = 20

    manager_cfg = BehaviorManagerConfig(
        push_linger_steps=args.push_linger_steps,
        unwedge_linger_steps=args.unwedge_linger_steps,
        attach_reward_threshold=args.attach_reward_threshold,
        sticky_push=True,
        activate_push_on_ir=True,
    )

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=140)

    for stage_idx, stage in enumerate(curriculum):
        if stage.episodes <= 0:
            continue

        print(
            f"\n[Stage {stage_idx + 1}/{len(curriculum)}] {stage.name} | "
            f"episodes={stage.episodes} difficulty={stage.difficulty} "
            f"wall={stage.wall_obstacles} box_speed={stage.box_speed} max_steps={stage.max_steps} "
            f"curiosity={'OFF' if args.no_curiosity else 'FIND-only'} "
            f"repr={'latent' if args.latent_learning else 'raw'}"
        )

        make_fns = [
            make_env_fn(OBELIX, args, stage=stage, worker_seed=args.seed + stage_idx * 10_000 + i)
            for i in range(args.n_envs)
        ]
        vec = VecEnv(make_fns=make_fns, reward_shaping_fn=None)

        managers = [BehaviorManager(manager_cfg) for _ in range(args.n_envs)]
        hidden = {
            behavior: nets[behavior].init_hidden(args.n_envs, device)
            for behavior in BEHAVIORS
        }

        init_seeds = [args.seed + stage_idx * 10_000 + i for i in range(args.n_envs)]
        obs_arr = np.array(vec.reset(seeds=init_seeds), dtype=np.float32)
        for i in range(args.n_envs):
            managers[i].reset(obs_arr[i])

        prev_action_arr = np.full(args.n_envs, -1, dtype=np.int64)
        prev_reward_arr = np.zeros(args.n_envs, dtype=np.float32)
        prev_done_arr = np.zeros(args.n_envs, dtype=np.float32)
        ep_ret = np.zeros(args.n_envs, dtype=np.float32)
        ep_steps = np.zeros(args.n_envs, dtype=np.int32)
        last_done = np.zeros(args.n_envs, dtype=bool)
        stage_episodes_done = 0
        window_returns: List[float] = []
        window_steps: List[int] = []

        while stage_episodes_done < stage.episodes and episodes_done < args.episodes:
            buf.clear()
            buf.store_init_hidden(hidden)

            for _ in range(args.rollout_len):
                behavior_ids = np.array([managers[i].current_behavior(obs_arr[i]) for i in range(args.n_envs)], dtype=object)
                action_idx = np.zeros(args.n_envs, dtype=np.int64)
                log_probs = np.zeros(args.n_envs, dtype=np.float32)
                values = np.zeros(args.n_envs, dtype=np.float32)
                belief_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

                obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
                prev_action_t = torch.tensor(prev_action_arr, dtype=torch.long, device=device)
                prev_reward_t = torch.tensor(prev_reward_arr, dtype=torch.float32, device=device)
                prev_done_t = torch.tensor(prev_done_arr, dtype=torch.float32, device=device)

                for behavior in BEHAVIORS:
                    idx = np.where(behavior_ids == behavior)[0]
                    if idx.size == 0:
                        continue
                    idx_t = torch.tensor(idx, dtype=torch.long, device=device)
                    h_b, c_b = hidden[behavior]
                    with torch.no_grad():
                        (
                            a_t,
                            logp_t,
                            _,
                            v_t,
                            (new_h, new_c),
                            belief_t,
                            _,
                        ) = nets[behavior].get_action(
                            obs_t[idx_t],
                            prev_action_t[idx_t],
                            prev_reward_t[idx_t],
                            prev_done_t[idx_t],
                            (h_b[:, idx_t, :].contiguous(), c_b[:, idx_t, :].contiguous()),
                        )
                    action_idx[idx] = a_t.cpu().numpy()
                    log_probs[idx] = logp_t.cpu().numpy()
                    values[idx] = v_t.cpu().numpy()
                    h_b[:, idx_t, :] = new_h
                    c_b[:, idx_t, :] = new_c
                    hidden[behavior] = (h_b, c_b)
                    belief_cache[behavior] = (idx_t, belief_t)

                results = vec.step([ACTIONS[a] for a in action_idx])
                next_obs_arr = np.array([r[0] for r in results], dtype=np.float32)
                raw_rewards = np.array([r[1] for r in results], dtype=np.float32)
                dones = np.array([r[2] for r in results], dtype=bool)

                ext_rewards = raw_rewards / args.reward_scale
                intrinsic_rewards = np.zeros(args.n_envs, dtype=np.float32)
                if not args.no_curiosity and FIND in belief_cache:
                    idx_t, belief_t = belief_cache[FIND]
                    with torch.no_grad():
                        intrinsic_raw = nets[FIND].intrinsic_reward(
                            belief_t=belief_t,
                            action_idx=torch.tensor(action_idx[idx_t.cpu().numpy()], dtype=torch.long, device=device),
                            next_obs=torch.tensor(next_obs_arr[idx_t.cpu().numpy()], dtype=torch.float32, device=device),
                        ).cpu().numpy()
                    curiosity_rms.update(intrinsic_raw)
                    intrinsic_norm = np.clip(
                        curiosity_rms.normalize(intrinsic_raw),
                        -args.intrinsic_clip,
                        args.intrinsic_clip,
                    ).astype(np.float32)
                    intrinsic_rewards[idx_t.cpu().numpy()] = intrinsic_norm

                total_rewards = ext_rewards.copy()
                find_mask = behavior_ids == FIND
                total_rewards[find_mask] += args.intrinsic_coef * intrinsic_rewards[find_mask]

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
                total_steps += args.n_envs
                last_done[:] = dones

                next_prev_action_arr = action_idx.astype(np.int64)
                next_prev_reward_arr = total_rewards.astype(np.float32)
                next_prev_done_arr = dones.astype(np.float32)

                for i in range(args.n_envs):
                    if not dones[i]:
                        managers[i].update(next_obs_arr[i], float(raw_rewards[i]), False)
                        continue

                    success = bool(raw_rewards[i] >= 100.0)
                    success_count += int(success)
                    recent_successes.append(int(success))
                    window_returns.append(float(ep_ret[i]))
                    window_steps.append(int(ep_steps[i]))

                    if ep_ret[i] > best_return:
                        best_return = float(ep_ret[i])
                        save_checkpoints(args.out_dir + ".best_return", {b: nets[b].cpu() for b in BEHAVIORS}, args)
                        for b in BEHAVIORS:
                            nets[b].to(device)

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
                    managers[i].reset(next_obs_arr[i])
                    next_prev_action_arr[i] = -1
                    next_prev_reward_arr[i] = 0.0
                    next_prev_done_arr[i] = 1.0
                    for behavior in BEHAVIORS:
                        hidden[behavior][0][:, i, :] = 0.0
                        hidden[behavior][1][:, i, :] = 0.0
                    ep_ret[i] = 0.0
                    ep_steps[i] = 0

                obs_arr = next_obs_arr
                prev_action_arr = next_prev_action_arr
                prev_reward_arr = next_prev_reward_arr
                prev_done_arr = next_prev_done_arr

                if stage_episodes_done >= stage.episodes or episodes_done >= args.episodes:
                    break

            tensors = buf.tensors(device)
            last_behavior_seq = np.array([managers[i].current_behavior(obs_arr[i]) for i in range(args.n_envs)], dtype=object)

            update_metrics = {}
            for behavior in BEHAVIORS:
                last_values = np.zeros(args.n_envs, dtype=np.float32)
                idx = np.where(last_behavior_seq == behavior)[0]
                if idx.size > 0:
                    idx_t = torch.tensor(idx, dtype=torch.long, device=device)
                    h_b, c_b = hidden[behavior]
                    with torch.no_grad():
                        _, v_last, _, _, _ = nets[behavior].forward_step(
                            torch.tensor(obs_arr[idx], dtype=torch.float32, device=device),
                            torch.tensor(prev_action_arr[idx], dtype=torch.long, device=device),
                            torch.tensor(prev_reward_arr[idx], dtype=torch.float32, device=device),
                            torch.tensor(prev_done_arr[idx], dtype=torch.float32, device=device),
                            (h_b[:, idx_t, :].contiguous(), c_b[:, idx_t, :].contiguous()),
                        )
                    last_values[idx] = v_last.cpu().numpy()

                metrics = ppo_update_behavior(
                    behavior=behavior,
                    net=nets[behavior],
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
                    obs_coef=args.obs_coef,
                    reward_coef=args.reward_coef,
                    done_coef=args.done_coef,
                    inverse_coef=args.inverse_coef,
                    forward_coef=args.forward_coef,
                )
                if metrics:
                    update_metrics[behavior] = metrics
                    for k, v in metrics.items():
                        recent_metrics[behavior][k].append(v)
                    schedulers[behavior].step()

            update_count += 1

            if len(recent_successes) >= 50:
                rolling_success = 100.0 * np.mean(recent_successes)
                save_checkpoints(args.out_dir + ".best_success", {b: nets[b].cpu() for b in BEHAVIORS}, args)
                for b in BEHAVIORS:
                    nets[b].to(device)

            if (episodes_done // LOG_EVERY) > (last_log_ep // LOG_EVERY) and window_returns:
                elapsed = time.time() - train_start
                rolling_success = 100.0 * np.mean(recent_successes) if recent_successes else 0.0
                tqdm.write(
                    f"\n┌─ ep {episodes_done - len(window_returns) + 1:>4d}–{episodes_done:<4d} "
                    f"({elapsed:.0f}s) | stage={stage.name} | curiosity={'OFF' if args.no_curiosity else 'FIND'}"
                )
                tqdm.write(
                    f"│ Avg Return    : {np.mean(window_returns):8.2f}   "
                    f"Best Return : {best_return:8.2f}   Rolling Success(200 ep): {rolling_success:6.2f}%"
                )
                tqdm.write(
                    f"│ Avg Steps     : {np.mean(window_steps):8.1f}   Updates : {update_count}   Total steps : {total_steps:,}"
                )
                for behavior in BEHAVIORS:
                    if not recent_metrics[behavior]:
                        continue
                    tqdm.write(
                        f"│ {behavior:>7s} | policy={np.mean(recent_metrics[behavior]['policy_loss']):8.4f} "
                        f"value={np.mean(recent_metrics[behavior]['value_loss']):8.4f} "
                        f"forward={np.mean(recent_metrics[behavior]['forward_loss']):8.4f} "
                        f"inverse={np.mean(recent_metrics[behavior]['inverse_loss']):8.4f} "
                        f"active={np.mean(recent_metrics[behavior]['n_active']):7.1f}"
                    )
                tqdm.write(f"└{'─' * 112}")
                last_log_ep = episodes_done
                window_returns.clear()
                window_steps.clear()

        vec.close()
        if episodes_done >= args.episodes:
            break

    pbar.close()
    save_checkpoints(args.out_dir, {b: nets[b].cpu() for b in BEHAVIORS}, args)
    elapsed = time.time() - train_start
    print(f"\nSaved final checkpoints to: {args.out_dir}")
    print(f"Time  : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(
        f"Stats : {episodes_done} episodes | {total_steps:,} steps | "
        f"{success_count} successes ({100 * success_count / max(1, episodes_done):.1f}%)"
    )


if __name__ == "__main__":
    main()
