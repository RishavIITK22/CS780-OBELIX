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
from three_policy_hetero.find_d3qn import (
    FindRewardShapeConfig,
    PrioritizedHistoryReplay,
    d3qn_update,
    select_action as select_find_action,
    shape_find_reward,
)
from three_policy_hetero.direction_estimator import DirectionalDuelingQNet, obs_direction_label
from three_policy_hetero.models import MLPActorCritic, RecurrentActorCritic
from three_policy_hetero.reward_hooks import (
    PushRewardHook,
    PushShapeConfig,
    UnwedgeRewardHook,
    UnwedgeShapeConfig,
)
from three_policy_hetero.trainer import GlobalRolloutBuffer, ppo_update_behavior
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


@dataclass
class PPOSpec:
    architecture: str
    hidden_dim: int
    stack_k: int = 1


def build_ppo_components(args, n_envs: int, device: torch.device):
    push_spec = PPOSpec(args.push_arch, args.push_hidden, stack_k=args.push_stack)
    unwedge_spec = PPOSpec(args.unwedge_arch, args.unwedge_hidden, stack_k=args.unwedge_stack)

    def build(spec: PPOSpec):
        if spec.architecture == "stack_mlp":
            adapter = FrameStackAdapter(n_envs, stack_k=spec.stack_k, include_prev_action=True, include_prev_reward=False, include_prev_done=False)
            model = MLPActorCritic(adapter.output_dim, hidden_dim=spec.hidden_dim).to(device)
        elif spec.architecture == "mlp":
            adapter = IdentityAdapter(n_envs, include_prev_action=True, include_prev_reward=False, include_prev_done=False)
            model = MLPActorCritic(adapter.output_dim, hidden_dim=spec.hidden_dim).to(device)
        elif spec.architecture in {"lstm", "gru", "rnn"}:
            adapter = IdentityAdapter(n_envs, include_prev_action=True, include_prev_reward=True, include_prev_done=True)
            model = RecurrentActorCritic(adapter.output_dim, hidden_dim=spec.hidden_dim, kind=spec.architecture).to(device)
        else:
            raise ValueError(f"Unsupported PPO architecture: {spec.architecture}")
        return model, adapter

    push_model, push_adapter = build(push_spec)
    unwedge_model, unwedge_adapter = build(unwedge_spec)
    models = {PUSH: push_model, UNWEDGE: unwedge_model}
    adapters = {PUSH: push_adapter, UNWEDGE: unwedge_adapter}
    specs = {PUSH: push_spec, UNWEDGE: unwedge_spec}
    return models, adapters, specs


def save_checkpoint_dir(out_dir: str, find_net, ppo_models, args, ppo_specs, suffix: str = "") -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": find_net.state_dict(),
            "config": {
                "behavior": FIND,
                "architecture": "d3qn_history",
                "hidden": args.find_hidden,
                "history_len": args.find_history,
                "history_encoder": args.find_history_encoder,
                "reward_scale": args.reward_scale,
            },
        },
        os.path.join(out_dir, f"weights_{FIND}{suffix}.pth"),
    )
    for behavior in (PUSH, UNWEDGE):
        model = ppo_models[behavior]
        spec = ppo_specs[behavior]
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "behavior": behavior,
                    "architecture": spec.architecture,
                    "hidden": spec.hidden_dim,
                    "stack_k": spec.stack_k,
                    "reward_scale": args.reward_scale,
                },
            },
            os.path.join(out_dir, f"weights_{behavior}{suffix}.pth"),
        )


def load_checkpoint_dir(load_dir: str, find_net, ppo_models, device: torch.device) -> None:
    missing = []
    for behavior in BEHAVIORS:
        path = os.path.join(load_dir, f"weights_{behavior}.pth")
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError("Missing warm-start checkpoints:\n" + "\n".join(missing))

    find_payload = torch.load(os.path.join(load_dir, f"weights_{FIND}.pth"), map_location=device)
    find_state = find_payload["state_dict"] if isinstance(find_payload, dict) and "state_dict" in find_payload else find_payload
    find_net.load_state_dict(find_state, strict=True)

    for behavior in (PUSH, UNWEDGE):
        payload = torch.load(os.path.join(load_dir, f"weights_{behavior}.pth"), map_location=device)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        ppo_models[behavior].load_state_dict(state, strict=True)

    print(f"[warm-start] Loaded checkpoints from: {load_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Hybrid three-policy trainer: FIND uses D3QN+PER with direction-estimator history, PUSH/UNWEDGE use PPO"
    )
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="three_policy_hybrid_weights")
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
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--rollout_len", type=int, default=256)
    ap.add_argument("--mini_batch", type=int, default=64)
    ap.add_argument("--max_grad", type=float, default=0.5)

    ap.add_argument("--find_history", type=int, default=8)
    ap.add_argument("--find_hidden", type=int, default=128)
    ap.add_argument("--find_history_encoder", type=str, default="gru", choices=["gru", "lstm"])
    ap.add_argument("--find_lr", type=float, default=2e-4)
    ap.add_argument("--find_replay", type=int, default=200_000)
    ap.add_argument("--find_batch", type=int, default=128)
    ap.add_argument("--find_warmup", type=int, default=2000)
    ap.add_argument("--find_update_every", type=int, default=4)
    ap.add_argument("--find_target_update", type=int, default=2000)
    ap.add_argument("--find_dir_coef", type=float, default=0.2)
    ap.add_argument("--find_eps_start", type=float, default=1.0)
    ap.add_argument("--find_eps_end", type=float, default=0.05)
    ap.add_argument("--find_eps_decay_steps", type=int, default=300_000)
    ap.add_argument("--find_per_alpha", type=float, default=0.6)
    ap.add_argument("--find_per_beta_start", type=float, default=0.4)
    ap.add_argument("--find_per_beta_end", type=float, default=1.0)
    ap.add_argument("--find_conf_gain_coef", type=float, default=0.05)
    ap.add_argument("--find_front_progress_bonus", type=float, default=0.05)
    ap.add_argument("--find_front_conf_bonus_coef", type=float, default=0.05)
    ap.add_argument("--find_stuck_penalty", type=float, default=0.05)

    ap.add_argument("--push_arch", type=str, default="stack_mlp", choices=["mlp", "stack_mlp", "lstm", "gru", "rnn"])
    ap.add_argument("--unwedge_arch", type=str, default="gru", choices=["mlp", "stack_mlp", "lstm", "gru", "rnn"])
    ap.add_argument("--push_hidden", type=int, default=256)
    ap.add_argument("--unwedge_hidden", type=int, default=128)
    ap.add_argument("--push_stack", type=int, default=4)
    ap.add_argument("--unwedge_stack", type=int, default=1)

    ap.add_argument("--reward_scale", type=float, default=20.0)
    ap.add_argument("--min_samples_per_behavior", type=int, default=64)
    ap.add_argument("--same_turn_threshold", type=int, default=4)
    ap.add_argument("--same_turn_penalty", type=float, default=0.01)
    ap.add_argument("--alternating_window", type=int, default=6)
    ap.add_argument("--alternating_penalty", type=float, default=0.015)
    ap.add_argument("--turn_ratio_window", type=int, default=8)
    ap.add_argument("--turn_ratio_threshold", type=float, default=0.75)
    ap.add_argument("--turn_ratio_penalty", type=float, default=0.01)
    ap.add_argument("--low_progress_obs_delta", type=float, default=0.05)
    ap.add_argument("--no_anti_spin", action="store_true")

    ap.add_argument("--push_contact_bonus", type=float, default=0.01)
    ap.add_argument("--push_forward_bonus", type=float, default=0.005)
    ap.add_argument("--push_stuck_penalty", type=float, default=0.01)
    ap.add_argument("--unwedge_recover_bonus", type=float, default=0.05)
    ap.add_argument("--unwedge_repeat_turn_penalty", type=float, default=0.01)

    ap.add_argument("--push_linger_steps", type=int, default=5)
    ap.add_argument("--unwedge_linger_steps", type=int, default=5)
    ap.add_argument("--attach_reward_threshold", type=float, default=90.0)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    vec = VecEnv(
        make_fns=[make_env_fn(OBELIX, args, worker_seed=args.seed + i) for i in range(args.n_envs)],
        reward_shaping_fn=None,
    )

    find_net = DirectionalDuelingQNet(
        history_len=args.find_history,
        hidden_dim=args.find_hidden,
        history_kind=args.find_history_encoder,
    ).to(device)
    target_find_net = DirectionalDuelingQNet(
        history_len=args.find_history,
        hidden_dim=args.find_hidden,
        history_kind=args.find_history_encoder,
    ).to(device)
    target_find_net.load_state_dict(find_net.state_dict())
    find_opt = optim.Adam(find_net.parameters(), lr=args.find_lr, eps=1e-5)
    replay = PrioritizedHistoryReplay(
        capacity=args.find_replay,
        history_len=args.find_history,
        obs_dim=OBS_DIM,
        alpha=args.find_per_alpha,
    )
    find_shape_cfg = FindRewardShapeConfig(
        confidence_gain_coef=args.find_conf_gain_coef,
        front_progress_bonus=args.find_front_progress_bonus,
        front_conf_bonus_coef=args.find_front_conf_bonus_coef,
        stuck_penalty=args.find_stuck_penalty,
    )

    ppo_models, ppo_adapters, ppo_specs = build_ppo_components(args, args.n_envs, device)
    if args.load_dir:
        load_checkpoint_dir(args.load_dir, find_net, ppo_models, device)
    ppo_opts = {b: optim.Adam(ppo_models[b].parameters(), lr=args.lr, eps=1e-5) for b in (PUSH, UNWEDGE)}
    ppo_buf = GlobalRolloutBuffer(args.rollout_len, args.n_envs)

    managers = [BehaviorManager(BehaviorManagerConfig(
        push_linger_steps=args.push_linger_steps,
        unwedge_linger_steps=args.unwedge_linger_steps,
        attach_reward_threshold=args.attach_reward_threshold,
        sticky_push=True,
        activate_push_on_ir=True,
    )) for _ in range(args.n_envs)]
    anti_spin = [AntiSpinPenaltyTracker(AntiSpinPenaltyConfig(
        same_turn_threshold=args.same_turn_threshold,
        same_turn_penalty=args.same_turn_penalty,
        alternating_window=args.alternating_window,
        alternating_penalty=args.alternating_penalty,
        turn_ratio_window=args.turn_ratio_window,
        turn_ratio_threshold=args.turn_ratio_threshold,
        turn_ratio_penalty=args.turn_ratio_penalty,
        low_progress_obs_delta=args.low_progress_obs_delta,
    )) for _ in range(args.n_envs)]
    push_hooks = [PushRewardHook(PushShapeConfig(
        contact_bonus=args.push_contact_bonus,
        forward_bonus=args.push_forward_bonus,
        stuck_penalty=args.push_stuck_penalty,
    )) for _ in range(args.n_envs)]
    unwedge_hooks = [UnwedgeRewardHook(UnwedgeShapeConfig(
        recover_bonus=args.unwedge_recover_bonus,
        repeat_turn_penalty=args.unwedge_repeat_turn_penalty,
    )) for _ in range(args.n_envs)]

    hidden = {b: ppo_models[b].init_hidden(args.n_envs, device) for b in (PUSH, UNWEDGE)}
    obs_arr = np.asarray(vec.reset(seeds=[args.seed + i for i in range(args.n_envs)]), dtype=np.float32)
    histories = [
        collections.deque([np.zeros(OBS_DIM, dtype=np.float32) for _ in range(args.find_history)], maxlen=args.find_history)
        for _ in range(args.n_envs)
    ]
    prev_action_arr = np.full(args.n_envs, -1, dtype=np.int64)
    prev_reward_arr = np.zeros(args.n_envs, dtype=np.float32)
    prev_done_arr = np.zeros(args.n_envs, dtype=np.float32)
    ep_ret = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps = np.zeros(args.n_envs, dtype=np.int32)
    for i in range(args.n_envs):
        managers[i].reset(obs_arr[i])
        anti_spin[i].reset(obs_arr[i])
        push_hooks[i].reset(obs_arr[i])
        unwedge_hooks[i].reset(obs_arr[i])
        for _ in range(args.find_history):
            histories[i].append(np.asarray(obs_arr[i], dtype=np.float32).reshape(-1))
        for b in (PUSH, UNWEDGE):
            ppo_adapters[b].reset_env(i)

    behavior_steps = {b: 0 for b in BEHAVIORS}
    episodes_done = 0
    total_steps = 0
    update_count = 0
    success_count = 0
    best_avg_return = -float("inf")
    recent_successes = collections.deque(maxlen=200)
    recent_metrics = {b: collections.defaultdict(lambda: collections.deque(maxlen=20)) for b in BEHAVIORS}
    window_returns: List[float] = []
    window_steps: List[int] = []
    window_spin: List[float] = []
    train_start = time.time()
    last_log_ep = 0
    log_every = 20

    print(
        f"\n[Three-Policy Hybrid] difficulty={args.difficulty} wall={args.wall_obstacles} "
        f"find=D3QN+PER({args.find_history_encoder},{args.find_history}) "
        f"push={args.push_arch}/{args.push_hidden} unwedge={args.unwedge_arch}/{args.unwedge_hidden}"
    )

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=140)

    while episodes_done < args.episodes:
        ppo_buf.clear()
        ppo_buf.store_init_hidden({b: hidden[b] for b in (PUSH, UNWEDGE)})

        for _ in range(args.rollout_len):
            behavior_ids = np.array([managers[i].current_behavior(obs_arr[i]) for i in range(args.n_envs)], dtype=object)
            action_idx = np.zeros(args.n_envs, dtype=np.int64)
            log_probs = np.zeros(args.n_envs, dtype=np.float32)
            values = np.zeros(args.n_envs, dtype=np.float32)
            find_cur_probs = np.zeros((args.n_envs, 4), dtype=np.float32)
            find_cur_conf = np.zeros(args.n_envs, dtype=np.float32)

            find_eps = float(
                np.interp(total_steps, [0, args.find_eps_decay_steps], [args.find_eps_start, args.find_eps_end])
            )
            find_beta = float(
                np.interp(total_steps, [0, args.find_eps_decay_steps], [args.find_per_beta_start, args.find_per_beta_end])
            )

            for i, behavior in enumerate(behavior_ids):
                behavior_steps[str(behavior)] += 1
                if behavior == FIND:
                    hist = np.stack(histories[i], axis=0)
                    a, conf, dir_probs = select_find_action(find_net, hist, find_eps, rng, device)
                    action_idx[i] = a
                    find_cur_conf[i] = conf
                    find_cur_probs[i] = dir_probs
                else:
                    adapter = ppo_adapters[str(behavior)]
                    feat = adapter.transform_step(
                        i, obs_arr[i], int(prev_action_arr[i]), float(prev_reward_arr[i]), float(prev_done_arr[i])
                    )
                    x = torch.tensor(feat[None, ...], dtype=torch.float32, device=device)
                    h_b, c_b = hidden[str(behavior)]
                    idx_t = torch.tensor([i], dtype=torch.long, device=device)
                    with torch.no_grad():
                        a_t, logp_t, _, v_t, (new_h, new_c) = ppo_models[str(behavior)].get_action(
                            x, (h_b[:, idx_t, :].contiguous(), c_b[:, idx_t, :].contiguous())
                        )
                    action_idx[i] = int(a_t.item())
                    log_probs[i] = float(logp_t.item())
                    values[i] = float(v_t.item())
                    h_b[:, idx_t, :] = new_h
                    c_b[:, idx_t, :] = new_c
                    hidden[str(behavior)] = (h_b, c_b)

            results = vec.step([ACTIONS[a] for a in action_idx])
            next_obs_arr = np.asarray([r[0] for r in results], dtype=np.float32)
            raw_rewards = np.asarray([r[1] for r in results], dtype=np.float32)
            dones = np.asarray([r[2] for r in results], dtype=bool)
            scaled_rewards = raw_rewards / args.reward_scale
            anti_spin_penalties = np.zeros(args.n_envs, dtype=np.float32)
            if not args.no_anti_spin:
                for i in range(args.n_envs):
                    anti_spin_penalties[i] = anti_spin[i].step(
                        behavior=str(behavior_ids[i]),
                        obs=obs_arr[i],
                        action_idx=int(action_idx[i]),
                        next_obs=next_obs_arr[i],
                        raw_reward=float(raw_rewards[i]),
                        attach_reward_threshold=args.attach_reward_threshold,
                    )

            total_rewards = scaled_rewards.copy()
            ppo_rewards = scaled_rewards.copy()

            for i, behavior in enumerate(behavior_ids):
                if behavior == FIND:
                    cur_hist = np.stack(histories[i], axis=0)
                    next_hist_deque = collections.deque(histories[i], maxlen=args.find_history)
                    next_hist_deque.append(np.asarray(next_obs_arr[i], dtype=np.float32).reshape(-1))
                    next_hist = np.stack(next_hist_deque, axis=0)
                    with torch.no_grad():
                        _, next_dir_logits = find_net(
                            torch.tensor(next_hist[None, ...], dtype=torch.float32, device=device)
                        )
                        next_dir_probs = torch.softmax(next_dir_logits, dim=-1).squeeze(0).cpu().numpy()
                    shaped = shape_find_reward(find_shape_cfg, find_cur_probs[i], next_dir_probs, next_obs_arr[i])
                    reward = scaled_rewards[i] + anti_spin_penalties[i] + shaped
                    total_rewards[i] = reward
                    replay.add(
                        hist=cur_hist,
                        action=int(action_idx[i]),
                        reward=float(reward),
                        next_hist=next_hist,
                        done=bool(dones[i]),
                        dir_target=obs_direction_label(next_obs_arr[i]),
                    )
                elif behavior == PUSH:
                    shaped = push_hooks[i].shape(obs_arr[i], int(action_idx[i]), next_obs_arr[i], float(raw_rewards[i]), bool(dones[i]))
                    reward = scaled_rewards[i] + shaped
                    total_rewards[i] = reward
                    ppo_rewards[i] = reward
                else:
                    shaped = unwedge_hooks[i].shape(obs_arr[i], int(action_idx[i]), next_obs_arr[i], float(raw_rewards[i]), bool(dones[i]))
                    reward = scaled_rewards[i] + shaped
                    total_rewards[i] = reward
                    ppo_rewards[i] = reward

            ppo_buf.add_batch(
                obs=obs_arr,
                prev_actions=prev_action_arr,
                prev_rewards=prev_reward_arr,
                prev_dones=prev_done_arr,
                actions=action_idx,
                behavior_ids=behavior_ids,
                log_probs=log_probs,
                values=values,
                rewards=ppo_rewards,
                ext_rewards=scaled_rewards,
                dones=dones.astype(np.float32),
                next_obs=next_obs_arr,
            )

            ep_ret += total_rewards
            ep_steps += 1
            window_spin.extend(anti_spin_penalties.tolist())
            total_steps += args.n_envs

            next_prev_action_arr = action_idx.astype(np.int64)
            next_prev_reward_arr = total_rewards.astype(np.float32)
            next_prev_done_arr = dones.astype(np.float32)

            for i, behavior in enumerate(behavior_ids):
                if not dones[i]:
                    histories[i].append(np.asarray(next_obs_arr[i], dtype=np.float32).reshape(-1))
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

                reset_obs = vec.reset_one(i, seed=args.seed + args.n_envs + episodes_done)
                obs_arr[i] = np.asarray(reset_obs, dtype=np.float32)
                histories[i].clear()
                for _ in range(args.find_history):
                    histories[i].append(np.asarray(obs_arr[i], dtype=np.float32).reshape(-1))
                managers[i].reset(obs_arr[i])
                anti_spin[i].reset(obs_arr[i])
                push_hooks[i].reset(obs_arr[i])
                unwedge_hooks[i].reset(obs_arr[i])
                for b in (PUSH, UNWEDGE):
                    ppo_adapters[b].reset_env(i)
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

            if len(replay) >= max(args.find_warmup, args.find_batch) and total_steps % args.find_update_every == 0:
                find_metrics = d3qn_update(
                    online_net=find_net,
                    target_net=target_find_net,
                    optimizer=find_opt,
                    replay=replay,
                    batch_size=args.find_batch,
                    beta=find_beta,
                    gamma=args.gamma,
                    dir_coef=args.find_dir_coef,
                    device=device,
                    max_grad=args.max_grad,
                )
                for k, v in find_metrics.items():
                    recent_metrics[FIND][k].append(v)
            if total_steps % args.find_target_update == 0:
                target_find_net.load_state_dict(find_net.state_dict())

            if episodes_done >= args.episodes:
                break

        ppo_tensors = ppo_buf.tensors(device)
        last_behavior_seq = np.array([managers[i].current_behavior(obs_arr[i]) for i in range(args.n_envs)], dtype=object)

        for behavior in (PUSH, UNWEDGE):
            if int(np.sum(ppo_tensors["behavior_seq"] == behavior)) < args.min_samples_per_behavior:
                continue
            last_values = np.zeros(args.n_envs, dtype=np.float32)
            idx = np.where(last_behavior_seq == behavior)[0]
            if idx.size > 0:
                feats = np.stack(
                    [
                        ppo_adapters[behavior].peek_step(
                            i, obs_arr[i], int(prev_action_arr[i]), float(prev_reward_arr[i]), float(prev_done_arr[i])
                        )
                        for i in idx
                    ],
                    axis=0,
                )
                x = torch.tensor(feats, dtype=torch.float32, device=device)
                idx_t = torch.tensor(idx, dtype=torch.long, device=device)
                h_b, c_b = hidden[behavior]
                with torch.no_grad():
                    _, v_last, _ = ppo_models[behavior].forward_step(
                        x, (h_b[:, idx_t, :].contiguous(), c_b[:, idx_t, :].contiguous())
                    )
                last_values[idx] = v_last.cpu().numpy()
            last_values = np.where(prev_done_arr > 0.5, 0.0, last_values)

            metrics = ppo_update_behavior(
                behavior=behavior,
                model=ppo_models[behavior],
                adapter=ppo_adapters[behavior],
                opt=ppo_opts[behavior],
                buf=ppo_buf,
                tensors=ppo_tensors,
                last_values=last_values,
                last_behavior_seq=last_behavior_seq,
                last_done=(prev_done_arr > 0.5),
                device=device,
                gamma=args.gamma,
                gae_lam=args.gae_lam,
                clip_eps=args.clip_eps,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                n_epochs=args.n_epochs,
                n_mini_batches=max(1, args.rollout_len // max(args.mini_batch, 1)),
                max_grad=args.max_grad,
                target_kl=0.03,
            )
            for k, v in metrics.items():
                recent_metrics[behavior][k].append(v)

        update_count += 1
        if window_returns and (episodes_done // log_every) > (last_log_ep // log_every):
            avg_return = float(np.mean(window_returns))
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                save_checkpoint_dir(args.out_dir + ".best", find_net, ppo_models, args, ppo_specs)
            rolling_success = 100.0 * np.mean(recent_successes) if recent_successes else 0.0
            elapsed = time.time() - train_start
            tqdm.write(
                f"\n┌─ ep {episodes_done - len(window_returns) + 1:>4d}–{episodes_done:<4d} ({elapsed:.0f}s) "
                f"────────────────────────────────────────────────────────"
            )
            tqdm.write(
                f"│ Avg Return : {avg_return:8.2f}   Avg Steps : {np.mean(window_steps):6.1f}   Rolling Success(200 ep): {rolling_success:6.2f}%"
            )
            if window_spin:
                tqdm.write(f"│ Avg anti-spin penalty/step : {np.mean(window_spin):8.4f}")
            tqdm.write(
                f"│ Behavior steps : find={behavior_steps[FIND]:,} push={behavior_steps[PUSH]:,} unwedge={behavior_steps[UNWEDGE]:,}   Updates : {update_count}"
            )
            if recent_metrics[FIND]:
                tqdm.write(
                    f"│   find | td={np.mean(recent_metrics[FIND]['find_td_loss']):8.4f} dir={np.mean(recent_metrics[FIND]['find_dir_loss']):8.4f} "
                    f"conf={np.mean(recent_metrics[FIND]['find_conf']):8.4f} q={np.mean(recent_metrics[FIND]['find_q']):8.4f}"
                )
            for behavior in (PUSH, UNWEDGE):
                if recent_metrics[behavior]:
                    tqdm.write(
                        f"│ {behavior:>6s} | policy={np.mean(recent_metrics[behavior]['policy_loss']):8.4f} "
                        f"value={np.mean(recent_metrics[behavior]['value_loss']):8.4f} entropy={np.mean(recent_metrics[behavior]['entropy']):8.4f}"
                    )
            tqdm.write(f"└{'─' * 112}")
            last_log_ep = episodes_done
            window_returns.clear()
            window_steps.clear()
            window_spin.clear()

    pbar.close()
    vec.close()
    save_checkpoint_dir(args.out_dir, find_net, ppo_models, args, ppo_specs)
    elapsed = time.time() - train_start
    print(f"\nSaved final checkpoints to: {args.out_dir}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min) | Episodes: {episodes_done} | Total steps: {total_steps:,}")


if __name__ == "__main__":
    main()
