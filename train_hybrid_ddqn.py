"""Train the hybrid OBELIX residual controller with Double DQN.

This is intentionally not PPO and not recurrent. The deterministic graph in
``hybrid_agent.py`` handles obvious SEARCH/ALIGN/PUSH/UNWEDGE structure. DDQN
only learns preferences among the candidate actions exposed by that graph.

Example:
    python train_hybrid_ddqn.py --obelix_py ./obelix.py --episodes 2000 --n_envs 16
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import random
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from hybrid_agent import (
    ACTIONS,
    ACTION_TO_IDX,
    FEATURE_DIM,
    HybridController,
    make_q_network,
)
from replay_buffer import ReplayBuffer
from vec_env import VecEnv


def get_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import OBELIX from {path}")
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
        original_step = env.step
        env.step = lambda action: original_step(action, render=False)
        return env

    return _make


def candidate_mask(candidates) -> np.ndarray:
    mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    for action in candidates:
        mask[ACTION_TO_IDX[action]] = True
    return mask


def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    masked = np.asarray(q_values, dtype=np.float32).copy()
    masked[~mask] = -1.0e9
    return int(np.argmax(masked))


def eps_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    frac = step / float(max(decay_steps, 1))
    return start + frac * (end - start)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="hybrid_weights.pth")
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--n_envs", type=int, default=16)

    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=250_000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=1000)
    ap.add_argument("--train_every", type=int, default=1)
    ap.add_argument("--reward_scale", type=float, default=100.0)
    ap.add_argument("--eps_start", type=float, default=0.35)
    ap.add_argument("--eps_end", type=float, default=0.03)
    ap.add_argument("--eps_decay_steps", type=int, default=150_000)
    ap.add_argument(
        "--supervisor_prob",
        type=float,
        default=0.25,
        help="Probability of taking the graph controller's default action during training.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"[Hybrid DDQN] device={device} envs={args.n_envs} feature_dim={FEATURE_DIM}")

    OBELIX = import_obelix(args.obelix_py)
    vec = VecEnv(
        [
            make_env_fn(OBELIX, args, seed=args.seed + i)
            for i in range(args.n_envs)
        ],
        reward_shaping_fn=None,
    )

    q = make_q_network().to(device)
    target = make_q_network().to(device)
    if args.load:
        q.load_state_dict(torch.load(args.load, map_location=device))
    target.load_state_dict(q.state_dict())
    target.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayBuffer(
        capacity=args.replay,
        fields={
            "obs": (torch.float32, (FEATURE_DIM,)),
            "action": (torch.long, ()),
            "reward": (torch.float32, ()),
            "next_obs": (torch.float32, (FEATURE_DIM,)),
            "done": (torch.float32, ()),
            "next_mask": (torch.bool, (len(ACTIONS),)),
        },
        pin_memory=(device.type == "cuda"),
    )

    controllers = [HybridController(max_steps=args.max_steps) for _ in range(args.n_envs)]
    obs_list = vec.reset([args.seed + i for i in range(args.n_envs)])
    decisions = [controllers[i].propose(obs_list[i]) for i in range(args.n_envs)]

    ep_raw_returns = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps = np.zeros(args.n_envs, dtype=np.int32)
    episodes_done = 0
    env_steps = 0
    learn_steps = 0
    best_mean = -float("inf")
    recent_returns = collections.deque(maxlen=100)
    recent_success = collections.deque(maxlen=100)
    recent_loss = collections.deque(maxlen=200)
    start_time = time.time()

    pbar = tqdm(total=args.episodes, desc="Hybrid-DDQN", unit="ep", ncols=120)

    try:
        while episodes_done < args.episodes:
            feats = np.stack([d.features for d in decisions]).astype(np.float32)

            with torch.no_grad():
                q_batch = q(torch.tensor(feats, dtype=torch.float32, device=device)).cpu().numpy()

            eps = eps_by_step(env_steps, args.eps_start, args.eps_end, args.eps_decay_steps)
            action_idxs: List[int] = []
            action_strs: List[str] = []

            for i, decision in enumerate(decisions):
                mask = candidate_mask(decision.candidates)
                if decision.forced:
                    action_idx = ACTION_TO_IDX[decision.action]
                elif np.random.random() < args.supervisor_prob:
                    action_idx = ACTION_TO_IDX[decision.action]
                elif np.random.random() < eps:
                    valid = np.flatnonzero(mask)
                    action_idx = int(valid[np.random.randint(0, len(valid))])
                else:
                    action_idx = masked_argmax(q_batch[i], mask)

                action = str(ACTIONS[action_idx])
                controllers[i].commit(action, decision.sensor, decision.mode)
                action_idxs.append(action_idx)
                action_strs.append(action)

            results = vec.step(action_strs)
            env_steps += args.n_envs

            for i, (next_obs, raw_reward, done) in enumerate(results):
                ep_raw_returns[i] += float(raw_reward)
                ep_steps[i] += 1

                scaled_reward = float(raw_reward) / args.reward_scale

                if done:
                    next_feat = np.zeros(FEATURE_DIM, dtype=np.float32)
                    next_mask = np.ones(len(ACTIONS), dtype=np.bool_)

                    success = float(raw_reward) > 0.0
                    recent_returns.append(float(ep_raw_returns[i]))
                    recent_success.append(float(success))

                    episodes_done += 1
                    pbar.update(1)

                    controllers[i].reset()
                    new_seed = args.seed + args.n_envs + episodes_done
                    reset_obs = vec.reset_one(i, new_seed)
                    decisions[i] = controllers[i].propose(reset_obs)

                    ep_raw_returns[i] = 0.0
                    ep_steps[i] = 0
                else:
                    decisions[i] = controllers[i].propose(next_obs)
                    next_feat = decisions[i].features
                    next_mask = candidate_mask(decisions[i].candidates)

                replay.add(
                    obs=feats[i],
                    action=action_idxs[i],
                    reward=scaled_reward,
                    next_obs=next_feat,
                    done=float(done),
                    next_mask=next_mask,
                )

            if len(replay) >= max(args.warmup, args.batch) and env_steps % args.train_every == 0:
                batch = replay.sample(args.batch, device)

                pred = q(batch["obs"]).gather(1, batch["action"].unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_online = q(batch["next_obs"])
                    next_online = next_online.masked_fill(~batch["next_mask"], -1.0e9)
                    next_action = torch.argmax(next_online, dim=1)
                    next_value = target(batch["next_obs"]).gather(
                        1, next_action.unsqueeze(1)
                    ).squeeze(1)
                    y = batch["reward"] + args.gamma * (1.0 - batch["done"]) * next_value

                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                learn_steps += 1
                recent_loss.append(float(loss.item()))
                if learn_steps % args.target_sync == 0:
                    target.load_state_dict(q.state_dict())

            if recent_returns:
                mean_ret = float(np.mean(recent_returns))
                if mean_ret > best_mean and len(recent_returns) >= min(20, args.episodes):
                    best_mean = mean_ret
                    torch.save(q.cpu().state_dict(), args.out + ".best")
                    q.to(device)

                pbar.set_postfix(
                    {
                        "ret100": f"{mean_ret:.1f}",
                        "succ100": f"{np.mean(recent_success):.2f}",
                        "eps": f"{eps:.3f}",
                        "loss": f"{np.mean(recent_loss) if recent_loss else 0.0:.3f}",
                    }
                )

    finally:
        pbar.close()
        vec.close()

    torch.save(q.cpu().state_dict(), args.out)
    elapsed = time.time() - start_time
    print(
        f"Saved {args.out}. episodes={episodes_done} env_steps={env_steps} "
        f"learn_steps={learn_steps} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
