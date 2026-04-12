"""collect_fsm_dataset.py — Collect expert trajectories using the FSM policy.

The expert is CompactBeliefState.fsm_suggest(), a hand-coded reactive policy
that uses persistent non-decaying belief state. It is imperfect but
consistently achieves find → push → partial-push on Level 0.

Each saved episode is a dict:
  obs        : float32 (T, 39)  — EPB input [raw(18)|belief(16)|fsm_onehot(5)]
  actions    : int64   (T,)     — FSM action index in ACTIONS list
  rewards    : float32 (T,)     — raw environment reward
  dones      : bool    (T,)
  enable_push: bool    (T,)     — env.enable_push at each step (push phase ground truth)

Usage:
  python collect_fsm_dataset.py --obelix_py obelix.py --n_episodes 3000 \\
      --out dataset_fsm.pkl
  python collect_fsm_dataset.py --obelix_py obelix.py --n_episodes 5000 \\
      --out dataset_fsm.pkl --curriculum
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import random
import time
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from compact_belief import ACTIONS, IN_DIM, CompactBeliefState, build_input


# ── Curriculum ─────────────────────────────────────────────────────────────────
# Each stage: (arena_size, difficulty, wall_obstacles, max_steps, n_episodes)
# Ordered easy → hard so the dataset covers the full task distribution.
CURRICULUM = [
    dict(arena_size=300, difficulty=0, wall_obstacles=False, max_steps=600,  n_episodes=500),
    dict(arena_size=400, difficulty=0, wall_obstacles=False, max_steps=800,  n_episodes=600),
    dict(arena_size=500, difficulty=0, wall_obstacles=False, max_steps=1000, n_episodes=700),
    dict(arena_size=500, difficulty=0, wall_obstacles=True,  max_steps=1000, n_episodes=500),
    dict(arena_size=500, difficulty=2, wall_obstacles=False, max_steps=1000, n_episodes=400),
    dict(arena_size=500, difficulty=2, wall_obstacles=True,  max_steps=1000, n_episodes=300),
]


def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def collect_episode(OBELIX, arena_size: int, difficulty: int,
                    wall_obstacles: bool, max_steps: int,
                    seed: int, scaling_factor: int = 5) -> Dict:
    """Run one episode under the FSM policy and return the trajectory dict."""
    env = OBELIX(
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=2,
        seed=seed,
    )
    raw = env.reset(seed=seed)

    belief = CompactBeliefState(max_steps=max_steps)

    obs_list: List[np.ndarray] = []
    act_list: List[int]        = []
    rew_list: List[float]      = []
    done_list: List[bool]      = []
    ep_list: List[bool]        = []

    done = False
    while not done:
        # Build EPB input before the FSM chooses an action
        x = build_input(raw, belief)          # (39,)

        # FSM action
        fsm_act_str = belief.fsm_suggest(raw)
        a_idx = ACTIONS.index(fsm_act_str)

        # Record (before step so x aligns with the action taken)
        obs_list.append(x)
        act_list.append(a_idx)

        # Environment step
        raw2, reward, done = env.step(fsm_act_str, render=True)

        rew_list.append(float(reward))
        done_list.append(bool(done))
        ep_list.append(bool(getattr(env, "enable_push", False)))

        # Update belief AFTER step with the action we chose
        belief.update(raw2, fsm_act_str)
        raw = raw2

    return {
        "obs":         np.array(obs_list,  dtype=np.float32),   # (T, 39)
        "actions":     np.array(act_list,  dtype=np.int64),      # (T,)
        "rewards":     np.array(rew_list,  dtype=np.float32),    # (T,)
        "dones":       np.array(done_list, dtype=bool),          # (T,)
        "enable_push": np.array(ep_list,   dtype=bool),          # (T,)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str, required=True)
    ap.add_argument("--n_episodes",     type=int, default=3000,
                    help="Total episodes (uniform single-stage if --curriculum not set)")
    ap.add_argument("--out",            type=str, default="dataset_fsm.pkl")
    ap.add_argument("--curriculum",     action="store_true",
                    help="Use the built-in multi-stage curriculum instead of a single config")
    ap.add_argument("--difficulty",     type=int, default=0)
    ap.add_argument("--arena_size",     type=int, default=500)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--max_steps",      type=int, default=1000)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    # Build list of (stage_config, n_episodes) pairs
    if args.curriculum:
        stages = [(s, s["n_episodes"]) for s in CURRICULUM]
    else:
        stages = [(
            dict(arena_size=args.arena_size, difficulty=args.difficulty,
                 wall_obstacles=args.wall_obstacles, max_steps=args.max_steps),
            args.n_episodes,
        )]

    dataset: List[Dict] = []
    total_steps = 0
    total_success = 0
    ep_global = 0
    t0 = time.time()

    for stage_cfg, n_eps in stages:
        desc = (f"arena={stage_cfg['arena_size']} diff={stage_cfg['difficulty']} "
                f"walls={stage_cfg['wall_obstacles']}")
        for i in tqdm(range(n_eps), desc=desc):
            seed_i = args.seed * 100003 + ep_global
            ep = collect_episode(OBELIX, seed=seed_i, **stage_cfg)
            dataset.append(ep)

            total_steps  += len(ep["actions"])
            total_success += int(ep["rewards"].sum() > 0)   # rough success proxy
            ep_global    += 1

    elapsed = time.time() - t0
    print(f"\n[dataset] {ep_global} episodes | {total_steps:,} steps | "
          f"~{total_success/ep_global*100:.1f}% positive-return | "
          f"{elapsed:.1f}s")
    print(f"[dataset] obs shape per ep (example): {dataset[0]['obs'].shape}")
    print(f"[dataset] action distribution: "
          + " | ".join(f"{a}:{(np.concatenate([e['actions'] for e in dataset])==i).mean()*100:.1f}%"
                       for i, a in enumerate(ACTIONS)))

    with open(args.out, "wb") as f:
        pickle.dump(dataset, f, protocol=4)
    print(f"[dataset] saved → {args.out}")


if __name__ == "__main__":
    main()
