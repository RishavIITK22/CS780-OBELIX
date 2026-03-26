import argparse
import csv
import importlib.util
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType
from typing import Callable, List

import numpy as np

from obelix import OBELIX


ActionFn = Callable[[np.ndarray, np.random.Generator], str]


@dataclass
class EvalResult:
    agent_name: str
    mean_score: float
    std_score: float
    runs: int
    max_steps: int
    scaling_factor: int
    arena_size: int
    wall_obstacles: bool
    difficulty: int
    box_speed: int


def load_agent_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("submitted_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_agent(
    agent_policy: ActionFn,
    *,
    agent_name: str,
    runs: int,
    base_seed: int,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    wall_obstacles: bool,
    difficulty: int,
    box_speed: int,
) -> EvalResult:
    scores: List[float] = []

    env = OBELIX(
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
        seed=base_seed,
    )

    for i in range(runs):
        seed = base_seed + i
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        total = 0.0
        done = False
        while not done:
            action = agent_policy(obs, rng)
            obs, reward, done = env.step(action, render=True)
            total += float(reward)

        scores.append(total)

    mean = float(np.mean(scores))
    std = float(np.std(scores))

    return EvalResult(
        agent_name=agent_name,
        mean_score=mean,
        std_score=std,
        runs=runs,
        max_steps=max_steps,
        scaling_factor=scaling_factor,
        arena_size=arena_size,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
    )


def append_leaderboard(path: str, result: EvalResult) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "agent_name",
                "mean_score",
                "std_score",
                "runs",
                "max_steps",
                "scaling_factor",
                "arena_size",
                "wall_obstacles",
                "difficulty",
                "box_speed",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "agent_name": result.agent_name,
                "mean_score": f"{result.mean_score:.6f}",
                "std_score": f"{result.std_score:.6f}",
                "runs": result.runs,
                "max_steps": result.max_steps,
                "scaling_factor": result.scaling_factor,
                "arena_size": result.arena_size,
                "wall_obstacles": int(result.wall_obstacles),
                "difficulty": result.difficulty,
                "box_speed": result.box_speed,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_file", type=str, required=True)
    parser.add_argument("--agent_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="difficulty level: 0=static, 2=blinking box, 3=moving+blinking",
    )
    parser.add_argument(
        "--box_speed",
        type=int,
        default=2,
        help="speed of moving box (pixels/step) for difficulty>=3",
    )

    parser.add_argument("--leaderboard_csv", type=str, default="leaderboard.csv")

    args = parser.parse_args()

    agent_mod = load_agent_module(args.agent_file)
    if not hasattr(agent_mod, "policy"):
        raise AttributeError("Submission must define: policy(obs, rng) -> action_str")

    policy_fn = getattr(agent_mod, "policy")
    agent_name = (
        args.agent_name or os.path.splitext(os.path.basename(args.agent_file))[0]
    )

    result = evaluate_agent(
        policy_fn,
        agent_name=agent_name,
        runs=args.runs,
        base_seed=args.seed,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    print(
        f"agent={result.agent_name} mean={result.mean_score:.3f} std={result.std_score:.3f} "
        f"runs={result.runs} steps={result.max_steps} arena={result.arena_size} "
        f"wall_obstacles={result.wall_obstacles} difficulty={result.difficulty} box_speed={result.box_speed}"
    )

    append_leaderboard(args.leaderboard_csv, result)
    print(f"Appended to {args.leaderboard_csv}")


if __name__ == "__main__":
    main()
