"""
Evaluation script for the Reinforcement Learning Challenge.

This script is run automatically by the evaluation server (Codabench).

What you must submit:
---------------------
A single Python file that defines:

    policy(obs: np.ndarray, rng: np.random.Generator) -> str

This function will be called at every environment step to decide the action.

IMPORTANT REQUIREMENTS:
----------------------
• Your code must run on CPU-only machines.
• Do NOT require CUDA/GPU for inference.
• If you use PyTorch, load weights with map_location="cpu".
• Your policy() must be fast and deterministic for a given RNG.
"""

import importlib.util
import os
import sys
from typing import Callable

import numpy as np

# Allow importing obelix.py from this folder
sys.path.append(os.path.dirname(__file__))
from obelix import OBELIX  # Environment simulator


# =========================================================
# Step 1 — Find the policy() function
# =========================================================
def find_policy(submission_dir: str) -> Callable[[np.ndarray, np.random.Generator], str]:
    """
    Search the submission directory for a Python file containing `policy()`.

    Codabench extracts the  zip into:
        input_dir/res/

    We scan all .py files and import them dynamically.
    The first file that defines a callable `policy` is used.

    Returns:
        policy function provided by the 
    """

    for fname in os.listdir(submission_dir):

        # Ignore non-python files
        if not fname.lower().endswith(".py"):
            continue

        fpath = os.path.join(submission_dir, fname)

        # Dynamically import the  file
        spec = importlib.util.spec_from_file_location("submitted_agent", fpath)

        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if policy() exists
            if hasattr(module, "policy"):
                policy_fn = getattr(module, "policy")

                if callable(policy_fn):
                    return policy_fn

    # If no valid policy found → submission fails
    raise RuntimeError(
        "No valid policy() function found. Submit a single .py file containing policy()."
    )


# =========================================================
# Step 2 — Run the agent in the OBELIX environment
# =========================================================
def evaluate_agent(policy_fn: Callable[[np.ndarray, np.random.Generator], str]) -> dict:
    """
    Evaluate the  policy on multiple difficulty levels.

    For each difficulty:
        • Run multiple episodes with different seeds
        • Call policy(obs, rng) every step
        • Accumulate total reward
        • Compute mean and standard deviation

    Final score = average performance across all levels.

    Returns:
        Dictionary of metrics written to scores.txt
    """

    # Number of episodes per difficulty
    runs = 10

    # Optional fast test mode for local debugging only
    if os.environ.get("LOCAL_QUICK", "0") == "1":
        runs = int(os.environ.get("LOCAL_QUICK_RUNS", "2"))
        max_steps = int(os.environ.get("LOCAL_QUICK_STEPS", "200"))

    # Fixed evaluation settings (same for everyone → fair comparison)
    base_seed = 0
    scaling_factor = 5
    arena_size = 500
    max_steps = 1000
    wall_obstacles = True

    # Difficulty levels tested
    # 0 = static
    # 2 = blinking
    # 3 = moving + blinking (hardest)
    difficulty_levels = [0, 2, 3]

    box_speed = 2

    results: dict[str, float] = {}
    all_scores: list[float] = []

    # -----------------------------------------------------
    # Run evaluation for each difficulty level
    # -----------------------------------------------------
    for difficulty in difficulty_levels:

        level_scores: list[float] = []

        for i in range(runs):

            seed = base_seed + i

            # Create environment
            env = OBELIX(
                scaling_factor=scaling_factor,
                arena_size=arena_size,
                max_steps=max_steps,
                wall_obstacles=wall_obstacles,
                difficulty=difficulty,
                box_speed=box_speed,
                seed=seed,
            )

            # Reset environment
            obs = env.reset(seed=seed)

            # Random generator passed to  policy
            rng = np.random.default_rng(seed)

            total = 0.0
            done = False

            # -------------------------------------------------
            # Main interaction loop
            # This is where YOUR policy() is called
            # -------------------------------------------------
            while not done:
                action = policy_fn(obs, rng)   # <---  function called
                obs, reward, done = env.step(action, render=False)
                total += float(reward)

            level_scores.append(total)
            all_scores.append(total)

        # Compute statistics for this difficulty
        mean_score = float(np.mean(level_scores))
        std_score = float(np.std(level_scores))

        results[f"mean_score_{difficulty}"] = mean_score
        results[f"std_score_{difficulty}"] = std_score

    # Overall aggregated metrics
    results["mean_score"] = float(np.mean(all_scores))
    results["std_score"] = float(np.std(all_scores))

    return results


# =========================================================
# Step 3 — Main entry point (called by Codabench)
# =========================================================
def main() -> None:
    """
    Codabench runs:
        python evaluate.py input_dir output_dir

    input_dir:
        ref/  (unused here)
        res/  ( submission)

    output_dir:
        scores.txt must be written here
    """

    if len(sys.argv) < 3:
        raise ValueError("Usage: evaluate.py <input_dir> <output_dir>")

    input_dir, output_dir = sys.argv[1:3]

    submit_dir = os.path.join(input_dir, "res")

    os.makedirs(output_dir, exist_ok=True)

    # Load  policy
    policy_fn = find_policy(submit_dir)

    # Evaluate performance
    results = evaluate_agent(policy_fn)

    # Write scores for leaderboard
    scores_file = os.path.join(output_dir, "scores.txt")

    with open(scores_file, "w", encoding="utf-8") as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.3f}\n")


if __name__ == "__main__":
    main()