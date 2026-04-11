import argparse
import inspect
import numpy as np

from obelix import OBELIX
import agent_template as agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    if hasattr(agent, "reset_agent"):
        agent.reset_agent()

    rng = np.random.default_rng(args.seed)
    obs = env.reset(seed=args.seed)
    done = False
    total_reward = 0.0
    step = 0
    policy_sig = inspect.signature(agent.policy)
    use_rng = len(policy_sig.parameters) >= 2

    while not done:
        action = agent.policy(obs, rng) if use_rng else agent.policy(obs)
        obs, reward, done = env.step(action, render=True)
        total_reward += reward
        step += 1
        print(
            f"step={step:04d} action={action:<3} reward={reward:7.1f} "
            f"return={total_reward:8.1f}"
        )

    print(f"Episode finished in {step} steps | total return = {total_reward:.1f}")


if __name__ == "__main__":
    main()
