import argparse
import time

import cv2

from obelix import OBELIX


MOVE_CHOICES = ["L45", "L22", "FW", "R22", "R45"]
KEY_TO_ACTION = {
    ord("q"): "L45",
    ord("a"): "L22",
    ord("s"): "IDLE",
    ord("w"): "FW",
    ord("d"): "R22",
    ord("e"): "R45",
}

KEY_TOGGLE_RUN = ord(" ")
KEY_RESET = ord("r")
KEY_QUIT = 27  # ESC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument(
        "--difficulty",
        help="difficulty level: 0=static, 2=blinking box, 3=moving+blinking",
        type=int,
        default=0,
    )
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--tick_ms", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def build_env(seed_offset=0):
        seed = args.seed + seed_offset
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=seed,
        )
        obs = env.reset(seed=seed)
        env.render_frame()
        return env, obs

    env, obs = build_env()
    running = True
    episode_reward = 0.0
    step = 0
    episode_idx = 0
    last_status_print = 0.0

    print("Controls:")
    print("  q a s w d e : single-step action L45 L22 IDLE FW R22 R45")
    print("  SPACE     : pause/resume stepping")
    print("  r         : reset episode")
    print("  ESC       : quit")
    print(f"Tick={args.tick_ms}ms, default action=IDLE")

    while True:
        key = cv2.waitKey(args.tick_ms) & 0xFF

        if key == KEY_QUIT:
            break
        if key == KEY_TOGGLE_RUN:
            running = not running
            print(f"[play] {'resumed' if running else 'paused'}")
            continue
        if key == KEY_RESET:
            episode_idx += 1
            env, obs = build_env(seed_offset=episode_idx)
            episode_reward = 0.0
            step = 0
            print("[play] reset")
            continue
        if not running:
            env.render_frame()
            continue

        action = KEY_TO_ACTION[key] if key in KEY_TO_ACTION else "IDLE"
        obs, reward, done = env.step(action)
        episode_reward += reward
        step += 1

        now = time.time()
        if now - last_status_print > 0.2 or done:
            print(
                f"step={step:04d} action={action:<4} reward={reward:7.1f} "
                f"return={episode_reward:8.1f} visible={int(env.box_visible)} "
                f"ir={int(obs[16])} stuck={int(obs[17])} push={int(env.enable_push)}"
            )
            last_status_print = now

        if done:
            print(f"[play] episode done, total score={episode_reward:.1f}")
            episode_idx += 1
            env, obs = build_env(seed_offset=episode_idx)
            episode_reward = 0.0
            step = 0

    cv2.waitKey(1)
    cv2.destroyAllWindows()
