import argparse
import json
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from obelix import OBELIX


MOVE_CHOICES = ["L45", "L22", "FW", "R22", "R45"]
RECORDED_ACTIONS = ["L45", "L22", "IDLE", "FW", "R22", "R45"]
KEY_TO_ACTION = {
    ord("q"): "L45",
    ord("a"): "L22",
    ord("s"): "IDLE",
    ord("w"): "FW",
    ord("d"): "R22",
    ord("e"): "R45",
}

KEY_SAVE_RESET = ord("r")
KEY_DISCARD_RESET = ord("x")
KEY_TOGGLE_RUN = ord(" ")
KEY_QUIT = 27  # ESC

STATE_TO_INT = {"F": 0, "P": 1, "U": 2}


@dataclass
class EpisodeRecorder:
    obs: list = field(default_factory=list)
    next_obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    ir_before: list = field(default_factory=list)
    ir_after: list = field(default_factory=list)
    stuck_before: list = field(default_factory=list)
    stuck_after: list = field(default_factory=list)
    enable_push_before: list = field(default_factory=list)
    enable_push_after: list = field(default_factory=list)
    active_state_before: list = field(default_factory=list)
    active_state_after: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    episode_return: float = 0.0
    attach_step: int = -1
    success: bool = False

    def add(self, obs, action_idx, reward, next_obs, done, env, step_idx, ts):
        before_push = bool(env.enable_push)
        before_state = STATE_TO_INT.get(env.active_state, -1)

        self.obs.append(np.asarray(obs, dtype=np.float32).copy())
        self.actions.append(int(action_idx))
        self.rewards.append(float(reward))
        self.next_obs.append(np.asarray(next_obs, dtype=np.float32).copy())
        self.dones.append(bool(done))
        self.ir_before.append(int(obs[16]))
        self.ir_after.append(int(next_obs[16]))
        self.stuck_before.append(int(obs[17]))
        self.stuck_after.append(int(next_obs[17]))
        self.enable_push_before.append(int(before_push))
        self.enable_push_after.append(int(env.enable_push))
        self.active_state_before.append(before_state)
        self.active_state_after.append(STATE_TO_INT.get(env.active_state, -1))
        self.timestamps.append(float(ts))
        self.episode_return += float(reward)

        if self.attach_step < 0 and (reward >= 100.0 or env.enable_push):
            self.attach_step = int(step_idx)
        if done and reward >= 1000.0:
            self.success = True

    def empty(self):
        return len(self.actions) == 0

    def to_payload(self, episode_idx, config):
        return {
            "episode_index": int(episode_idx),
            "obs": np.asarray(self.obs, dtype=np.float32),
            "next_obs": np.asarray(self.next_obs, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.bool_),
            "ir_before": np.asarray(self.ir_before, dtype=np.uint8),
            "ir_after": np.asarray(self.ir_after, dtype=np.uint8),
            "stuck_before": np.asarray(self.stuck_before, dtype=np.uint8),
            "stuck_after": np.asarray(self.stuck_after, dtype=np.uint8),
            "enable_push_before": np.asarray(self.enable_push_before, dtype=np.uint8),
            "enable_push_after": np.asarray(self.enable_push_after, dtype=np.uint8),
            "active_state_before": np.asarray(self.active_state_before, dtype=np.int8),
            "active_state_after": np.asarray(self.active_state_after, dtype=np.int8),
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
            "episode_return": np.float32(self.episode_return),
            "attach_step": np.int32(self.attach_step),
            "success": np.bool_(self.success),
            "action_names": np.asarray(RECORDED_ACTIONS),
            "config_json": np.asarray(json.dumps(config)),
        }


def next_episode_index(out_dir: str) -> int:
    existing = []
    for name in os.listdir(out_dir):
        if name.startswith("episode_") and name.endswith(".npz"):
            try:
                existing.append(int(name[len("episode_") : -len(".npz")]))
            except ValueError:
                continue
    return (max(existing) + 1) if existing else 0


def save_episode(out_dir: str, episode_idx: int, recorder: EpisodeRecorder, config: dict):
    if recorder.empty():
        return False

    path = os.path.join(out_dir, f"episode_{episode_idx:06d}.npz")
    np.savez_compressed(path, **recorder.to_payload(episode_idx, config))

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    summary = {
        "episode_index": episode_idx,
        "file": os.path.basename(path),
        "num_steps": len(recorder.actions),
        "episode_return": recorder.episode_return,
        "success": recorder.success,
        "attach_step": recorder.attach_step,
        "difficulty": config["difficulty"],
        "wall_obstacles": config["wall_obstacles"],
        "box_speed": config["box_speed"],
        "max_steps": config["max_steps"],
        "saved_at": time.time(),
    }
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    return True


def build_env(args, seed_offset=0):
    seed = args.seed + seed_offset if args.seed is not None else None
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
    return env, obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="offline_manual_dataset")
    parser.add_argument("--save_partial_on_reset", action="store_true")
    parser.add_argument("--tick_ms", type=int, default=80)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = {
        "scaling_factor": args.scaling_factor,
        "arena_size": args.arena_size,
        "max_steps": args.max_steps,
        "wall_obstacles": args.wall_obstacles,
        "difficulty": args.difficulty,
        "box_speed": args.box_speed,
        "seed": args.seed,
        "collector": "manual_keyboard",
    }

    episode_idx = next_episode_index(args.out_dir)
    env, obs = build_env(args, seed_offset=episode_idx)
    recorder = EpisodeRecorder()
    env.render_frame()
    episode_reward = 0.0
    step_idx = 0
    running = True
    last_status_print = 0.0

    print("Controls:")
    print("  q a s w d e : single-step action L45 L22 IDLE FW R22 R45")
    print("  SPACE     : pause/resume stepping")
    print("  r         : save current episode and reset")
    print("  x         : discard current episode and reset")
    print("  ESC       : quit")
    print(f"Dataset dir: {args.out_dir}")
    print(f"Tick={args.tick_ms}ms, default action=IDLE")

    while True:
        key = cv2.waitKey(args.tick_ms) & 0xFF

        if key == KEY_QUIT:
            if args.save_partial_on_reset and not recorder.empty():
                saved = save_episode(args.out_dir, episode_idx, recorder, config)
                if saved:
                    print(f"[saved] episode_{episode_idx:06d}.npz on quit")
                    episode_idx += 1
            break

        if key == KEY_TOGGLE_RUN:
            running = not running
            print(f"[record] {'resumed' if running else 'paused'}")
            continue

        if key == KEY_SAVE_RESET:
            saved = save_episode(args.out_dir, episode_idx, recorder, config)
            if saved:
                print(
                    f"[saved] episode_{episode_idx:06d}.npz | "
                    f"steps={len(recorder.actions)} return={recorder.episode_return:.1f} "
                    f"success={int(recorder.success)}"
                )
                episode_idx += 1
            else:
                print("[save] skipped empty episode")
            env, obs = build_env(args, seed_offset=episode_idx)
            recorder = EpisodeRecorder()
            env.render_frame()
            episode_reward = 0.0
            step_idx = 0
            continue

        if key == KEY_DISCARD_RESET:
            print(
                f"[discard] steps={len(recorder.actions)} return={recorder.episode_return:.1f}"
            )
            env, obs = build_env(args, seed_offset=episode_idx)
            recorder = EpisodeRecorder()
            env.render_frame()
            episode_reward = 0.0
            step_idx = 0
            continue

        if not running:
            env.render_frame()
            continue

        if key not in KEY_TO_ACTION and key != 255:
            continue

        action = KEY_TO_ACTION[key] if key in KEY_TO_ACTION else "IDLE"
        action_idx = RECORDED_ACTIONS.index(action)
        next_obs, reward, done = env.step(action)
        recorder.add(
            obs=obs,
            action_idx=action_idx,
            reward=reward,
            next_obs=next_obs,
            done=done,
            env=env,
            step_idx=step_idx,
            ts=time.time(),
        )
        episode_reward += reward
        step_idx += 1

        now = time.time()
        if now - last_status_print > 0.2 or done:
            print(
                f"step={step_idx:04d} action={action:<3} reward={reward:7.1f} "
                f"return={episode_reward:8.1f} visible={int(env.box_visible)} "
                f"ir={int(next_obs[16])} stuck={int(next_obs[17])} "
                f"push={int(env.enable_push)} state={env.active_state}"
            )
            last_status_print = now

        obs = next_obs

        if done:
            saved = save_episode(args.out_dir, episode_idx, recorder, config)
            if saved:
                print(
                    f"[saved] episode_{episode_idx:06d}.npz | "
                    f"steps={len(recorder.actions)} return={recorder.episode_return:.1f} "
                    f"success={int(recorder.success)} attach_step={recorder.attach_step}"
                )
                episode_idx += 1
            env, obs = build_env(args, seed_offset=episode_idx)
            recorder = EpisodeRecorder()
            env.render_frame()
            episode_reward = 0.0
            step_idx = 0

    cv2.waitKey(1)
    cv2.destroyAllWindows()
