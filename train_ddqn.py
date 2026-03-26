"""Double DQN trainer (GPU/CPU) for OBELIX — with parallel environments.
Usage: python train_dqn.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --n_envs 8
"""

from __future__ import annotations
import argparse, collections, random, time
from multiprocessing import Process, Pipe
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] CPU")
    return device

DEVICE  = get_device()
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


# ── Network ───────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────
class TensorReplay:
    """Circular replay buffer on pinned CPU memory for fast async GPU transfers."""

    def __init__(self, cap: int, obs_dim: int):
        self.cap, self.ptr, self.size = cap, 0, 0
        self.s          = torch.zeros(cap, obs_dim, dtype=torch.float32).pin_memory()
        self.s2         = torch.zeros(cap, obs_dim, dtype=torch.float32).pin_memory()
        self.a          = torch.zeros(cap, dtype=torch.long).pin_memory()
        self.r      = torch.zeros(cap, dtype=torch.float32).pin_memory()
        self.failed = torch.zeros(cap, dtype=torch.float32).pin_memory()

    def add(self, s, a, r, s2, failed: bool):
        i = self.ptr
        self.s     [i] = torch.as_tensor(s,  dtype=torch.float32)
        self.s2    [i] = torch.as_tensor(s2, dtype=torch.float32)
        self.a     [i] = a
        self.r     [i] = r
        self.failed[i] = float(failed)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, device: torch.device):
        idx = torch.randint(0, self.size, (batch,))
        kw  = dict(device=device, non_blocking=True)
        return (
            self.s     [idx].to(**kw),
            self.a     [idx].to(**kw),
            self.r     [idx].to(**kw),
            self.s2    [idx].to(**kw),
            self.failed[idx].to(**kw),
        )

    def __len__(self):
        return self.size


# ── Parallel environments ─────────────────────────────────────────────────────
def _env_worker(make_fn, conn, reward_scale: float, living_penalty: float):
    """Runs in a subprocess. Receives (cmd, data) and sends back results."""
    env = make_fn()
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            conn.send(env.reset(seed=data))
        elif cmd == "step":
            obs, reward, done = env.step(data, render=False)
            failed = done and (float(reward) < -50)
            r = float(reward) / reward_scale - living_penalty
            conn.send((obs, r, done, failed))
        elif cmd == "close":
            conn.close()
            break


class VecEnv:
    """Runs N OBELIX envs in separate processes and steps them in parallel."""

    def __init__(self, make_fns, reward_scale: float = 100.0, living_penalty: float = 0.005):
        self.n = len(make_fns)
        self.parents, workers = zip(*[Pipe() for _ in make_fns])
        self.procs = [
            Process(target=_env_worker, args=(fn, w, reward_scale, living_penalty), daemon=True)
            for fn, w in zip(make_fns, workers)
        ]
        for p in self.procs:
            p.start()

    def reset(self, seeds: List[int]) -> List[np.ndarray]:
        for p, s in zip(self.parents, seeds):
            p.send(("reset", s))
        return [p.recv() for p in self.parents]

    def reset_one(self, idx: int, seed: int) -> np.ndarray:
        self.parents[idx].send(("reset", seed))
        return self.parents[idx].recv()

    def step(self, actions: List[str]) -> List[tuple]:
        for p, a in zip(self.parents, actions):
            p.send(("step", a))
        return [p.recv() for p in self.parents]

    def close(self):
        for p in self.parents:
            p.send(("close", None))
        for proc in self.procs:
            proc.join()


# ── Helpers ───────────────────────────────────────────────────────────────────
def import_obelix(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights.pth")
    ap.add_argument("--episodes",        type=int,   default=2000)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--batch",           type=int,   default=356)
    ap.add_argument("--replay",          type=int,   default=500_000)
    ap.add_argument("--warmup",          type=int,   default=2000)
    ap.add_argument("--target_sync",     type=int,   default=2000)
    ap.add_argument("--reward_scale",    type=float, default=100.0)
    ap.add_argument("--living_penalty",  type=float, default=0.005)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int,   default=200_000)
    ap.add_argument("--seed",            type=int,   default=0)
    ap.add_argument("--device",          type=str,   default=None)
    ap.add_argument("--n_envs",          type=int,   default=8,
                    help="Number of parallel environments (default: 8)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    print(f"[VecEnv] Running {args.n_envs} parallel environments")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q   = DQN().to(device)
    tgt = DQN().to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = TensorReplay(cap=args.replay, obs_dim=18)

    steps       = 0
    learn_steps = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        return args.eps_start + (t / args.eps_decay_steps) * (args.eps_end - args.eps_start)

    ep_ret   = [0.0] * args.n_envs
    ep_steps = [0]   * args.n_envs

    window_returns: List[float] = []
    window_steps:   List[int]   = []
    recent_losses: collections.deque = collections.deque(maxlen=1000)
    best_return   = -float("inf")
    episodes_done = 0
    train_start   = time.time()

    def make_fn(seed):
        return lambda: OBELIX(
            scaling_factor=args.scaling_factor, arena_size=args.arena_size,
            max_steps=args.max_steps, wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty, box_speed=args.box_speed, seed=seed,
        )

    vec    = VecEnv(
        [make_fn(args.seed + i) for i in range(args.n_envs)],
        reward_scale=args.reward_scale,
        living_penalty=args.living_penalty,
    )
    states = vec.reset([args.seed + i for i in range(args.n_envs)])

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=100)

    while episodes_done < args.episodes:
        eps = eps_by_step(learn_steps)

        # ── Select actions ────────────────────────────────────────────────────
        with torch.no_grad():
            s_batch = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            q_vals  = q(s_batch).cpu().numpy()

        actions = []
        for i in range(args.n_envs):
            if np.random.rand() < eps:
                actions.append(np.random.randint(len(ACTIONS)))
            else:
                actions.append(int(np.argmax(q_vals[i])))

        # ── Step all envs ─────────────────────────────────────────────────────
        results = vec.step([ACTIONS[a] for a in actions])

        for i, (s2, r, done, failed) in enumerate(results):
            ep_steps[i] += 1
            steps       += 1

            ep_ret[i] += r
            replay.add(states[i], actions[i], r, s2, failed)

            if done:
                window_returns.append(ep_ret[i])
                window_steps.append(ep_steps[i])
                if ep_ret[i] > best_return:
                    best_return = ep_ret[i]
                    torch.save(q.cpu().state_dict(), args.out + ".best")
                    q.to(device)
                episodes_done += 1

                pbar.set_postfix({
                    "ret":  f"{ep_ret[i]:.1f}",
                    "best": f"{best_return:.1f}",
                    "ε":    f"{eps:.3f}",
                    "buf":  f"{len(replay)}",
                })
                pbar.update(1)

                states[i]   = vec.reset_one(i, args.seed + episodes_done + i)
                ep_ret[i]   = 0.0
                ep_steps[i] = 0

                if episodes_done >= args.episodes:
                    break
            else:
                states[i] = s2

        # ── Learning step (once per vector step) ──────────────────────────────
        if len(replay) >= max(args.warmup, args.batch):
            sb_t, ab_t, rb_t, s2b_t, failed_t = replay.sample(args.batch, device)

            with torch.no_grad():
                next_a   = torch.argmax(q(s2b_t), dim=1)
                next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                y        = rb_t + args.gamma * (1.0 - failed_t) * next_val

            pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 5.0)
            opt.step()

            learn_steps += 1
            recent_losses.append(loss.item())

            if learn_steps % args.target_sync == 0:
                tgt.load_state_dict(q.state_dict())

        if episodes_done % 10 == 0 and episodes_done > 0 and window_returns:
            tqdm.write(
                f"\n┌─ Episodes {episodes_done-9:>4d}–{episodes_done:<4d} "
                f"({'%.0f' % (time.time()-train_start)}s) ───────────────────\n"
                f"│  Avg Return : {np.mean(window_returns):>8.2f}   Best : {best_return:.2f}\n"
                f"│  Avg Loss   : {np.mean(recent_losses) if recent_losses else 0.0:>8.4f}"
                f"   Avg Steps : {np.mean(window_steps):.1f}\n"
                f"│  Epsilon    : {eps:>8.4f}   Buf : {len(replay):,}   "
                f"EnvSteps : {steps:,}   LearnSteps : {learn_steps:,}\n"
                f"└{'─'*60}"
            )
            window_returns.clear()
            window_steps.clear()

    pbar.close()
    vec.close()

    torch.save(q.cpu().state_dict(), args.out)
    t = time.time() - train_start
    print(f"\nSaved: {args.out}  |  {t:.1f}s ({t/60:.1f} min)")


if __name__ == "__main__":
    main()