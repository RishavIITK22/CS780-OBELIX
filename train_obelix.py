"""
train_obelix.py  —  DQN training for OBELIX (FSM-guided)
=========================================================
Saves:  weights.pth   (put next to agent_template.py for submission)

--- CURRICULUM mode (default, recommended) ---
Trains progressively through all difficulty levels:
    Stage 1  Level 1 static box            2000 eps
    Stage 2  Level 1 + wall                1000 eps
    Stage 3  Level 2 blinking              2000 eps
    Stage 4  Level 2 + wall                1000 eps
    Stage 5  Level 3 moving+blinking       2000 eps
    Stage 6  Level 3 + wall                1000 eps
    Stage 7  Mixed all levels              2000 eps
    Stage 8  Mixed + wall                   500 eps

    python train_obelix.py
    python train_obelix.py --mode curriculum

--- FLAT mode (single difficulty, fast experiments) ---
Trains directly on one level without staged progression.

    python train_obelix.py --mode flat --difficulty 1 --episodes 3000
    python train_obelix.py --mode flat --difficulty 3 --wall --episodes 5000
    python train_obelix.py --mode flat --difficulty 2 --episodes 4000 --eps-start 0.8

--- Common options (both modes) ---
    --save      PATH   output weights file         [weights.pth]
    --resume    PATH   continue from a checkpoint  [off]
    --eval-only PATH   skip training, just evaluate a saved checkpoint
    --wall             enable wall obstacles (flat mode; curriculum manages its own)
    --box-speed N      box speed for level 3        [2]
    --lr        F      learning rate                [3e-4]
    --batch     N      batch size                   [256]
"""

import os, sys, random, argparse, time
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from obelix import OBELIX
from agent_template import (
    DQN, featurize, _phase_onehot, _action_onehot,
    _fsm_suggest, _fsm_update_phase, reset_agent, _fsm,
    ACTIONS, ACTION_IDX, N_INPUT, N_ACTIONS, DEVICE,
)

# -----------------------------------------------------------------------
# Fixed hyper-parameters
# -----------------------------------------------------------------------
GAMMA          = 0.99
REPLAY_SIZE    = 100_000
MIN_REPLAY     = 2_000
TARGET_UPDATE  = 500
GRAD_CLIP      = 10.0
FSM_BIAS_TRAIN = 2.0
SCALING        = 1
MAX_STEPS      = 2000

# -----------------------------------------------------------------------
# Curriculum schedule
# (difficulty, wall, n_episodes, eps_start, eps_end, label)
# difficulty=None  →  random choice of 1/2/3 each episode
# -----------------------------------------------------------------------
CURRICULUM = [
    (1,    False, 2000, 1.00, 0.15, "L1 static"),
    (1,    True,  1000, 0.30, 0.10, "L1 static + wall"),
    (2,    False, 2000, 0.50, 0.10, "L2 blinking"),
    (2,    True,  1000, 0.25, 0.08, "L2 blinking + wall"),
    (3,    False, 2000, 0.40, 0.08, "L3 moving+blinking"),
    (3,    True,  1000, 0.25, 0.05, "L3 moving+blinking + wall"),
    (None, False, 2000, 0.15, 0.03, "Mixed fine-tune"),
    (None, True,   500, 0.10, 0.02, "Mixed + wall final"),
]

# -----------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------
Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done"])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


# -----------------------------------------------------------------------
# Reward shaping
# -----------------------------------------------------------------------
def _parse_local(obs):
    obs = np.asarray(obs, dtype=int)
    lf  = int(obs[0]  or obs[2])
    ln  = int(obs[1]  or obs[3])
    ff  = int(obs[4]  or obs[6]  or obs[8]  or obs[10])
    fn  = int(obs[5]  or obs[7]  or obs[9]  or obs[11])
    rf  = int(obs[12] or obs[14])
    rn  = int(obs[13] or obs[15])
    ir  = int(obs[16])
    stk = int(obs[17])
    return lf, ln, ff, fn, rf, rn, ir, stk


def shaped_reward(raw, obs, prev_obs, phase):
    bonus = 0.0
    lf,  ln,  ff,  fn,  rf,  rn,  ir,  stk  = _parse_local(obs)
    plf, pln, pff, pfn, prf, prn, pir, pstk = _parse_local(prev_obs)

    curr = ir*5 + fn*3 + ff*2 + (ln or rn) + (lf or rf)*0.5
    prev = pir*5 + pfn*3 + pff*2 + (pln or prn) + (plf or prf)*0.5
    if curr > prev:
        bonus += 1.5
    elif curr < prev and phase == "find":
        bonus -= 0.5

    if (ff or fn or ir) and not stk:
        bonus += 0.5
    if not any(int(b) for b in list(obs)[:17]):
        bonus -= 0.1

    return raw + bonus


# -----------------------------------------------------------------------
# Build network input tensor
# -----------------------------------------------------------------------
def make_tensor(obs, phase, fsm_act):
    vec = np.concatenate([featurize(obs), _phase_onehot(phase), _action_onehot(fsm_act)])
    return torch.tensor(vec, dtype=torch.float32, device=DEVICE)


# -----------------------------------------------------------------------
# Optimisation step (Double DQN)
# -----------------------------------------------------------------------
def optimise(policy_net, target_net, optimiser, replay, batch_size):
    if len(replay) < batch_size:
        return None
    batch      = replay.sample(batch_size)
    obs_b      = torch.stack([t.obs      for t in batch])
    next_obs_b = torch.stack([t.next_obs for t in batch])
    act_b      = torch.tensor([t.action  for t in batch], dtype=torch.long,    device=DEVICE)
    rew_b      = torch.tensor([t.reward  for t in batch], dtype=torch.float32, device=DEVICE)
    done_b     = torch.tensor([t.done    for t in batch], dtype=torch.float32, device=DEVICE)

    q_curr = policy_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_acts = policy_net(next_obs_b).argmax(1)
        q_next    = target_net(next_obs_b).gather(1, next_acts.unsqueeze(1)).squeeze(1)
        target    = rew_b + GAMMA * q_next * (1 - done_b)

    loss = nn.SmoothL1Loss()(q_curr, target)
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
    optimiser.step()
    return loss.item()


# -----------------------------------------------------------------------
# Single episode rollout
# -----------------------------------------------------------------------
def run_episode(env, policy_net, replay, epsilon):
    obs      = env.reset()
    reset_agent()
    prev_obs = obs.copy()
    ep_raw   = 0.0
    done     = False
    steps    = 0

    while not done:
        phase   = _fsm["phase"]
        fsm_act = _fsm_suggest(obs)
        st      = make_tensor(obs, phase, fsm_act)

        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            with torch.no_grad():
                q = policy_net(st.unsqueeze(0)).squeeze(0).cpu().numpy()
            q[ACTION_IDX[fsm_act]] += FSM_BIAS_TRAIN
            action = ACTIONS[int(np.argmax(q))]

        next_obs, raw, done = env.step(action, render=False)
        ep_raw += raw
        reward  = shaped_reward(raw, next_obs, prev_obs, phase)

        _fsm_update_phase(obs, action)
        nphase  = _fsm["phase"]
        nfsm    = _fsm_suggest(next_obs)
        nst     = make_tensor(next_obs, nphase, nfsm)

        replay.push(st, ACTION_IDX[action], reward, nst, float(done))
        prev_obs = obs
        obs      = next_obs
        steps   += 1

    return ep_raw, steps


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------
def evaluate(policy_net, difficulty, wall, box_speed, n_eval=10):
    policy_net.eval()
    rewards = []
    for seed in range(n_eval):
        env = OBELIX(scaling_factor=SCALING, max_steps=MAX_STEPS,
                     wall_obstacles=wall, difficulty=difficulty,
                     box_speed=box_speed, seed=seed + 88888)
        obs = env.reset()
        reset_agent()
        done = False; total = 0.0
        while not done:
            phase   = _fsm["phase"]
            fsm_act = _fsm_suggest(obs)
            st      = make_tensor(obs, phase, fsm_act).unsqueeze(0)
            with torch.no_grad():
                q = policy_net(st).squeeze(0).cpu().numpy()
            q[ACTION_IDX[fsm_act]] += FSM_BIAS_TRAIN
            action = ACTIONS[int(np.argmax(q))]
            obs, reward, done = env.step(action, render=False)
            _fsm_update_phase(obs, action)
            total += reward
        rewards.append(total)
    policy_net.train()
    return float(np.mean(rewards)), float(np.std(rewards))


# -----------------------------------------------------------------------
# Stage runner  (shared by both curriculum and flat)
# -----------------------------------------------------------------------
def run_stage(policy_net, target_net, optimiser, replay,
              difficulty, wall, n_eps, eps_start, eps_end,
              label, box_speed, batch_size, total_steps, log_every):
    """
    Train for one stage. Returns (total_steps, final_epsilon).
    difficulty=None → random 1/2/3 each episode.
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  episodes={n_eps}  eps {eps_start:.3f}→{eps_end:.3f}  "
          f"wall={wall}  diff={difficulty if difficulty else 'mixed'}")
    print(f"{'='*60}")

    eps_decay     = (eps_end / max(eps_start, 1e-9)) ** (1.0 / max(n_eps, 1))
    epsilon       = eps_start
    stage_rewards = []
    stage_steps   = []
    stage_success = []
    losses        = []
    start_time    = time.time()

    for ep in range(n_eps):
        diff = random.choice([1, 2, 3]) if difficulty is None else difficulty
        env  = OBELIX(scaling_factor=SCALING, max_steps=MAX_STEPS,
                      wall_obstacles=wall, difficulty=diff, box_speed=box_speed)

        ep_reward, steps = run_episode(env, policy_net, replay, epsilon)
        total_steps     += steps
        stage_rewards.append(ep_reward)
        stage_steps.append(steps)
        stage_success.append(float(ep_reward >= 1000.0))
        epsilon = max(eps_end, epsilon * eps_decay)

        if len(replay) >= MIN_REPLAY:
            for _ in range(max(1, steps // 4)):
                loss = optimise(policy_net, target_net, optimiser, replay, batch_size)
                if loss is not None:
                    losses.append(loss)

        if total_steps % TARGET_UPDATE < steps:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % log_every == 0 or ep == 0:
            window = min(log_every, ep + 1)
            mr = np.mean(stage_rewards[-window:])
            ms = np.mean(stage_steps[-window:])
            ml = np.mean(losses[-window:]) if losses else 0.0
            succ = 100.0 * np.mean(stage_success[-window:])
            best = np.max(stage_rewards)
            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"  ep {ep+1:>4}/{n_eps} | "
                f"avg_return={mr:>8.1f} best={best:>8.1f} | "
                f"avg_steps={ms:>6.1f} success={succ:>5.1f}% | "
                f"loss={ml:.4f} eps={epsilon:.4f} buf={len(replay)} | "
                f"total_steps={total_steps} speed={eps_per_sec:.2f} ep/s"
            )

    return total_steps, epsilon


# -----------------------------------------------------------------------
# Curriculum training
# -----------------------------------------------------------------------
def train_curriculum(args, policy_net, target_net, optimiser, replay):
    total_steps = 0
    for idx, (diff, wall, n_eps, eps_s, eps_e, label) in enumerate(CURRICULUM):
        total_steps, _ = run_stage(
            policy_net, target_net, optimiser, replay,
            difficulty=diff, wall=wall, n_eps=n_eps,
            eps_start=eps_s, eps_end=eps_e, label=f"Stage {idx+1}/{len(CURRICULUM)}: {label}",
            box_speed=args.box_speed, batch_size=args.batch,
            total_steps=total_steps, log_every=args.log_every,
        )

        eval_diff = diff if diff is not None else 3
        m, s = evaluate(policy_net, eval_diff, wall, args.box_speed)
        print(f"\n  >> Stage eval  diff={eval_diff} wall={wall}: {m:.1f} ± {s:.1f}")

        ckpt_path = f"weights_stage{idx+1}.pth"
        torch.save({"model_state_dict": policy_net.state_dict(),
                    "stage": idx+1, "mean_reward": m}, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")


# -----------------------------------------------------------------------
# Flat training
# -----------------------------------------------------------------------
def train_flat(args, policy_net, target_net, optimiser, replay):
    run_stage(
        policy_net, target_net, optimiser, replay,
        difficulty=args.difficulty,
        wall=args.wall,
        n_eps=args.episodes,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        label=f"Flat  diff={args.difficulty}  wall={args.wall}",
        box_speed=args.box_speed,
        batch_size=args.batch,
        total_steps=0,
        log_every=args.log_every,
    )


# -----------------------------------------------------------------------
# Final eval across all levels
# -----------------------------------------------------------------------
def final_eval(policy_net, box_speed):
    print("\n" + "="*60)
    print("FINAL EVALUATION  (greedy, 10 episodes each)")
    print("="*60)
    for diff in [1, 2, 3]:
        for wall in [False, True]:
            m, s = evaluate(policy_net, diff, wall, box_speed)
            print(f"  L{diff}  wall={str(wall):<5}  {m:>8.1f} ± {s:.1f}")


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # --- mode ---
    p.add_argument(
        "--mode", choices=["curriculum", "flat"], default="curriculum",
        help="'curriculum' (default): staged training across all levels. "
             "'flat': single-level training, use --difficulty to pick level.",
    )

    # --- flat-mode options ---
    flat = p.add_argument_group("flat mode options (ignored in curriculum mode)")
    flat.add_argument("--difficulty", type=int, choices=[1, 2, 3], default=1,
                      help="Difficulty level for flat mode (default: 1)")
    flat.add_argument("--episodes",  type=int, default=3000,
                      help="Number of training episodes in flat mode (default: 3000)")
    flat.add_argument("--eps-start", type=float, default=1.0,
                      help="Starting epsilon for flat mode (default: 1.0)")
    flat.add_argument("--eps-end",   type=float, default=0.05,
                      help="Final epsilon for flat mode (default: 0.05)")
    flat.add_argument("--wall",      action="store_true", default=False,
                      help="Enable wall obstacles in flat mode (default: off)")

    # --- common options ---
    common = p.add_argument_group("common options")
    common.add_argument("--save",      default="weights.pth",
                        help="Path to save final weights (default: weights.pth)")
    common.add_argument("--resume",    default=None, metavar="PATH",
                        help="Resume training from an existing .pth checkpoint")
    common.add_argument("--eval-only", default=None, metavar="PATH",
                        help="Skip training; just evaluate this checkpoint and exit")
    common.add_argument("--box-speed", type=int, default=2,
                        help="Box speed for level 3 (default: 2)")
    common.add_argument("--lr",        type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    common.add_argument("--batch",     type=int, default=256,
                        help="Batch size (default: 256)")
    common.add_argument("--log-every", type=int, default=50,
                        help="Print rolling training logs every N episodes (default: 50)")

    return p.parse_args()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # ---- eval-only shortcut ----
    if args.eval_only:
        net = DQN(N_INPUT, N_ACTIONS).to(DEVICE)
        ckpt = torch.load(args.eval_only, map_location=DEVICE)
        net.load_state_dict(ckpt.get("model_state_dict", ckpt))
        net.eval()
        print(f"Loaded {args.eval_only} for evaluation only.")
        final_eval(net, args.box_speed)
        sys.exit(0)

    # ---- build networks ----
    policy_net = DQN(N_INPUT, N_ACTIONS).to(DEVICE)
    target_net = DQN(N_INPUT, N_ACTIONS).to(DEVICE)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        sd   = ckpt.get("model_state_dict", ckpt)
        policy_net.load_state_dict(sd)
        print(f"Resumed from {args.resume}")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimiser = optim.Adam(policy_net.parameters(), lr=args.lr)
    replay    = ReplayBuffer(REPLAY_SIZE)

    # ---- train ----
    print(f"\nMode: {args.mode.upper()}")
    if args.mode == "curriculum":
        train_curriculum(args, policy_net, target_net, optimiser, replay)
    else:
        train_flat(args, policy_net, target_net, optimiser, replay)

    # ---- save ----
    torch.save({"model_state_dict": policy_net.state_dict()}, args.save)
    print(f"\nWeights saved: {args.save}")

    # ---- final evaluation ----
    final_eval(policy_net, args.box_speed)
