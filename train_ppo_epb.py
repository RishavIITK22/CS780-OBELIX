"""train_ppo_epb.py — PPO trainer for the Explicit Phase Belief (EPB) agent.

Key innovations vs. prior attempts:
  1. No GRU — CompactBeliefState provides explicit non-decaying memory.
     The last-known box direction persists across hundreds of blank steps.
  2. Dual-head actor (find + push) with a soft learned gate.
     Both heads share a trunk; the gate is supervised by env.enable_push.
  3. Small-arena curriculum: starts 300×300 so the robot finds the box quickly
     and the push head gets dense gradient signal from the very beginning.
  4. Reward shaping via VecEnv shaping_fn that also passes enable_push back
     as the "failed" flag so the gate can be supervised without env changes.

Usage:
    python train_ppo_epb.py --obelix_py obelix.py --out weights_epb.pth
    python train_ppo_epb.py --obelix_py obelix.py --out weights_epb.pth \\
        --no_curriculum --difficulty 2 --wall_obstacles  # direct hard mode
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from compact_belief import (
    ACTIONS, BELIEF_DIM, IN_DIM, CompactBeliefState, build_input, parse_obs
)
from vec_env import VecEnv

N_ACTIONS = len(ACTIONS)


# ── curriculum ────────────────────────────────────────────────────────────────

# Each stage: (arena_size, difficulty, wall_obstacles, max_steps, n_episodes)
# Total = 6 000 episodes.  Start small so the box is found quickly → dense
# push-head gradients from episode 1.
CURRICULUM = [
    dict(arena_size=300, difficulty=0, wall_obstacles=False, max_steps=600,  episodes=800),
    dict(arena_size=400, difficulty=0, wall_obstacles=False, max_steps=800,  episodes=1000),
    dict(arena_size=500, difficulty=0, wall_obstacles=False, max_steps=1000, episodes=1200),
    dict(arena_size=500, difficulty=0, wall_obstacles=True,  max_steps=1000, episodes=1200),
    dict(arena_size=500, difficulty=2, wall_obstacles=True,  max_steps=1000, episodes=1000),
    dict(arena_size=500, difficulty=3, wall_obstacles=True,  max_steps=1000, episodes=800),
]
CURRICULUM_START_EPS = []   # filled in main()


# ── network ───────────────────────────────────────────────────────────────────

class DualHeadActorCritic(nn.Module):
    """MLP with two actor heads (find / push) and a soft learned gate.

    Forward returns (logits, value, gate):
      logits = (1-gate) * find_logits + gate * push_logits
      gate   in [0,1]; supervised toward env.enable_push during training.
    """

    def __init__(self, in_dim: int = IN_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.find_actor = nn.Linear(hidden, N_ACTIONS)
        self.push_actor = nn.Linear(hidden, N_ACTIONS)
        self.gate_fc    = nn.Linear(hidden, 1)
        self.critic     = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.find_actor.weight, gain=0.01)
        nn.init.zeros_(self.find_actor.bias)
        nn.init.orthogonal_(self.push_actor.weight, gain=0.01)
        nn.init.zeros_(self.push_actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        nn.init.orthogonal_(self.gate_fc.weight, gain=0.01)
        nn.init.zeros_(self.gate_fc.bias)

    def forward(self, x: torch.Tensor):
        h           = self.trunk(x)
        gate        = torch.sigmoid(self.gate_fc(h))        # [B, 1]
        find_logits = self.find_actor(h)
        push_logits = self.push_actor(h)
        # Blend: gate≈0 → find behaviour; gate≈1 → push behaviour
        logits      = (1.0 - gate) * find_logits + gate * push_logits
        value       = self.critic(h).squeeze(-1)
        return logits, value, gate.squeeze(-1)


# ── rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Flat (non-sequential) buffer; no need for BPTT since there's no GRU."""

    def __init__(self, rollout_len: int, n_envs: int, in_dim: int,
                 gamma: float, gae_lam: float) -> None:
        self.rollout_len = rollout_len
        self.n_envs      = n_envs
        self.in_dim      = in_dim
        self.gamma       = gamma
        self.gae_lam     = gae_lam
        self.clear()

    def clear(self) -> None:
        T, N, D = self.rollout_len, self.n_envs, self.in_dim
        self.obs          = np.zeros((T, N, D), dtype=np.float32)
        self.actions      = np.zeros((T, N),    dtype=np.int64)
        self.log_probs    = np.zeros((T, N),    dtype=np.float32)
        self.rewards      = np.zeros((T, N),    dtype=np.float32)
        self.values       = np.zeros((T, N),    dtype=np.float32)
        self.dones        = np.zeros((T, N),    dtype=np.float32)
        self.gate_targets = np.zeros((T, N),    dtype=np.float32)
        self.ptr          = 0

    def add(self, obs, actions, log_probs, rewards, values,
            dones, gate_targets) -> None:
        t = self.ptr
        self.obs[t]          = obs
        self.actions[t]      = actions
        self.log_probs[t]    = log_probs
        self.rewards[t]      = rewards
        self.values[t]       = values
        self.dones[t]        = dones
        self.gate_targets[t] = gate_targets
        self.ptr += 1

    def compute_advantages(self, last_values: np.ndarray):
        T        = self.ptr
        adv      = np.zeros((T, self.n_envs), dtype=np.float32)
        last_gae = np.zeros(self.n_envs,      dtype=np.float32)
        nxt_v    = last_values.astype(np.float32)
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.dones[t]
            delta       = self.rewards[t] + self.gamma * nxt_v * nonterminal - self.values[t]
            last_gae    = delta + self.gamma * self.gae_lam * nonterminal * last_gae
            adv[t]      = last_gae
            nxt_v       = self.values[t]
        returns = adv + self.values[:T]
        return adv, returns


# ── PPO update (flat mini-batches) ────────────────────────────────────────────

def ppo_update(
    net:        DualHeadActorCritic,
    opt:        optim.Optimizer,
    buf:        RolloutBuffer,
    last_values: np.ndarray,
    *,
    clip_eps:   float,
    vf_coef:    float,
    ent_coef:   float,
    gate_coef:  float,
    n_epochs:   int,
    batch_size: int,
    max_grad:   float,
    device:     torch.device,
) -> dict:
    T, N = buf.ptr, buf.n_envs
    D    = buf.in_dim

    adv, returns = buf.compute_advantages(last_values)
    # Normalise advantages over the whole rollout
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs_f     = torch.tensor(buf.obs[:T].reshape(-1, D),     device=device)
    actions_f = torch.tensor(buf.actions[:T].reshape(-1),    dtype=torch.long,  device=device)
    old_lp_f  = torch.tensor(buf.log_probs[:T].reshape(-1),  device=device)
    adv_f     = torch.tensor(adv.reshape(-1),                device=device)
    ret_f     = torch.tensor(returns.reshape(-1),            device=device)
    gate_t_f  = torch.tensor(buf.gate_targets[:T].reshape(-1), device=device)

    total   = T * N
    metrics = collections.defaultdict(list)

    for _ in range(n_epochs):
        perm = torch.randperm(total, device=device)
        for start in range(0, total, batch_size):
            idx = perm[start : start + batch_size]

            logits, values, gate = net(obs_f[idx])
            dist    = Categorical(logits=logits)
            new_lp  = dist.log_prob(actions_f[idx])
            entropy = dist.entropy()

            ratio  = torch.exp(new_lp - old_lp_f[idx])
            surr1  = ratio * adv_f[idx]
            surr2  = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_f[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.mse_loss(values, ret_f[idx])
            entropy_loss = -entropy.mean()
            # Gate auxiliary loss: BCE toward env.enable_push ground truth
            gate_loss   = F.binary_cross_entropy(gate, gate_t_f[idx])

            loss = (policy_loss
                    + vf_coef    * value_loss
                    + ent_coef   * entropy_loss
                    + gate_coef  * gate_loss)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad)
            opt.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            metrics["gate_loss"].append(gate_loss.item())
            metrics["approx_kl"].append(approx_kl)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ── environment helpers ───────────────────────────────────────────────────────

def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def make_env_fn(OBELIX_cls, stage: dict, scaling_factor: int,
                box_speed: int, seed: int):
    def _make():
        env = OBELIX_cls(
            scaling_factor = scaling_factor,
            arena_size     = stage["arena_size"],
            max_steps      = stage["max_steps"],
            wall_obstacles = stage["wall_obstacles"],
            difficulty     = stage["difficulty"],
            box_speed      = box_speed,
            seed           = seed,
        )
        orig_step = env.step
        env.step  = lambda action: orig_step(action, render=False)
        return env
    return _make


def make_shaping_fn(reward_scale: float):
    """Shaping that also tunnels env.enable_push back via the 'failed' slot.

    The VecEnv worker calls: shaped_r, failed = shaping_fn(raw_r, done, env)
    We repurpose 'failed' as is_attached (bool) so the main loop can supervise
    the gate without modifying the environment or VecEnv protocol.
    """
    def _shape(raw_reward: float, done: bool, env) -> tuple:
        r = raw_reward / reward_scale

        # Dense push-progress bonus: reward every FW step in push-state
        if env.enable_push and env.active_state == "P" and not env.stuck_flag:
            r += 1.5 / reward_scale

        # Mild escalating penalty for being stuck while pushing
        if env.stuck_flag and env.enable_push:
            r -= 0.3 / reward_scale

        # Pass enable_push as 'failed' flag back to main loop
        return r, bool(env.enable_push)

    return _shape


def build_vec_env(OBELIX_cls, stage: dict, n_envs: int, scaling_factor: int,
                  box_speed: int, base_seed: int) -> VecEnv:
    make_fns = [
        make_env_fn(OBELIX_cls, stage, scaling_factor, box_speed, base_seed + i)
        for i in range(n_envs)
    ]
    return VecEnv(make_fns=make_fns,
                  reward_shaping_fn=make_shaping_fn(20.0))


# ── checkpoint ────────────────────────────────────────────────────────────────

def save_bundle(path: str, net: DualHeadActorCritic,
                opt: optim.Optimizer | None, config: dict) -> None:
    payload = {
        "model_state_dict": net.state_dict(),
        "config": config,
    }
    if opt is not None:
        payload["optimizer_state_dict"] = opt.state_dict()
    torch.save(payload, path)
    print(f"[save] {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def get_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps"); print("[device] Apple MPS")
    else:
        d = torch.device("cpu"); print("[device] CPU")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO-EPB: Explicit Phase Belief agent")
    # paths
    ap.add_argument("--obelix_py",   type=str,  required=True)
    ap.add_argument("--out",         type=str,  default="weights_epb.pth")
    ap.add_argument("--resume",      type=str,  default=None)
    # curriculum / difficulty
    ap.add_argument("--no_curriculum", action="store_true",
                    help="Skip curriculum; use fixed env params below")
    ap.add_argument("--episodes",    type=int,  default=6000)
    ap.add_argument("--arena_size",  type=int,  default=500)
    ap.add_argument("--max_steps",   type=int,  default=1000)
    ap.add_argument("--difficulty",  type=int,  default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",   type=int,  default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    # training
    ap.add_argument("--n_envs",      type=int,  default=8)
    ap.add_argument("--seed",        type=int,  default=0)
    ap.add_argument("--device",      type=str,  default=None)
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--gamma",       type=float, default=0.995)
    ap.add_argument("--gae_lam",     type=float, default=0.97)
    ap.add_argument("--clip_eps",    type=float, default=0.2)
    ap.add_argument("--vf_coef",     type=float, default=0.5)
    ap.add_argument("--ent_coef",    type=float, default=0.008)
    ap.add_argument("--gate_coef",   type=float, default=0.05)
    ap.add_argument("--rollout_len", type=int,  default=256)
    ap.add_argument("--n_epochs",    type=int,  default=4)
    ap.add_argument("--batch_size",  type=int,  default=256)
    ap.add_argument("--max_grad",    type=float, default=0.5)
    ap.add_argument("--hidden",      type=int,  default=128)
    ap.add_argument("--fsm_bias",    type=float, default=2.0)
    ap.add_argument("--log_every",   type=int,  default=20)
    args = ap.parse_args()

    # ── seeding ───────────────────────────────────────────────────────
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ── curriculum setup ──────────────────────────────────────────────
    if args.no_curriculum:
        stages = [dict(
            arena_size=args.arena_size, difficulty=args.difficulty,
            wall_obstacles=args.wall_obstacles, max_steps=args.max_steps,
            episodes=args.episodes,
        )]
    else:
        stages = CURRICULUM

    # Pre-compute cumulative episode counts for each stage boundary
    stage_start_eps = [0]
    for s in stages:
        stage_start_eps.append(stage_start_eps[-1] + s["episodes"])
    total_episodes = stage_start_eps[-1]

    # ── network ───────────────────────────────────────────────────────
    net = DualHeadActorCritic(in_dim=IN_DIM, hidden=args.hidden).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[resume] loaded {args.resume}")

    OBELIX_cls = import_obelix(args.obelix_py)

    # ── per-env state ─────────────────────────────────────────────────
    beliefs           = [CompactBeliefState(max_steps=1000) for _ in range(args.n_envs)]
    is_attached_actual = np.zeros(args.n_envs, dtype=bool)

    # ── build first stage VecEnv ──────────────────────────────────────
    current_stage_idx = 0
    stage = stages[0]
    vec = build_vec_env(OBELIX_cls, stage, args.n_envs,
                        args.scaling_factor, args.box_speed, args.seed)
    seeds = [args.seed + i for i in range(args.n_envs)]
    raw_obs = vec.reset(seeds=seeds)

    # ── episode tracking ──────────────────────────────────────────────
    ep_returns      = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps        = np.zeros(args.n_envs, dtype=np.int32)
    episodes_done   = 0
    total_steps     = 0
    updates         = 0
    best_return     = -float("inf")
    window_returns: List[float] = []
    window_steps:   List[int]   = []
    success_window  = collections.deque(maxlen=200)
    gate_window     = collections.deque(maxlen=200)
    train_start     = time.time()
    seed_counter    = args.seed + args.n_envs

    config = dict(in_dim=IN_DIM, hidden=args.hidden, fsm_bias=args.fsm_bias)
    print(f"\n[PPO-EPB] in_dim={IN_DIM} hidden={args.hidden} "
          f"n_envs={args.n_envs} rollout={args.rollout_len}")
    print(f"[PPO-EPB] curriculum={'OFF' if args.no_curriculum else 'ON'} "
          f"total_eps={total_episodes}\n")

    pbar = tqdm(total=total_episodes, desc="Training", unit="ep", ncols=120)

    while episodes_done < total_episodes:

        # ── curriculum stage advancement ──────────────────────────────
        while (current_stage_idx + 1 < len(stages) and
               episodes_done >= stage_start_eps[current_stage_idx + 1]):
            current_stage_idx += 1
            stage = stages[current_stage_idx]
            print(f"\n[stage {current_stage_idx}] "
                  f"arena={stage['arena_size']} diff={stage['difficulty']} "
                  f"walls={stage['wall_obstacles']} steps={stage['max_steps']}")
            vec.close()
            vec = build_vec_env(OBELIX_cls, stage, args.n_envs,
                                args.scaling_factor, args.box_speed,
                                seed_counter)
            seed_counter += args.n_envs
            raw_obs = vec.reset(seeds=list(range(seed_counter - args.n_envs,
                                                  seed_counter)))
            for b in beliefs:
                b.reset()
            is_attached_actual[:] = False
            ep_returns[:] = 0
            ep_steps[:] = 0

        # ── FSM bias: decays from fsm_bias → 0.3× over first 60% of training
        progress       = episodes_done / max(total_episodes - 1, 1)
        fsm_bias_now   = args.fsm_bias * max(0.3, 1.0 - progress / 0.6)
        # Entropy also anneals: helps with exploitation later
        ent_coef_now   = args.ent_coef * max(0.25, 1.0 - progress / 0.7)

        # ── collect rollout ───────────────────────────────────────────
        buf = RolloutBuffer(args.rollout_len, args.n_envs, IN_DIM,
                            args.gamma, args.gae_lam)

        for _ in range(args.rollout_len):
            obs_inputs = np.array(
                [build_input(raw_obs[i], beliefs[i]) for i in range(args.n_envs)],
                dtype=np.float32
            )
            obs_t = torch.tensor(obs_inputs, device=device)

            with torch.no_grad():
                logits, values, gates = net(obs_t)

                # Apply FSM prior as soft bias (find phase only)
                bias = torch.zeros_like(logits)
                for i in range(args.n_envs):
                    if not is_attached_actual[i]:
                        # FSM suggestion is the last 5 dims (one-hot) of obs_inputs
                        fsm_idx = int(np.argmax(obs_inputs[i, -N_ACTIONS:]))
                        bias[i, fsm_idx] += fsm_bias_now

                dist      = Categorical(logits=logits + bias)
                actions   = dist.sample()
                log_probs = dist.log_prob(actions)

            action_strs = [ACTIONS[a] for a in actions.cpu().numpy()]
            results     = vec.step(action_strs)

            next_raw_obs         = [r[0] for r in results]
            rewards              = np.array([r[1] for r in results], dtype=np.float32)
            dones                = np.array([r[2] for r in results], dtype=bool)
            is_attached_from_env = np.array([r[3] for r in results], dtype=np.float32)

            # Update ground-truth attachment (from env.enable_push via shaping_fn)
            is_attached_actual[:] = is_attached_from_env.astype(bool)

            buf.add(
                obs          = obs_inputs,
                actions      = actions.cpu().numpy(),
                log_probs    = log_probs.cpu().numpy(),
                rewards      = rewards,
                values       = values.cpu().numpy(),
                dones        = dones.astype(np.float32),
                gate_targets = is_attached_from_env,
            )

            ep_returns += rewards
            ep_steps   += 1
            total_steps += args.n_envs

            for i in range(args.n_envs):
                beliefs[i].update(raw_obs[i], action_strs[i])
                gate_window.append(float(gates[i].item()))

                if dones[i]:
                    window_returns.append(float(ep_returns[i]))
                    window_steps.append(int(ep_steps[i]))
                    # Scaled success threshold: 80 ≈ (2000+100-~200) / 20
                    success_window.append(float(ep_returns[i] > 80.0))
                    best_return = max(best_return, float(ep_returns[i]))
                    episodes_done += 1
                    pbar.update(1)

                    seed_counter += 1
                    new_obs = vec.reset_one(i, seed_counter)
                    raw_obs[i] = new_obs
                    beliefs[i].reset()
                    is_attached_actual[i] = False
                    ep_returns[i] = 0.0
                    ep_steps[i]   = 0
                else:
                    raw_obs[i] = next_raw_obs[i]

            if episodes_done >= total_episodes:
                break

        # ── bootstrap last values ─────────────────────────────────────
        with torch.no_grad():
            obs_last = np.array(
                [build_input(raw_obs[i], beliefs[i]) for i in range(args.n_envs)],
                dtype=np.float32
            )
            _, last_v, _ = net(torch.tensor(obs_last, device=device))
            last_values  = last_v.cpu().numpy()
            # Zero out values for envs that just finished an episode
            for i in range(args.n_envs):
                if ep_steps[i] == 0:
                    last_values[i] = 0.0

        # ── PPO update ────────────────────────────────────────────────
        metrics = ppo_update(
            net         = net,
            opt         = opt,
            buf         = buf,
            last_values = last_values,
            clip_eps    = args.clip_eps,
            vf_coef     = args.vf_coef,
            ent_coef    = ent_coef_now,
            gate_coef   = args.gate_coef,
            n_epochs    = args.n_epochs,
            batch_size  = args.batch_size,
            max_grad    = args.max_grad,
            device      = device,
        )
        updates += 1

        # ── logging ───────────────────────────────────────────────────
        if episodes_done > 0 and episodes_done % args.log_every == 0:
            win     = min(args.log_every, len(window_returns))
            avg_ret = float(np.mean(window_returns[-win:])) if win else 0.0
            avg_st  = float(np.mean(window_steps[-win:]))   if win else 0.0
            succ    = 100.0 * float(np.mean(success_window)) if success_window else 0.0
            avg_g   = float(np.mean(gate_window))            if gate_window   else 0.0
            speed   = episodes_done / max(time.time() - train_start, 1e-6)
            print(
                f"\n[ep {episodes_done:5d}|stage {current_stage_idx}] "
                f"avg_ret={avg_ret:.2f} best={best_return:.2f} "
                f"avg_steps={avg_st:.1f} succ(200)={succ:.1f}% "
                f"gate={avg_g:.3f}"
            )
            print(
                f"  policy={metrics.get('policy_loss',0):.4f} "
                f"value={metrics.get('value_loss',0):.4f} "
                f"entropy={metrics.get('entropy',0):.4f} "
                f"gate_loss={metrics.get('gate_loss',0):.4f} "
                f"kl={metrics.get('approx_kl',0):.4f} "
                f"fsm_bias={fsm_bias_now:.2f} "
                f"steps={total_steps} {speed:.1f}ep/s"
            )
            pbar.set_postfix(
                ret  = f"{avg_ret:.1f}",
                best = f"{best_return:.1f}",
                succ = f"{succ:.0f}%",
                gate = f"{avg_g:.2f}",
            )

        # ── checkpoint: save on improvement ──────────────────────────
        if best_return > -float("inf") and episodes_done % 100 == 0:
            save_bundle(args.out, net, opt, config)

    # ── final save ────────────────────────────────────────────────────
    vec.close()
    save_bundle(args.out, net, opt, config)
    pbar.close()
    print(f"\n[done] {episodes_done} episodes | best_return={best_return:.2f}")


if __name__ == "__main__":
    main()
