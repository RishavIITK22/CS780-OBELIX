"""PPO trainer for OBELIX — vectorized envs, frame stacking, reward shaping.

Uses VecEnv (vec_env.py) to run N independent OBELIX instances in parallel
subprocesses and collect rollouts N× faster than a single-env loop.

Usage:
    python train_ppo.py --obelix_py ./obelix.py --n_envs 8
    python train_ppo.py --obelix_py ./obelix.py --n_envs 8 --out weights.pth \\
        --episodes 3000 --stack 8 --rollout_len 512

Architecture notes:
  - One VecEnv holds n_envs workers, each in its own subprocess.
  - One FrameStack and one RewardShaper are kept per worker in the MAIN
    process — both are stateful (deques, seen-pattern sets) and are not
    cleanly picklable, so keeping them here is simpler and safer.
  - Reward shaping is NOT passed to VecEnv as reward_shaping_fn; VecEnv
    returns raw (obs, reward, done) 3-tuples and shaping happens below.
  - push_active[i] is a sticky per-worker latch: once obs[16] (IR sensor)
    fires it stays True for the rest of that episode, mirroring how
    env.enable_push works inside the subprocess.
  - The rollout buffer stores rollout_len × n_envs transitions. GAE is
    computed per-worker column (shape: T_actual × n_envs) then flattened.
  - Total transitions per PPO update = rollout_len × n_envs (or fewer on
    the last partial rollout when training ends mid-rollout).
"""

from __future__ import annotations

import argparse
import collections
import time
import importlib.util
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from vec_env import VecEnv
from reward_shaper import RewardShaper
from state_encoder import BeliefStateEncoder


# ── Device ─────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        d = torch.device("cpu")
        print("[Device] CPU")
    return d


DEVICE  = get_device()
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACT   = len(ACTIONS)



# ── Actor-Critic network ───────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """Shared-trunk MLP Actor-Critic for PPO.

    Input : stacked obs  (obs_dim * k,)
    Trunk : two Tanh hidden layers, shared
    Actor : logits over 5 discrete actions
    Critic: scalar V(s)
    """

    def __init__(self, in_dim: int, hidden: int = 128, n_actions: int = N_ACT):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        logits, value = self(x)
        dist   = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor):
        logits, values = self(x)
        dist    = Categorical(logits=logits)
        log_p   = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_p, entropy, values


# ── Rollout buffer (vectorized) ────────────────────────────────────────────────
class RolloutBuffer:
    """Stores rollout_len × n_envs transitions, computes per-worker GAE.

    Layout
    ------
    add_batch() appends all N workers' data at each timestep in worker order:
    [t0_w0, t0_w1, ..., t0_wN-1, t1_w0, ...].
    Reshaping to (T_actual, N) gives the per-worker time axis so GAE never
    bleeds across worker boundaries.

    Partial fills
    -------------
    The rollout loop may break early when episodes_done >= args.episodes.
    T_actual = len(rewards) // N may therefore be less than rollout_len.
    Both compute_gae() and tensors() trim to T_actual * N to stay consistent.
    """

    def __init__(self, rollout_len: int, n_envs: int, obs_dim: int):
        self.T       = rollout_len
        self.N       = n_envs
        self.total   = rollout_len * n_envs
        self.obs_dim = obs_dim
        self.clear()

    def clear(self) -> None:
        self.obs       = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []
        self.ptr       = 0

    def add_batch(
        self,
        obs:       np.ndarray,   # (N, obs_dim)
        actions:   np.ndarray,   # (N,)
        log_probs: np.ndarray,   # (N,)
        rewards:   np.ndarray,   # (N,)
        values:    np.ndarray,   # (N,)
        dones:     np.ndarray,   # (N,)  float 0.0 or 1.0
    ) -> None:
        for i in range(self.N):
            self.obs      .append(obs[i])
            self.actions  .append(actions[i])
            self.log_probs.append(log_probs[i])
            self.rewards  .append(rewards[i])
            self.values   .append(values[i])
            self.dones    .append(dones[i])
        self.ptr += self.N

    def is_full(self) -> bool:
        return self.ptr >= self.total

    def _t_actual(self) -> int:
        """Number of complete timestep rows stored (may be < self.T)."""
        return len(self.rewards) // self.N

    def compute_gae(
        self,
        last_values: np.ndarray,   # (N,)
        gamma:       float = 0.99,
        gae_lam:     float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-worker GAE then flatten to (T_actual * N,).

        FIX (Bug 1 — original): The previous code used `t == self.T - 1`
        to decide when to use last_values as the bootstrap.  When the buffer
        is partially filled (T_actual < self.T), that condition is never True
        and last_values is never substituted, so the final timestep's TD error
        uses values[T_actual] — an out-of-bounds access — instead.
        Fix: use `t == T_actual - 1`.
        """
        T_actual = self._t_actual()
        n_trim   = T_actual * self.N

        rewards = np.array(self.rewards[:n_trim], dtype=np.float32).reshape(T_actual, self.N)
        values  = np.array(self.values [:n_trim], dtype=np.float32).reshape(T_actual, self.N)
        dones   = np.array(self.dones  [:n_trim], dtype=np.float32).reshape(T_actual, self.N)

        advantages = np.zeros((T_actual, self.N), dtype=np.float32)
        last_gae   = np.zeros(self.N,             dtype=np.float32)

        for t in reversed(range(T_actual)):
            # FIX (Bug 1): use T_actual - 1, not self.T - 1
            next_val = last_values if t == T_actual - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta    = rewards[t] + gamma * next_val * not_done - values[t]
            last_gae = delta + gamma * gae_lam * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return (
            torch.from_numpy(advantages.reshape(-1)),
            torch.from_numpy(returns   .reshape(-1)),
        )

    def tensors(self, device: torch.device):
        """Return obs/actions/log_probs tensors trimmed to complete rows."""
        n_trim    = self._t_actual() * self.N
        obs       = torch.tensor(np.array(self.obs      [:n_trim]), dtype=torch.float32, device=device)
        actions   = torch.tensor(np.array(self.actions  [:n_trim]), dtype=torch.long,    device=device)
        log_probs = torch.tensor(np.array(self.log_probs[:n_trim]), dtype=torch.float32, device=device)
        return obs, actions, log_probs


# ── PPO update ─────────────────────────────────────────────────────────────────
def ppo_update(
    net:         ActorCritic,
    opt:         optim.Optimizer,
    buf:         RolloutBuffer,
    last_values: np.ndarray,
    device:      torch.device,
    gamma:       float = 0.99,
    gae_lam:     float = 0.95,
    clip_eps:    float = 0.2,
    vf_coef:     float = 0.5,
    ent_coef:    float = 0.01,
    n_epochs:    int   = 4,
    mini_batch:  int   = 64,
    max_grad:    float = 0.5,
) -> dict:
    """Core PPO update over the full (T_actual × N) flattened batch.

    FIX (Bug 2 — original): The previous code set T_total = buf.total
    (always rollout_len * n_envs) and passed it to randperm(), but
    compute_gae() and tensors() both trim to T_actual * N, which may be
    smaller on a partial fill.  randperm(buf.total) then generates indices
    beyond the tensor bounds → IndexError.
    Fix: derive T_total from the actual trimmed length after compute_gae.
    """
    advantages, returns = buf.compute_gae(last_values, gamma, gae_lam)

    # advantages/returns shape: (T_actual * N,)
    # FIX (Bug 2): use the actual length, not buf.total
    T_total = advantages.shape[0]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.to(device)
    returns    = returns   .to(device)

    obs, actions, old_log_probs = buf.tensors(device)

    metrics = collections.defaultdict(list)

    for _ in range(n_epochs):
        idx = torch.randperm(T_total)   # FIX (Bug 2): randperm over actual size

        for start in range(0, T_total, mini_batch):
            mb = idx[start : start + mini_batch]

            new_log_p, entropy, values = net.evaluate(obs[mb], actions[mb])

            ratio       = torch.exp(new_log_p - old_log_probs[mb])
            adv_mb      = advantages[mb]
            surr1       = ratio * adv_mb
            surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss   = nn.functional.mse_loss(values, returns[mb])
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad)
            opt.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"] .append(value_loss .item())
            metrics["entropy"]    .append(-entropy_loss.item())

            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            metrics["approx_kl"].append(approx_kl)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ── Helpers ────────────────────────────────────────────────────────────────────
def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def make_env_fn(OBELIX, args, worker_seed: int):
    """Zero-argument factory for one OBELIX subprocess instance.

    Patches env.step to always pass render=False so VecEnv can call
    env.step(action_str) without knowing about the render kwarg.
    """
    def _make():
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=worker_seed,
        )
        _orig_step = env.step
        env.step = lambda action: _orig_step(action, render=False)
        return env
    return _make


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
    description="PPO + VecEnv + frame stacking trainer for OBELIX"
    )
    # ── Environment ───────────────────────────────────────────────────────
    ap.add_argument("--obelix_py",      type=str, required=True)
    ap.add_argument("--out",            type=str, default="weights.pth")
    ap.add_argument("--load",           type=str, default=None,
                    help="Path to weights to warm-start from (curriculum training)")
    ap.add_argument("--episodes",       type=int, default=3000)
    ap.add_argument("--max_steps",      type=int, default=1000)
    ap.add_argument("--difficulty",     type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",      type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size",     type=int, default=500)
    ap.add_argument("--seed",           type=int, default=0)
    ap.add_argument("--device",         type=str, default=None)

    # ── Parallelism ────────────────────────────────────────────────────────
    ap.add_argument("--n_envs", type=int, default=8)

    # ── PPO hyperparameters ────────────────────────────────────────────────
    ap.add_argument("--lr",          type=float, default=1e-4,
                    help="Adam LR — lowered from 3e-4 to avoid over-fitting early")
    ap.add_argument("--gamma",       type=float, default=0.99)
    ap.add_argument("--gae_lam",     type=float, default=0.95)
    ap.add_argument("--clip_eps",    type=float, default=0.2)
    ap.add_argument("--vf_coef",     type=float, default=0.5)
    ap.add_argument("--ent_coef",    type=float, default=0.05,
                    help="Entropy bonus — raised to 0.05 to discourage spin convergence")
    ap.add_argument("--n_epochs",    type=int,   default=4)
    ap.add_argument("--rollout_len", type=int,   default=512)
    ap.add_argument("--mini_batch",  type=int,   default=64)
    ap.add_argument("--max_grad",    type=float, default=0.5)
    ap.add_argument("--hidden",      type=int,   default=128)

    # ── Frame stacking ─────────────────────────────────────────────────────
    ap.add_argument("--stack", type=int, default=8)

    # ── Reward shaping ─────────────────────────────────────────────────────
    ap.add_argument("--reward_scale",     type=float, default=20.0)
    ap.add_argument("--approach_scale",   type=float, default=3.0)
    ap.add_argument("--push_scale",       type=float, default=4.0)
    ap.add_argument("--stuck_penalty",    type=float, default=2.0)
    ap.add_argument("--spin_penalty",     type=float, default=2.0)
    ap.add_argument("--spin_threshold",   type=int,   default=6)
    ap.add_argument("--forward_bonus",    type=float, default=1.0)
    ap.add_argument("--efficiency_bonus", type=float, default=10.0)
    ap.add_argument("--no_shaping",       action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else DEVICE

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    make_fns = [
        make_env_fn(OBELIX, args, worker_seed=args.seed + i)
        for i in range(args.n_envs)
    ]
    vec = VecEnv(make_fns=make_fns, reward_shaping_fn=None)

    # BeliefStateEncoder handles both encoding and frame-stacking internally.
    encoders = [BeliefStateEncoder(stack_k=args.stack) for _ in range(args.n_envs)]
    shapers = [
        RewardShaper(
            reward_scale     = args.reward_scale,
            approach_scale   = args.approach_scale,
            push_scale       = args.push_scale,
            stuck_penalty    = args.stuck_penalty,
            spin_penalty     = args.spin_penalty,
            spin_threshold   = args.spin_threshold,
            forward_bonus    = args.forward_bonus,
            efficiency_bonus = args.efficiency_bonus,
            max_steps        = args.max_steps,
        )
        for _ in range(args.n_envs)
    ]

    obs_dim = encoders[0].output_dim
    net     = ActorCritic(in_dim=obs_dim, hidden=args.hidden).to(device)

    if args.load is not None:
        sd = torch.load(args.load, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict(sd, strict=False)
        print(f"[warm-start] Loaded weights from {args.load}")

    opt     = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    total_updates = max(
        (args.episodes * args.max_steps) // (args.rollout_len * args.n_envs), 1
    )
    # scheduler = optim.lr_scheduler.LinearLR(
    #     opt, start_factor=1.0, end_factor=0.01, total_iters=total_updates
    # )
    scheduler = optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=999999)

    buf = RolloutBuffer(
        rollout_len=args.rollout_len,
        n_envs=args.n_envs,
        obs_dim=obs_dim,
    )

    best_return    = -float("inf")
    episodes_done  = 0
    total_steps    = 0
    update_count   = 0
    train_start    = time.time()

    window_returns: List[float] = []
    window_steps:   List[int]   = []
    success_count  = 0

    recent_metrics = collections.defaultdict(lambda: collections.deque(maxlen=20))

    # FIX (Bug 3): track the last episode index at which we logged so the
    # log fires reliably even when episodes_done jumps by n_envs per tick.
    # The old `episodes_done % 10 == 0` check is skipped entirely whenever
    # n_envs doesn't divide 10 (e.g. n_envs=8: counter goes 0→8→16→... and
    # never lands on 10, 30, 50, ...).
    last_log_ep = 0
    LOG_EVERY   = 10   # log once per this many completed episodes

    init_seeds   = [args.seed + i for i in range(args.n_envs)]
    raw_obs_list = vec.reset(seeds=init_seeds)

    for enc in encoders:
        enc.reset()
    obs_arr = np.array(
        [encoders[i].encode(raw_obs_list[i]) for i in range(args.n_envs)],
        dtype=np.float32,
    )
    for sh in shapers:
        sh.reset()

    ep_ret      = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps    = np.zeros(args.n_envs, dtype=np.int32)
    last_done   = np.zeros(args.n_envs, dtype=bool)

    # Sticky per-worker push flag — mirrors env.enable_push which latches True
    # on first IR contact and stays True for the rest of the episode.
    # obs[16] (IR bit) alone is NOT sticky: it only fires when the sensor
    # line overlaps the box in the current frame, and can briefly drop to 0
    # mid-push.  Using obs[16] raw would give wrong signals to RewardShaper.
    push_active = np.zeros(args.n_envs, dtype=bool)

    print(f"\n[PPO-VecEnv] n_envs={args.n_envs}  obs_dim={obs_dim}  "
          f"rollout_len={args.rollout_len}  "
          f"total_per_update={args.rollout_len * args.n_envs}")
    print(f"[PPO-VecEnv] episodes={args.episodes}  device={device}  "
          f"reward_shaping={'OFF' if args.no_shaping else 'ON'}\n")

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=110)

    while episodes_done < args.episodes:

        buf.clear()

        for _ in range(args.rollout_len):

            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)

            with torch.no_grad():
                actions_t, log_probs_t, _, values_t = net.get_action(obs_t)

            action_indices = actions_t.cpu().numpy()
            action_strs    = [ACTIONS[a] for a in action_indices]

            results       = vec.step(action_strs)
            raw_obs2_list = [r[0] for r in results]
            raw_rewards   = np.array([r[1] for r in results], dtype=np.float32)
            dones         = np.array([r[2] for r in results], dtype=bool)

            new_obs_arr    = np.empty_like(obs_arr)
            shaped_rewards = np.empty(args.n_envs, dtype=np.float32)

            for i in range(args.n_envs):
                # encode() updates internal state and returns stacked vector
                new_obs_arr[i] = encoders[i].encode(raw_obs2_list[i], action_indices[i])

                scaled = float(raw_rewards[i]) / args.reward_scale

                # Use reward spike to latch push_active — IR bit is NOT sticky
                # and can drop to 0 mid-push, so raw_obs2_list[i][16] is wrong.
                if raw_rewards[i] >= 90.0:
                    push_active[i] = True

                if args.no_shaping:
                    shaped_rewards[i] = scaled
                else:
                    # encode_single() reads updated state for 38-dim shaper input
                    enc_single = encoders[i].encode_single(raw_obs2_list[i], action_indices[i])
                    shaped_rewards[i] = shapers[i].shape(
                        raw_reward=scaled,
                        encoded_obs=enc_single,
                        done=bool(dones[i]),
                        enable_push=bool(push_active[i]),
                        action=action_strs[i],
                    )

            buf.add_batch(
                obs       = obs_arr,
                actions   = action_indices,
                log_probs = log_probs_t.cpu().numpy(),
                rewards   = shaped_rewards,
                values    = values_t.cpu().numpy(),
                dones     = dones.astype(np.float32),
            )

            ep_ret   += shaped_rewards
            ep_steps += 1
            total_steps += args.n_envs
            last_done[:] = dones

            for i in range(args.n_envs):
                if not dones[i]:
                    continue

                if raw_rewards[i] >= 100.0:
                    success_count += 1

                window_returns.append(float(ep_ret[i]))
                window_steps  .append(int(ep_steps[i]))

                if ep_ret[i] > best_return:
                    best_return = float(ep_ret[i])
                    torch.save(net.cpu().state_dict(), args.out + ".best")
                    net.to(device)

                episodes_done += 1
                pbar.set_postfix({
                    "ret":  f"{ep_ret[i]:.1f}",
                    "best": f"{best_return:.1f}",
                    "suc":  success_count,
                })
                pbar.update(1)

                new_seed = args.seed + args.n_envs + episodes_done
                raw_reset_obs = vec.reset_one(i, seed=new_seed)
                encoders[i].reset()
                new_obs_arr[i] = encoders[i].encode(raw_reset_obs)
                shapers[i].reset()

                ep_ret[i]      = 0.0
                ep_steps[i]    = 0
                push_active[i] = False

            obs_arr = new_obs_arr

            if episodes_done >= args.episodes:
                break

        # ── PPO update ────────────────────────────────────────────────────────
        with torch.no_grad():
            obs_t     = torch.tensor(obs_arr, dtype=torch.float32, device=device)
            _, last_v = net(obs_t)
            last_vals = last_v.cpu().numpy()

        last_vals = np.where(last_done, 0.0, last_vals)

        metrics = ppo_update(
            net         = net,
            opt         = opt,
            buf         = buf,
            last_values = last_vals,
            device      = device,
            gamma       = args.gamma,
            gae_lam     = args.gae_lam,
            clip_eps    = args.clip_eps,
            vf_coef     = args.vf_coef,
            ent_coef    = args.ent_coef,
            n_epochs    = args.n_epochs,
            mini_batch  = args.mini_batch,
            max_grad    = args.max_grad,
        )

        scheduler.step()
        update_count += 1

        for k, v in metrics.items():
            recent_metrics[k].append(v)

        # FIX (Bug 3): fire whenever we've crossed another LOG_EVERY boundary,
        # not just when episodes_done happens to be exactly divisible by 10.
        # With n_envs=8, episodes_done steps by 8 and skips most multiples of 10.
        # FIX (Bug 4): label the window by its actual size (len(window_returns))
        # not by a hardcoded offset of 9, which is wrong whenever n_envs > 1.
        if (episodes_done // LOG_EVERY) > (last_log_ep // LOG_EVERY) \
                and episodes_done > 0 and window_returns:
            elapsed    = time.time() - train_start
            n_in_window = len(window_returns)
            tqdm.write(
                f"\n┌─ ep {episodes_done - n_in_window + 1:>4d}–{episodes_done:<4d} "
                f"({elapsed:.0f}s) | workers={args.n_envs} "
                f"────────────────────────────\n"
                f"│  Avg Return   : {np.mean(window_returns):>8.2f}   "
                f"Best : {best_return:.2f}   Successes : {success_count}\n"
                f"│  Avg Steps    : {np.mean(window_steps):.1f}   "
                f"Total steps : {total_steps:,}   Updates : {update_count}\n"
                f"│  Policy loss  : {np.mean(recent_metrics['policy_loss']):>8.4f}   "
                f"Value loss : {np.mean(recent_metrics['value_loss']):.4f}\n"
                f"│  Entropy      : {np.mean(recent_metrics['entropy']):>8.4f}   "
                f"Approx KL  : {np.mean(recent_metrics['approx_kl']):.4f}   "
                f"LR : {scheduler.get_last_lr()[0]:.2e}\n"
                f"└{'─'*64}"
            )
            last_log_ep = episodes_done
            window_returns.clear()
            window_steps.clear()

    pbar.close()
    vec.close()

    torch.save(net.cpu().state_dict(), args.out)
    elapsed = time.time() - train_start
    print(f"\nSaved : {args.out}  (best: {args.out}.best)")
    print(f"Time  : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Stats : {episodes_done} episodes | {total_steps:,} steps | "
          f"{success_count} successes ({100*success_count/max(1,episodes_done):.1f}%)")


if __name__ == "__main__":
    main()