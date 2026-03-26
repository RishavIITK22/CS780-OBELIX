from __future__ import annotations

import argparse
import collections
import importlib.util
import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from vec_env import VecEnv


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
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


DEVICE = get_device()

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACT = len(ACTIONS)
OBS_DIM = 18


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic imports
# ──────────────────────────────────────────────────────────────────────────────
def import_class_from_path(path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("dynamic_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def import_obelix(path: str):
    return import_class_from_path(path, "OBELIX")


def import_reward_shaper(path: str, class_name: str = "RewardShaper"):
    return import_class_from_path(path, class_name)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for policy input
# x_t = [obs_t, onehot(a_{t-1}), r_{t-1}]
# This is very useful in POMDPs.
# ──────────────────────────────────────────────────────────────────────────────
POLICY_INPUT_DIM = OBS_DIM + N_ACT + 1


def one_hot_actions(action_idx: np.ndarray, n_actions: int = N_ACT) -> np.ndarray:
    out = np.zeros((len(action_idx), n_actions), dtype=np.float32)
    out[np.arange(len(action_idx)), action_idx] = 1.0
    return out


def build_policy_input(
    obs_arr: np.ndarray,               # (N, OBS_DIM)
    prev_action_oh: np.ndarray,        # (N, N_ACT)
    prev_reward: np.ndarray,           # (N,)
) -> np.ndarray:
    prev_reward_col = prev_reward.reshape(-1, 1).astype(np.float32)
    return np.concatenate(
        [obs_arr.astype(np.float32), prev_action_oh.astype(np.float32), prev_reward_col],
        axis=1,
    ).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Actor-Critic LSTM
# ──────────────────────────────────────────────────────────────────────────────
class ActorCriticLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = POLICY_INPUT_DIM,
        hidden_dim: int = 128,
        n_actions: int = N_ACT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def init_hidden(
        self, batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.hidden_dim, device=device)
        c = torch.zeros(1, batch, self.hidden_dim, device=device)
        return h, c

    def forward(
        self,
        x: torch.Tensor,   # (N, D)
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        enc = self.encoder(x).unsqueeze(1)      # (N, 1, 64)
        out, hidden = self.lstm(enc, hidden)    # (N, 1, H)
        out = out.squeeze(1)                    # (N, H)
        logits = self.actor(out)                # (N, A)
        value = self.critic(out).squeeze(-1)    # (N,)
        return logits, value, hidden

    def get_action(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ):
        logits, value, hidden = self.forward(x, hidden)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, hidden

    def evaluate_sequence(
        self,
        x_seq: torch.Tensor,       # (N_mb, T, D)
        init_h: torch.Tensor,      # (1, N_mb, H)
        init_c: torch.Tensor,      # (1, N_mb, H)
        dones_seq: torch.Tensor,   # (N_mb, T)
    ):
        N_mb, T, _ = x_seq.shape

        h, c = init_h, init_c
        enc_seq = self.encoder(x_seq)  # (N_mb, T, 64)
        outputs = []

        for t in range(T):
            x_t = enc_seq[:, t, :].unsqueeze(1)      # (N_mb, 1, 64)
            out, (h, c) = self.lstm(x_t, (h, c))    # (N_mb, 1, H)
            out = out.squeeze(1)
            outputs.append(out)

            if t < T - 1:
                mask = (1.0 - dones_seq[:, t]).view(1, N_mb, 1)
                h = h * mask
                c = c * mask

        out_seq = torch.stack(outputs, dim=1)             # (N_mb, T, H)
        logits = self.actor(out_seq)                      # (N_mb, T, A)
        values = self.critic(out_seq).squeeze(-1)         # (N_mb, T)

        logits_flat = logits.permute(1, 0, 2).reshape(T * N_mb, -1)
        values_flat = values.permute(1, 0).reshape(T * N_mb)

        dist = Categorical(logits=logits_flat)
        entropy_flat = dist.entropy()

        return logits_flat, values_flat, entropy_flat


# ──────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# Stores policy inputs, not just raw observations.
# ──────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, rollout_len: int, n_envs: int, input_dim: int):
        self.T = rollout_len
        self.N = n_envs
        self.input_dim = input_dim
        self._init_h: Optional[torch.Tensor] = None
        self._init_c: Optional[torch.Tensor] = None
        self.clear()

    def clear(self) -> None:
        self.x = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store_init_hidden(self, h: torch.Tensor, c: torch.Tensor) -> None:
        self._init_h = h.detach().clone()
        self._init_c = c.detach().clone()

    def get_init_hidden(self, device: torch.device):
        assert self._init_h is not None and self._init_c is not None
        return self._init_h.to(device), self._init_c.to(device)

    def add_batch(
        self,
        x: np.ndarray,             # (N, D)
        actions: np.ndarray,       # (N,)
        log_probs: np.ndarray,     # (N,)
        rewards: np.ndarray,       # (N,)
        values: np.ndarray,        # (N,)
        dones: np.ndarray,         # (N,) float
    ) -> None:
        for i in range(self.N):
            self.x.append(x[i])
            self.actions.append(actions[i])
            self.log_probs.append(log_probs[i])
            self.rewards.append(rewards[i])
            self.values.append(values[i])
            self.dones.append(dones[i])

    def _t_actual(self) -> int:
        return len(self.rewards) // self.N

    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float = 0.995,
        gae_lam: float = 0.97,
    ):
        T_actual = self._t_actual()
        n_trim = T_actual * self.N

        rewards = np.array(self.rewards[:n_trim], dtype=np.float32).reshape(T_actual, self.N)
        values = np.array(self.values[:n_trim], dtype=np.float32).reshape(T_actual, self.N)
        dones = np.array(self.dones[:n_trim], dtype=np.float32).reshape(T_actual, self.N)

        advantages = np.zeros((T_actual, self.N), dtype=np.float32)
        last_gae = np.zeros(self.N, dtype=np.float32)

        for t in reversed(range(T_actual)):
            next_val = last_values if t == T_actual - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * not_done - values[t]
            last_gae = delta + gamma * gae_lam * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return (
            torch.from_numpy(advantages.reshape(-1)),
            torch.from_numpy(returns.reshape(-1)),
        )

    def tensors(self, device: torch.device):
        T_actual = self._t_actual()
        n_trim = T_actual * self.N

        x_flat = torch.tensor(
            np.array(self.x[:n_trim]), dtype=torch.float32, device=device
        )  # (T*N, D)

        x_seq = x_flat.reshape(T_actual, self.N, self.input_dim).permute(1, 0, 2).contiguous()
        dones_flat = torch.tensor(
            np.array(self.dones[:n_trim]), dtype=torch.float32, device=device
        )
        dones_seq = dones_flat.reshape(T_actual, self.N).permute(1, 0).contiguous()

        actions = torch.tensor(
            np.array(self.actions[:n_trim]), dtype=torch.long, device=device
        )
        old_log_probs = torch.tensor(
            np.array(self.log_probs[:n_trim]), dtype=torch.float32, device=device
        )

        return x_flat, x_seq, dones_seq, actions, old_log_probs


# ──────────────────────────────────────────────────────────────────────────────
# PPO update
# ──────────────────────────────────────────────────────────────────────────────
def ppo_update(
    net: ActorCriticLSTM,
    opt: optim.Optimizer,
    buf: RolloutBuffer,
    last_values: np.ndarray,
    device: torch.device,
    gamma: float = 0.995,
    gae_lam: float = 0.97,
    clip_eps: float = 0.15,
    vf_coef: float = 0.5,
    ent_coef: float = 0.02,
    n_epochs: int = 4,
    n_mini_batches: int = 4,
    max_grad: float = 0.5,
    target_kl: float = 0.02,
):
    advantages, returns = buf.compute_gae(last_values, gamma, gae_lam)
    advantages = advantages.to(device)
    returns = returns.to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    _, x_seq, dones_seq, actions, old_log_probs = buf.tensors(device)
    init_h, init_c = buf.get_init_hidden(device)

    T_actual = buf._t_actual()
    N = buf.N

    metrics = collections.defaultdict(list)
    early_stop = False

    for epoch in range(n_epochs):
        worker_indices = torch.randperm(N, device=device)
        mb_size = max(1, N // n_mini_batches)

        for mb_start in range(0, N, mb_size):
            wb = worker_indices[mb_start: mb_start + mb_size]
            N_mb = wb.shape[0]

            logits_flat, values_flat, entropy_flat = net.evaluate_sequence(
                x_seq[wb],
                init_h[:, wb, :].contiguous(),
                init_c[:, wb, :].contiguous(),
                dones_seq[wb],
            )

            t_idx = torch.arange(T_actual, device=device)
            global_idx = (t_idx.unsqueeze(0) * N + wb.unsqueeze(1))
            flat_idx = global_idx.permute(1, 0).reshape(-1)

            adv_mb = advantages[flat_idx]
            ret_mb = returns[flat_idx]
            act_mb = actions[flat_idx]
            old_logp_mb = old_log_probs[flat_idx]

            dist = Categorical(logits=logits_flat)
            new_logp = dist.log_prob(act_mb)
            entropy = entropy_flat.mean()

            ratio = torch.exp(new_logp - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values_flat, ret_mb)
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad)
            opt.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["approx_kl"].append(approx_kl)
            metrics["clip_frac"].append(clip_frac)

            if approx_kl > 1.5 * target_kl:
                early_stop = True
                break

        if early_stop:
            break

    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    metrics["early_stop"] = float(early_stop)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Env factory
# ──────────────────────────────────────────────────────────────────────────────
def make_env_fn(OBELIX, args, worker_seed: int):
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
        original_step = env.step
        env.step = lambda action: original_step(action, render=False)
        return env
    return _make


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Robust PPO+LSTM trainer for OBELIX POMDP")

    # Files
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--reward_shaper_py", type=str, required=True)
    ap.add_argument("--reward_shaper_class", type=str, default="RewardShaper")

    # Training
    ap.add_argument("--out", type=str, default="weights_ppo_lstm_robust.pth")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)

    # Env
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--n_envs", type=int, default=16)

    # PPO
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae_lam", type=float, default=0.97)
    ap.add_argument("--clip_eps", type=float, default=0.15)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.02)
    ap.add_argument("--target_kl", type=float, default=0.02)
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--rollout_len", type=int, default=256)
    ap.add_argument("--n_mini_batches", type=int, default=4)
    ap.add_argument("--max_grad", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)

    # Reward shaping args passed into external RewardShaper
    ap.add_argument("--reward_scale", type=float, default=10.0)
    ap.add_argument("--approach_scale", type=float, default=2.0)
    ap.add_argument("--push_scale", type=float, default=3.0)
    ap.add_argument("--stuck_penalty", type=float, default=1.0)
    ap.add_argument("--explore_scale", type=float, default=0.4)
    ap.add_argument("--efficiency_bonus", type=float, default=50.0)
    ap.add_argument("--forward_bonus", type=float, default=1.0)
    ap.add_argument("--spin_penalty", type=float, default=3.0)
    ap.add_argument("--spin_threshold", type=int, default=5)
    ap.add_argument("--boundary_penalty", type=float, default=3.0)
    ap.add_argument("--boundary_threshold", type=int, default=10)

    ap.add_argument("--track_scale", type=float, default=1.5)
    ap.add_argument("--track_memory_steps", type=int, default=6)
    ap.add_argument("--ir_bonus", type=float, default=2.0)
    ap.add_argument("--recovery_bonus", type=float, default=2.5)
    ap.add_argument("--oscillation_penalty", type=float, default=1.5)
    ap.add_argument("--stability_bonus", type=float, default=0.4)
    ap.add_argument("--persistence_penalty", type=float, default=2.0)
    ap.add_argument("--persistence_threshold", type=int, default=6)
    ap.add_argument("--push_forward_bonus", type=float, default=1.5)
    ap.add_argument("--blind_push_penalty", type=float, default=1.0)

    ap.add_argument("--no_shaping", action="store_true")

    args = ap.parse_args()

    device = torch.device(args.device) if args.device else DEVICE

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    RewardShaperClass = import_reward_shaper(
        args.reward_shaper_py,
        args.reward_shaper_class,
    )

    make_fns = [
        make_env_fn(OBELIX, args, worker_seed=args.seed + i)
        for i in range(args.n_envs)
    ]
    vec = VecEnv(make_fns=make_fns, reward_shaping_fn=None)

    shapers = [
        RewardShaperClass(
            max_steps=args.max_steps,
            reward_scale=args.reward_scale,
            approach_scale=args.approach_scale,
            push_scale=args.push_scale,
            stuck_penalty=args.stuck_penalty,
            explore_scale=args.explore_scale,
            efficiency_bonus=args.efficiency_bonus,
            forward_bonus=args.forward_bonus,
            spin_penalty=args.spin_penalty,
            spin_threshold=args.spin_threshold,
            boundary_penalty=args.boundary_penalty,
            boundary_threshold=args.boundary_threshold,
            track_scale=args.track_scale,
            track_memory_steps=args.track_memory_steps,
            ir_bonus=args.ir_bonus,
            recovery_bonus=args.recovery_bonus,
            oscillation_penalty=args.oscillation_penalty,
            stability_bonus=args.stability_bonus,
            persistence_penalty=args.persistence_penalty,
            persistence_threshold=args.persistence_threshold,
            push_forward_bonus=args.push_forward_bonus,
            blind_push_penalty=args.blind_push_penalty,
        )
        for _ in range(args.n_envs)
    ]

    net = ActorCriticLSTM(
        input_dim=POLICY_INPUT_DIM,
        hidden_dim=args.hidden,
        n_actions=N_ACT,
    ).to(device)

    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    total_updates = max(
        (args.episodes * args.max_steps) // (args.rollout_len * args.n_envs), 1
    )
    scheduler = optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.1, total_iters=total_updates
    )

    buf = RolloutBuffer(
        rollout_len=args.rollout_len,
        n_envs=args.n_envs,
        input_dim=POLICY_INPUT_DIM,
    )

    h, c = net.init_hidden(args.n_envs, device)

    best_return = -float("inf")
    best_success_rate = -1.0
    episodes_done = 0
    total_steps = 0
    update_count = 0
    success_count = 0
    train_start = time.time()

    window_returns: List[float] = []
    window_steps: List[int] = []
    recent_successes = collections.deque(maxlen=200)
    recent_metrics = collections.defaultdict(lambda: collections.deque(maxlen=20))

    # Reset envs
    init_seeds = [args.seed + i for i in range(args.n_envs)]
    obs_arr = np.array(vec.reset(seeds=init_seeds), dtype=np.float32)

    for sh in shapers:
        sh.reset()

    ep_ret = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps = np.zeros(args.n_envs, dtype=np.int32)
    last_done = np.zeros(args.n_envs, dtype=bool)
    push_active = np.zeros(args.n_envs, dtype=bool)

    # For POMDP input
    prev_action_oh = np.zeros((args.n_envs, N_ACT), dtype=np.float32)
    prev_reward = np.zeros(args.n_envs, dtype=np.float32)

    LOG_EVERY = 20
    last_log_ep = 0

    print(
        f"\n[Robust PPO-LSTM] envs={args.n_envs}  rollout={args.rollout_len}  "
        f"hidden={args.hidden}  input_dim={POLICY_INPUT_DIM}"
    )
    print(
        f"[Robust PPO-LSTM] gamma={args.gamma}  gae={args.gae_lam}  "
        f"clip={args.clip_eps}  ent={args.ent_coef}  lr={args.lr}"
    )
    print(
        f"[Robust PPO-LSTM] reward_shaping={'OFF' if args.no_shaping else 'ON'}  "
        f"difficulty={args.difficulty}  wall={args.wall_obstacles}\n"
    )

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=120)

    while episodes_done < args.episodes:
        buf.clear()
        buf.store_init_hidden(h, c)

        for _ in range(args.rollout_len):
            x_arr = build_policy_input(obs_arr, prev_action_oh, prev_reward)
            x_t = torch.tensor(x_arr, dtype=torch.float32, device=device)

            with torch.no_grad():
                actions_t, logp_t, _, values_t, (h, c) = net.get_action(x_t, (h, c))

            action_idx = actions_t.cpu().numpy()
            action_strs = [ACTIONS[a] for a in action_idx]

            results = vec.step(action_strs)
            next_obs_arr = np.array([r[0] for r in results], dtype=np.float32)
            raw_rewards = np.array([r[1] for r in results], dtype=np.float32)
            dones = np.array([r[2] for r in results], dtype=bool)

            shaped_rewards = np.empty(args.n_envs, dtype=np.float32)

            for i in range(args.n_envs):
                scaled = float(raw_rewards[i]) / args.reward_scale

                if args.no_shaping:
                    shaped_rewards[i] = scaled
                else:
                    if next_obs_arr[i][16]:
                        push_active[i] = True

                    shaped_rewards[i] = shapers[i].shape(
                        raw_reward=scaled,
                        obs=next_obs_arr[i],
                        done=bool(dones[i]),
                        enable_push=bool(push_active[i]),
                        action=action_strs[i],
                    )

            buf.add_batch(
                x=x_arr,
                actions=action_idx,
                log_probs=logp_t.cpu().numpy(),
                rewards=shaped_rewards,
                values=values_t.cpu().numpy(),
                dones=dones.astype(np.float32),
            )

            ep_ret += shaped_rewards
            ep_steps += 1
            total_steps += args.n_envs
            last_done[:] = dones

            # Prepare next-step prev_action / prev_reward
            next_prev_action_oh = one_hot_actions(action_idx)
            next_prev_reward = shaped_rewards.copy()

            for i in range(args.n_envs):
                if not dones[i]:
                    continue

                success = bool(raw_rewards[i] >= 100.0)
                success_count += int(success)
                recent_successes.append(int(success))

                window_returns.append(float(ep_ret[i]))
                window_steps.append(int(ep_steps[i]))

                episodes_done += 1
                pbar.update(1)
                pbar.set_postfix({
                    "ret": f"{ep_ret[i]:.1f}",
                    "best": f"{best_return:.1f}",
                    "succ": success_count,
                })

                # save best by episodic return
                if ep_ret[i] > best_return:
                    best_return = float(ep_ret[i])
                    torch.save(net.cpu().state_dict(), args.out + ".best_return")
                    net.to(device)

                new_seed = args.seed + args.n_envs + episodes_done
                reset_obs = vec.reset_one(i, seed=new_seed)
                next_obs_arr[i] = np.array(reset_obs, dtype=np.float32)

                shapers[i].reset()
                push_active[i] = False

                ep_ret[i] = 0.0
                ep_steps[i] = 0

                # zero recurrent state
                h[:, i, :] = 0.0
                c[:, i, :] = 0.0

                # zero prev action/reward after reset
                next_prev_action_oh[i, :] = 0.0
                next_prev_reward[i] = 0.0

                if episodes_done >= args.episodes:
                    break

            obs_arr = next_obs_arr
            prev_action_oh = next_prev_action_oh
            prev_reward = next_prev_reward

            if episodes_done >= args.episodes:
                break

        with torch.no_grad():
            x_arr = build_policy_input(obs_arr, prev_action_oh, prev_reward)
            x_t = torch.tensor(x_arr, dtype=torch.float32, device=device)
            _, last_v, _ = net(x_t, (h, c))
            last_vals = last_v.cpu().numpy()

        last_vals = np.where(last_done, 0.0, last_vals)

        metrics = ppo_update(
            net=net,
            opt=opt,
            buf=buf,
            last_values=last_vals,
            device=device,
            gamma=args.gamma,
            gae_lam=args.gae_lam,
            clip_eps=args.clip_eps,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            n_epochs=args.n_epochs,
            n_mini_batches=args.n_mini_batches,
            max_grad=args.max_grad,
            target_kl=args.target_kl,
        )

        scheduler.step()
        update_count += 1

        for k, v in metrics.items():
            recent_metrics[k].append(v)

        # save best by rolling success rate
        if len(recent_successes) > 0:
            rolling_success = 100.0 * np.mean(recent_successes)
            if rolling_success > best_success_rate:
                best_success_rate = rolling_success
                torch.save(net.cpu().state_dict(), args.out + ".best_success")
                net.to(device)

        if (episodes_done // LOG_EVERY) > (last_log_ep // LOG_EVERY) and len(window_returns) > 0:
            elapsed = time.time() - train_start
            rolling_success = 100.0 * np.mean(recent_successes) if len(recent_successes) > 0 else 0.0

            tqdm.write(
                f"\n┌─ ep {episodes_done - len(window_returns) + 1:>4d}–{episodes_done:<4d} "
                f"({elapsed:.0f}s) ─────────────────────────────────────────────\n"
                f"│ Avg Return    : {np.mean(window_returns):8.2f}   "
                f"Best Return : {best_return:8.2f}\n"
                f"│ Avg Steps     : {np.mean(window_steps):8.1f}   "
                f"Rolling Success(200 ep): {rolling_success:6.2f}%\n"
                f"│ Policy Loss   : {np.mean(recent_metrics['policy_loss']):8.4f}   "
                f"Value Loss : {np.mean(recent_metrics['value_loss']):8.4f}\n"
                f"│ Entropy       : {np.mean(recent_metrics['entropy']):8.4f}   "
                f"KL : {np.mean(recent_metrics['approx_kl']):8.4f}   "
                f"ClipFrac : {np.mean(recent_metrics['clip_frac']):6.3f}\n"
                f"│ LR            : {scheduler.get_last_lr()[0]:.2e}   "
                f"Updates : {update_count}   Total steps : {total_steps:,}\n"
                f"└{'─'*84}"
            )

            last_log_ep = episodes_done
            window_returns.clear()
            window_steps.clear()

    pbar.close()
    vec.close()

    torch.save(net.cpu().state_dict(), args.out)

    elapsed = time.time() - train_start
    final_success = 100.0 * success_count / max(1, episodes_done)

    print(f"\nSaved final model        : {args.out}")
    print(f"Saved best return model  : {args.out}.best_return")
    print(f"Saved best success model : {args.out}.best_success")
    print(f"Time                     : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(
        f"Stats                    : {episodes_done} episodes | "
        f"{total_steps:,} steps | {success_count} successes "
        f"({final_success:.2f}%)"
    )


if __name__ == "__main__":
    main()