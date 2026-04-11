"""PPO trainer for OBELIX using agent_template-style features plus GRU memory.

This keeps the strongest ideas from agent_template.py:
    - 24-d engineered observation features
    - explicit FSM phase prior (find/push/unwedge)
    - FSM suggested action prior

Then adds learned temporal memory through a GRU policy/value network and trains
it end-to-end with PPO over vectorized environments.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import importlib.util
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from vec_env import VecEnv


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
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
PHASES = ["find", "push", "unwedge"]
FSM_BIAS = 2.0
N_ACTIONS = len(ACTIONS)
FEAT_DIM = 24 + len(PHASES) + len(ACTIONS)


def _phase_onehot(phase: str) -> np.ndarray:
    v = np.zeros(len(PHASES), dtype=np.float32)
    v[PHASES.index(phase)] = 1.0
    return v


def _action_onehot(action: str) -> np.ndarray:
    v = np.zeros(len(ACTIONS), dtype=np.float32)
    v[ACTION_IDX[action]] = 1.0
    return v


def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


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


def parse_obs(obs):
    obs = np.asarray(obs, dtype=int)
    lf = int(obs[0] or obs[2])
    ln = int(obs[1] or obs[3])
    ff = int(obs[4] or obs[6] or obs[8] or obs[10])
    fn = int(obs[5] or obs[7] or obs[9] or obs[11])
    rf = int(obs[12] or obs[14])
    rn = int(obs[13] or obs[15])
    ir = int(obs[16])
    stk = int(obs[17])
    any_s = int(any(obs[:17]))
    return lf, ln, ff, fn, rf, rn, ir, stk, any_s


def featurize(obs) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    lf, ln, ff, fn, rf, rn, ir, stk, any_s = parse_obs(obs)
    return np.array([
        lf, ln, ff, fn, rf, rn, ir, stk,
        float(lf or ln),
        float(rf or rn),
        float(ff or fn or ir),
        float(ln or fn or rn or ir),
        float(lf or ff or rf),
        float(ln and not rn),
        float(rn and not ln),
        float(fn and not (ln or rn)),
        float(lf and not rf),
        float(rf and not lf),
        float(ir * 5 + fn * 3 + ff * 2 + (ln or rn) + (lf or rf) * 0.5) / 5.0,
        float(not any_s),
        float(ff or fn or ir),
        float((lf + ln) - (rf + rn)) / 2.0,
        float(stk),
        float(ln or fn or rn or ir),
    ], dtype=np.float32)


@dataclasses.dataclass
class FSMState:
    phase: str = "find"
    steps_in_phase: int = 0
    steps_no_sensor: int = 0
    spin_dir: str = "L45"
    push_stuck_cnt: int = 0
    find_stuck_cnt: int = 0
    episode_step: int = 0

    def reset(self):
        self.phase = "find"
        self.steps_in_phase = 0
        self.steps_no_sensor = 0
        self.spin_dir = random.choice(["L45", "R45"])
        self.push_stuck_cnt = 0
        self.find_stuck_cnt = 0
        self.episode_step = 0

    def suggest(self, obs) -> str:
        phase = self.phase
        lf, ln, ff, fn, rf, rn, ir, stk, _ = parse_obs(obs)

        if phase == "find":
            if stk:
                self.find_stuck_cnt += 1
                return "L45" if self.find_stuck_cnt % 2 == 0 else "R45"
            self.find_stuck_cnt = 0
            if ir:
                return "FW"
            if fn:
                return "FW"
            if ff:
                if rn and not ln:
                    return "L22"
                if ln and not rn:
                    return "R22"
                return "FW"
            if ln and not rn:
                return "L22"
            if rn and not ln:
                return "R22"
            if lf and not rf:
                return "L22"
            if rf and not lf:
                return "R22"
            if lf and rf:
                return self.spin_dir
            self.steps_no_sensor += 1
            return self.spin_dir if (self.steps_no_sensor % 16) < 6 else "FW"

        if phase == "push":
            if stk:
                self.push_stuck_cnt += 1
                c = self.push_stuck_cnt
                if c <= 2:
                    return "L22"
                if c <= 4:
                    return "R22"
                if c <= 6:
                    return "L45"
                self.push_stuck_cnt = 0
                return "R45"
            self.push_stuck_cnt = 0
            return "FW"

        if stk:
            return "L45" if (self.steps_in_phase % 4) < 2 else "R45"
        return "FW"

    def update(self, obs, action: str):
        _, _, _, fn, _, _, ir, stk, _ = parse_obs(obs)
        self.steps_in_phase += 1
        self.episode_step += 1

        if self.phase == "find":
            if (ir and action == "FW") or (fn and self.steps_in_phase > 3 and action == "FW"):
                self.phase = "push"
                self.steps_in_phase = 0
                self.push_stuck_cnt = 0
        elif self.phase == "push":
            if stk:
                self.phase = "unwedge"
                self.steps_in_phase = 0
        elif self.phase == "unwedge":
            if not stk:
                self.phase = "push"
                self.steps_in_phase = 0


def build_input(obs, fsm: FSMState) -> np.ndarray:
    fsm_act = fsm.suggest(obs)
    vec = np.concatenate(
        [featurize(obs), _phase_onehot(fsm.phase), _action_onehot(fsm_act)]
    ).astype(np.float32)
    return vec


class GRUActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int, gru_hidden: int):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=False)
        self.actor = nn.Linear(gru_hidden, N_ACTIONS)
        self.critic = nn.Linear(gru_hidden, 1)

        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(p, gain=np.sqrt(2))
            elif "weight_hh" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def zero_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def forward_step(self, x: torch.Tensor, h: torch.Tensor):
        enc = self.encoder(x).unsqueeze(0)
        out, h_new = self.gru(enc, h)
        out = out.squeeze(0)
        logits = self.actor(out)
        values = self.critic(out).squeeze(-1)
        return logits, values, h_new

    def forward_sequence(self, x_seq: torch.Tensor, starts: torch.Tensor, h0: torch.Tensor):
        # x_seq: [T, B, D], starts: [T, B] bool indicating reset before step t
        T, B, _ = x_seq.shape
        h = h0
        logits_all = []
        values_all = []
        for t in range(T):
            start_mask = starts[t].view(1, B, 1)
            h = h * (~start_mask)
            logits_t, values_t, h = self.forward_step(x_seq[t], h)
            logits_all.append(logits_t)
            values_all.append(values_t)
        return torch.stack(logits_all, dim=0), torch.stack(values_all, dim=0), h


class RolloutBuffer:
    def __init__(self, rollout_len: int, n_envs: int, in_dim: int, gamma: float, gae_lam: float):
        self.rollout_len = rollout_len
        self.n_envs = n_envs
        self.in_dim = in_dim
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.clear()

    def clear(self):
        T, N, D = self.rollout_len, self.n_envs, self.in_dim
        self.obs = np.zeros((T, N, D), dtype=np.float32)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.starts = np.zeros((T, N), dtype=bool)
        self.ptr = 0

    def add(self, obs, actions, log_probs, rewards, values, dones, starts):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones
        self.starts[t] = starts
        self.ptr += 1

    def compute_advantages(self, last_values: np.ndarray):
        T = self.ptr
        adv = np.zeros((T, self.n_envs), dtype=np.float32)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        next_values = last_values.astype(np.float32)
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_values * nonterminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lam * nonterminal * last_gae
            adv[t] = last_gae
            next_values = self.values[t]
        returns = adv + self.values[:T]
        return adv, returns


def ppo_update(
    net: GRUActorCritic,
    opt: optim.Optimizer,
    buf: RolloutBuffer,
    last_values: np.ndarray,
    h0_all: torch.Tensor,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    n_epochs: int,
    mini_batch_envs: int,
    max_grad: float,
    device: torch.device,
):
    T, N = buf.ptr, buf.n_envs
    adv, returns = buf.compute_advantages(last_values)
    valid_adv = adv.reshape(-1)
    adv = (adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)

    obs = torch.tensor(buf.obs[:T], dtype=torch.float32, device=device)
    actions = torch.tensor(buf.actions[:T], dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buf.log_probs[:T], dtype=torch.float32, device=device)
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    starts = torch.tensor(buf.starts[:T], dtype=torch.bool, device=device)

    metrics = collections.defaultdict(list)

    for _ in range(n_epochs):
        env_perm = torch.randperm(N, device=device)
        for start in range(0, N, mini_batch_envs):
            env_idx = env_perm[start : start + mini_batch_envs]
            obs_mb = obs[:, env_idx]
            act_mb = actions[:, env_idx]
            old_lp_mb = old_log_probs[:, env_idx]
            adv_mb = adv_t[:, env_idx]
            ret_mb = returns_t[:, env_idx]
            starts_mb = starts[:, env_idx]

            h0 = h0_all[:, env_idx].detach()
            logits, values, _ = net.forward_sequence(obs_mb, starts_mb, h0)
            dist = Categorical(logits=logits.reshape(-1, N_ACTIONS))
            new_log_probs = dist.log_prob(act_mb.reshape(-1)).view(T, -1)
            entropy = dist.entropy().view(T, -1)
            values = values.view(T, -1)

            ratio = torch.exp(new_log_probs - old_lp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, ret_mb)
            entropy_loss = -entropy.mean()
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad)
            opt.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
            metrics["approx_kl"].append(approx_kl)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def save_bundle(path: str, net: GRUActorCritic, optimizer: optim.Optimizer | None, args):
    payload = {
        "model_state_dict": net.state_dict(),
        "config": {
            "in_dim": FEAT_DIM,
            "hidden": args.hidden,
            "gru_hidden": args.gru_hidden,
            "fsm_bias": args.fsm_bias,
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser(description="PPO with agent_template features + GRU memory")
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights_template_memory.pth")
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae_lam", type=float, default=0.97)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.005)
    ap.add_argument("--rollout_len", type=int, default=256)
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--mini_batch_envs", type=int, default=4)
    ap.add_argument("--max_grad", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gru_hidden", type=int, default=128)
    ap.add_argument("--reward_scale", type=float, default=20.0)
    ap.add_argument("--fsm_bias", type=float, default=FSM_BIAS)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    make_fns = [make_env_fn(OBELIX, args, args.seed + i) for i in range(args.n_envs)]
    vec = VecEnv(make_fns=make_fns, reward_shaping_fn=None)

    net = GRUActorCritic(FEAT_DIM, hidden=args.hidden, gru_hidden=args.gru_hidden).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[resume] loaded {args.resume}")
    elif args.load:
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"[warm-start] loaded {args.load}")

    fsm_states = [FSMState() for _ in range(args.n_envs)]
    for fsm in fsm_states:
        fsm.reset()

    seeds = [args.seed + i for i in range(args.n_envs)]
    raw_obs = vec.reset(seeds=seeds)
    obs_inputs = np.asarray([build_input(raw_obs[i], fsm_states[i]) for i in range(args.n_envs)], dtype=np.float32)
    hidden = net.zero_hidden(args.n_envs, device)
    episode_starts = np.ones(args.n_envs, dtype=bool)

    ep_returns = np.zeros(args.n_envs, dtype=np.float32)
    ep_steps = np.zeros(args.n_envs, dtype=np.int32)
    window_returns: List[float] = []
    window_steps: List[int] = []
    success_window: collections.deque = collections.deque(maxlen=200)
    episodes_done = 0
    total_steps = 0
    updates = 0
    best_return = -float("inf")
    train_start = time.time()

    print(
        f"\n[PPO Template Memory] envs={args.n_envs} obs_dim={FEAT_DIM} "
        f"hidden={args.hidden} gru={args.gru_hidden} rollout={args.rollout_len}"
    )
    print(
        f"[PPO Template Memory] difficulty={args.difficulty} wall={args.wall_obstacles} "
        f"reward_scale={args.reward_scale} fsm_bias={args.fsm_bias}\n"
    )

    pbar = tqdm(total=args.episodes, desc="Training", unit="ep", ncols=120)

    while episodes_done < args.episodes:
        buf = RolloutBuffer(args.rollout_len, args.n_envs, FEAT_DIM, args.gamma, args.gae_lam)
        rollout_start_hidden = hidden.clone()

        for _ in range(args.rollout_len):
            obs_t = torch.tensor(obs_inputs, dtype=torch.float32, device=device)
            with torch.no_grad():
                start_mask = torch.tensor(episode_starts, dtype=torch.bool, device=device).view(1, args.n_envs, 1)
                hidden = hidden * (~start_mask)
                logits, values, hidden_new = net.forward_step(obs_t, hidden)
                # FSM prior as a soft policy bias during training-time action sampling
                bias = torch.zeros_like(logits)
                for i in range(args.n_envs):
                    act_idx = int(np.argmax(obs_inputs[i][-len(ACTIONS):]))
                    bias[i, act_idx] += args.fsm_bias
                dist = Categorical(logits=logits + bias)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            action_strs = [ACTIONS[a] for a in actions.cpu().numpy()]
            results = vec.step(action_strs)
            next_raw_obs = [r[0] for r in results]
            raw_rewards = np.asarray([r[1] for r in results], dtype=np.float32)
            dones = np.asarray([r[2] for r in results], dtype=bool)
            rewards = raw_rewards / args.reward_scale

            buf.add(
                obs=obs_inputs,
                actions=actions.cpu().numpy(),
                log_probs=log_probs.cpu().numpy(),
                rewards=rewards,
                values=values.cpu().numpy(),
                dones=dones.astype(np.float32),
                starts=episode_starts.copy(),
            )

            ep_returns += raw_rewards
            ep_steps += 1
            total_steps += args.n_envs

            next_episode_starts = np.zeros(args.n_envs, dtype=bool)
            for i in range(args.n_envs):
                fsm_states[i].update(raw_obs[i], action_strs[i])
                if dones[i]:
                    terminal_reward = raw_rewards[i]
                    window_returns.append(float(ep_returns[i]))
                    window_steps.append(int(ep_steps[i]))
                    success_window.append(float(terminal_reward >= 1000.0))
                    best_return = max(best_return, float(ep_returns[i]))
                    episodes_done += 1
                    pbar.update(1)

                    new_obs = vec.reset_one(i, args.seed + args.n_envs + episodes_done + i)
                    raw_obs[i] = new_obs
                    fsm_states[i].reset()
                    hidden[:, i : i + 1] = 0.0
                    ep_returns[i] = 0.0
                    ep_steps[i] = 0
                    next_episode_starts[i] = True
                else:
                    raw_obs[i] = next_raw_obs[i]
            obs_inputs = np.asarray([build_input(raw_obs[i], fsm_states[i]) for i in range(args.n_envs)], dtype=np.float32)
            hidden = hidden_new
            episode_starts = next_episode_starts

            if episodes_done >= args.episodes:
                break

        with torch.no_grad():
            obs_last = torch.tensor(obs_inputs, dtype=torch.float32, device=device)
            start_mask = torch.tensor(episode_starts, dtype=torch.bool, device=device).view(1, args.n_envs, 1)
            hidden_eval = hidden * (~start_mask)
            _, last_values_t, _ = net.forward_step(obs_last, hidden_eval)
            last_values = last_values_t.cpu().numpy() * (1.0 - episode_starts.astype(np.float32))

        metrics = ppo_update(
            net=net,
            opt=opt,
            buf=buf,
            last_values=last_values,
            h0_all=rollout_start_hidden,
            clip_eps=args.clip_eps,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            n_epochs=args.n_epochs,
            mini_batch_envs=args.mini_batch_envs,
            max_grad=args.max_grad,
            device=device,
        )
        updates += 1

        if episodes_done > 0 and (episodes_done % args.log_every == 0):
            win = min(args.log_every, len(window_returns))
            avg_ret = float(np.mean(window_returns[-win:])) if win else 0.0
            avg_steps = float(np.mean(window_steps[-win:])) if win else 0.0
            succ = 100.0 * float(np.mean(success_window)) if success_window else 0.0
            speed = episodes_done / max(time.time() - train_start, 1e-6)
            print(
                f"\n[ep {max(1, episodes_done-win+1)}-{episodes_done}] "
                f"avg_return={avg_ret:.2f} best={best_return:.2f} "
                f"avg_steps={avg_steps:.1f} rolling_success(200)={succ:.2f}%"
            )
            print(
                f"policy={metrics.get('policy_loss', 0.0):.4f} "
                f"value={metrics.get('value_loss', 0.0):.4f} "
                f"entropy={metrics.get('entropy', 0.0):.4f} "
                f"kl={metrics.get('approx_kl', 0.0):.4f} "
                f"updates={updates} total_steps={total_steps} speed={speed:.2f} ep/s"
            )
            pbar.set_postfix(
                ret=f"{avg_ret:.1f}",
                best=f"{best_return:.1f}",
                succ=f"{succ:.0f}",
            )

    vec.close()
    save_bundle(args.out, net, opt, args)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
