"""train_ppo_gru_bc.py — PPO + GRU fine-tuning warm-started from BC-Transformer.

Pipeline
--------
1.  collect_fsm_dataset.py → dataset_fsm.pkl    (FSM expert trajectories)
2.  bc_transformer.py      → weights_bc.pth     (BC pre-training)
3.  THIS SCRIPT            → weights_ppo_bc.pth (RL fine-tuning)

Architecture
------------
UnifiedGRUActorCritic  (single agent — no FSM phase switching):
  encoder   : Linear(39 → 128) + Tanh    ← weights from BCTransformer.embedding
  gru       : GRU(128 → 128)             ← random init  (new temporal module)
  actor     : Linear(128 → 5)            ← weights from BCTransformer.actor_head
  critic    : Linear(128 → 1)            ← random init

Weight transfer rationale
--------------------------
The BC-Transformer learned a useful linear feature map from 39-dim EPB input
to 128-dim features and a good action prior (actor_head) via supervised
imitation of the FSM expert.  The GRU is initialized randomly — it must learn
temporal patterns from RL interaction. The critic is always random (it has no
BC analogue).  The encoder + actor warm-start means the policy starts close to
the FSM prior and is refined via PPO rather than trained from scratch.

Training
--------
- Single-env loop (no VecEnv) so CompactBeliefState state is easy to maintain.
- GRURolloutBuffer stores the hidden state h at each step (truncated BPTT).
- No reward shaping: raw environment reward only (lesson learned from Phase 2).
- Curriculum over arena sizes: 300 → 500 px.

Usage:
  python train_ppo_gru_bc.py --obelix_py obelix.py --bc_weights weights_bc.pth \\
      --out weights_ppo_bc.pth
  python train_ppo_gru_bc.py --obelix_py obelix.py --bc_weights weights_bc.pth \\
      --out weights_ppo_bc.pth --episodes 5000 --no_curriculum
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.distributions import Categorical

from compact_belief import ACTIONS, IN_DIM, CompactBeliefState, build_input

N_ACTIONS = len(ACTIONS)   # 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ── Curriculum ─────────────────────────────────────────────────────────────────
# Each entry: (arena_size, difficulty, wall_obstacles, max_steps, n_episodes)
CURRICULUM = [
    dict(arena_size=300, difficulty=0, wall_obstacles=False, max_steps=600,  episodes=600),
    dict(arena_size=400, difficulty=0, wall_obstacles=False, max_steps=800,  episodes=700),
    dict(arena_size=500, difficulty=0, wall_obstacles=False, max_steps=1000, episodes=800),
    dict(arena_size=500, difficulty=0, wall_obstacles=True,  max_steps=1000, episodes=800),
    dict(arena_size=500, difficulty=2, wall_obstacles=False, max_steps=1000, episodes=600),
    dict(arena_size=500, difficulty=2, wall_obstacles=True,  max_steps=1000, episodes=500),
]


# ── Unified GRU Actor-Critic ───────────────────────────────────────────────────

GRU_HIDDEN  = 128   # must match BCTransformer d_model for weight transfer
GRU_CHUNK   = 32    # truncated-BPTT chunk length during PPO update


class UnifiedGRUActorCritic(nn.Module):
    """Single GRU agent that handles find / push / unwedge implicitly.

    No explicit phase switching — the GRU hidden state encodes phase context.
    Input: 39-dim EPB vector [raw_obs(18) | belief(16) | fsm_onehot(5)].

    Weight transfer from BCTransformer:
      encoder[0]  ←  bc.embedding   (Linear 39→128)
      actor       ←  bc.actor_head  (Linear 128→5)
    The GRU and critic are always randomly initialised.
    """

    def __init__(self, obs_dim: int = IN_DIM, n_actions: int = N_ACTIONS,
                 enc_dim: int = GRU_HIDDEN, gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.gru_hidden = gru_hidden

        # encoder[0] receives the BC embedding weights directly
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, enc_dim),
            nn.Tanh(),
        )
        self.gru    = nn.GRU(enc_dim, gru_hidden, batch_first=True)
        self.actor  = nn.Linear(gru_hidden, n_actions)
        self.critic = nn.Linear(gru_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(p, gain=math.sqrt(2))
            elif "weight_hh" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def zero_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.gru_hidden, device=device)

    def forward_step(self, obs: torch.Tensor, h: torch.Tensor):
        """Single-step inference during rollout.
        obs : (1, obs_dim)
        h   : (1, 1, gru_hidden)
        Returns  logits (1, n_actions),  value (1,),  h_new (1, 1, gru_hidden)
        """
        enc        = self.encoder(obs).unsqueeze(1)         # (1, 1, enc_dim)
        out, h_new = self.gru(enc, h.to(enc.dtype))         # (1, 1, gru_hidden)
        out        = out.squeeze(1)                          # (1, gru_hidden)
        return self.actor(out), self.critic(out).squeeze(-1), h_new

    def forward_sequence(self, obs_seq: torch.Tensor, h0: torch.Tensor):
        """Sequence forward for truncated BPTT during PPO update.
        obs_seq : (B, T, obs_dim)
        h0      : (1, B, gru_hidden)
        Returns  logits (B*T, n_actions),  values (B*T,)
        """
        B, T, _ = obs_seq.shape
        enc = self.encoder(obs_seq.view(B * T, -1)).view(B, T, -1)
        out, _ = self.gru(enc, h0.to(enc.dtype))            # (B, T, gru_hidden)
        out = out.contiguous().view(B * T, -1)
        return self.actor(out), self.critic(out).squeeze(-1)


# ── Weight transfer ────────────────────────────────────────────────────────────

def load_bc_weights(net: UnifiedGRUActorCritic, bc_path: str) -> None:
    """Transfer embedding and actor_head from BCTransformer checkpoint.

    Transferred layers
    ------------------
    bc.embedding.weight/bias  →  net.encoder[0].weight/bias   (Linear 39→128)
    bc.actor_head.weight/bias →  net.actor.weight/bias        (Linear 128→5)

    NOT transferred: GRU weights, critic — these must be learned from scratch
    via RL because BC has no temporal recurrence and no value target.
    """
    ck = torch.load(bc_path, map_location=device, weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck

    def _copy(src_key: str, dst_param: nn.Parameter, name: str):
        if src_key in sd:
            assert sd[src_key].shape == dst_param.shape, (
                f"Shape mismatch {name}: BC={sd[src_key].shape} PPO={dst_param.shape}"
            )
            dst_param.data.copy_(sd[src_key])
            print(f"  [transfer] {src_key} → {name}  shape={tuple(dst_param.shape)}")
        else:
            print(f"  [transfer] WARNING: {src_key} not found in BC checkpoint — skipped")

    _copy("embedding.weight", net.encoder[0].weight, "encoder[0].weight")
    _copy("embedding.bias",   net.encoder[0].bias,   "encoder[0].bias")
    _copy("actor_head.weight", net.actor.weight,      "actor.weight")
    _copy("actor_head.bias",   net.actor.bias,        "actor.bias")
    print(f"[bc_init] transfer complete  (GRU + critic remain randomly initialised)")


# ── GRU Rollout Buffer ─────────────────────────────────────────────────────────

class GRURolloutBuffer:
    """Stores one horizon of transitions with per-step GRU hidden states.

    Hidden states h are stored so the PPO update can replay exact initial
    conditions for each truncated-BPTT chunk without re-simulating episodes.
    """

    def __init__(self, horizon: int, obs_dim: int, gru_hidden: int,
                 gamma: float, lam: float, chunk_len: int = GRU_CHUNK):
        self.horizon    = horizon
        self.obs_dim    = obs_dim
        self.gru_hidden = gru_hidden
        self.gamma      = gamma
        self.lam        = lam
        self.chunk_len  = chunk_len
        self._pin       = device.type == "cuda"
        self.reset()

    def reset(self):
        self.obs      = np.zeros((self.horizon, self.obs_dim),    dtype=np.float32)
        self.actions  = np.zeros(self.horizon,                    dtype=np.int64)
        self.rewards  = np.zeros(self.horizon,                    dtype=np.float32)
        self.dones    = np.zeros(self.horizon,                    dtype=np.float32)
        self.logprobs = np.zeros(self.horizon,                    dtype=np.float32)
        self.values   = np.zeros(self.horizon,                    dtype=np.float32)
        self.hiddens  = np.zeros((self.horizon, self.gru_hidden), dtype=np.float32)
        self.ptr      = 0

    def add(self, obs, action, reward, done, logprob, value, h: torch.Tensor):
        if self.ptr >= self.horizon:
            return
        i = self.ptr
        self.obs[i]      = obs
        self.actions[i]  = action
        self.rewards[i]  = reward
        self.dones[i]    = float(done)
        self.logprobs[i] = logprob
        self.values[i]   = value
        self.hiddens[i]  = h.squeeze().cpu().numpy()
        self.ptr        += 1

    def _compute_gae(self, last_value: float):
        adv      = np.zeros(self.ptr, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            nv       = last_value if t == self.ptr - 1 else self.values[t + 1]
            delta    = self.rewards[t] + self.gamma * nv * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * last_gae
            adv[t]   = last_gae
        return adv, adv + self.values[:self.ptr]

    def get_chunks(self, last_value: float):
        """Yield sequential chunks for truncated BPTT.
        Each chunk: (obs (1,T,D), actions (T,), old_logprobs (T,),
                     advantages (T,), returns (T,), h0 (1,1,H))
        """
        if self.ptr < 2:
            return
        adv, ret = self._compute_gae(last_value)
        adv      = (adv - adv.mean()) / (adv.std() + 1e-8)

        def _t(arr, dtype=torch.float32):
            t = torch.from_numpy(arr)
            return (t.pin_memory() if self._pin else t).to(device, non_blocking=True).to(dtype)

        for start in range(0, self.ptr, self.chunk_len):
            end = min(start + self.chunk_len, self.ptr)
            sl  = slice(start, end)
            yield (
                _t(self.obs[sl]).unsqueeze(0),                  # (1, T, obs_dim)
                _t(self.actions[sl], dtype=torch.int64),        # (T,)
                _t(self.logprobs[sl]),                          # (T,)
                _t(adv[sl]),                                    # (T,)
                _t(ret[sl]),                                    # (T,)
                _t(self.hiddens[start]).view(1, 1, -1),         # (1, 1, gru_h)
            )


# ── PPO update ─────────────────────────────────────────────────────────────────

def ppo_update(net: UnifiedGRUActorCritic, opt, scaler,
               buf: GRURolloutBuffer, last_value: float,
               epochs: int, clip_eps: float, vf_coef: float,
               ent_coef: float, max_grad_norm: float, use_amp: bool):
    """Truncated-BPTT PPO update over the collected rollout."""
    chunks = list(buf.get_chunks(last_value))
    if not chunks:
        return 0.0, 0.0, 0.0

    total_pg = total_vf = total_ent = n = 0.0

    for _ in range(epochs):
        random.shuffle(chunks)
        for obs_c, act_c, old_lp_c, adv_c, ret_c, h0_c in chunks:
            with autocast(device_type=device.type, enabled=use_amp):
                logits, values = net.forward_sequence(obs_c, h0_c)
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(act_c)
                entropy = dist.entropy()

                ratio   = torch.exp(new_lp - old_lp_c)
                pg_loss = torch.max(
                    -adv_c * ratio,
                    -adv_c * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps),
                ).mean()
                vf_loss  = 0.5 * (values - ret_c).pow(2).mean()
                ent_loss = -entropy.mean()
                loss     = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()

            total_pg  += pg_loss.item()
            total_vf  += vf_loss.item()
            total_ent += (-ent_loss.item())
            n         += 1

    return (total_pg / n, total_vf / n, total_ent / n) if n > 0 else (0.0, 0.0, 0.0)


# ── Helpers ────────────────────────────────────────────────────────────────────

def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def _raw(net):
    return net._orig_mod if hasattr(net, "_orig_mod") else net


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    # Environment
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--bc_weights",     type=str,   default=None,
                    help="Path to weights_bc.pth from bc_transformer.py  "
                         "(omit for random init)")
    ap.add_argument("--episodes",       type=int,   default=5000)
    ap.add_argument("--max_steps",      type=int,   default=1000)
    ap.add_argument("--difficulty",     type=int,   default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",      type=int,   default=2)
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--arena_size",     type=int,   default=500)
    ap.add_argument("--no_curriculum",  action="store_true")
    # PPO
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lam",            type=float, default=0.95)
    ap.add_argument("--lr",             type=float, default=3e-4)
    ap.add_argument("--horizon",        type=int,   default=2048)
    ap.add_argument("--epochs",         type=int,   default=8)
    ap.add_argument("--clip_eps",       type=float, default=0.2)
    ap.add_argument("--vf_coef",        type=float, default=0.5)
    ap.add_argument("--ent_coef",       type=float, default=0.01)
    ap.add_argument("--max_grad_norm",  type=float, default=0.5)
    # Misc
    ap.add_argument("--out",            type=str,   default="weights_ppo_bc.pth")
    ap.add_argument("--resume",         type=str,   default=None)
    ap.add_argument("--seed",           type=int,   default=0)
    ap.add_argument("--no_amp",         action="store_true")
    ap.add_argument("--no_compile",     action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX  = import_obelix(args.obelix_py)
    use_amp = (device.type == "cuda") and (not args.no_amp)

    # ── Build network ─────────────────────────────────────────────────────────
    net = UnifiedGRUActorCritic().to(device)

    if args.bc_weights and not args.resume:
        print(f"[init] loading BC weights from {args.bc_weights}")
        load_bc_weights(_raw(net), args.bc_weights)
    elif not args.bc_weights:
        print("[init] no BC weights supplied — training from random initialisation")

    if (not args.no_compile) and device.type == "cuda":
        try:
            net = torch.compile(net)
            print("[opt] torch.compile enabled")
        except Exception as e:
            print(f"[opt] torch.compile unavailable: {e}")

    opt    = optim.Adam(_raw(net).parameters(), lr=args.lr, eps=1e-5)
    scaler = GradScaler("cuda", enabled=use_amp)
    buf    = GRURolloutBuffer(args.horizon, IN_DIM, GRU_HIDDEN, args.gamma, args.lam)

    start_ep = 0
    steps    = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        _raw(net).load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_ep = ck["episode"]
        steps    = ck["steps"]
        print(f"[resume] episode={start_ep}  steps={steps}")

    n_params = sum(p.numel() for p in _raw(net).parameters())
    print(f"[model] UnifiedGRUActorCritic  params={n_params:,}  "
          f"input={IN_DIM}  gru_hidden={GRU_HIDDEN}  n_actions={N_ACTIONS}")

    # ── Curriculum / single-stage ─────────────────────────────────────────────
    if args.no_curriculum:
        curriculum = [dict(arena_size=args.arena_size, difficulty=args.difficulty,
                           wall_obstacles=args.wall_obstacles,
                           max_steps=args.max_steps, episodes=args.episodes)]
    else:
        curriculum = CURRICULUM

    # ── Observation buffer on device ──────────────────────────────────────────
    obs_buf = torch.zeros(1, IN_DIM, dtype=torch.float32, device=device)

    def to_device(x_np: np.ndarray) -> torch.Tensor:
        obs_buf[0].copy_(torch.from_numpy(x_np), non_blocking=True)
        return obs_buf

    # ── Episode factory ───────────────────────────────────────────────────────
    def new_env(ep_idx, cfg):
        seed_i = args.seed * 100003 + ep_idx
        e = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=cfg["arena_size"],
            max_steps=cfg["max_steps"],
            wall_obstacles=cfg["wall_obstacles"],
            difficulty=cfg["difficulty"],
            box_speed=args.box_speed,
            seed=seed_i,
        )
        raw = e.reset(seed=seed_i)
        return e, np.asarray(raw, dtype=np.float32), seed_i

    # ── Training ──────────────────────────────────────────────────────────────
    ep          = start_ep
    rewards_log = deque(maxlen=100)
    t0          = time.time()

    for stage_cfg in curriculum:
        stage_eps = stage_cfg["episodes"]
        stage_end = ep + stage_eps
        print(f"\n[stage] arena={stage_cfg['arena_size']}  diff={stage_cfg['difficulty']}  "
              f"walls={stage_cfg['wall_obstacles']}  episodes={stage_eps}")

        while ep < stage_end:

            # Reset per-episode state
            env, raw, _        = new_env(ep, stage_cfg)
            belief             = CompactBeliefState(max_steps=stage_cfg["max_steps"])
            h                  = _raw(net).zero_hidden()   # (1, 1, GRU_HIDDEN)
            ep_reward          = 0.0
            buf.reset()

            horizon_steps = 0

            while horizon_steps < args.horizon:
                # Build 39-dim EPB input
                x = build_input(raw, belief)
                st = to_device(x)

                # Inference (GRU step)
                with torch.no_grad():
                    with autocast(device_type=device.type, enabled=use_amp):
                        logits, value, h_new = _raw(net).forward_step(st, h)
                        dist      = Categorical(logits=logits)
                        action_t  = dist.sample()
                        log_prob  = dist.log_prob(action_t)

                h_before = h
                h        = h_new

                a_idx   = int(action_t.item())
                act_str = ACTIONS[a_idx]

                raw2, env_r, done = env.step(act_str, render=False)
                raw2 = np.asarray(raw2, dtype=np.float32)

                # Update belief with the new observation and action taken
                belief.update(raw2, act_str)

                buf.add(
                    x, a_idx, float(env_r), done,
                    float(log_prob.item()), float(value.item()),
                    h=h_before,
                )

                ep_reward    += float(env_r)
                steps        += 1
                horizon_steps += 1
                raw           = raw2

                if done:
                    rewards_log.append(ep_reward)
                    avg100  = np.mean(rewards_log)
                    elapsed = time.time() - t0
                    speed   = steps / elapsed if elapsed > 0 else 0
                    print(f"[ep {ep+1}]  return={ep_reward:.1f}  "
                          f"avg100={avg100:.1f}  steps={steps}  "
                          f"arena={stage_cfg['arena_size']}  {speed:.0f}sps")
                    ep += 1

                    if ep >= stage_end:
                        break

                    # Reset for next episode
                    env, raw, _  = new_env(ep, stage_cfg)
                    belief       = CompactBeliefState(max_steps=stage_cfg["max_steps"])
                    h            = _raw(net).zero_hidden()
                    ep_reward    = 0.0

            # ── PPO update ────────────────────────────────────────────────────
            with torch.no_grad():
                x_lv = build_input(raw, belief)
                with autocast(device_type=device.type, enabled=use_amp):
                    _, lv_t, _ = _raw(net).forward_step(to_device(x_lv), h)
                last_val = float(lv_t.item()) * (1 - float(done))

            pg, vf, ent = ppo_update(
                _raw(net), opt, scaler, buf, last_val,
                args.epochs, args.clip_eps, args.vf_coef,
                args.ent_coef, args.max_grad_norm, use_amp,
            )
            print(f"  [ppo]  steps={buf.ptr}  pg={pg:.4f}  vf={vf:.4f}  ent={ent:.4f}")

            # Checkpoint every 200 episodes
            if ep > 0 and ep % 200 == 0:
                ck_path = args.out.replace(".pth", f"_ep{ep}.pth")
                torch.save({
                    "net":     _raw(net).state_dict(),
                    "opt":     opt.state_dict(),
                    "scaler":  scaler.state_dict(),
                    "episode": ep, "steps": steps,
                }, ck_path)
                print(f"  [ckpt] {ck_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    torch.save(_raw(net).state_dict(), args.out)
    print(f"\n[done] weights → {args.out}")

    # Also save an agent-compatible submission copy
    subm = args.out.replace(".pth", "_submission.pth")
    torch.save({
        "state_dict": _raw(net).state_dict(),
        "config": dict(obs_dim=IN_DIM, n_actions=N_ACTIONS, gru_hidden=GRU_HIDDEN),
    }, subm)
    print(f"[done] submission payload → {subm}")


if __name__ == "__main__":
    main()
