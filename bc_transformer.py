"""bc_transformer.py — Behavioural Cloning with a causal GPT-style Transformer.

Architecture
------------
Input sequence x_{1..T}  each x_t ∈ ℝ^{39}  (EPB input from compact_belief)

  embedding   : Linear(39 → d_model=128)          ← transferred to PPO encoder
  pos_embed   : Embedding(max_len=1024, 128)       (learned positional encoding)
  transformer : N × TransformerEncoderLayer        (causal / autoregressive mask)
                  d_model=128, nhead=4, ffn_dim=256, dropout=0.1
  actor_head  : Linear(128 → 5)                   ← transferred to PPO actor

Training: cross-entropy(actor_head(transformer(embed(x_t))), fsm_action_t)
          averaged over all non-padding positions.

Weight transfer
---------------
After BC training, train_ppo_gru_bc.py loads this checkpoint and copies:
  bc_transformer.embedding  →  unified_gru_net.encoder[0]  (Linear 39→128)
  bc_transformer.actor_head →  unified_gru_net.actor        (Linear 128→5)

Usage:
  python bc_transformer.py --dataset dataset_fsm.pkl --out weights_bc.pth
  python bc_transformer.py --dataset dataset_fsm.pkl --out weights_bc.pth \\
      --epochs 30 --d_model 128 --n_heads 4 --n_layers 3 --seq_len 128
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import time
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from compact_belief import ACTIONS, IN_DIM  # IN_DIM = 39

N_ACTIONS = len(ACTIONS)   # 5


# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))


# ── Model ──────────────────────────────────────────────────────────────────────

class BCTransformer(nn.Module):
    """Causal (GPT-style) Transformer for behaviour cloning.

    Key design choices:
    - Causal mask: position t attends only to 1..t (autoregressive).
    - Learned positional embedding (not sinusoidal) for flexibility.
    - Pre-LN (LayerNorm before attention): more stable training.
    - Shared embedding / actor_head names match those expected by
      train_ppo_gru_bc.py for weight transfer.
    """

    def __init__(self, in_dim: int = IN_DIM, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3,
                 ffn_dim: int = 256, dropout: float = 0.1,
                 max_len: int = 1024):
        super().__init__()
        self.d_model = d_model

        # ── Input embedding (Linear, no activation — mirrors PPO encoder[0])
        self.embedding = nn.Linear(in_dim, d_model)

        # ── Learned positional encoding
        self.pos_embed = nn.Embedding(max_len, d_model)

        # ── Transformer encoder (causal mask applied in forward)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)

        # ── Actor head (directly transferred to PPO unified GRU actor)
        self.actor_head = nn.Linear(d_model, N_ACTIONS)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular True mask: position i cannot attend to j > i."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask  # True = masked out in nn.TransformerEncoderLayer

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        x                : (B, T, in_dim)
        key_padding_mask : (B, T)  True = padding position (ignored by attention)

        Returns logits   : (B, T, N_ACTIONS)
        """
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)

        h = self.embedding(x) + self.pos_embed(pos)           # (B, T, d_model)
        h = self.transformer(
            h,
            mask=self._causal_mask(T, x.device),
            src_key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        return self.actor_head(h)                              # (B, T, N_ACTIONS)

    def predict_step(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Single-step greedy action for eval/visualization (no grad needed)."""
        with torch.no_grad():
            logits = self.forward(x)          # (1, t+1, N_ACTIONS)
            return logits[0, t].argmax(-1)    # scalar


# ── Dataset ────────────────────────────────────────────────────────────────────

class TrajectoryWindowDataset(Dataset):
    """Random fixed-length windows sampled from the FSM trajectory dataset.

    Each item is (obs_window, action_window, padding_mask):
      obs_window    : float32  (seq_len, 39)
      action_window : int64    (seq_len,)
      padding_mask  : bool     (seq_len,)  True = padding

    Episodes shorter than seq_len are right-padded with zeros / action=0.
    """

    def __init__(self, episodes: List[Dict], seq_len: int = 128,
                 samples_per_ep: int = 4):
        self.seq_len = seq_len
        self.windows: List[tuple] = []

        for ep in episodes:
            T = len(ep["actions"])
            if T < 2:
                continue
            # Sample multiple overlapping windows per episode
            n_samples = max(1, min(samples_per_ep, T // (seq_len // 2)))
            for _ in range(n_samples):
                start = random.randint(0, max(0, T - 1))
                end   = start + seq_len
                obs_w = ep["obs"][start:end]         # (≤seq_len, 39)
                act_w = ep["actions"][start:end]      # (≤seq_len,)
                length = len(act_w)

                # Pad to seq_len
                if length < seq_len:
                    pad = seq_len - length
                    obs_w = np.concatenate(
                        [obs_w, np.zeros((pad, IN_DIM), dtype=np.float32)], axis=0)
                    act_w = np.concatenate(
                        [act_w, np.zeros(pad, dtype=np.int64)], axis=0)

                mask = np.zeros(seq_len, dtype=bool)
                mask[length:] = True   # True = padding

                self.windows.append((obs_w, act_w, mask))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        obs_w, act_w, mask = self.windows[idx]
        return (
            torch.from_numpy(obs_w),
            torch.from_numpy(act_w),
            torch.from_numpy(mask),
        )


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    # ── Load dataset
    print(f"[data] loading {args.dataset} ...")
    with open(args.dataset, "rb") as f:
        episodes = pickle.load(f)
    print(f"[data] {len(episodes)} episodes loaded")

    random.shuffle(episodes)
    n_val  = max(1, int(len(episodes) * 0.05))
    val_ep = episodes[:n_val]
    trn_ep = episodes[n_val:]
    print(f"[data] train={len(trn_ep)} val={n_val} episodes")

    trn_ds = TrajectoryWindowDataset(trn_ep, seq_len=args.seq_len,
                                     samples_per_ep=args.samples_per_ep)
    val_ds = TrajectoryWindowDataset(val_ep, seq_len=args.seq_len, samples_per_ep=2)
    print(f"[data] train windows={len(trn_ds)}  val windows={len(val_ds)}")

    trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=1, pin_memory=(device.type == "cuda"))

    # ── Model
    model = BCTransformer(
        in_dim=IN_DIM, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ffn_dim=args.ffn_dim,
        dropout=args.dropout, max_len=args.seq_len + 16,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] BCTransformer  params={n_params:,}  d_model={args.d_model}  "
          f"n_heads={args.n_heads}  n_layers={args.n_layers}")

    opt       = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train
        model.train()
        trn_loss = trn_correct = trn_total = 0
        t0 = time.time()

        for obs_b, act_b, mask_b in trn_loader:
            obs_b  = obs_b.to(device, non_blocking=True)      # (B, T, 39)
            act_b  = act_b.to(device, non_blocking=True)      # (B, T)
            mask_b = mask_b.to(device, non_blocking=True)     # (B, T)  True=pad

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(obs_b, key_padding_mask=mask_b)  # (B, T, 5)
                # Flatten and mask out padding
                B, T, C = logits.shape
                logits_flat = logits.view(B * T, C)
                act_flat    = act_b.view(B * T)
                # Set padding targets to -1 so CrossEntropyLoss ignores them
                act_flat = act_flat.masked_fill(mask_b.view(B * T), -1)
                loss = criterion(logits_flat, act_flat)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # Accuracy (non-padding positions only)
            valid = ~mask_b.view(B * T)
            preds = logits_flat.detach().argmax(-1)
            trn_correct += int((preds[valid] == act_flat[valid]).sum())
            trn_total   += int(valid.sum())
            trn_loss    += loss.item()

        trn_loss /= len(trn_loader)
        trn_acc   = trn_correct / max(1, trn_total)

        # ── Validate
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for obs_b, act_b, mask_b in val_loader:
                obs_b  = obs_b.to(device, non_blocking=True)
                act_b  = act_b.to(device, non_blocking=True)
                mask_b = mask_b.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(obs_b, key_padding_mask=mask_b)
                B, T, C = logits.shape
                logits_flat = logits.view(B * T, C)
                act_flat    = act_b.view(B * T).masked_fill(mask_b.view(B * T), -1)
                val_loss   += criterion(logits_flat, act_flat).item()
                valid       = ~mask_b.view(B * T)
                preds       = logits_flat.argmax(-1)
                val_correct += int((preds[valid] == act_flat[valid]).sum())
                val_total   += int(valid.sum())

        val_loss /= max(1, len(val_loader))
        val_acc   = val_correct / max(1, val_total)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"[epoch {epoch:03d}/{args.epochs}]  "
              f"trn_loss={trn_loss:.4f}  trn_acc={trn_acc*100:.1f}%  |  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%  |  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  t={elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "val_acc":     val_acc,
                "state_dict":  model.state_dict(),
                "config": dict(in_dim=IN_DIM, d_model=args.d_model,
                               n_heads=args.n_heads, n_layers=args.n_layers,
                               ffn_dim=args.ffn_dim),
            }, args.out)
            print(f"  ↳ best val_acc={best_val_acc*100:.1f}%  saved → {args.out}")

    print(f"\n[done] best val acc = {best_val_acc*100:.1f}%  weights → {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",        type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_bc.pth")
    # Model
    ap.add_argument("--d_model",        type=int,   default=128)
    ap.add_argument("--n_heads",        type=int,   default=4)
    ap.add_argument("--n_layers",       type=int,   default=3)
    ap.add_argument("--ffn_dim",        type=int,   default=256)
    ap.add_argument("--dropout",        type=float, default=0.1)
    ap.add_argument("--seq_len",        type=int,   default=128,
                    help="Window length for training sequences")
    # Training
    ap.add_argument("--epochs",         type=int,   default=40)
    ap.add_argument("--batch_size",     type=int,   default=128)
    ap.add_argument("--lr",             type=float, default=3e-4)
    ap.add_argument("--samples_per_ep", type=int,   default=4,
                    help="Random windows sampled per episode per epoch")
    ap.add_argument("--seed",           type=int,   default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
