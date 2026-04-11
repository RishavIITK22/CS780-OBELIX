from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cpu")
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
PHASES = ["find", "push", "unwedge"]


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


def _phase_onehot(phase: str) -> np.ndarray:
    v = np.zeros(len(PHASES), dtype=np.float32)
    v[PHASES.index(phase)] = 1.0
    return v


def _action_onehot(action: str) -> np.ndarray:
    v = np.zeros(len(ACTIONS), dtype=np.float32)
    v[ACTION_IDX[action]] = 1.0
    return v


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


class FSMState:
    def __init__(self):
        self.phase = "find"
        self.steps_in_phase = 0
        self.steps_no_sensor = 0
        self.spin_dir = "L45"
        self.push_stuck_cnt = 0
        self.find_stuck_cnt = 0
        self.episode_step = 0

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
    return np.concatenate(
        [featurize(obs), _phase_onehot(fsm.phase), _action_onehot(fsm_act)]
    ).astype(np.float32)


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
        self.actor = nn.Linear(gru_hidden, len(ACTIONS))
        self.critic = nn.Linear(gru_hidden, 1)

    def zero_hidden(self):
        return torch.zeros(1, 1, self.gru_hidden, device=DEVICE)

    def forward_step(self, x: torch.Tensor, h: torch.Tensor):
        enc = self.encoder(x).unsqueeze(0)
        out, h_new = self.gru(enc, h)
        out = out.squeeze(0)
        return self.actor(out), self.critic(out).squeeze(-1), h_new


_here = os.path.dirname(os.path.abspath(__file__))
_weights_path = os.path.join(_here, "weights_template_memory.pth")
_payload = torch.load(_weights_path, map_location=DEVICE, weights_only=False)
_config = _payload.get("config", {})
_net = GRUActorCritic(
    in_dim=int(_config.get("in_dim", 32)),
    hidden=int(_config.get("hidden", 128)),
    gru_hidden=int(_config.get("gru_hidden", 128)),
).to(DEVICE)
_net.load_state_dict(_payload["model_state_dict"] if "model_state_dict" in _payload else _payload)
_net.eval()

_fsm = FSMState()
_fsm.reset()
_hidden = _net.zero_hidden()
_step_count = 0


def reset_agent():
    global _fsm, _hidden, _step_count
    _fsm.reset()
    _hidden = _net.zero_hidden()
    _step_count = 0


@torch.no_grad()
def policy(obs, rng=None):
    global _hidden, _step_count
    x_np = build_input(obs, _fsm)
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits, _, h_new = _net.forward_step(x, _hidden)
    _hidden = h_new
    action = ACTIONS[int(torch.argmax(logits, dim=-1).item())]
    _fsm.update(obs, action)
    _step_count += 1
    return action
