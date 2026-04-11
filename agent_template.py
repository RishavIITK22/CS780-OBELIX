"""
OBELIX Agent: FSM-Guided DQN
=============================
Architecture:
  - FSM provides: current phase (3-hot) + suggested action (5-hot)
  - Feature engineering converts 18-bit obs into 24 meaningful features
  - DQN network takes concat(features, phase, fsm_action) = 32-dim input
  - FSM bias added to Q-values so DRL only overrides FSM when confident
  - At inference: loads weights.pth from same directory

Submission:
  agent_template.py  +  weights.pth  (both in same folder)
"""

import os
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTIONS    = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS  = len(ACTIONS)
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
PHASES     = ["find", "push", "unwedge"]
N_FEAT     = 24
N_PHASE    = 3
N_FSM      = 5
N_INPUT    = N_FEAT + N_PHASE + N_FSM  # 32

FSM_BIAS   = 2.0   # Q bonus for FSM-suggested action
DEVICE     = torch.device("cpu")


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=N_INPUT, output_dim=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Feature Engineering: 18 raw bits -> 24 features
# ---------------------------------------------------------------------------
def _parse_obs(obs):
    obs = np.asarray(obs, dtype=int)
    lf  = int(obs[0]  or obs[2])
    ln  = int(obs[1]  or obs[3])
    ff  = int(obs[4]  or obs[6]  or obs[8]  or obs[10])
    fn  = int(obs[5]  or obs[7]  or obs[9]  or obs[11])
    rf  = int(obs[12] or obs[14])
    rn  = int(obs[13] or obs[15])
    ir  = int(obs[16])
    stk = int(obs[17])
    any_s = int(any(obs[:17]))
    return lf, ln, ff, fn, rf, rn, ir, stk, any_s


def featurize(obs) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    lf, ln, ff, fn, rf, rn, ir, stk, any_s = _parse_obs(obs)
    return np.array([
        # raw grouped (8)
        lf, ln, ff, fn, rf, rn, ir, stk,
        # directional summaries (5)
        float(lf or ln),
        float(rf or rn),
        float(ff or fn or ir),
        float(ln or fn or rn or ir),
        float(lf or ff or rf),
        # relative direction hints (5)
        float(ln and not rn),
        float(rn and not ln),
        float(fn and not (ln or rn)),
        float(lf and not rf),
        float(rf and not lf),
        # proximity score 0-1 (1)
        float(ir*5 + fn*3 + ff*2 + (ln or rn) + (lf or rf)*0.5) / 5.0,
        # no-signal flag (1)
        float(not any_s),
        # forward clear (1)
        float(ff or fn or ir),
        # side imbalance -1..1 (1)
        float((lf + ln) - (rf + rn)) / 2.0,
        # stuck emphasis (1)
        float(stk),
        # any near (1)
        float(ln or fn or rn or ir),
    ], dtype=np.float32)  # total = 24


def _phase_onehot(phase: str) -> np.ndarray:
    v = np.zeros(N_PHASE, dtype=np.float32)
    v[PHASES.index(phase)] = 1.0
    return v


def _action_onehot(action: str) -> np.ndarray:
    v = np.zeros(N_FSM, dtype=np.float32)
    v[ACTION_IDX[action]] = 1.0
    return v


def _build_input(obs, phase: str, fsm_act: str) -> torch.Tensor:
    vec = np.concatenate([featurize(obs), _phase_onehot(phase), _action_onehot(fsm_act)])
    return torch.tensor(vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)


# ---------------------------------------------------------------------------
# FSM  (provides phase tracking + default action suggestion)
# ---------------------------------------------------------------------------
_fsm = {
    "phase":          "find",
    "steps_in_phase": 0,
    "steps_no_sensor":0,
    "spin_dir":       "L45",
    "push_stuck_cnt": 0,
    "find_stuck_cnt": 0,
    "episode_step":   0,
}


def reset_agent():
    """Must be called at the start of each episode."""
    import random as _r
    _fsm["phase"]           = "find"
    _fsm["steps_in_phase"]  = 0
    _fsm["steps_no_sensor"] = 0
    _fsm["spin_dir"]        = _r.choice(["L45", "R45"])
    _fsm["push_stuck_cnt"]  = 0
    _fsm["find_stuck_cnt"]  = 0
    _fsm["episode_step"]    = 0


def _fsm_suggest(obs) -> str:
    phase = _fsm["phase"]
    lf, ln, ff, fn, rf, rn, ir, stk, _ = _parse_obs(obs)

    if phase == "find":
        if stk:
            _fsm["find_stuck_cnt"] += 1
            return "L45" if _fsm["find_stuck_cnt"] % 2 == 0 else "R45"
        _fsm["find_stuck_cnt"] = 0
        if ir:                 return "FW"
        if fn:                 return "FW"
        if ff:
            if rn and not ln:  return "L22"
            if ln and not rn:  return "R22"
            return "FW"
        if ln and not rn:      return "L22"
        if rn and not ln:      return "R22"
        if lf and not rf:      return "L22"
        if rf and not lf:      return "R22"
        if lf and rf:          return _fsm["spin_dir"]
        _fsm["steps_no_sensor"] += 1
        return _fsm["spin_dir"] if (_fsm["steps_no_sensor"] % 16) < 6 else "FW"

    elif phase == "push":
        if stk:
            _fsm["push_stuck_cnt"] += 1
            c = _fsm["push_stuck_cnt"]
            if   c <= 2: return "L22"
            elif c <= 4: return "R22"
            elif c <= 6: return "L45"
            else:
                _fsm["push_stuck_cnt"] = 0
                return "R45"
        _fsm["push_stuck_cnt"] = 0
        return "FW"

    else:  # unwedge
        if stk:
            return "L45" if (_fsm["steps_in_phase"] % 4) < 2 else "R45"
        return "FW"


def _fsm_update_phase(obs, action):
    lf, ln, ff, fn, rf, rn, ir, stk, _ = _parse_obs(obs)
    _fsm["steps_in_phase"] += 1
    _fsm["episode_step"]   += 1

    if _fsm["phase"] == "find":
        if (ir and action == "FW") or (fn and _fsm["steps_in_phase"] > 3 and action == "FW"):
            _fsm["phase"] = "push"; _fsm["steps_in_phase"] = 0; _fsm["push_stuck_cnt"] = 0
    elif _fsm["phase"] == "push":
        if stk:
            _fsm["phase"] = "unwedge"; _fsm["steps_in_phase"] = 0
    elif _fsm["phase"] == "unwedge":
        if not stk:
            _fsm["phase"] = "push"; _fsm["steps_in_phase"] = 0


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
_model: DQN = None


def _load_model():
    global _model
    _model = DQN(input_dim=N_INPUT, output_dim=N_ACTIONS).to(DEVICE)
    _model.eval()
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "weights.pth")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE)
        sd   = ckpt.get("model_state_dict", ckpt)
        _model.load_state_dict(sd)
        print(f"[OBELIX] Loaded weights from {path}")
    else:
        print(f"[OBELIX] WARNING: weights.pth not found. Using FSM-only + random net.")


_load_model()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def policy(obs) -> str:
    """Returns one of: 'L45', 'L22', 'FW', 'R22', 'R45'"""
    phase   = _fsm["phase"]
    fsm_act = _fsm_suggest(obs)

    net_in  = _build_input(obs, phase, fsm_act)
    with torch.no_grad():
        q = _model(net_in).squeeze(0).cpu().numpy()

    # FSM bias: DQN only overrides FSM when it's very confident
    q[ACTION_IDX[fsm_act]] += FSM_BIAS
    action = ACTIONS[int(np.argmax(q))]

    _fsm_update_phase(obs, action)
    return action