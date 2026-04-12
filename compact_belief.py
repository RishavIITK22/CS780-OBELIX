"""compact_belief.py — Explicit non-decaying belief state for OBELIX POMDP.

The core problem with GRU-based agents: after hundreds of blank (no-sensor)
steps, the hidden state is washed toward the zero attractor and the last known
box direction is forgotten.

This module provides CompactBeliefState — a 16-dim vector of hand-tracked
variables that NEVER decay on blank steps.  Key properties:
  * last_visible_* fields update ONLY when a sensor fires; they persist forever.
  * steps_since_visible counts up (never resets on blank steps).
  * attached_flag latches True on attachment and never goes back.

Feed [raw_obs (18) | belief (16) | fsm_onehot (5)] = 39-dim into an MLP.
"""

from __future__ import annotations
import random
import numpy as np

BELIEF_DIM = 16
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
IN_DIM = 18 + BELIEF_DIM + len(ACTIONS)   # 39


# ── observation parser ────────────────────────────────────────────────────────

def parse_obs(obs):
    """Compress 18-bit raw obs into 9 aggregate signals."""
    obs = np.asarray(obs, dtype=int)
    lf  = int(obs[0] or obs[2])
    ln  = int(obs[1] or obs[3])
    ff  = int(obs[4] or obs[6] or obs[8] or obs[10])
    fn  = int(obs[5] or obs[7] or obs[9] or obs[11])
    rf  = int(obs[12] or obs[14])
    rn  = int(obs[13] or obs[15])
    ir  = int(obs[16])
    stk = int(obs[17])
    any_s = int(any(obs[:17]))
    return lf, ln, ff, fn, rf, rn, ir, stk, any_s


# ── belief state ──────────────────────────────────────────────────────────────

class CompactBeliefState:
    """Explicit, non-decaying POMDP belief tracker.

    Call update(obs, action) after every environment step.
    Call to_vector() to get the 16-dim feature vector for the network.
    Call fsm_suggest(obs) to get the reactive FSM action (used as training bias).
    """

    def __init__(self, max_steps: int = 1000) -> None:
        self.max_steps = max_steps
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        # Phase tracking
        self.attached: bool = False        # latches True; never resets within episode
        self.ir_on_steps: int = 0          # consecutive steps IR has been active

        # Last-known box sighting — updated ONLY when any sensor fires.
        # These persist across hundreds of blank steps (the key innovation).
        self.last_visible_left: int = 0
        self.last_visible_front: int = 0
        self.last_visible_right: int = 0
        self.last_visible_ir: int = 0

        # Staleness counter (increments on blank steps, capped at 500)
        self.steps_since_visible: int = 500   # start as "haven't seen box yet"

        # Stuck tracking
        self.consecutive_stuck: int = 0
        self.just_recovered: bool = False
        self._prev_stuck: bool = False

        # Blind-exploration state
        self.spin_dir: int = random.choice([0, 1])  # 0=R, 1=L
        self.blind_steps: int = 0            # consecutive steps with no sensor

        # Progress & phase counters
        self.step_count: int = 0
        self.push_stuck_count: int = 0
        self.steps_in_phase: int = 0

        # Action memory
        self.last_action: str = "FW"

    # ------------------------------------------------------------------
    def update(self, obs, action: str) -> None:
        """Update belief state given the observation received this step
        and the action that was just chosen (for the next step)."""
        lf, ln, ff, fn, rf, rn, ir, stk, any_s = parse_obs(obs)

        self.step_count += 1

        # ── attachment detection ──────────────────────────────────────
        # Count consecutive steps IR is active, regardless of which action
        # was taken.  The old (ir AND last_action==FW) heuristic was too
        # strict: a single spin step before IR fired kept resetting the counter,
        # and on moving boxes (Level 3) the box drifts away in 1-2 steps.
        if not self.attached:
            if ir:
                self.ir_on_steps += 1
                if self.ir_on_steps >= 2:
                    self.attached = True
                    self.steps_in_phase = 0
                    self.push_stuck_count = 0
            else:
                self.ir_on_steps = 0

            # Backup: if stuck while IR was recently active → box is attached
            # but we missed it (robot is now pressing box against a wall).
            if stk and self.last_visible_ir and not self.attached:
                self.attached = True
                self.steps_in_phase = 0
                self.push_stuck_count = 0

        # ── last-visible direction ────────────────────────────────────
        # ONLY updated when sensor fires — preserves last sighting forever.
        if any_s:
            self.last_visible_left  = int(bool(lf or ln))
            self.last_visible_front = int(bool(ff or fn or ir))
            self.last_visible_right = int(bool(rf or rn))
            self.last_visible_ir    = ir
            self.steps_since_visible = 0
            self.blind_steps = 0
        else:
            self.steps_since_visible = min(self.steps_since_visible + 1, 500)
            if not self.attached:
                self.blind_steps += 1
                # Flip exploration spin direction every 100 blind steps
                # so the robot systematically sweeps the arena.
                if self.blind_steps > 0 and self.blind_steps % 100 == 0:
                    self.spin_dir = 1 - self.spin_dir

        # ── stuck tracking ────────────────────────────────────────────
        self.just_recovered = bool(self._prev_stuck and not stk)
        self._prev_stuck = bool(stk)

        if stk:
            self.consecutive_stuck += 1
        else:
            self.consecutive_stuck = 0

        if self.attached:
            if stk:
                self.push_stuck_count += 1
            else:
                self.push_stuck_count = 0

        self.steps_in_phase += 1
        self.last_action = action

    # ------------------------------------------------------------------
    def to_vector(self) -> np.ndarray:
        """Return the 16-dim belief feature vector (all in [0, 1])."""
        return np.array([
            float(self.attached),                               # 0  phase gate
            float(self.last_visible_left),                      # 1  last seen left
            float(self.last_visible_front),                     # 2  last seen front
            float(self.last_visible_right),                     # 3  last seen right
            float(self.last_visible_ir),                        # 4  last seen IR
            min(self.steps_since_visible / 200.0, 1.0),        # 5  staleness
            min(self.consecutive_stuck / 10.0, 1.0),           # 6  stuck severity
            float(self.just_recovered),                         # 7  recovery edge
            float(self.spin_dir),                               # 8  exploration dir
            min(self.step_count / self.max_steps, 1.0),        # 9  episode progress
            min(self.blind_steps / 100.0, 1.0),                # 10 blind duration
            min(self.push_stuck_count / 5.0, 1.0),             # 11 push-stuck count
            min(self.steps_in_phase / 200.0, 1.0),             # 12 time in phase
            float(self._prev_stuck),                            # 13 currently stuck
            min(self.ir_on_steps / 2.0, 1.0),                  # 14 attach confidence
            float(self.steps_since_visible == 0),               # 15 sensor on now
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def fsm_suggest(self, obs) -> str:
        """Deterministic FSM action suggestion used as a soft training-time prior.

        Priority order (highest → lowest):
          1. Attached → push forward; escape if stuck
          2. Wall collision (STUCK) → alternate-direction escape rotation
          3. IR or near-front active → FW (direct approach / attach)
          4. Moderate front signal → fine-tune alignment, then FW
          5. Asymmetric near-side → turn toward it
          6. Symmetric near-side (box broadly ahead) → FW
          7. Asymmetric far-side → gentle turn toward it
          8. Symmetric far-side → FW (box straight ahead or behind)
          9. Recent memory (steps_since_visible < 25) → navigate to last-seen dir
         10. Fully blind → boustrophedon (lawnmower): FW×40 → L22×4 → FW×6 → L22×4
             Two L22×4 bursts = 176° total ≈ 180° → sweeps back in adjacent lane.
             Half-cycle = 54 steps; ~80% forward, covers a fresh strip each pass.
        """
        lf, ln, ff, fn, rf, rn, ir, stk, _ = parse_obs(obs)

        # 1. Attached → push / unwedge
        if self.attached:
            if stk:
                return "L45" if (self.push_stuck_count % 4) < 2 else "R45"
            return "FW"

        # 2. Wall-stuck escape
        if stk:
            return "L45" if (self.consecutive_stuck % 4) < 2 else "R45"

        # 3. Strong direct signal → approach immediately
        if ir or fn:
            return "FW"

        # 4. Moderate front signal → align then drive in
        if ff:
            if rn and not ln:
                return "L22"
            if ln and not rn:
                return "R22"
            return "FW"

        # 5. Asymmetric near-side → turn toward closer side
        if ln and not rn:
            return "L22"
        if rn and not ln:
            return "R22"

        # 6. Symmetric near-side → box is broadly ahead, drive forward
        if ln and rn:
            return "FW"

        # 7. Asymmetric far-side → gentle turn
        if lf and not rf:
            return "L22"
        if rf and not lf:
            return "R22"

        # 8. Symmetric far-side → box ahead or directly behind, drive forward
        if lf and rf:
            return "FW"

        # 9. Recent memory: box was visible < 25 steps ago — navigate toward it
        #    Uses the non-decaying last_visible_* fields from CompactBeliefState.
        if self.steps_since_visible < 25:
            if self.last_visible_ir or self.last_visible_front:
                return "FW"
            if self.last_visible_left and not self.last_visible_right:
                return "L22"
            if self.last_visible_right and not self.last_visible_left:
                return "R22"
            return "FW"   # symmetric last sighting → drive forward

        # 10. Fully blind: boustrophedon (lawnmower) sweep
        #
        #  Half-cycle structure (54 steps):
        #    FW × 40  — sweep one lane
        #    ?? × 4   — first 88° turn (face perpendicular toward next lane)
        #    FW × 6   — step into the next lane
        #    ?? × 4   — second 88° turn (face back: 176° total from start)
        #
        #  The turn direction ALTERNATES each half-cycle:
        #    even halves → L22  (going, say, East → turns toward North)
        #    odd  halves → R22  (going West  → turns toward North again)
        #
        #  This ensures the lane-shift FW always advances in the SAME
        #  perpendicular direction regardless of sweep heading.
        #  If both turns were always L22, the odd-half shift would go South,
        #  partially cancelling the even-half North shift → near-zero net advance.
        #
        #  Net perpendicular advance per full cycle (108 steps): ~15 px.
        #  With sonar range ~50 px each side, strips overlap slightly → no gaps.
        #  85 % of blind steps are FW, giving ~15× more arena coverage vs spinning.
        _SWEEP = 40   # forward leg (~120 px at 3 px/step in a 500 px arena)
        _TURN  = 4    # 4 × 22° = 88° per burst
        _SHIFT = 6    # perpendicular lane-shift steps
        _HALF  = _SWEEP + _TURN + _SHIFT + _TURN   # 54 steps per half-cycle

        local    = self.blind_steps % _HALF
        half_num = (self.blind_steps // _HALF) % 2   # 0 = even, 1 = odd
        turn     = "L22" if half_num == 0 else "R22"

        if local < _SWEEP:
            return "FW"
        if local < _SWEEP + _TURN:
            return turn          # first turn: face perpendicular to sweep
        if local < _SWEEP + _TURN + _SHIFT:
            return "FW"          # shift one lane
        return turn              # second turn: face back (176° total)


# ── feature builder ───────────────────────────────────────────────────────────

def build_input(obs, belief: CompactBeliefState) -> np.ndarray:
    """Build 39-dim input: [raw_obs(18) | belief(16) | fsm_onehot(5)]."""
    obs_arr = np.asarray(obs, dtype=np.float32)
    fsm_act = belief.fsm_suggest(obs)
    fsm_oh = np.zeros(len(ACTIONS), dtype=np.float32)
    fsm_oh[ACTIONS.index(fsm_act)] = 1.0
    return np.concatenate([obs_arr, belief.to_vector(), fsm_oh])
