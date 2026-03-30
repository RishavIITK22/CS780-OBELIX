from __future__ import annotations

import numpy as np


class RewardShaper:
    """Stateful per-episode reward shaper for OBELIX.

    Takes the encoded observation from BeliefStateEncoder (38-dim) to access
    richer features than raw bits allow.

    Encoded layout (must match BeliefStateEncoder.encode output):
        [0:18]  raw obs  (sonar bits 0-15, IR bit 16, stuck bit 17)
        [18:26] sector strengths  (2*near + far, per sector)
        [26:29] direction summary (left, front, right)
        [29]    steps_since_seen  (normalised 0-1)
        [30]    stuck_steps       (normalised 0-1)
        [31]    just_got_ir       (1 on IR rising edge)
        [32]    just_recovered    (1 on stuck falling edge)
        [33:38] prev_action one-hot

    Shaping terms:
        1. Front alignment bonus  — front strength increasing (find phase)
        2. IR contact bonus       — one-shot on just_got_ir (pre-attach)
        3. Push progress          — each non-stuck FW step while attached
        4. Escalating stuck penalty — grows with stuck_steps duration
        5. Recovery bonus         — just_recovered fires
        6. Spin penalty           — escalating after spin_threshold rotations
        7. Forward bonus          — prefer FW over spinning (find phase)
        8. Search pressure        — gentle penalty when box unseen for long
        9. Efficiency bonus       — one-time at success, scales with steps saved
    """

    # Indices into the 38-dim encoded vector
    _IR_BIT         = 16
    _STUCK_BIT      = 17
    _STR_FRONT_START = 20   # strengths[2] = sectors 2..5 = front fan
    _STR_FRONT_END   = 24
    _DIR_FRONT      = 27    # direction_summary[1] = front aggregate
    _STEPS_SINCE_SEEN = 29
    _STUCK_STEPS    = 30
    _JUST_GOT_IR    = 31
    _JUST_RECOVERED = 32

    def __init__(
        self,
        reward_scale:       float = 20.0,
        approach_scale:     float = 2.0,   # general approach bonus (not just front)
        front_align_scale:  float = 3.0,   # approach via front specifically
        ir_contact_bonus:   float = 5.0,   # one-shot on IR rising edge
        push_scale:         float = 4.0,   # per non-stuck push step
        stuck_penalty:      float = 3.0,   # base; multiplied by stuck_steps
        recovery_bonus:     float = 2.0,   # on just_recovered
        spin_penalty:       float = 2.0,   # escalating per extra rotation
        spin_threshold:     int   = 6,
        forward_bonus:      float = 1.0,   # per FW step in find phase
        search_pressure:    float = 1.0,   # penalty scaled by steps_since_seen
        efficiency_bonus:   float = 10.0,
        max_steps:          int   = 1000,
    ):
        self.reward_scale      = reward_scale
        self.approach_scale    = approach_scale
        self.front_align_scale = front_align_scale
        self.ir_contact_bonus  = ir_contact_bonus
        self.push_scale        = push_scale
        self.stuck_penalty     = stuck_penalty
        self.recovery_bonus    = recovery_bonus
        self.spin_penalty      = spin_penalty
        self.spin_threshold    = spin_threshold
        self.forward_bonus     = forward_bonus
        self.search_pressure   = search_pressure
        self.efficiency_bonus  = efficiency_bonus
        self.max_steps         = max_steps
        self.reset()

    def reset(self) -> None:
        self._prev_front_strength = 0.0
        self._consec_rotations    = 0
        self._step_count          = 0

    def shape(
        self,
        raw_reward:   float,
        encoded_obs:  np.ndarray,   # 38-dim output of BeliefStateEncoder.encode()
        done:         bool,
        enable_push:  bool,
        action:       str = "FW",
    ) -> float:
        """
        raw_reward:  already divided by reward_scale by the caller.
        encoded_obs: full 38-dim encoded vector from BeliefStateEncoder.
        Returns the shaped reward (same scale as raw_reward).
        """
        r = float(raw_reward)
        s = self.reward_scale

        stuck          = bool(encoded_obs[self._STUCK_BIT])
        front_strength = float(encoded_obs[self._DIR_FRONT])
        steps_since_seen = float(encoded_obs[self._STEPS_SINCE_SEEN])   # normed 0-1
        stuck_steps    = float(encoded_obs[self._STUCK_STEPS])          # normed 0-1
        just_got_ir    = bool(encoded_obs[self._JUST_GOT_IR])
        just_recovered = bool(encoded_obs[self._JUST_RECOVERED])

        self._step_count += 1

        # 1. Front alignment bonus (find phase)
        # Reward the box moving into the front sector specifically — directly
        # shapes the agent to face the box, which is necessary for IR and attach.
        if not enable_push and front_strength > self._prev_front_strength:
            delta = front_strength - self._prev_front_strength
            r += self.front_align_scale * delta / s

        # 2. IR contact bonus (one-shot, find phase)
        # IR fires only when the box is directly ahead at near range.
        # A rising edge means the agent just aligned perfectly — strong signal.
        if just_got_ir and not enable_push:
            r += self.ir_contact_bonus / s

        # 3. Push progress
        if enable_push and not stuck and not done:
            r += self.push_scale / s

        # 4. Escalating stuck penalty
        # stuck_steps is normalised [0,1]: 0 = just got stuck, 1 = stuck for
        # max_stuck_steps. Flat penalty at low duration, grows as trap persists.
        if stuck:
            r -= self.stuck_penalty * (1.0 + stuck_steps) / s

        # 5. Recovery bonus
        if just_recovered:
            r += self.recovery_bonus / s

        # 6. Spin penalty (find phase)
        if action in {"L45", "L22", "R22", "R45"} and not enable_push:
            self._consec_rotations += 1
        else:
            self._consec_rotations = 0

        if self._consec_rotations > self.spin_threshold:
            excess = self._consec_rotations - self.spin_threshold
            r -= self.spin_penalty * excess / s

        # 7. Forward bonus (find phase)
        if action == "FW" and not stuck and not enable_push:
            r += self.forward_bonus / s

        # 8. Search pressure (find phase)
        # When the box has been invisible for a long time, gently penalise
        # standing still or spinning. steps_since_seen=1.0 means the agent
        # hasn't seen the box for max_steps_since_seen steps — escalate then.
        if not enable_push and steps_since_seen > 0.5:
            r -= self.search_pressure * (steps_since_seen - 0.5) / s

        # 9. Efficiency bonus at success
        if done and raw_reward > 0:
            efficiency = max(0.0, 1.0 - self._step_count / self.max_steps)
            r += self.efficiency_bonus * efficiency / s

        self._prev_front_strength = front_strength
        return r
