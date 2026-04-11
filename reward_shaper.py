from __future__ import annotations

import numpy as np


class RewardShaper:
    """Small stateful reward shaper for OBELIX.

    This intentionally keeps only the highest-signal shaping terms:
      - find: small bonus for useful sensor evidence, especially front/near
      - push: small bonus for forward progress after attachment
      - stuck: escalating penalty while stuck
      - recovery: small bonus when the stuck flag clears

    The public API stays compatible with the previous RewardShaper:
        shape(raw_reward, encoded_obs, done, enable_push, action="FW")

    `raw_reward` is expected to already be scaled by the caller, as in the old
    implementation. The shaping terms are divided by `reward_scale` too.
    """

    _IR_BIT = 16
    _STUCK_BIT = 17
    _DIR_LEFT = 26
    _DIR_FRONT = 27
    _DIR_RIGHT = 28
    _STUCK_STEPS = 30
    _JUST_RECOVERED = 32

    def __init__(
        self,
        reward_scale: float = 20.0,
        find_sensor_bonus: float = 0.4,
        find_front_bonus: float = 0.8,
        ir_bonus: float = 1.0,
        push_forward_bonus: float = 2.0,
        stuck_penalty: float = 2.0,
        stuck_penalty_growth: float = 2.0,
        recovery_bonus: float = 1.0,
        **_unused,
    ):
        self.reward_scale = reward_scale
        self.find_sensor_bonus = find_sensor_bonus
        self.find_front_bonus = find_front_bonus
        self.ir_bonus = ir_bonus
        self.push_forward_bonus = push_forward_bonus
        self.stuck_penalty = stuck_penalty
        self.stuck_penalty_growth = stuck_penalty_growth
        self.recovery_bonus = recovery_bonus
        self.reset()

    def reset(self) -> None:
        self._step_count = 0

    def shape(
        self,
        raw_reward: float,
        encoded_obs: np.ndarray,
        done: bool,
        enable_push: bool,
        action: str = "FW",
    ) -> float:
        r = float(raw_reward)
        s = max(float(self.reward_scale), 1e-6)
        obs = np.asarray(encoded_obs, dtype=np.float32)

        ir_on = bool(obs[self._IR_BIT])
        stuck = bool(obs[self._STUCK_BIT])
        left_strength = float(obs[self._DIR_LEFT])
        front_strength = float(obs[self._DIR_FRONT])
        right_strength = float(obs[self._DIR_RIGHT])
        stuck_steps = float(obs[self._STUCK_STEPS])
        just_recovered = bool(obs[self._JUST_RECOVERED])

        self._step_count += 1

        if not enable_push:
            side_strength = max(left_strength, right_strength)
            r += self.find_sensor_bonus * side_strength / s
            r += self.find_front_bonus * front_strength / s
            if ir_on:
                r += self.ir_bonus / s

        if enable_push and action == "FW" and not stuck and not done:
            r += self.push_forward_bonus / s

        if stuck:
            r -= (self.stuck_penalty + self.stuck_penalty_growth * stuck_steps) / s

        if just_recovered:
            r += self.recovery_bonus / s

        return r
