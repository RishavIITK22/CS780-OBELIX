import numpy as np


class RewardShaper:
    """Per-episode stateful reward shaper for OBELIX.

    Drop-in replacement for the current shaper.
    Designed for PPO + LSTM in a partially observable setting.

    Main ideas added:
      1. Memory-style shaping for blinking box:
         reward short-horizon motion consistent with the last seen box direction.
      2. Anti-oscillation:
         penalise rapid left-right alternation.
      3. Recovery bonus:
         reward escaping from stuck states.
      4. IR progress:
         reward first acquisition / reacquisition of frontal IR.
      5. Wall-vs-box disambiguation:
         penalise persistent unchanged detections when not attached.
      6. Push alignment:
         slightly reward sensible forward pushing once attached.
    """

    def __init__(
        self,
        max_steps: int = 1000,
        reward_scale: float = 10.0,

        # Existing core shaping
        approach_scale: float = 2.0,
        push_scale: float = 3.0,
        stuck_penalty: float = 1.0,
        explore_scale: float = 0.5,
        efficiency_bonus: float = 50.0,
        forward_bonus: float = 1.0,
        spin_penalty: float = 3.0,
        spin_threshold: int = 5,
        boundary_penalty: float = 3.0,
        boundary_threshold: int = 10,

        # New POMDP-focused shaping
        track_scale: float = 1.5,              # reward short-horizon motion toward last seen box dir
        track_memory_steps: int = 6,           # how long to keep rewarding tracking after box vanishes
        ir_bonus: float = 2.0,                 # bonus when IR is acquired/reacquired
        recovery_bonus: float = 2.5,           # reward when agent escapes stuck state
        oscillation_penalty: float = 1.5,      # penalty for L<->R oscillation
        stability_bonus: float = 0.4,          # small bonus for stable consistent detections
        persistence_penalty: float = 2.0,      # penalty for unchanged detection too long (likely wall)
        persistence_threshold: int = 6,        # unchanged steps before persistence penalty starts
        push_forward_bonus: float = 1.5,       # reward for moving forward while attached
        blind_push_penalty: float = 1.0,       # pushing blindly without useful frontal evidence
    ):
        self.max_steps = max_steps
        self.reward_scale = reward_scale

        self.approach_scale = approach_scale
        self.push_scale = push_scale
        self.stuck_penalty = stuck_penalty
        self.explore_scale = explore_scale
        self.efficiency_bonus = efficiency_bonus
        self.forward_bonus = forward_bonus
        self.spin_penalty = spin_penalty
        self.spin_threshold = spin_threshold
        self.boundary_penalty = boundary_penalty
        self.boundary_threshold = boundary_threshold

        self.track_scale = track_scale
        self.track_memory_steps = track_memory_steps
        self.ir_bonus = ir_bonus
        self.recovery_bonus = recovery_bonus
        self.oscillation_penalty = oscillation_penalty
        self.stability_bonus = stability_bonus
        self.persistence_penalty = persistence_penalty
        self.persistence_threshold = persistence_threshold
        self.push_forward_bonus = push_forward_bonus
        self.blind_push_penalty = blind_push_penalty

        # Internal episode state
        self.reset()

    def reset(self) -> None:
        self._prev_sonar_count = 0
        self._prev_ir = 0
        self._prev_stuck = 0
        self._seen_patterns = set()
        self._step_count = 0

        self._consec_rotations = 0
        self._consec_stuck = 0
        self._steps_at_boundary = 0

        self._prev_action = None

        # Tracking memory for blinking / moving target
        self._last_seen_dir = None
        self._steps_since_seen = 10**9

        # Persistence / stability
        self._prev_pattern = None
        self._same_pattern_steps = 0
        self._stable_detection_steps = 0

    # ------------------------------------------------------------------
    # Direction helpers
    # ------------------------------------------------------------------
    def _sector_strengths(self, obs: np.ndarray) -> np.ndarray:
        """Collapse 16 sonar bits into 8 directional sector strengths.

        For each sector:
            strength = 2 * near_bit + 1 * far_bit
        So near detections are treated as stronger / more reliable than far.
        """
        near = obs[:8].astype(np.float32)
        far = obs[8:16].astype(np.float32)
        return 2.0 * near + 1.0 * far

    def _best_sector(self, obs: np.ndarray):
        """Return most likely box direction sector index in [0..7], or None."""
        strengths = self._sector_strengths(obs)
        if strengths.sum() <= 0:
            return None
        return int(np.argmax(strengths))

    def _action_matches_direction(self, action: str, sector: int) -> bool:
        """Coarse action-direction heuristic.

        We do not know exact geometric mapping of sensor sector IDs here,
        so use a simple front/left/right partition:

          sectors near front:        0, 1, 7
          sectors on left side:      2, 3
          sectors on right side:     5, 6
          ambiguous / rear-ish:      4

        Then reward actions that roughly make sense for that remembered side.
        """
        front = {0, 1, 7}
        left = {2, 3}
        right = {5, 6}
        rearish = {4}

        if sector in front:
            return action == "FW"
        if sector in left:
            return action in {"L22", "L45"}
        if sector in right:
            return action in {"R22", "R45"}
        if sector in rearish:
            return action in {"L45", "R45"}
        return False

    def _is_opposite_turn(self, prev_action: str, action: str) -> bool:
        opposite_pairs = {
            ("L22", "R22"), ("R22", "L22"),
            ("L45", "R45"), ("R45", "L45"),
            ("L22", "R45"), ("R45", "L22"),
            ("L45", "R22"), ("R22", "L45"),
        }
        return (prev_action, action) in opposite_pairs

    # ------------------------------------------------------------------
    def shape(
        self,
        raw_reward: float,
        obs: np.ndarray,
        done: bool,
        enable_push: bool,
        action: str = "FW",
    ) -> float:
        """Return shaped reward.

        Parameters
        ----------
        raw_reward : float
            In your trainer this is already scaled before calling shape():
                scaled = raw_reward / reward_scale
            so here it is the working reward we modify.
        obs : np.ndarray
            18-dim observation.
        done : bool
        enable_push : bool
            External attachment/push mode flag tracked by trainer.
        action : str
            One of {"L45", "L22", "FW", "R22", "R45"}.
        """
        r = float(raw_reward)

        sonar_bits = obs[:16]
        ir_bit = int(obs[16])
        stuck_bit = int(obs[17])

        self._step_count += 1

        sonar_count = int(sonar_bits.sum())
        pattern = tuple(obs.astype(int))
        sector = self._best_sector(obs)

        # --------------------------------------------------------------
        # 1. Approach bonus: more sonar evidence before push
        # --------------------------------------------------------------
        if sonar_count > self._prev_sonar_count and not enable_push:
            delta = sonar_count - self._prev_sonar_count
            r += self.approach_scale * delta / self.reward_scale

        # --------------------------------------------------------------
        # 2. Tracking bonus for blinking / temporarily invisible box
        # --------------------------------------------------------------
        # If box is visible now, refresh remembered direction.
        if sector is not None:
            self._last_seen_dir = sector
            self._steps_since_seen = 0
        else:
            self._steps_since_seen += 1

        # If box vanished recently, reward actions consistent with last seen side.
        if (
            not enable_push
            and self._last_seen_dir is not None
            and 0 < self._steps_since_seen <= self.track_memory_steps
        ):
            if self._action_matches_direction(action, self._last_seen_dir):
                r += self.track_scale / self.reward_scale

        # --------------------------------------------------------------
        # 3. Push progress
        # --------------------------------------------------------------
        if enable_push and not stuck_bit and not done:
            r += self.push_scale / self.reward_scale

        # --------------------------------------------------------------
        # 4. Push alignment shaping
        # --------------------------------------------------------------
        if enable_push:
            if action == "FW" and not stuck_bit:
                r += self.push_forward_bonus / self.reward_scale

            # Blind / weakly informed pushing penalty before terminal success
            # Encourage agent to avoid random pushing when frontal evidence is poor.
            front_near = int(obs[0]) if len(obs) >= 1 else 0
            front_far = int(obs[8]) if len(obs) >= 9 else 0
            if action == "FW" and (front_near + front_far + ir_bit == 0) and not done:
                r -= self.blind_push_penalty / self.reward_scale

        # --------------------------------------------------------------
        # 5. Stuck penalty + boundary penalty
        # --------------------------------------------------------------
        if stuck_bit:
            self._consec_stuck += 1
            r -= self.stuck_penalty / self.reward_scale

            if not enable_push:
                self._steps_at_boundary += 1
            else:
                self._steps_at_boundary = 0
        else:
            self._consec_stuck = 0
            self._steps_at_boundary = 0

        if self._steps_at_boundary >= self.boundary_threshold and not enable_push:
            extra = (
                self.boundary_penalty
                * (self._steps_at_boundary - self.boundary_threshold + 1)
                / self.reward_scale
            )
            r -= extra

        # --------------------------------------------------------------
        # 6. Recovery bonus: escaped from stuck
        # --------------------------------------------------------------
        if self._prev_stuck == 1 and stuck_bit == 0:
            r += self.recovery_bonus / self.reward_scale

        # --------------------------------------------------------------
        # 7. Forward movement bonus while searching
        # --------------------------------------------------------------
        if action == "FW" and not stuck_bit and not enable_push:
            r += self.forward_bonus / self.reward_scale

        # --------------------------------------------------------------
        # 8. Spin penalty
        # --------------------------------------------------------------
        ROTATE = {"L45", "L22", "R22", "R45"}
        if action in ROTATE and not enable_push:
            self._consec_rotations += 1
        else:
            self._consec_rotations = 0

        if self._consec_rotations > self.spin_threshold and not enable_push:
            r -= (
                self.spin_penalty
                * (self._consec_rotations - self.spin_threshold)
                / self.reward_scale
            )

        # --------------------------------------------------------------
        # 9. Anti-oscillation penalty: L-R-L-R behavior
        # --------------------------------------------------------------
        if (
            not enable_push
            and self._prev_action is not None
            and self._is_opposite_turn(self._prev_action, action)
        ):
            r -= self.oscillation_penalty / self.reward_scale

        # --------------------------------------------------------------
        # 10. IR progress bonus
        # --------------------------------------------------------------
        if ir_bit == 1 and self._prev_ir == 0 and not enable_push:
            r += self.ir_bonus / self.reward_scale

        # --------------------------------------------------------------
        # 11. Stability bonus for short consistent detections
        # --------------------------------------------------------------
        # Reward brief consistency, but not endless persistence.
        current_sonar_pattern = tuple(sonar_bits.astype(int))
        if sonar_count > 0 and current_sonar_pattern == self._prev_pattern:
            self._stable_detection_steps += 1
            if 1 <= self._stable_detection_steps <= 3 and not enable_push:
                r += self.stability_bonus / self.reward_scale
        else:
            self._stable_detection_steps = 0

        # --------------------------------------------------------------
        # 12. Persistence penalty: likely staring at wall
        # --------------------------------------------------------------
        # If same detection repeats too long without push/attachment,
        # discourage continuing the same behavior.
        if current_sonar_pattern == self._prev_pattern and sonar_count > 0 and not enable_push:
            self._same_pattern_steps += 1
        else:
            self._same_pattern_steps = 0

        if self._same_pattern_steps >= self.persistence_threshold and not enable_push:
            extra = (
                self.persistence_penalty
                * (self._same_pattern_steps - self.persistence_threshold + 1)
                / self.reward_scale
            )
            r -= extra

        # --------------------------------------------------------------
        # 13. Novelty bonus
        # --------------------------------------------------------------
        if not enable_push:
            sonar_pattern_only = tuple(sonar_bits.astype(int))
            if sonar_pattern_only not in self._seen_patterns:
                self._seen_patterns.add(sonar_pattern_only)
                r += self.explore_scale / self.reward_scale

        # --------------------------------------------------------------
        # 14. Efficiency bonus on successful termination
        # --------------------------------------------------------------
        # In your trainer, the shaped reward function receives raw_reward already
        # divided by reward_scale, so success detection must still use the scaled
        # threshold. Your original code used raw_reward >= 1.0 for scale=10. :contentReference[oaicite:1]{index=1}
        if done and raw_reward >= 1.0:
            efficiency = max(0.0, 1.0 - (self._step_count / self.max_steps))
            r += self.efficiency_bonus * efficiency / self.reward_scale

        # --------------------------------------------------------------
        # Update internal state for next step
        # --------------------------------------------------------------
        self._prev_sonar_count = sonar_count
        self._prev_ir = ir_bit
        self._prev_stuck = stuck_bit
        self._prev_pattern = current_sonar_pattern
        self._prev_action = action

        return r