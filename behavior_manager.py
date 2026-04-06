from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FIND     = "find"
PUSH     = "push"
UNWEDGE  = "unwedge"
BEHAVIORS = (FIND, PUSH, UNWEDGE)


@dataclass
class BehaviorManagerConfig:
    push_linger_steps:       int   = 5
    unwedge_linger_steps:    int   = 5
    attach_reward_threshold: float = 90.0
    sticky_push:             bool  = True
    activate_push_on_ir:     bool  = False


class BehaviorManager:
    """Paper-style behavior arbitration for OBELIX.

    Priority order (highest first):
        unwedge > push > find

    Caller contract (must be followed every step):
        behavior  = manager.current_behavior(obs)       # read before acting
        action    = policy[behavior](obs)
        next_obs, raw_reward, done = env.step(action)
        manager.update(next_obs, raw_reward, done)      # write after stepping
        # next iteration uses next_obs

    Notes
    -----
    Push activation
        Attachment is most reliably detected from the reward spike on first
        contact (raw_reward >= attach_reward_threshold, default 90.0).
        Optionally, setting activate_push_on_ir=True also activates push
        when obs[16] (IR bit) first fires — useful if reward spike is delayed.

    sticky_push (default True)
        When True, push stays active until episode reset once attachment is
        detected.  The env latches enable_push internally, so this mirrors the
        simulator's own behaviour.  When False, push deactivates if the IR bit
        stays low for push_linger_steps consecutive steps after activation.

    Push linger for sticky_push=False (FIX applied)
        The push_timer is now set to push_linger_steps on the FIRST step that
        IR drops to 0 after attachment, not refreshed while IR is high.
        This means push stays active for exactly push_linger_steps steps after
        IR contact is lost, which is the intended linger behaviour.

    Unwedge linger
        Once a stuck event fires, unwedge stays active for unwedge_linger_steps
        steps even after obs[17] returns to 0 — gives the robot time to clear
        the obstacle before resuming find or push.
    """

    def __init__(self, config: Optional[BehaviorManagerConfig] = None):
        self.config = config or BehaviorManagerConfig()
        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        """Reset all state.  Call at every episode start.

        Parameters
        ----------
        obs : array-like, optional
            Initial observation.  If provided and obs[17]==1 (stuck at reset),
            the unwedge timer is pre-armed so the first action is already
            correct.
        """
        self._push_active      = False
        self._push_timer       = 0
        self._unwedge_timer    = 0
        self._prev_ir          = False

        if obs is not None:
            self._prev_ir = self._has_ir(obs)
            if self._is_stuck(obs):
                self._unwedge_timer = self.config.unwedge_linger_steps

    def current_behavior(self, obs: np.ndarray) -> str:
        """Return the currently active behavior given the latest observation.

        Must be called with the observation BEFORE env.step() for the current
        timestep (see caller contract in class docstring).
        """
        if self._is_stuck(obs) or self._unwedge_timer > 0:
            return UNWEDGE
        if self._push_active or self._push_timer > 0:
            return PUSH
        return FIND

    def update(self, obs: np.ndarray, raw_reward: float, done: bool) -> None:
        """Update internal state from the result of env.step().

        Must be called with next_obs and raw_reward AFTER env.step() and
        BEFORE the next call to current_behavior() (see caller contract).

        Parameters
        ----------
        obs        : next observation returned by env.step()
        raw_reward : raw (unscaled) reward returned by env.step()
        done       : episode termination flag
        """
        if done:
            # Clear internal state defensively so a missed external reset()
            # cannot leak the previous episode's latent phase into the next one.
            self.reset()
            return

        stuck = self._is_stuck(obs)
        ir_on = self._has_ir(obs)
        ir_rising = ir_on and not self._prev_ir

        # ── Push state machine ────────────────────────────────────────────────
        attach_event = (
            raw_reward >= self.config.attach_reward_threshold
            or (self.config.activate_push_on_ir and ir_rising)
        )

        if attach_event:
            self._push_active = True
            if not self.config.sticky_push:
                self._push_timer = self.config.push_linger_steps

        elif self._push_active and not self.config.sticky_push:
            if ir_on:
                pass
            elif self._prev_ir and not ir_on:
                self._push_timer = self.config.push_linger_steps
            elif self._push_timer > 0:
                self._push_timer -= 1
                if self._push_timer == 0:
                    self._push_active = False
            else:
                self._push_active = False

        # ── Unwedge state machine ─────────────────────────────────────────────
        if stuck:
            self._unwedge_timer = self.config.unwedge_linger_steps
        elif self._unwedge_timer > 0:
            self._unwedge_timer -= 1

        self._prev_ir = ir_on

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _has_ir(obs: np.ndarray) -> bool:
        return bool(np.asarray(obs).reshape(-1)[16])

    @staticmethod
    def _is_stuck(obs: np.ndarray) -> bool:
        return bool(np.asarray(obs).reshape(-1)[17])
