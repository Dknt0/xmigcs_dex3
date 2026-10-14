"""Integer control-tick scheduling; seconds are never converted back to ticks."""
import math
import operator


class TickDecimator:
    def __init__(self, period_ticks, policy_dt, control_dt=None):
        if isinstance(period_ticks, bool):
            raise ValueError("decimation must be a positive integer, not bool")
        try:
            self.period_ticks = operator.index(period_ticks)
        except TypeError as exc:
            raise ValueError("decimation must be a positive integer") from exc
        if self.period_ticks < 1:
            raise ValueError("decimation must be a positive integer")
        if not math.isfinite(policy_dt) or policy_dt <= 0:
            raise ValueError("policy dt must be finite and positive")
        if control_dt is not None and (
            not math.isfinite(control_dt) or control_dt <= 0
            or not math.isclose(policy_dt, control_dt, rel_tol=1e-12, abs_tol=0.0)
        ):
            raise ValueError("policy dt must match the main control-loop dt")
        self.reset()

    def reset(self):
        # Each state entry produces a fresh target on its first control tick.
        self._phase = 0

    def step(self):
        due = self._phase == 0
        self._phase = (self._phase + 1) % self.period_ticks
        return due
