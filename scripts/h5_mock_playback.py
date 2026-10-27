"""ROS-free trajectory sampling and state machine for software mock checks."""
import numpy as np


def smoothstep(u):
    u = float(np.clip(u, 0., 1.))
    return u ** 3 * (10. - 15. * u + 6. * u ** 2)


def transition_duration(start, target, velocity_limits, minimum=5.):
    delta = np.abs(target - start)
    duration = max(float(minimum), float(np.max(1.875 * delta / velocity_limits)))
    arm = np.r_[0:6, 28:34]
    acceleration = np.tile([3., 3., 4.5, 8., 8., 4.5], 2)
    jerk = np.tile([30., 30., 45., 80., 80., 45.], 2)
    duration = max(duration, float(np.max(np.sqrt(5.774 * delta[arm] / acceleration))),
                   float(np.max(np.cbrt(60. * delta[arm] / jerk))))
    return duration * 1.05


def sample_trajectory(q, fps, elapsed):
    index = np.clip(elapsed * fps, 0., len(q) - 1.)
    lo = int(np.floor(index)); hi = min(lo + 1, len(q) - 1)
    return q[lo] * (1. - (index - lo)) + q[hi] * (index - lo)


class MockPlayback:
    """Wall-clock playback; paused time is never counted as trajectory progress."""
    def __init__(self, q, fps, velocity_limits, minimum_transition=5.):
        self.q, self.fps, self.velocity_limits = q, fps, velocity_limits
        self.minimum_transition = minimum_transition
        self.duration = (len(q) - 1) / fps
        self.phase = "new"
        self.elapsed = 0.

    def enter(self, actual, now):
        self.start = np.array(actual, copy=True)
        self.target = sample_trajectory(self.q, self.fps, self.elapsed)
        self.enter_duration = transition_duration(self.start, self.target, self.velocity_limits, self.minimum_transition)
        self.phase_start, self.phase = now, "entering"

    def pause(self, now):
        if self.phase == "playing":
            self.elapsed = min(self.duration, max(0., now - self.play_start))
        self.phase = "paused"

    def resume(self, actual, now):
        if self.phase != "paused":
            raise ValueError("Resume requires paused state")
        self.enter(actual, now)

    def command(self, now):
        if self.phase == "entering":
            u = (now - self.phase_start) / self.enter_duration
            if u >= 1.:
                self.phase = "playing"
                self.play_start = now - self.elapsed
                return self.target.copy()
            return self.start + (self.target - self.start) * smoothstep(u)
        if self.phase == "playing":
            self.elapsed = min(self.duration, max(0., now - self.play_start))
            if self.elapsed >= self.duration:
                self.phase, self.settle_start = "settling", now
            return sample_trajectory(self.q, self.fps, self.elapsed)
        if self.phase == "settling":
            return self.q[-1].copy()
        raise ValueError("No playback command in state " + self.phase)
