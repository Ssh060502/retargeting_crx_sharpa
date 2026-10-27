"""ROS-free small, relative single-arm path for transport commissioning."""
import numpy as np
from scipy.interpolate import make_interp_spline


def build_relative_test(qpos56, side, frames=30, max_delta=.01745329252, leg_seconds=10., rate=50.):
    if side not in ("left", "right"):
        raise ValueError("Select left or right explicitly")
    if not 0 < max_delta <= .02 or leg_seconds < 10. or rate != 50.:
        raise ValueError("Commissioning caps: max_delta <= .02rad, leg >= 10s, rate=50Hz")
    if not 6 <= frames <= 60:
        raise ValueError("Use 6..60 source frames for a short segment")
    qpos56 = np.asarray(qpos56)
    if qpos56.ndim != 2 or qpos56.shape[1] != 56:
        raise ValueError("Expected qpos56 with 56 columns")
    offset = 0 if side == "left" else 28
    arm = np.asarray(qpos56, dtype=float)[:frames, offset:offset+6]
    if arm.ndim != 2 or len(arm) < 6 or arm.shape[1] != 6 or not np.isfinite(arm).all():
        raise ValueError("Need at least six finite arm frames")
    delta = arm - arm[0]
    spline = make_interp_spline(np.linspace(0,1,len(delta)), delta, k=5,
                               bc_type=([(1,np.zeros(6)),(2,np.zeros(6))], [(1,np.zeros(6)),(2,np.zeros(6))]))
    probe = np.linspace(0,1,10001)
    amplitude = float(np.max(np.abs(spline(probe))))
    if amplitude < 1e-7:
        raise ValueError("Selected segment has no measurable motion; select another segment")
    scale = min(1., .98 * max_delta / amplitude)
    # Conservative low planning budgets; not manufacturer-certified limits.
    duration = max(leg_seconds,
                   float(np.max(np.abs(spline(probe,1))*scale))/.008,
                   np.sqrt(float(np.max(np.abs(spline(probe,2))*scale))/.008),
                   np.cbrt(float(np.max(np.abs(spline(probe,3))*scale))/.02)) * 1.1
    count = int(np.ceil(duration*rate))
    outward = spline(np.linspace(0,1,count+1))*scale
    path = np.concatenate([outward, outward[-2::-1]])
    path[0] = path[-1] = 0.
    if np.max(np.abs(path)) > max_delta:
        raise ValueError("Smoothed path exceeds excursion cap")
    return path, rate


def anchor_path(relative, start, lower, upper, margin=.05):
    start = np.asarray(start, dtype=float)
    if start.shape != (6,) or not np.isfinite(start).all():
        raise ValueError("Need six finite measured joints")
    result = start + relative
    if np.any(result < np.asarray(lower)+margin) or np.any(result > np.asarray(upper)-margin):
        raise ValueError("Current pose/path too near joint limits; no command sent")
    return result
