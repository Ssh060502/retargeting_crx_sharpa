"""Play a saved bimanual trajectory ONLY on independently verified software mock.

Default is validation only. --run-mock opts into command publication in isolated
ROS_DOMAIN_ID=187. This is not a real-robot replay entry point.
"""
import argparse
import csv
import queue
import sys
import threading
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from h5_replay_geometry import UrdfModel
from h5_mock_playback import MockPlayback
from run_h5_replay import load_crx_side_config


def load_motion(path):
    cfg = {s: load_crx_side_config(s) for s in ("left", "right")}
    names = tuple(tuple(cfg[s]["actuated_joints"]) for s in ("left", "right"))
    lower, upper, vmax = [], [], []
    for side in ("left", "right"):
        model = UrdfModel(cfg[side]["urdf_path"])
        joints = {j.get("name"): j for j in ET.parse(cfg[side]["urdf_path"]).getroot().findall("joint")}
        for name in cfg[side]["actuated_joints"]:
            lo, hi = model.joint_limits[model.joint_index(name)]
            lower.append(lo); upper.append(hi)
            vmax.append(float(joints[name].find("limit").get("velocity")))
    lower, upper, vmax = map(np.asarray, (lower, upper, vmax))
    with np.load(path, allow_pickle=False) as data:
        if "preview_only" in data and bool(data["preview_only"]):
            raise ValueError("Display-only entry preview cannot be executed")
        q, fps = data["qpos56"].copy(), float(data["fps"])
        if data["sides"].tolist() != ["left", "right"]:
            raise ValueError("Mock runner requires both retargeted sides in left/right order")
        if q.ndim != 2 or q.shape[1] != 56 or len(q) < 3 or not np.isfinite(q).all() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid trajectory")
        for side in ("left", "right"):
            diag = data[side + "_diagnostics"]
            if diag.shape != (len(q), 4) or not np.isfinite(diag).all() or np.any(diag[:, 0] >= 1e-4) or np.any(diag[:, 1] >= 1e-3) or np.any(diag[:, 3] != 0):
                raise ValueError("Failed or missing IK validation: " + side)
    if np.any(q < lower - 1e-7) or np.any(q > upper + 1e-7):
        raise ValueError("Trajectory has out-of-bounds joints")
    q = np.clip(q, lower, upper)  # Snap only <= 1e-7 representation noise.
    if np.max(np.abs(np.diff(q, axis=0) * fps) / vmax) > 1. + 1e-7:
        raise ValueError("Trajectory exceeds URDF velocity limits")
    return q, fps, names, lower, upper, vmax


def keyboard(commands):
    for line in sys.stdin:
        command = line.strip().lower()
        if command in ("p", "r", "q"):
            commands.put(command)
        if command == "q":
            return
    commands.put("q")  # Closing stdin stops; never silently continue unattended.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--run-mock", action="store_true")
    parser.add_argument("--transition-seconds", type=float, default=5.)
    parser.add_argument("--log", default="outputs/hammer/mock_replay.csv")
    args = parser.parse_args()
    if not np.isfinite(args.transition_seconds) or args.transition_seconds < 2.:
        parser.error("--transition-seconds must be finite and at least 2")
    q, fps, names, lower, upper, vmax = load_motion(args.trajectory)
    rate = 50.
    print(f"Validated {len(q)} frames; recorded fps={fps:.6f}; playback duration={(len(q)-1)/fps:.2f}s; command rate={rate:g}Hz")
    print("Mock only. Resampling is linear; this checks transport, not contact physics or hardware acceleration feasibility.")
    if not args.run_mock:
        print("Validation only. No ROS imports, connections or commands. Add --run-mock only with isolated software mock running.")
        return
    if not sys.stdin.isatty():
        parser.error("--run-mock needs an interactive terminal for p/r/q controls")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from h5_mock_guard import MockGuard
    from retargeting_ros.sharpa_joint import SharpaJointBackend
    guard = MockGuard()  # Must succeed before backend/command publishers are created.
    backend = None
    commands = queue.Queue()
    reason = "not started"
    path = Path(args.log); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        guard.check()
        backend = SharpaJointBackend(robot_names=names, initial_qpos=np.r_[q[0,:28], q[0,28:]],
                                     lower=lower, upper=upper, control_period=1/rate,
                                     startup_timeout=8., target_timeout=.3, strict_stop=True)
        guard.check()
        actual = backend.get_joint_pos()
        machine = MockPlayback(q, fps, vmax, args.transition_seconds)
        started = time.monotonic(); machine.enter(actual, started)
        print(f"Verified mock. Initial transition {machine.enter_duration:.2f}s.")
        print("Type p + Enter to pause; r + Enter to re-enter and resume; q + Enter or Ctrl+C to stop.")
        threading.Thread(target=keyboard, args=(commands,), daemon=True).start()
        last_phase, last_command, last_send = None, actual.copy(), started
        settled_since = None
        deadline = started
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["elapsed_s", "phase", "trajectory_s", "max_error_rad"] + ["target_"+str(i) for i in range(56)] + ["actual_"+str(i) for i in range(56)])
            while True:
                now = time.monotonic(); guard.check()
                if now - deadline > .2:
                    raise RuntimeError("Playback missed command deadline; refusing catch-up burst")
                while not commands.empty():
                    action = commands.get_nowait()
                    if action == "q":
                        reason = "user stop"; machine.phase = "stopped"; break
                    if action == "p" and machine.phase not in ("paused", "stopped"):
                        machine.pause(now); backend.pause_tracking()
                    if action == "r" and machine.phase == "paused":
                        if not backend.resume_tracking():
                            raise RuntimeError("Feedback unavailable for resume")
                        actual = backend.get_joint_pos()
                        machine.resume(actual, now)
                        last_command, last_send = actual.copy(), now
                        settled_since = None
                if machine.phase == "stopped":
                    break
                actual = backend.get_joint_pos()
                if machine.phase != last_phase:
                    print("Phase:", machine.phase, flush=True); last_phase = machine.phase
                if machine.phase != "paused":
                    target = machine.command(now)
                    dt = max(now-last_send, 1/rate)
                    if np.max(np.abs(target-last_command)/dt/vmax) > 1.05:
                        raise RuntimeError("Command step exceeds URDF velocity budget")
                    backend.execute(target)
                    last_command, last_send = target.copy(), now
                    error = float(np.max(np.abs(actual-target)))
                    if error > .5:
                        raise RuntimeError("Mock tracking error exceeded 0.5rad; stopping")
                    writer.writerow([now-started,machine.phase,machine.elapsed,error,*target,*actual])
                    if machine.phase == "settling":
                        arm = np.r_[0:6,28:34]
                        hand = np.r_[6:28,34:56]
                        settled = np.max(np.abs(actual[arm]-q[-1,arm])) < .02 and np.max(np.abs(actual[hand]-q[-1,hand])) < .05
                        settled_since = now if settled and settled_since is None else (settled_since if settled else None)
                        if settled_since is not None and now-settled_since >= .5:
                            reason = "completed and settled"; break
                        if now-machine.settle_start > 10.:
                            raise RuntimeError("Final pose did not settle within 10s")
                stream.flush()
                # No catch-up loop; every publication is separated by an actual wait.
                deadline = now + 1/rate
                time.sleep(max(0., deadline-time.monotonic()))
    except KeyboardInterrupt:
        reason = "Ctrl+C"; print("Interrupted; requesting measured hold.")
    except Exception:
        reason = "error"; raise
    finally:
        try:
            if backend is not None:
                backend.close()  # Existing backend sends one measured hold when feedback is valid.
        finally:
            guard.close()
            print("Stopped:", reason, "Log:", path)


if __name__ == "__main__":
    main()
