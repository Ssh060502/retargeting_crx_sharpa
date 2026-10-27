"""Explicit physical bimanual replay, starting with an object placed on the table.

Default is offline plan review only. This is open-loop joint replay, not policy
inference, physical grasp verification, collision certification or an e-stop.
"""
import argparse
import csv
import json
from pathlib import Path
import queue
import sys
import threading
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from h5_mock_playback import MockPlayback
from run_h5_mock import load_motion, keyboard

ARM = np.r_[0:6, 28:34]
HAND = np.r_[6:28, 34:56]


def conservative_rate(q, fps):
    """Additional first-run slowdown, not a manufacturer limit or safety proof."""
    if len(q) < 4:
        raise ValueError("Need at least four frames for acceleration/jerk rate review")
    vmax = np.full(56, .4); vmax[ARM] = .2
    velocity = np.diff(q, axis=0) * fps
    acceleration = np.diff(velocity[:, ARM], axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    factor = max(1., float(np.max(np.abs(velocity) / vmax)),
                 float(np.sqrt(np.max(np.abs(acceleration) / .5))),
                 float(np.cbrt(np.max(np.abs(jerk) / 2.))))
    return fps / (factor * 1.1), factor * 1.1


def object_placement(trajectory, reference):
    with np.load(trajectory, allow_pickle=False) as data, h5py.File(reference, "r") as ref:
        poses = data["object_object_0_scene_poses"]
        if poses.ndim != 3 or poses.shape[1:] != (4,4) or not np.isfinite(poses).all():
            raise ValueError("Missing/invalid object_0 placement")
        pose = poses[0]
        vertices = ref["object_tracks/0/vertices"][:]
        if not np.isfinite(vertices).all():
            raise ValueError("Invalid object mesh")
        world = vertices @ pose[:3,:3].T + pose[:3,3]
        result = {"coordinates": "shared scene frame (configured left-base frame)",
                  "object_origin_xyz_m": pose[:3,3].tolist(),
                  "object_orientation_xyzw": Rotation.from_matrix(pose[:3,:3]).as_quat().tolist(),
                  "object_mesh_bounds_min_m": world.min(0).tolist(),
                  "object_mesh_bounds_max_m": world.max(0).tolist(),
                  "object_mesh_vertex_mean_m": world.mean(0).tolist(),
                  "note": "Object origin is not a directly measurable corner or grasp point. Align actual tool geometry/orientation; no perception correction is provided."}
        if "table_height" in data:
            result["assumed_table_z_m"] = float(data["table_height"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--reference-h5", required=True, help="Reference object mesh for first-pose placement review")
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--transition-seconds", type=float, default=15.)
    parser.add_argument("--out-dir", default="outputs/hammer/real_review")
    args = parser.parse_args()
    if not np.isfinite(args.transition_seconds) or args.transition_seconds < 15.:
        parser.error("Initial physical transition must be at least 15s")
    q, saved_fps, names, lower, upper, model_vmax = load_motion(args.trajectory)
    fps, slowdown = conservative_rate(q, saved_fps)
    placement = object_placement(args.trajectory, args.reference_h5)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plan = {"trajectory": str(Path(args.trajectory).resolve()), "frames": len(q),
            "saved_fps": saved_fps, "execution_fps": fps, "extra_slowdown": slowdown,
            "duration_s": (len(q)-1)/fps, "object_first_pose": placement,
            "unresolved": ["hammer near left hand", "sub-mm mesh/table penetration and measurement uncertainty", "entry/inter-arm/swept collision", "actual mounting and object pose accuracy", "grasp and contact physics"],
            "hardware_approved": False}
    (out/"plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    print("Fingers will follow recorded trajectory, including opening/closing. Start EMPTY; hammer must be on the table at the reviewed initial pose.")
    if not args.run_real:
        print("Plan review only. No ROS imports/connections/commands. Review plan.json and object placement before physical operation."); return
    if not sys.stdin.isatty():
        parser.error("Physical replay requires an interactive local operator terminal")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
    from h5_real_guard import RealGuard
    from retargeting_ros.sharpa_joint import SharpaJointBackend
    guard = RealGuard()
    backend = None
    motion_started = False
    reason = "not started"
    try:
        guard.check()
        backend = SharpaJointBackend(robot_names=names, initial_qpos=q[0], lower=lower, upper=upper,
                                     control_period=.02, startup_timeout=8., target_timeout=.3, strict_stop=True)
        start = backend.get_joint_pos()
        transition_vmax = np.minimum(model_vmax, .3)
        transition_vmax[ARM] = np.minimum(transition_vmax[ARM], .1)
        machine = MockPlayback(q, fps, transition_vmax, args.transition_seconds)
        machine.enter(start, time.monotonic())
        np.savez_compressed(out/"reviewed_start.npz", measured_start=start, first_target=q[0],
                            transition_seconds=machine.enter_duration, trajectory=q, fps=fps)
        for side, offset in (("left",0),("right",28)):
            print(side, "measured arm degrees:", np.rad2deg(start[offset:offset+6]))
            print(side, "first target arm degrees:", np.rad2deg(q[0,offset:offset+6]))
            print(side, "first target hand degrees:", np.rad2deg(q[0,offset+6:offset+28]))
        print("Initial measured-to-first transition seconds:",machine.enter_duration)
        print("Review the ENTIRE entry path, tool placement, empty hands and local hardware stop with the responsible operator.")
        print("Known close hammer/left-hand motion and table uncertainty are NOT resolved by this program. No force/grasp verification is performed.")
        # No command is sent while waiting. Feedback and physical-stack verification keep running.
        confirmed = queue.Queue()
        def confirm():
            try: confirmed.put(input("Type START REAL TABLETOP REPLAY to execute this reviewed plan; anything else cancels: ").strip())
            except EOFError: confirmed.put("")
        threading.Thread(target=confirm,daemon=True).start()
        while confirmed.empty():
            guard.check(); backend.get_joint_pos(); time.sleep(.05)
        if confirmed.get() != "START REAL TABLETOP REPLAY":
            reason = "cancelled"; return
        guard.check()
        actual = backend.get_joint_pos()
        if np.max(np.abs(actual-start)) > .01:
            raise RuntimeError("Robot changed since plan review; rerun and review new start. No trajectory command sent.")
        machine.enter(actual, time.monotonic())
        commands = queue.Queue()
        threading.Thread(target=keyboard,args=(commands,),daemon=True).start()
        print("Physical replay starting. p+Enter pause, q+Enter stop. Resume after a physical pause is DISABLED: inspect tool state first.")
        began = time.monotonic(); deadline=began; last_send=began; last_command=actual.copy()
        last_phase=None; settled_since=None
        vmax=np.minimum(model_vmax,.4); vmax[ARM]=np.minimum(vmax[ARM],.2)
        with (out/"tracking.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream)
            writer.writerow(["elapsed_s","phase","trajectory_s","arm_error_rad","hand_error_rad"]+["target_"+str(i) for i in range(56)]+["actual_"+str(i) for i in range(56)])
            while True:
                now=time.monotonic();guard.check()
                if now-deadline > .1:raise RuntimeError("Missed publication deadline; no catch-up burst")
                while not commands.empty():
                    command=commands.get_nowait()
                    if command=="q":reason="user stop";machine.phase="stopped";break
                    if command=="p" and machine.phase!="paused":
                        machine.pause(now);backend.pause_tracking()
                    if command=="r":print("Resume disabled for physical contact replay; stop, inspect and replan.",flush=True)
                if machine.phase=="stopped":break
                actual=backend.get_joint_pos()
                if last_phase!=machine.phase:
                    print("Phase:",machine.phase,flush=True);last_phase=machine.phase
                if machine.phase!="paused":
                    target=machine.command(now)
                    if np.max(np.abs(target-last_command)/max(now-last_send,.02)/vmax)>1.05:
                        raise RuntimeError("Command step exceeds first-run speed budget")
                    arm_error=float(np.max(np.abs(actual[ARM]-target[ARM])))
                    hand_error=float(np.max(np.abs(actual[HAND]-target[HAND])))
                    if arm_error>.15 or hand_error>.25:
                        raise RuntimeError("Tracking error threshold exceeded; may indicate lag/contact. Stopping, not forcing grasp.")
                    motion_started = True
                    backend.execute(target);last_send=now;last_command=target.copy()
                    writer.writerow([now-began,machine.phase,machine.elapsed,arm_error,hand_error,*target,*actual]);stream.flush()
                    if machine.phase=="settling":
                        settled=arm_error<.02 and hand_error<.08
                        settled_since=now if settled and settled_since is None else (settled_since if settled else None)
                        if settled_since is not None and now-settled_since>.5:
                            reason="joint replay completed and settled; task success not verified";break
                        if now-machine.settle_start>10.:raise RuntimeError("Final joint state did not settle")
                deadline=now+.02;time.sleep(max(0.,deadline-time.monotonic()))
    except KeyboardInterrupt:
        reason="Ctrl+C"
    except Exception:
        reason="error";raise
    finally:
        try:
            if backend is not None:backend.close(send_hold=motion_started)
        finally:
            guard.close()
            print("Stopped:",reason,"Records:",out)
            print("Software hold/close is not an emergency stop and does not guarantee that an object remains gripped.")


if __name__=="__main__":
    main()
