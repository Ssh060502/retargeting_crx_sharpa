"""Retarget a ConTrack GRAB h5 (reference or policy-rollout) qpos trajectory from
RBY1A+Sharpa onto CRX-5iA+Sharpa, and optionally stream it to the real/mock robot.

The h5 schema (qpos (T,H,29), urdf_name (H,), is_rhand (H,), fps (H,)) is produced both
by ConTrack's original GRAB-retargeted reference files and by play.py --export_rollout,
so this script works on either without changes.

Pipeline per active hand side:
  1. FK on the RBY1A+Sharpa URDF (ConTrack asset) using qpos[:, h, 0:7] (arm) to get the
     pose of the Sharpa hand's own root frame "{side}_hand_C_MC" in the RBY1A base frame.
     This frame, not CRX's "{side}_retarget_wrist", is the point that is physically
     identical on both robots (same hand hardware), so it is the correct FK/IK target.
  2. Iterative damped-least-squares IK on the CRX-5iA+Sharpa URDF (this repo's asset,
     which already bakes in the measured hand-mount offset) to find the 6 CRX arm joint
     angles that put ITS OWN "{side}_hand_C_MC" frame at that same pose.
  3. qpos[:, h, 7:29] (Sharpa finger angles) are copied through unchanged -- both robots
     use the identical Sharpa hand, so no hand retargeting is needed, only the arm IK.
  4. The result is assembled into the 56-value [left(6+22), right(6+22)] layout expected
     by retargeting_ros.sharpa_joint.SharpaJointBackend.execute(), with the unused side
     held at its configured initial_qpos the whole time.

By default this is a dry run: it only computes diagnostics and saves the retargeted
qpos to an .npz file. Pass --execute to actually stream it to the ROS backend (requires
bimanual_system.launch.py already running with the real or mock drivers).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

ARM_DOF = 6
HAND_DOF = 22
SIDE_DOF = ARM_DOF + HAND_DOF  # 28
RBY1A_ARM_DOF = 7
RBY1A_HAND_DOF = 22

SHARPA_SUFFIXES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)


class SidePinocchioModel:
    """Thin wrapper mirroring retargeting.core.kinematics.pinocchio_model.RobotPinocchio,
    duplicated here (instead of imported) so this script has no import-time dependency on
    a specific installed package layout for the kinematics-only pieces it needs."""

    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def dof_joint_names(self) -> list[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    def joint_index(self, name: str) -> int:
        return self.dof_joint_names.index(name)

    def frame_index(self, name: str) -> int:
        for i, frame in enumerate(self.model.frames):
            if frame.name == name:
                return i
        raise ValueError(f"frame {name!r} not found; have: {[f.name for f in self.model.frames]}")

    @property
    def joint_limits(self) -> np.ndarray:
        return np.stack([self.model.lowerPositionLimit, self.model.upperPositionLimit], axis=1)

    def frame_pose(self, frame_name: str, qpos: np.ndarray) -> np.ndarray:
        pin.framesForwardKinematics(self.model, self.data, qpos)
        return self.data.oMf[self.frame_index(frame_name)].homogeneous.copy()

    def frame_jacobian(self, frame_name: str, qpos: np.ndarray) -> np.ndarray:
        return pin.computeFrameJacobian(
            self.model, self.data, q=qpos, frame_id=self.frame_index(frame_name),
            reference_frame=pin.LOCAL_WORLD_ALIGNED,
        )


def load_h5_reference(path: str) -> dict:
    with h5py.File(path, "r") as f:
        qpos = f["qpos"][:].astype(np.float64)  # (T, H, 29)
        urdf_name = [s.decode("utf-8") for s in np.asarray(f["urdf_name"])]
        is_rhand = np.asarray(f["is_rhand"], dtype=np.uint8).astype(bool)
        fps = f["fps"][:].astype(np.float64)
    return {"qpos": qpos, "urdf_name": urdf_name, "is_rhand": is_rhand, "fps": fps}


def rby1a_hand_wrist_poses(urdf_path: str, side: str, arm_qpos: np.ndarray) -> np.ndarray:
    """FK the RBY1A+Sharpa URDF to get "{side}_hand_C_MC" poses for every frame.

    Args:
        urdf_path: Path to ConTrack's assets/urdf/rby1a_sharpa.urdf.
        side: "left" or "right".
        arm_qpos: (T, 7) RBY1A arm joint angles for this side, in right_arm_0..6 /
            left_arm_0..6 order (matches ConTrack's RBY1A_*_JOINT_NAMES order, which is
            how the h5's first 7 qpos columns per hand are laid out).

    Returns:
        (T, 4, 4) homogeneous poses of the hand root frame in the RBY1A base frame.
    """
    model = SidePinocchioModel(urdf_path)
    arm_names = [f"{side}_arm_{i}" for i in range(RBY1A_ARM_DOF)]
    arm_idx = [model.joint_index(n) for n in arm_names]
    frame_name = f"{side}_hand_C_MC"

    full_q = np.zeros(model.dof)
    poses = np.empty((arm_qpos.shape[0], 4, 4))
    for t in range(arm_qpos.shape[0]):
        full_q[arm_idx] = arm_qpos[t]
        poses[t] = model.frame_pose(frame_name, full_q)
    return poses


def load_crx_side_config(side: str) -> dict:
    yaml_path = REPO_ROOT / "configs" / "robots" / f"crx5ia_sharpa_wave_{side}.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    urdf_path = str((REPO_ROOT / data["model"]["path"]).resolve())
    return {
        "actuated_joints": list(data["actuated_joints"]),
        "initial_qpos": np.asarray(data["initial_qpos"], dtype=np.float64),
        "urdf_path": urdf_path,
    }


def solve_crx_arm_ik(
    urdf_path: str,
    side: str,
    target_poses: np.ndarray,
    hand_qpos: np.ndarray,
    seed_arm_qpos: np.ndarray,
    max_iters: int = 150,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    damping: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame damped-least-squares IK for the 6 CRX arm joints only.

    Args:
        urdf_path: Path to this repo's crx5ia_sharpa_wave_{side} URDF.
        side: "left" or "right".
        target_poses: (T, 4, 4) target poses for "{side}_hand_C_MC", from RBY1A FK.
        hand_qpos: (T, 22) Sharpa finger angles for this side (passthrough; only needed
            to fill out a valid full-model qpos vector -- they do not affect the arm's
            own FK since the hand is downstream of hand_C_MC, not between it and the arm).
        seed_arm_qpos: (6,) initial guess for frame 0 (e.g. this side's initial_qpos arm
            slice); each later frame is seeded from the previous frame's solution.
        max_iters, pos_tol, rot_tol, damping: IK solver tuning.

    Returns:
        Tuple of (arm_qpos (T, 6), per_frame_diagnostics) where diagnostics has columns
        [final_pos_error_m, final_rot_error_rad, iterations_used, hit_joint_limit].
    """
    model = SidePinocchioModel(urdf_path)
    frame_name = f"{side}_hand_C_MC"
    # Resolve joint-name -> pinocchio q-index explicitly. Declaration order in the YAML
    # (or the URDF) is not guaranteed to match pinocchio's internal q-vector order, so
    # never assume the 6 arm joints land at contiguous indices 0..5.
    arm_names = [f"J{i}" for i in range(1, ARM_DOF + 1)]
    hand_names = [f"{side}_{suffix}" for suffix in SHARPA_SUFFIXES]
    arm_idx = [model.joint_index(n) for n in arm_names]
    hand_idx = [model.joint_index(n) for n in hand_names]
    limits = model.joint_limits[arm_idx]

    n_frames = target_poses.shape[0]
    arm_qpos = np.empty((n_frames, ARM_DOF))
    diagnostics = np.empty((n_frames, 4))

    q_arm = seed_arm_qpos.copy()
    full_q = np.zeros(model.dof)
    for t in range(n_frames):
        full_q[hand_idx] = hand_qpos[t]
        target = target_poses[t]
        target_pos, target_rot = target[:3, 3], target[:3, :3]

        hit_limit = False
        it_used = max_iters
        for it in range(max_iters):
            full_q[arm_idx] = q_arm
            pose = model.frame_pose(frame_name, full_q)
            pos_err = target_pos - pose[:3, 3]
            rot_err = pin.log3(target_rot @ pose[:3, :3].T)
            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
                it_used = it
                break
            jac = model.frame_jacobian(frame_name, full_q)[:, arm_idx]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + (damping**2) * np.eye(6), err)
            q_arm = q_arm + dq
            clipped = np.clip(q_arm, limits[:, 0], limits[:, 1])
            if not np.allclose(clipped, q_arm):
                hit_limit = True
            q_arm = clipped
        else:
            it_used = max_iters

        full_q[arm_idx] = q_arm
        pose = model.frame_pose(frame_name, full_q)
        final_pos_err = float(np.linalg.norm(target_pos - pose[:3, 3]))
        final_rot_err = float(np.linalg.norm(pin.log3(target_rot @ pose[:3, :3].T)))

        arm_qpos[t] = q_arm
        diagnostics[t] = [final_pos_err, final_rot_err, it_used, float(hit_limit)]

    return arm_qpos, diagnostics


def retarget_hand(
    ref: dict,
    rby1a_urdf: str,
) -> dict:
    """Retarget every hand present in the h5 reference to CRX-5iA+Sharpa.

    Returns:
        Dict with "qpos56" (T, 56) in the SharpaJointBackend execute() layout, "sides"
        (which side(s) were actually retargeted), "fps", and "diagnostics" per retargeted
        side.
    """
    qpos = ref["qpos"]  # (T, H, 29)
    is_rhand = ref["is_rhand"]
    n_frames = qpos.shape[0]

    crx_cfg = {s: load_crx_side_config(s) for s in ("left", "right")}
    qpos56 = np.tile(
        np.concatenate([crx_cfg["left"]["initial_qpos"], crx_cfg["right"]["initial_qpos"]]),
        (n_frames, 1),
    )  # (T, 56), unused side(s) stay at their home pose the whole time

    retargeted_sides = []
    diagnostics = {}
    for h in range(qpos.shape[1]):
        side = "right" if bool(is_rhand[h]) else "left"
        retargeted_sides.append(side)
        offset = 0 if side == "left" else SIDE_DOF

        rby1a_arm = qpos[:, h, :RBY1A_ARM_DOF]
        sharpa_hand = qpos[:, h, RBY1A_ARM_DOF:RBY1A_ARM_DOF + RBY1A_HAND_DOF]

        target_poses = rby1a_hand_wrist_poses(rby1a_urdf, side, rby1a_arm)
        seed = crx_cfg[side]["initial_qpos"][:ARM_DOF]
        arm_qpos, diag = solve_crx_arm_ik(
            crx_cfg[side]["urdf_path"], side, target_poses, sharpa_hand, seed,
        )
        qpos56[:, offset:offset + ARM_DOF] = arm_qpos
        qpos56[:, offset + ARM_DOF:offset + SIDE_DOF] = sharpa_hand
        diagnostics[side] = diag

    return {
        "qpos56": qpos56,
        "sides": retargeted_sides,
        "fps": float(np.max(ref["fps"])),
        "diagnostics": diagnostics,
        "crx_cfg": crx_cfg,
    }


def print_diagnostics(result: dict) -> bool:
    """Print IK quality diagnostics. Returns True if the trajectory looks safe to send."""
    ok = True
    for side, diag in result["diagnostics"].items():
        pos_err, rot_err, iters, hit_limit = diag[:, 0], diag[:, 1], diag[:, 2], diag[:, 3]
        n_hit_limit = int(hit_limit.sum())
        n_no_converge = int((iters >= 150).sum())  # == max_iters default in solve_crx_arm_ik
        print(
            f"[{side}] frames={len(diag)} "
            f"pos_err(m): mean={pos_err.mean():.5f} max={pos_err.max():.5f}  "
            f"rot_err(rad): mean={rot_err.mean():.5f} max={rot_err.max():.5f}  "
            f"joint_limit_hits={n_hit_limit}  non_converged={n_no_converge}"
        )
        # CRX-5iA repeatability is far below 1mm, so IK should converge much tighter than
        # this; 1cm / ~3deg here means something is substantively wrong (unreachable
        # target, wrong frame, bad seed), not just solver noise.
        if pos_err.max() > 0.01 or rot_err.max() > 0.05 or n_hit_limit or n_no_converge:
            print(f"[{side}] WARNING: one or more safety thresholds exceeded; do not --execute "
                  f"this trajectory until investigated.")
            ok = False
    return ok


def stream_to_robot(result: dict, args: argparse.Namespace) -> None:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from retargeting_ros.sharpa_joint import SharpaJointBackend

    names = tuple(
        tuple(result["crx_cfg"][side]["actuated_joints"]) for side in ("left", "right")
    )
    limits = np.concatenate([
        _actuated_limits(result["crx_cfg"][side]["urdf_path"], result["crx_cfg"][side]["actuated_joints"])
        for side in ("left", "right")
    ])
    control_period = 1.0 / result["fps"]

    backend = SharpaJointBackend(
        robot_names=names,
        initial_qpos=np.concatenate(
            [result["crx_cfg"]["left"]["initial_qpos"], result["crx_cfg"]["right"]["initial_qpos"]]
        ),
        lower=limits[:, 0],
        upper=limits[:, 1],
        control_period=control_period,
        startup_timeout=args.startup_timeout,
        target_timeout=max(control_period * 4.0, 0.25),
        publish_hz=None,
        crx_namespace=args.crx_namespace,
        sharpa_namespace=args.sharpa_namespace,
    )
    print(f"Connected to real/mock backend. Streaming {result['qpos56'].shape[0]} frames "
          f"at {result['fps']:.2f} fps ({control_period * 1000:.1f} ms/frame). Ctrl+C to stop.")
    try:
        next_t = time.monotonic()
        for frame in result["qpos56"]:
            backend.execute(frame)
            next_t += control_period
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
        print("Replay finished; holding final pose. Ctrl+C or close the launch to stop.")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        backend.close()


def _actuated_limits(urdf_path: str, actuated_joints: list[str]) -> np.ndarray:
    model = SidePinocchioModel(urdf_path)
    idx = [model.joint_index(n) for n in actuated_joints]
    return model.joint_limits[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a ConTrack GRAB h5 "
                        "(reference or play.py --export_rollout output)")
    parser.add_argument("--rby1a-urdf", default=str(Path.home() / "ConTrack" / "assets" / "urdf" / "rby1a_sharpa.urdf"),
                        help="Path to ConTrack's assets/urdf/rby1a_sharpa.urdf on this machine")
    parser.add_argument("--out", default=None, help="Where to save the retargeted qpos56 "
                        ".npz (default: alongside --data)")
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Actually stream to the ROS backend. Without this flag, only "
                        "computes and saves diagnostics (dry run, no ROS needed).")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Allow --execute even if diagnostics look unsafe.")
    parser.add_argument("--crx-namespace", default="crx5ia")
    parser.add_argument("--sharpa-namespace", default="sharpa")
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    args = parser.parse_args()

    if not Path(args.rby1a_urdf).is_file():
        parser.error(f"--rby1a-urdf not found: {args.rby1a_urdf}")

    ref = load_h5_reference(args.data)
    print(f"Loaded {args.data}: T={ref['qpos'].shape[0]} H={ref['qpos'].shape[1]} "
          f"fps={ref['fps']}")

    result = retarget_hand(ref, args.rby1a_urdf)
    print(f"Retargeted side(s): {result['sides']}")
    safe = print_diagnostics(result)

    out_path = Path(args.out) if args.out else Path(args.data).with_suffix(".crx_retarget.npz")
    np.savez_compressed(out_path, qpos56=result["qpos56"], fps=result["fps"], sides=result["sides"])
    print(f"Saved retargeted qpos56 to {out_path}")

    if args.execute:
        if not safe and not args.force:
            print("Refusing to --execute: diagnostics above exceeded safety thresholds. "
                  "Pass --force to override after you have reviewed them.")
            return
        stream_to_robot(result, args)
    else:
        print("Dry run only (no ROS, no hardware touched). Re-run with --execute once you "
              "have reviewed the diagnostics above and bimanual_system.launch.py is running.")


if __name__ == "__main__":
    main()
