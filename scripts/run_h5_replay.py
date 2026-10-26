"""Offline RBY1A+Sharpa to CRX replay review.

Transforms are xyz + fixed-axis roll/pitch/yaw (radians). Source hand poses
are moved into the environment using the source base, then transformed by
one common scene transform and expressed in each destination robot base.
No ROS or hardware execution is available in this review tool.
Requires numpy, scipy, h5py, pyyaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from h5_replay_geometry import UrdfModel, solve, transform
from scipy.spatial.transform import Rotation
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


SideUrdfModel = UrdfModel


def load_h5_reference(path: str, reference_path=None, trim_reset=False) -> dict:
    with h5py.File(path, "r") as f:
        qpos = f["qpos"][:].astype(np.float64)  # (T, H, 29)
        urdf_name = [s.decode("utf-8") for s in np.asarray(f["urdf_name"])]
        is_rhand = np.asarray(f["is_rhand"], dtype=np.uint8).astype(bool)
        fps = f["fps"][:].astype(np.float64)
        if "robot_base" in f:
            if f.attrs.get("robot_base_frame", "base") != "base":
                raise ValueError("robot_base must describe the URDF base link")
            base_pos = f["robot_base/translations"][:]
            base_quat = f["robot_base/orientations_xyzw"][:]
        else:
            if reference_path is None:
                raise ValueError("Missing robot_base: provide --reference for legacy rollout")
            with h5py.File(reference_path, "r") as ref:
                offsets = ref["base_translation"][:]
                if not np.allclose(offsets, offsets[0]):
                    raise ValueError("Reference has different base translations per hand")
                base_pos = np.tile(offsets[0], (len(qpos), 1))
                base_quat = np.tile([0., 0., 0., 1.], (len(qpos), 1))
            print("Using reference base translation and configured identity rotation (inferred)")
        objects = {}
        if "object_tracks" in f:
            for key, group in f["object_tracks"].items():
                objects[key] = (group["translations"][:], group["orientations_xyzw"][:])
    if qpos.ndim != 3 or qpos.shape[2] != 29 or len(qpos) < 3:
        raise ValueError("Expected qpos (T,H,29), T >= 3")
    if len(is_rhand) != qpos.shape[1] or len(set(is_rhand.tolist())) != len(is_rhand):
        raise ValueError("Side metadata mismatch or duplicate sides")
    if len(urdf_name) != qpos.shape[1] or any(Path(n).name != "rby1a_sharpa.urdf" for n in urdf_name):
        raise ValueError("Source URDF metadata does not match RBY1A model")
    if fps.shape != (qpos.shape[1],) or not np.isfinite(fps).all() or np.any(fps <= 0) or not np.allclose(fps, fps[0]):
        raise ValueError("Expected equal, finite positive hand sample rates")
    if base_pos.shape != (len(qpos), 3) or base_quat.shape != (len(qpos), 4):
        raise ValueError("Base pose length/shape mismatch")
    for pos, quat in [(base_pos, base_quat), *objects.values()]:
        if pos.shape != (len(qpos), 3) or quat.shape != (len(qpos), 4):
            raise ValueError("Pose length/shape mismatch")
        if not np.isfinite(pos).all() or not np.isfinite(quat).all() or not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-3):
            raise ValueError("Invalid pose values/quaternions")
    if not np.isfinite(qpos).all():
        raise ValueError("Non-finite qpos")
    if trim_reset:
        if reference_path is None:
            raise ValueError("--trim-reset-last requires --reference")
        with h5py.File(reference_path, "r") as ref:
            if not np.allclose(qpos[-1], ref["qpos"][0], atol=1e-7):
                raise ValueError("Last qpos does not match reference initial reset state")
        qpos, base_pos, base_quat = qpos[:-1], base_pos[:-1], base_quat[:-1]
        objects = {k: (pos[:-1], quat[:-1]) for k, (pos, quat) in objects.items()}
        print("Removed confirmed reset final frame")
    base = np.tile(np.eye(4), (len(qpos), 1, 1))
    base[:, :3, 3] = base_pos
    base[:, :3, :3] = Rotation.from_quat(base_quat).as_matrix()
    return {"qpos": qpos, "urdf_name": urdf_name, "is_rhand": is_rhand, "fps": fps,
            "base_poses": base, "objects": objects}


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
    model = SideUrdfModel(urdf_path)
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
    """Per-frame bounded least-squares IK for the 6 CRX arm joints only.

    Args:
        urdf_path: Path to this repo's crx5ia_sharpa_wave_{side} URDF.
        side: "left" or "right".
        target_poses: (T, 4, 4) target poses for "{side}_hand_C_MC", from RBY1A FK.
        hand_qpos: (T, 22) Sharpa finger angles for this side (passthrough; only needed
            to fill out a valid full-model qpos vector -- they do not affect the arm's
            own FK since the hand is downstream of hand_C_MC, not between it and the arm).
        seed_arm_qpos: (6,) initial guess for frame 0 (e.g. this side's initial_qpos arm
            slice); each later frame is seeded from the previous frame's solution.
        max_iters bounds function evaluations per attempt; pos_tol/rot_tol define convergence.
        damping is retained for call compatibility and unused by the bounded solver.

    Returns:
        Tuple of (arm_qpos (T, 6), per_frame_diagnostics) where diagnostics has columns
        [final_pos_error_m, final_rot_error_rad, function_evaluations, near_joint_limit].
    """
    return solve(SideUrdfModel(urdf_path), side, target_poses,
                 hand_qpos, seed_arm_qpos, max_nfev=max_iters, pos_tol=pos_tol, rot_tol=rot_tol)


def retarget_hand(
    ref: dict,
    rby1a_urdf: str,
    args,
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
    if args.side != "both" and not np.any(is_rhand == (args.side == "right")):
        raise ValueError("Requested side not present in input")

    crx_cfg = {s: load_crx_side_config(s) for s in ("left", "right")}
    qpos56 = np.tile(
        np.concatenate([crx_cfg["left"]["initial_qpos"], crx_cfg["right"]["initial_qpos"]]),
        (n_frames, 1),
    )  # (T, 56), unused side(s) stay at their home pose the whole time

    retargeted_sides = []
    diagnostics = {}
    for h in range(qpos.shape[1]):
        side = "right" if bool(is_rhand[h]) else "left"
        if args.side != "both" and side != args.side:
            continue
        retargeted_sides.append(side)
        offset = 0 if side == "left" else SIDE_DOF

        rby1a_arm = qpos[:, h, :RBY1A_ARM_DOF]
        sharpa_hand = qpos[:, h, RBY1A_ARM_DOF:RBY1A_ARM_DOF + RBY1A_HAND_DOF]

        if args.clip_hand_tolerance > 0:
            hand_model = SideUrdfModel(crx_cfg[side]["urdf_path"])
            hand_indices = [hand_model.joint_index(side + "_" + suffix) for suffix in SHARPA_SUFFIXES]
            hand_bounds = hand_model.joint_limits[hand_indices]
            clipped = np.clip(sharpa_hand, hand_bounds[:, 0], hand_bounds[:, 1])
            correction = float(np.max(np.abs(clipped - sharpa_hand)))
            if correction > args.clip_hand_tolerance:
                raise ValueError("Hand limit violation exceeds explicit clipping tolerance: " + str(correction))
            print(side, "hand limit clipping: max correction rad", correction,
                  "values changed", int(np.count_nonzero(clipped != sharpa_hand)))
            sharpa_hand = clipped
        target_poses = rby1a_hand_wrist_poses(rby1a_urdf, side, rby1a_arm)
        # One scene transform for all hands/objects; each CRX has its own base.
        target_poses = np.linalg.inv(transform(getattr(args, side + "_base"))) @ transform(args.scene_transform) @ ref["base_poses"] @ target_poses
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
    """Report geometric IK checks only; this does not certify hardware execution."""
    ok = True
    for side, diag in result["diagnostics"].items():
        pos_err, rot_err, iters, hit_limit = diag[:, 0], diag[:, 1], diag[:, 2], diag[:, 3]
        n_hit_limit = int(hit_limit.sum())
        n_no_converge = int(((pos_err >= 1e-4) | (rot_err >= 1e-3)).sum())
        print(
            f"[{side}] frames={len(diag)} "
            f"pos_err(m): mean={pos_err.mean():.3e} max={pos_err.max():.3e}  "
            f"rot_err(rad): mean={rot_err.mean():.3e} max={rot_err.max():.3e}  "
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a ConTrack GRAB h5 "
                        "(reference or play.py --export_rollout output)")
    parser.add_argument("--rby1a-urdf", default=str(Path.home() / "ConTrack" / "assets" / "urdf" / "rby1a_sharpa.urdf"),
                        help="Path to ConTrack's assets/urdf/rby1a_sharpa.urdf on this machine")
    parser.add_argument("--out", default=None, help="Where to save the retargeted qpos56 "
                        ".npz (default: alongside --data)")
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Unavailable in this offline version; always rejects execution.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Legacy flag; cannot override the disabled execution path.")
    parser.add_argument("--crx-namespace", default="crx5ia")
    parser.add_argument("--sharpa-namespace", default="sharpa")
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--clip-hand-tolerance", type=float, default=0., help="Explicit maximum finger correction in radians; 0 preserves raw angles")
    parser.add_argument("--time-scale", type=float, default=1., help="Duration multiplier >= 1; does not certify acceleration/collision safety")
    parser.add_argument("--search-placement", action="store_true", help="Offline single-arm 9-candidate scene placement search; not physical calibration")
    parser.add_argument("--reference", help="Reference H5 for legacy base pose/reset verification")
    parser.add_argument("--trim-reset-last", action="store_true")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--scene-transform", nargs=6, type=float, default=[0.] * 6, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    parser.add_argument("--left-base", nargs=6, type=float)
    parser.add_argument("--right-base", nargs=6, type=float)
    args = parser.parse_args()
    if not np.isfinite(args.clip_hand_tolerance) or args.clip_hand_tolerance < 0:
        parser.error("--clip-hand-tolerance must be finite and nonnegative")
    if not np.isfinite(args.time_scale) or args.time_scale < 1:
        parser.error("--time-scale must be finite and >= 1")
    if args.execute:
        parser.error("Execution disabled in this offline review version: transition, collision and controller checks still required")
    for values in (args.scene_transform, args.left_base, args.right_base):
        if values is not None and not np.isfinite(values).all():
            parser.error("Transforms must be finite; angles in radians")

    if not Path(args.rby1a_urdf).is_file():
        parser.error(f"--rby1a-urdf not found: {args.rby1a_urdf}")

    if args.side == "both" and (args.left_base is None or args.right_base is None):
        parser.error("Bimanual replay needs explicit --left-base and --right-base xyz rpy in one scene frame")
    for side in ("left", "right"):
        if getattr(args, side + "_base") is None:
            setattr(args, side + "_base", [0.] * 6)
    ref = load_h5_reference(args.data, args.reference, args.trim_reset_last)
    print(f"Loaded {args.data}: T={ref['qpos'].shape[0]} H={ref['qpos'].shape[1]} "
          f"fps={ref['fps']}")

    if args.search_placement:
        if args.side == "both":
            parser.error("Placement search currently requires one explicit --side")
        h = int(np.flatnonzero(ref["is_rhand"] == (args.side == "right"))[0])
        cfg = load_crx_side_config(args.side)
        source = ref["base_poses"][0] @ rby1a_hand_wrist_poses(args.rby1a_urdf, args.side, ref["qpos"][:1, h, :7])[0]
        model = SideUrdfModel(cfg["urdf_path"])
        indices = [model.joint_index("J" + str(i)) for i in range(1, 7)]
        sampled = dict(ref)
        keep = np.unique(np.linspace(0, len(ref["qpos"]) - 1, 12).astype(int))
        sampled["qpos"] = ref["qpos"][keep]
        sampled["base_poses"] = ref["base_poses"][keep]
        best = None
        for j1 in (-.5, 0., .5):
            for j2 in (-.3, 0., .3):
                full = np.zeros(model.dof)
                arm = cfg["initial_qpos"][:6].copy()
                arm[:2] += [j1, j2]
                full[indices] = arm
                placement = transform(getattr(args, args.side + "_base")) @ model.frame_pose(args.side + "_hand_C_MC", full) @ np.linalg.inv(source)
                args.scene_transform = np.r_[placement[:3, 3], Rotation.from_matrix(placement[:3, :3]).as_euler("xyz")].tolist()
                trial = retarget_hand(sampled, args.rby1a_urdf, args)
                diag = trial["diagnostics"][args.side]
                failed = int(((diag[:, 0] >= 1e-4) | (diag[:, 1] >= 1e-3)).sum())
                score = (failed, float(diag[:, 0].max() + .2 * diag[:, 1].max()))
                print("placement candidate", j1, j2, "score", score, flush=True)
                if best is None or score < best[0]:
                    best = (score, args.scene_transform.copy())
        args.scene_transform = best[1]
        print("Selected scene transform xyz/rpy:", args.scene_transform, flush=True)
    result = retarget_hand(ref, args.rby1a_urdf, args)
    result["fps"] /= args.time_scale
    print(f"Retargeted side(s): {result['sides']}")
    print_diagnostics(result)
    for side in result["sides"]:
        offset = 0 if side == "left" else SIDE_DOF
        arm = result["qpos56"][:, offset:offset + ARM_DOF]
        speed = np.diff(arm, axis=0) * result["fps"]
        accel = np.diff(speed, axis=0) * result["fps"]
        import xml.etree.ElementTree as ET
        joints = {j.get("name"): j for j in ET.parse(result["crx_cfg"][side]["urdf_path"]).getroot().findall("joint")}
        velocity_limits = np.array([float(joints["J" + str(i)].find("limit").get("velocity")) for i in range(1, 7)])
        side_q = result["qpos56"][:, offset:offset + SIDE_DOF]
        names = result["crx_cfg"][side]["actuated_joints"]
        bounds = np.array([[float(joints[name].find("limit").get(k)) for k in ("lower", "upper")] for name in names])
        invalid = np.any((side_q < bounds[:, 0] - 1e-7) | (side_q > bounds[:, 1] + 1e-7), axis=0)
        print(side, "joints outside position limits", [name for name, bad in zip(names, invalid) if bad])
        all_vmax = np.array([float(joints[name].find("limit").get("velocity")) for name in names])
        all_ratio = float(np.max(np.abs(np.diff(side_q, axis=0) * result["fps"]) / all_vmax))
        print(side, "all joints speed / URDF limit max ratio", all_ratio)
        ratio = float(np.max(np.abs(speed) / velocity_limits))
        print(side, "arm speed / URDF velocity limit max ratio", ratio,
              "minimum total duration multiplier for arm speed only", max(1., args.time_scale * ratio))
        print(side, "max arm speed rad/s", np.abs(speed).max(axis=0),
              "max arm acceleration rad/s^2", np.abs(accel).max(axis=0))

    out_path = Path(args.out) if args.out else Path(args.data).with_suffix(".crx_retarget.npz")
    extra = {}
    for key, (pos, quat) in ref["objects"].items():
        obj = np.tile(np.eye(4), (len(pos), 1, 1))
        obj[:, :3, 3], obj[:, :3, :3] = pos, Rotation.from_quat(quat).as_matrix()
        extra["object_" + key + "_scene_poses"] = transform(args.scene_transform) @ obj
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, qpos56=result["qpos56"], fps=result["fps"], sides=result["sides"],
                        time_scale=args.time_scale, clip_hand_tolerance=args.clip_hand_tolerance, scene_transform=args.scene_transform, left_base=args.left_base, right_base=args.right_base,
                        **{side + "_diagnostics": diag for side, diag in result["diagnostics"].items()}, **extra)
    print(f"Saved retargeted qpos56 to {out_path}")

    print("Offline review only; output has not been collision-checked or approved for hardware.")


if __name__ == "__main__":
    main()
