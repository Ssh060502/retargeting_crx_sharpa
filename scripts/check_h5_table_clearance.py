"""Offline mesh clearance against a finite tabletop footprint.

Negative clearance flags geometry below the top surface within its xy footprint.
This is a conservative downward-column test, not complete collision certification.
No ROS, hardware, third-party mesh library, or graphics are used.
"""
import argparse
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from h5_replay_geometry import UrdfModel, transform
from run_h5_replay import load_crx_side_config


def load_stl(path):
    raw = Path(path).read_bytes()
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if len(raw) == 84 + count * 50:
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
            triangles = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].astype(float)
            if not len(triangles) or not np.isfinite(triangles).all():
                raise ValueError("Empty/nonfinite STL: " + str(path))
            return triangles
    vertices = [list(map(float, line.split()[1:])) for line in raw.decode("ascii").splitlines()
                if line.strip().startswith("vertex ")]
    triangles = np.asarray(vertices, dtype=float).reshape(-1, 3, 3)
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("Empty/nonfinite STL: " + str(path))
    return triangles


def minimum_over_footprint(triangles, bounds):
    """Exact triangle-surface minimum z after clipping to a rectangular xy column."""
    xmin, xmax, ymin, ymax = bounds
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    possible = (high[:, 0] >= xmin) & (low[:, 0] <= xmax) & (high[:, 1] >= ymin) & (low[:, 1] <= ymax)
    triangles, low, high = triangles[possible], low[possible], high[possible]
    inside = (low[:, 0] >= xmin) & (high[:, 0] <= xmax) & (low[:, 1] >= ymin) & (high[:, 1] <= ymax)
    best = float(triangles[inside, :, 2].min()) if inside.any() else float("inf")
    for triangle in triangles[~inside]:
        polygon = list(triangle)
        for axis, edge, sign in ((0, xmin, 1), (0, xmax, -1), (1, ymin, 1), (1, ymax, -1)):
            clipped = []
            if not polygon:
                break
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                da, db = sign * (a[axis] - edge), sign * (b[axis] - edge)
                if da >= 0:
                    clipped.append(a)
                if (da >= 0) != (db >= 0):
                    clipped.append(a + (b - a) * da / (da - db))
            polygon = clipped
        if polygon:
            best = min(best, min(point[2] for point in polygon))
    return best


def collision_meshes(urdf):
    root = ET.parse(urdf).getroot()
    meshes, uncovered = [], []
    for link in root.findall("link"):
        collisions = link.findall("collision")
        if not collisions:
            uncovered.append(link.get("name"))
        for collision in collisions:
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Unsupported non-mesh collision geometry: " + link.get("name"))
            filename = mesh.get("filename")
            if "://" in filename:
                raise ValueError("Resolve package mesh URI before checking: " + filename)
            path = (Path(urdf).parent / filename).resolve()
            if path.suffix.lower() != ".stl":
                raise ValueError("Only STL collision meshes supported: " + str(path))
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            origin = collision.find("origin")
            xyz = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            local = load_stl(path) * scale
            pose = transform(np.r_[xyz, rpy])
            local = local @ pose[:3, :3].T + pose[:3, 3]
            meshes.append((link.get("name"), local))
    if not meshes:
        raise ValueError("No collision meshes in " + str(urdf))
    return meshes, uncovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--reference-h5", required=True, help="Object mesh source, not object motion source")
    parser.add_argument("--object-map", nargs="+", default=["object_0=0"], help="rollout object key=reference mesh key")
    parser.add_argument("--table", nargs=5, type=float, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "Z"))
    parser.add_argument("--tolerance", type=float, default=.002, help="Report penetrations deeper than this many meters")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    xmin, xmax, ymin, ymax, z = args.table
    if not np.isfinite([*args.table, args.tolerance]).all() or xmin >= xmax or ymin >= ymax or args.tolerance < 0:
        parser.error("Invalid table bounds/tolerance")
    report = {"scope": "sampled mesh/table-footprint downward-column clearance only",
              "not_checked": ["inter-arm/self/object-robot collisions", "between-frame swept motion", "initial/final transitions", "real calibration accuracy", "support stands and unmodeled obstacles"],
              "table": args.table, "tolerance_m": args.tolerance, "entities": [], "links_without_collision_geometry": {}}
    def record(name, values):
        values = np.asarray(values)
        finite = np.isfinite(values)
        bad = np.flatnonzero(values < -args.tolerance)
        entry = {"name": name, "frames_with_xy_overlap": int(finite.sum()),
                 "minimum_clearance_m": float(values[finite].min()) if finite.any() else None,
                 "worst_frame": int(np.argmin(values)) if finite.any() else None,
                 "penetration_frames": bad.tolist()}
        report["entities"].append(entry)
        if len(bad):
            print(name, "below surface:", len(bad), "frames; min clearance m", entry["minimum_clearance_m"], "worst frame", entry["worst_frame"], flush=True)
    with np.load(args.trajectory, allow_pickle=False) as data:
        qpos, fps = data["qpos56"], float(data["fps"])
        if qpos.ndim != 2 or qpos.shape[1] != 56 or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid trajectory")
        report.update(frames=len(qpos), fps=fps)
        for side in data["sides"].tolist():
            cfg = load_crx_side_config(side)
            model = UrdfModel(cfg["urdf_path"])
            meshes, uncovered = collision_meshes(cfg["urdf_path"])
            report["links_without_collision_geometry"][side] = uncovered
            indices = [model.joint_index(name) for name in cfg["actuated_joints"]]
            base = transform(data[side + "_base"])
            offset = 0 if side == "left" else 28
            clearances = {i: [] for i in range(len(meshes))}
            for frame in qpos:
                q = np.zeros(model.dof); q[indices] = frame[offset:offset + 28]
                for i, (link, triangles) in enumerate(meshes):
                    pose = base @ model.frame_pose(link, q)
                    world = triangles @ pose[:3, :3].T + pose[:3, 3]
                    clearances[i].append(minimum_over_footprint(world, args.table[:4]) - z)
            for i, (link, _) in enumerate(meshes):
                record(side + "/" + link + "/collision_" + str(i), clearances[i])
            print("Checked", side, len(meshes), "collision meshes", flush=True)
        covered = set()
        with h5py.File(args.reference_h5, "r") as reference:
            for mapping in args.object_map:
                target, source = mapping.split("=", 1)
                poses = data["object_" + target + "_scene_poses"]
                group = reference["object_tracks/" + source]
                vertices, faces = group["vertices"][:], group["faces"][:]
                if not np.isfinite(vertices).all() or len(poses) != len(qpos):
                    raise ValueError("Invalid object mesh/pose alignment")
                triangles = vertices[faces]
                values = []
                for pose in poses:
                    world = triangles @ pose[:3, :3].T + pose[:3, 3]
                    values.append(minimum_over_footprint(world, args.table[:4]) - z)
                record("object/" + target, values)
                covered.add("object_" + target + "_scene_poses")
        expected = {key for key in data.files if key.startswith("object_") and key.endswith("_scene_poses")}
        if covered != expected:
            raise ValueError("Object mesh mappings must cover every object pose track")
    report["flagged_entities"] = sum(bool(entry["penetration_frames"]) for entry in report["entities"])
    report["hardware_approved"] = False
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Flagged entities:", report["flagged_entities"], "Report:", path)
    print("Not a full collision check or hardware approval. Near-zero object clearance may represent intended support contact.")


if __name__ == "__main__":
    main()
