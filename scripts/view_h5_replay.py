"""Passive browser preview of qpos56, object meshes and measured tabletop.

Uses collision meshes as the robot appearance. No ROS imports or command output.
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from h5_replay_geometry import UrdfModel, transform
from check_h5_table_clearance import collision_meshes
from run_h5_replay import load_crx_side_config


def wxyz(pose):
    return np.roll(Rotation.from_matrix(pose[:3, :3]).as_quat(), 1)


def load_preview(trajectory, reference, mappings):
    with np.load(trajectory, allow_pickle=False) as data:
        qpos = data["qpos56"].copy()
        fps = float(data["fps"])
        if qpos.ndim != 2 or qpos.shape[1] != 56 or len(qpos) < 2 or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Expected finite qpos56 (T,56), T >= 2, and positive fps")
        robots = {}
        for side in data["sides"].tolist():
            if side not in ("left", "right") or side in robots:
                raise ValueError("Invalid side metadata")
            cfg = load_crx_side_config(side)
            model = UrdfModel(cfg["urdf_path"])
            meshes, uncovered = collision_meshes(cfg["urdf_path"])
            base_values = data[side + "_base"]
            if base_values.shape != (6,) or not np.isfinite(base_values).all():
                raise ValueError("Invalid base pose")
            robots[side] = dict(model=model, meshes=meshes, base=transform(base_values),
                                indices=[model.joint_index(n) for n in cfg["actuated_joints"]],
                                uncovered=uncovered)
        if not robots:
            raise ValueError("No active robot sides")
        objects = {}
        with h5py.File(reference, "r") as ref:
            for mapping in mappings:
                target, source = mapping.split("=", 1)
                poses = data["object_" + target + "_scene_poses"].copy()
                vertices = ref["object_tracks/" + source + "/vertices"][:]
                faces = ref["object_tracks/" + source + "/faces"][:]
                if poses.shape != (len(qpos), 4, 4) or not np.isfinite(poses).all() or not np.isfinite(vertices).all():
                    raise ValueError("Invalid object pose/mesh data")
                objects[target] = dict(poses=poses, vertices=vertices, faces=faces)
        expected = {k for k in data.files if k.startswith("object_") and k.endswith("_scene_poses")}
        if {"object_" + k + "_scene_poses" for k in objects} != expected:
            raise ValueError("Object mappings must cover all saved object tracks")
    return qpos, fps, robots, objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--reference-h5", required=True)
    parser.add_argument("--object-map", nargs="+", default=["object_0=0"])
    parser.add_argument("--table", nargs=5, type=float, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "Z"))
    parser.add_argument("--port", type=int, default=9219)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--validate-only", action="store_true", help="Load inputs/meshes and exit without opening a server")
    args = parser.parse_args()
    xmin, xmax, ymin, ymax, z = args.table
    if not np.isfinite(args.table).all() or xmin >= xmax or ymin >= ymax:
        parser.error("Invalid table bounds")
    qpos, fps, robots, objects = load_preview(args.trajectory, args.reference_h5, args.object_map)
    print(f"Loaded {len(qpos)} frames, {fps:.6f} fps; duration {(len(qpos)-1)/fps:.2f}s")
    print("Robot display uses collision meshes, not photorealistic visual geometry.")
    print("Passive preview only. No ROS, controllers, physics, or collision certification.")
    if args.validate_only:
        for side, robot in robots.items():
            print(side, "collision meshes", len(robot["meshes"]))
        return
    try:
        import viser
    except ImportError as exc:
        raise SystemExit("Viewer needs the repository's optional viser dependency: .venv/bin/python -m pip install viser") from exc
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (1.7, 1.1, 1.1)
    server.initial_camera.look_at = (.45, -.45, .05)
    server.scene.add_frame("/world", axes_length=.2, axes_radius=.003)
    thickness = .03  # Display thickness only; not a measured table underside.
    vertices = np.array([[x,y,h] for h in (z-thickness,z) for y in (ymin,ymax) for x in (xmin,xmax)])
    faces = np.array([[0,2,1],[1,2,3],[4,5,6],[5,7,6],[0,1,4],[1,5,4],
                      [2,6,3],[3,6,7],[0,4,2],[2,4,6],[1,3,5],[3,7,5]])
    table_handle = server.scene.add_mesh_simple("/table", vertices, faces, color=(150,120,85), opacity=.55)
    handles = {}
    for side, robot in robots.items():
        base = robot["base"]
        server.scene.add_frame("/bases/"+side, position=base[:3,3], wxyz=wxyz(base), axes_length=.12, axes_radius=.003)
        for index, (link, triangles) in enumerate(robot["meshes"]):
            name = f"/robots/{side}/mesh_{index}"
            handles[(side,index)] = server.scene.add_frame(name, show_axes=False)
            verts = triangles.reshape(-1,3)
            server.scene.add_mesh_simple(name+"/geometry", verts, np.arange(len(verts)).reshape(-1,3),
                                         color=(205,215,225) if side=="left" else (140,195,165))
    for name, obj in objects.items():
        obj["handle"] = server.scene.add_frame("/objects/"+name, show_axes=False)
        server.scene.add_mesh_simple("/objects/"+name+"/mesh", obj["vertices"], obj["faces"], color=(225,155,55))
    server.gui.add_markdown("**Offline preview** ? no robot commands. Collision geometry display; not collision validation.")
    playing = server.gui.add_checkbox("Playing", initial_value=False)
    frame = server.gui.add_slider("Frame", min=0, max=len(qpos)-1, step=1, initial_value=0)
    speed = server.gui.add_slider("Playback speed", min=.25, max=4., step=.25, initial_value=1.)
    loop = server.gui.add_checkbox("Loop", initial_value=False)
    table_visible = server.gui.add_checkbox("Show table", initial_value=True)
    # Visibility changes do not modify playback data.
    last, next_step = None, time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if not playing.value:
                next_step = now
            elif now >= next_step:
                if frame.value >= len(qpos)-1:
                    if loop.value:
                        frame.value = 0
                    else:
                        playing.value = False
                else:
                    frame.value += 1
                next_step = now + 1/(fps*float(speed.value))
            state = (int(frame.value), bool(table_visible.value))
            if state != last:
                with server.atomic():
                    # Handles are updated from recorded poses only, with no controller path.
                    for side, robot in robots.items():
                        q = np.zeros(robot["model"].dof)
                        offset = 0 if side=="left" else 28
                        q[robot["indices"]] = qpos[state[0],offset:offset+28]
                        for index, (link, _) in enumerate(robot["meshes"]):
                            pose = robot["base"] @ robot["model"].frame_pose(link,q)
                            h = handles[(side,index)]; h.position = pose[:3,3]; h.wxyz = wxyz(pose)
                    for obj in objects.values():
                        pose = obj["poses"][state[0]]
                        obj["handle"].position = pose[:3,3]; obj["handle"].wxyz = wxyz(pose)
                    table_handle.visible = state[1]
                last = state
            time.sleep(.005)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
