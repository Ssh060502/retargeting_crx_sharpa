"""Offline URDF serial-chain FK and bounded IK; no ROS or mesh loading."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(xyzrpy):
    result = np.eye(4)
    result[:3, 3] = xyzrpy[:3]
    result[:3, :3] = Rotation.from_euler("xyz", xyzrpy[3:]).as_matrix()
    return result


class UrdfModel:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = root.findall("joint")
        self.names = [j.get("name") for j in self.joints if j.get("type") != "fixed"]
        if any(j.get("type") not in ("fixed", "revolute", "prismatic") for j in self.joints):
            raise ValueError("Offline model supports fixed, revolute and prismatic joints only")
        if any(j.find("mimic") is not None for j in self.joints):
            raise ValueError("Mimic joints require explicit model support")
        self.parents = {j.find("child").get("link"): j for j in self.joints}
        self.links = {link.get("name") for link in root.findall("link")}
        self.dof = len(self.names)
        self.joint_limits = np.array([
            [float(j.find("limit").get(k)) for k in ("lower", "upper")]
            for j in self.joints if j.get("type") != "fixed"
        ])
        self._chains = {}

    def joint_index(self, name):
        return self.names.index(name)

    def frame_pose(self, frame, q):
        if frame not in self._chains:
            if frame not in self.links:
                raise ValueError("Unknown URDF link: " + frame)
            chain = []
            link = frame
            while link in self.parents:
                joint = self.parents[link]
                origin = joint.find("origin")
                xyz = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                axis_node = joint.find("axis")
                axis = np.array([1., 0., 0.]) if axis_node is None else np.fromstring(axis_node.get("xyz"), sep=" ")
                index = None if joint.get("type") == "fixed" else self.joint_index(joint.get("name"))
                chain.append((transform(np.r_[xyz, rpy]), axis, index, joint.get("type")))
                link = joint.find("parent").get("link")
            self._chains[frame] = list(reversed(chain))
        pose = np.eye(4)
        for origin, axis, index, kind in self._chains[frame]:
            pose = pose @ origin
            if index is not None:
                motion = np.eye(4)
                if kind == "revolute":
                    motion[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * q[index]).as_matrix()
                else:
                    motion[:3, 3] = axis / np.linalg.norm(axis) * q[index]
                pose = pose @ motion
        return pose


def solve(model, side, targets, fingers, seed, max_nfev=200, pos_tol=1e-4, rot_tol=1e-3):
    arm_idx = [model.joint_index("J" + str(i)) for i in range(1, 7)]
    limits = model.joint_limits[arm_idx]
    q = np.zeros(model.dof)
    frame = side + "_hand_C_MC"
    seed = np.clip(seed, limits[:, 0] + 1e-8, limits[:, 1] - 1e-8)
    previous = seed.copy()
    results, diagnostics = [], []
    for target in targets:
        def residual(arm):
            q[arm_idx] = arm
            pose = model.frame_pose(frame, q)
            return np.r_[pose[:3, 3] - target[:3, 3],
                         .2 * Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec()]
        best = None
        for start in (previous, seed, (limits[:, 0] + limits[:, 1]) / 2):
            fit = least_squares(residual, start, bounds=(limits[:, 0], limits[:, 1]),
                                max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            error = residual(fit.x)
            pos, rot = np.linalg.norm(error[:3]), np.linalg.norm(error[3:]) / .2
            candidate = (np.linalg.norm(error), fit, pos, rot)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if pos < pos_tol and rot < rot_tol:
                break
        _, fit, pos, rot = best
        near_limit = bool(np.any(np.minimum(fit.x - limits[:, 0], limits[:, 1] - fit.x) < 1e-4))
        converged = pos < pos_tol and rot < rot_tol
        results.append(fit.x.copy())
        diagnostics.append([pos, rot, fit.nfev, float(near_limit)])
        if converged:
            previous = fit.x.copy()
    return np.asarray(results), np.asarray(diagnostics)
