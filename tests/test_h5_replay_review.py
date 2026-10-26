import sys
from pathlib import Path
import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from h5_replay_geometry import UrdfModel, solve, transform
from run_h5_replay import load_h5_reference, load_crx_side_config


def test_analytic_fk(tmp_path):
    urdf = tmp_path / "arm.urdf"
    urdf.write_text('''<robot name="test"><link name="base"/><link name="arm"/><link name="tip"/>
    <joint name="q" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/></joint>
    <joint name="offset" type="fixed"><parent link="arm"/><child link="tip"/>
    <origin xyz="2 0 0"/></joint></robot>''')
    pose = UrdfModel(urdf).frame_pose("tip", np.array([np.pi / 2]))
    np.testing.assert_allclose(pose[:3, 3], [1, 2, 0], atol=1e-12)
    np.testing.assert_allclose(pose[:3, 0], [0, 1, 0], atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
def test_crx_reachable_and_unreachable(side):
    cfg = load_crx_side_config(side)
    model = UrdfModel(cfg["urdf_path"])
    indices = [model.joint_index("J" + str(i)) for i in range(1, 7)]
    seed = cfg["initial_qpos"][:6]
    full = np.zeros(model.dof)
    full[indices] = seed + [.05, -.03, .02, .04, .03, -.01]
    target = model.frame_pose(side + "_hand_C_MC", full)
    _, diag = solve(model, side, np.array([target]), None, seed)
    assert diag[0, 0] < 1e-4 and diag[0, 1] < 1e-3
    target[:3, 3] = [10, 10, 10]
    arm, diag = solve(model, side, np.array([target]), None, seed, max_nfev=30)
    assert diag[0, 0] > 1
    assert np.all(arm >= model.joint_limits[indices, 0])
    assert np.all(arm <= model.joint_limits[indices, 1])


def test_common_transform_preserves_object_relative_hand():
    hand = transform([.4, .1, .2, .1, .2, .3])
    obj = transform([.5, 0, .1, -.2, .1, -.3])
    scene = transform([.1, .2, .3, -.5, .4, 1.])
    np.testing.assert_allclose(np.linalg.inv(scene @ obj) @ (scene @ hand), np.linalg.inv(obj) @ hand, atol=1e-12)


def test_legacy_base_and_reset_validation(tmp_path):
    reference = tmp_path / "reference.h5"
    rollout = tmp_path / "rollout.h5"
    q = np.zeros((3, 2, 29)); q[1, 0, 0] = .1
    with h5py.File(reference, "w") as f:
        f["base_translation"] = [[0, 0, -.85], [0, 0, -.85]]
        f["qpos"] = q[:1]
    with h5py.File(rollout, "w") as f:
        f["qpos"] = q
        f["urdf_name"] = np.array([b"rby1a_sharpa.urdf"] * 2)
        f["is_rhand"] = [0, 1]
        f["fps"] = [60., 60.]
    with pytest.raises(ValueError, match="Missing robot_base"):
        load_h5_reference(rollout)
    result = load_h5_reference(rollout, reference, True)
    assert len(result["qpos"]) == 2
    np.testing.assert_allclose(result["base_poses"][:, 2, 3], -.85)
    with h5py.File(rollout, "a") as f:
        f["qpos"][-1, 0, 0] = .2
    with pytest.raises(ValueError, match="does not match"):
        load_h5_reference(rollout, reference, True)


@pytest.mark.parametrize("extra, expected", [
    (["--execute", "--force"], "Execution disabled"),
    ([], "Bimanual replay needs explicit"),
])
def test_cli_rejects_execution_and_missing_bases(extra, expected):
    import subprocess
    command = [sys.executable, str(ROOT / "scripts" / "run_h5_replay.py"),
               "--data", "unused.h5", "--rby1a-urdf",
               str(ROOT / "assets/robots/crx5ia_sharpa_wave_right/urdf/crx5ia_sharpa_wave_right.urdf")]
    result = subprocess.run(command + extra, capture_output=True, text=True)
    assert result.returncode != 0
    assert expected in result.stderr
