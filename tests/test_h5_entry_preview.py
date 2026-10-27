import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_h5_entry_preview import prepare_preview


def inputs(tmp_path):
    source=tmp_path/"trajectory.npz";review=tmp_path/"start.npz"
    q=np.ones((3,56))*.1
    poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[.5,.6,.7]
    np.savez(source,qpos56=q,sides=["left","right"],left_base=np.zeros(6),right_base=np.zeros(6),object_object_0_scene_poses=poses,table_height=-.175)
    np.savez(review,measured_start=np.zeros(56),first_target=q[0],trajectory=q,transition_seconds=2.1)
    return source,review


def test_entry_endpoints_timing_and_stationary_object(tmp_path):
    source,review=inputs(tmp_path);output=tmp_path/"preview.npz"
    prepare_preview(source,review,output)
    with np.load(output) as f:
        np.testing.assert_allclose(f["qpos56"][0],0.)
        np.testing.assert_allclose(f["qpos56"][-1],.1)
        assert (len(f["qpos56"])-1)/float(f["fps"])==pytest.approx(2.1)
        assert bool(f["preview_only"])
        np.testing.assert_allclose(f["object_object_0_scene_poses"][:,0,3],.5)
        assert float(f["table_height"])==-.175
        assert "left_diagnostics" not in f
    from run_h5_mock import load_motion
    with pytest.raises(ValueError,match="Display-only"):
        load_motion(output)


def test_wrong_trajectory_and_input_overwrite_rejected(tmp_path):
    source,review=inputs(tmp_path)
    with pytest.raises(ValueError,match="overwrite"):
        prepare_preview(source,review,source)
    np.savez(review,measured_start=np.zeros(56),first_target=np.ones(56)*.2,trajectory=np.ones((3,56))*.2,transition_seconds=2.)
    with pytest.raises(ValueError,match="different trajectory"):
        prepare_preview(source,review,tmp_path/"preview.npz")
