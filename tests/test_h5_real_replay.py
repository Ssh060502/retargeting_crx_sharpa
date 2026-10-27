import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from h5_real_guard import require_real_domain,validate_plugins
from run_h5_real import conservative_rate


def test_physical_guard_rejects_mock_and_unknown_plugins():
    for plugin in ("mock_components/GenericSystem","unknown"):
        with pytest.raises(RuntimeError):
            validate_plugins([NS(plugin_name=plugin,state=NS(label="active"))])
    validate_plugins([NS(plugin_name="fanuc_robot_driver/FanucHardwareInterface",state=NS(label="active"))])


def test_physical_domain_does_not_reuse_mock(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID","187")
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE","LOCALHOST")
    with pytest.raises(RuntimeError):require_real_domain()
    monkeypatch.setenv("ROS_DOMAIN_ID","87")
    require_real_domain()


def test_additional_slowdown_covers_arm_and_finger_rates():
    q=np.zeros((20,56));q[:,0]=np.linspace(0,1,20);q[:,7]=np.linspace(0,2,20)
    fps,factor=conservative_rate(q,60.)
    assert factor>=1 and fps<60
    velocity=np.diff(q,axis=0)*fps
    assert np.max(np.abs(velocity[:,0]))<=.2
    assert np.max(np.abs(velocity[:,7]))<=.4


def test_module_import_does_not_load_ros_or_publish():
    root=Path(__file__).resolve().parents[1]
    code="import sys; sys.path.insert(0,'scripts'); import run_h5_real; assert 'rclpy' not in sys.modules; assert 'retargeting_ros.sharpa_joint' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr



def test_publisher_error_preserves_endpoint_evidence_without_relaxing_rule():
    from h5_real_guard import validate_command_publishers
    own=NS(node_name="retargeting_sharpa_joint",node_namespace="/",endpoint_gid=[1,2])
    other=NS(node_name="move_to_default_pose",node_namespace="/",endpoint_gid=[3,4])
    validate_command_publishers("/target", [])
    validate_command_publishers("/target", [own])
    with pytest.raises(RuntimeError,match="move_to_default_pose"):
        validate_command_publishers("/target", [own,other])
    with pytest.raises(RuntimeError,match="count=2"):
        validate_command_publishers("/target", [own,own])
    with pytest.raises(RuntimeError,match="gid"):
        validate_command_publishers("/target", [other])
