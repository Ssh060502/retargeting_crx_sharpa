"""Import regressions must run in a fresh process, before native solver imports."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_basic_backend_import_never_loads_solver_or_ros():
    run("""
import sys
from importlib.abc import MetaPathFinder
class BlockOptional(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pinocchio', 'mr_utils', 'nlopt', 'rclpy', 'torch'):
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from retargeting_ros.sharpa_joint import SharpaJointBackend
from teleoperation.backends.base import BackendStepResult
assert 'teleoperation.flow' not in sys.modules
assert 'retargeting.core' not in sys.modules
assert 'rclpy' not in sys.modules
""")


def test_public_exports_remain_lazy_and_cached():
    run("""
import sys, types
import teleoperation
assert 'ExecutionFlow' in teleoperation.__all__
assert 'ExecutionFlow' in dir(teleoperation)
assert 'teleoperation.flow' not in sys.modules
stub = types.ModuleType('teleoperation.flow')
stub.ExecutionFlow = type('ExecutionFlow', (), {})
sys.modules['teleoperation.flow'] = stub
from teleoperation import ExecutionFlow
assert ExecutionFlow is stub.ExecutionFlow
assert teleoperation.ExecutionFlow is ExecutionFlow
try:
    teleoperation.not_a_public_symbol
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attribute must fail')
""")
