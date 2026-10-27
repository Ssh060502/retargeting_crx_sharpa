import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from h5_mock_playback import MockPlayback, sample_trajectory, smoothstep, transition_duration
from h5_mock_guard import require_mock_domain, validate_plugins


def test_mock_domain_is_explicit_and_local(monkeypatch):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    with pytest.raises(RuntimeError):
        require_mock_domain()
    monkeypatch.setenv("ROS_DOMAIN_ID", "87")
    with pytest.raises(RuntimeError):
        require_mock_domain()
    monkeypatch.setenv("ROS_DOMAIN_ID", "187")
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    require_mock_domain()


@pytest.mark.parametrize("plugins", [[], ["fanuc_hardware/FanucSystem"], ["mock_components/GenericSystem", "real/plugin"]])
def test_guard_rejects_unverified_hardware(plugins):
    with pytest.raises(RuntimeError):
        validate_plugins([NS(plugin_name=p, state=NS(label="active")) for p in plugins])


def test_guard_accepts_only_active_mock():
    validate_plugins([NS(plugin_name="mock_components/GenericSystem", state=NS(label="active"))])
    with pytest.raises(RuntimeError):
        validate_plugins([NS(plugin_name="mock_components/GenericSystem", state=NS(label="inactive"))])


def test_transition_has_exact_endpoints_and_limits():
    start, target, limits = np.zeros(56), np.ones(56), np.ones(56)
    duration = transition_duration(start, target, limits)
    times = np.linspace(0, duration, 2001)
    positions = np.stack([start + (target-start)*smoothstep(t/duration) for t in times])
    np.testing.assert_allclose(positions[0], start)
    np.testing.assert_allclose(positions[-1], target)
    assert np.max(np.abs(np.diff(positions, axis=0) / np.diff(times)[:,None])) <= 1.
    assert smoothstep(-1.) == 0. and smoothstep(2.) == 1.


def test_pause_resume_reenters_from_feedback_and_preserves_progress():
    q = np.tile(np.linspace(.1, .2, 11)[:,None], (1,56))
    m = MockPlayback(q, 2., np.ones(56))
    m.enter(np.zeros(56), 0.)
    end_enter = m.enter_duration
    np.testing.assert_allclose(m.command(end_enter), q[0])
    m.command(end_enter + 2.)
    m.pause(end_enter + 2.)
    assert m.elapsed == pytest.approx(2.)
    with pytest.raises(ValueError):
        m.command(100.)
    actual = np.full(56, .07)
    m.resume(actual, 100.)
    np.testing.assert_allclose(m.command(100.), actual)
    np.testing.assert_allclose(m.command(100.+m.enter_duration), sample_trajectory(q,2.,2.))
    assert m.elapsed == pytest.approx(2.)
    m.command(100.+m.enter_duration+10.)
    assert m.phase == "settling"
    np.testing.assert_allclose(m.command(200.), q[-1])


def test_sampling_does_not_change_duration_or_overrun_end():
    q=np.tile(np.array([0., .1, .2])[:,None],(1,56))
    np.testing.assert_allclose(sample_trajectory(q,2.,.25), .05)
    np.testing.assert_allclose(sample_trajectory(q,2.,100.), .2)
