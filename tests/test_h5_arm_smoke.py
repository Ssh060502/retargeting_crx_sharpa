import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from h5_arm_smoke import build_relative_test, anchor_path


def test_bounded_slow_return_path_selected_arm_only():
    source=np.zeros((30,56))
    source[:,28:34]=np.linspace(0,1,30)[:,None]*np.array([1.,-.8,.6,-.3,.2,.1])
    source[:,:6]=100.  # Unselected side must have no effect.
    q,rate=build_relative_test(source,"right")
    np.testing.assert_allclose(q[0],0.)
    np.testing.assert_allclose(q[-1],0.)
    assert np.abs(q).max()<=np.deg2rad(1.)
    v=np.diff(q,axis=0)*rate;a=np.diff(v,axis=0)*rate;j=np.diff(a,axis=0)*rate
    assert np.abs(v).max()<.01
    assert np.abs(a).max()<.01
    assert np.abs(j).max()<.03
    anchored=anchor_path(q,np.zeros(6),-np.ones(6),np.ones(6))
    np.testing.assert_allclose(anchored,q)


def test_anchor_near_limit_rejected():
    with pytest.raises(ValueError,match="too near joint limits"):
        anchor_path(np.zeros((10,6)),np.full(6,.98),-np.ones(6),np.ones(6))


@pytest.mark.parametrize("side,frames,amplitude",[("both",30,.01),("right",4,.01),("right",61,.01),("right",30,.2)])
def test_invalid_test_parameters_fail(side,frames,amplitude):
    with pytest.raises(ValueError):
        build_relative_test(np.zeros((100,56)),side,frames,max_delta=amplitude)


def test_no_motion_is_not_silently_reported_as_test():
    with pytest.raises(ValueError,match="no measurable motion"):
        build_relative_test(np.zeros((30,56)),"right")
