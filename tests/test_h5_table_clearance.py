import sys
from pathlib import Path
import struct
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_h5_table_clearance import minimum_over_footprint, load_stl


def test_triangle_crossing_without_inside_vertices():
    tri = np.array([[[-2., 0., -.1], [2., 0., -.1], [0., 3., -.1]]])
    assert minimum_over_footprint(tri, [-.5, .5, .2, .6]) == pytest.approx(-.1)


def test_outside_footprint_is_not_penetration():
    tri = np.array([[[2., 2., -2.], [3., 2., -2.], [2., 3., -2.]]])
    assert np.isinf(minimum_over_footprint(tri, [0, 1, 0, 1]))


def test_clipping_interpolates_height():
    tri = np.array([[[0., 0., 0.], [2., 0., 2.], [0., 2., 0.]]])
    assert minimum_over_footprint(tri, [1, 1.5, 0, .1]) == pytest.approx(1.)


def test_binary_stl_units_preserved(tmp_path):
    path = tmp_path / "mesh.stl"
    path.write_bytes(b"solid" + b" " * 75 + struct.pack("<I", 1) + struct.pack("<12fH", 0,0,1, 0,0,0, 2,0,0, 0,2,0, 0))
    mesh = load_stl(path)
    assert mesh.shape == (1, 3, 3)
    assert mesh.max() == 2.
