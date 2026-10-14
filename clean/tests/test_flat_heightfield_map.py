"""Check map coverage and reflection without requiring the FEM environment."""
import numpy as np
import pytest

from flat_heightfield_map import trajectory_field


def radial_trajectory():
    angles = np.arange(80)*2*np.pi/80
    radius = np.linspace(3, 2, 21)[:, None]
    speed = -np.ones((21, 80))
    parameters = dict(k_a=57., k_r=57., theta_a=2.1, theta_r=.87, casr_smoothing_rad=.001)
    theta = parameters["theta_r"] - .001*np.log(np.expm1(-speed/(57*.001)))
    states = np.c_[radius*np.cos(angles), radius*np.sin(angles), theta]
    return dict(out_t=np.linspace(0, 1, 21), out_x=states), dict(nodes=80, physical_parameters=parameters)


def test_straight_radial_motion_has_zero_chirality_only_in_swept_region():
    results, manifest = radial_trajectory()
    *_, interpolate = trajectory_field(results, manifest)
    evaluated = interpolate(np.array([0., 2.5, 3.2]), np.array([0., 0.2, 0.]))
    assert np.ma.getmaskarray(evaluated).tolist() == [True, False, True]
    assert evaluated[1] == pytest.approx(0, abs=1e-12)


def test_nonuniform_recession_preserves_reflection_in_interpolated_map():
    results, manifest = radial_trajectory()
    n = manifest["nodes"]
    angles = np.arange(n)*2*np.pi/n
    speed = np.broadcast_to(-1-.2*np.sin(angles), (21, n))
    results["out_x"][:, 2*n:] = .87-.001*np.log(np.expm1(-speed/(57*.001)))
    *_, interpolate = trajectory_field(results, manifest)
    x = np.linspace(-1.5, 1.5, 60)
    y = np.sqrt(2.5**2-x**2)
    actual = interpolate(x, y)
    assert not np.any(np.ma.getmaskarray(actual))
    np.testing.assert_allclose(actual, -actual[::-1], atol=1e-11)


def test_nonmonotone_saved_times_are_rejected():
    results, manifest = radial_trajectory()
    results["out_t"][2] = results["out_t"][1]
    with pytest.raises(ValueError, match="trajectory"):
        trajectory_field(results, manifest)
