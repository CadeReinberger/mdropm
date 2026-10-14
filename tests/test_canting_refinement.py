"""Fast numerical-identity tests; no FEniCS/MPI dependency."""
import json
import pickle
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from canting_estimator import K_ALPHA_MM, average_samples, normal_motion_curvature, estimate_kinematic


def circle(nodes=80, radius=3.0):
    angle = np.arange(nodes)[None, :] * (2 * np.pi / nodes)
    return angle, radius * np.cos(angle), radius * np.sin(angle)


def test_radial_motion_is_straight_and_stationary_motion_has_no_area():
    _, x, y = circle()
    alpha, area = normal_motion_curvature(x, y, -np.ones_like(x))
    np.testing.assert_allclose(alpha, 0, atol=2e-15)
    assert np.all(area > 0)
    alpha, area = normal_motion_curvature(x, y, np.zeros_like(x))
    assert np.all(np.isnan(alpha))
    assert np.all(area == 0)


def test_discrete_turning_rate_matches_finite_time_change_of_normal():
    angle, x, y = circle()
    speed = -1 + .2 * np.cos(angle)
    def velocity(x, y):
        tx = (np.roll(x, -1, axis=1) - np.roll(x, 1, axis=1))/2
        ty = (np.roll(y, -1, axis=1) - np.roll(y, 1, axis=1))/2
        return speed * ty / np.hypot(tx, ty), -speed * tx / np.hypot(tx, ty)
    vx, vy = velocity(x, y)
    step = 1e-5
    forward = velocity(x + step*vx, y + step*vy)
    backward = velocity(x - step*vx, y - step*vy)
    ax, ay = [(a-b)/(2*step) for a, b in zip(forward, backward)]
    expected = K_ALPHA_MM * (vx*ay-vy*ax) / np.hypot(vx, vy)**3
    actual, _ = normal_motion_curvature(x, y, speed)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=2e-10)


def test_curvature_converges_quadratically_to_circle_normal_motion_identity():
    errors = []
    for nodes in (40, 80, 160):
        angle, x, y = circle(nodes)
        speed = -1 + .2*np.cos(angle)
        alpha, _ = normal_motion_curvature(x, y, speed)
        expected = K_ALPHA_MM * .2 * np.sin(angle) / (3 * np.abs(speed))
        errors.append(np.max(np.abs(alpha-expected)))
    assert errors[0] / errors[1] > 3.9
    assert errors[1] / errors[2] > 3.9


def test_time_reparameterization_does_not_change_curvature():
    angle, x, y = circle()
    speed = -1 + .2*np.cos(angle)
    alpha, area = normal_motion_curvature(x, y, speed)
    fast_alpha, fast_area = normal_motion_curvature(x, y, speed*100)
    np.testing.assert_allclose(fast_alpha, alpha, atol=1e-14)
    np.testing.assert_allclose(fast_area, area*100)


def test_spatial_average_preserves_constant_field_and_ignores_pinned_samples():
    values = np.array([3., 3., 3., 3., 3., 1e12])
    d2 = np.arange(6) * .01
    areas = np.arange(1, 7, dtype=float)
    speeds = np.array([1., 1., 1., 1., 1., 0.])
    assert average_samples((values, d2, areas, speeds, 80)) == pytest.approx(3)
    with pytest.raises(ValueError, match='no moving'):
        average_samples((values, d2, areas, speeds*0, 80))


def test_degenerate_geometry_fails_explicitly():
    with pytest.raises(ValueError, match='degenerate'):
        normal_motion_curvature(np.zeros((3, 12)), np.zeros((3, 12)), np.ones((3, 12)))


def test_saved_time_sampling_does_not_change_the_physical_average(tmp_path):
    angle, x, y = circle()
    speed = -.1 * (1 + .2*np.sin(angle[0]))
    p = dict(k_a=57., k_r=57., theta_a=2.1, theta_r=.87, casr_smoothing_rad=.001)
    theta = p['theta_r'] - .001*np.log(np.expm1(-speed/(p['k_r']*.001)))
    def rhs(t, state):
        xx, yy = np.split(state, 2)
        tx, ty = np.roll(xx, -1)-np.roll(xx, 1), np.roll(yy, -1)-np.roll(yy, 1)
        return np.r_[speed*ty/np.hypot(tx, ty), -speed*tx/np.hypot(tx, ty)]
    solution = solve_ivp(rhs, (0, 3), np.r_[x[0], y[0]], dense_output=True, rtol=1e-11, atol=1e-12)
    (tmp_path/'manifest.json').write_text(json.dumps({'physical_parameters': p}))
    estimates = []
    for count in (151, 301, 601):
        times = np.linspace(0, 3, count)
        states = np.c_[solution.sol(times).T, np.broadcast_to(theta, (count, len(theta))) ]
        path = tmp_path/'results.pkl'
        with path.open('wb') as stream:
            pickle.dump(dict(out_t=times, out_x=states), stream)
        estimates.append(estimate_kinematic(path)[0])
    np.testing.assert_allclose(estimates, estimates[-1], rtol=2e-5)
