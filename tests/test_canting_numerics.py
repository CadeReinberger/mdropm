import numpy as np

import heightscape
from ca_bdry_computer import compute_p_star_over_gamma
from geo_util import compute_casr
from physical_params import physical_params
from canting_plotter import saturating_response


def test_smoothed_casr_matches_piecewise_law_away_from_transition():
    params = physical_params()
    hard, _ = compute_casr(params)
    params.casr_smoothing_rad = 1e-3
    smooth, smooth_prime = compute_casr(params)

    for angle in (params.theta_r - 0.01, params.theta_r + 0.01, params.theta_a - 0.01, params.theta_a + 0.01):
        assert np.isclose(smooth(angle), hard(angle), atol=3e-6)
        assert np.isfinite(smooth_prime(angle))


def test_initial_pressure_uses_named_radius_and_half_length():
    params = physical_params()
    landscape = heightscape.constructors.rect_interp_htscape((10, 10), (0.1, 0.3, 0.3, 0.1))
    correct = compute_p_star_over_gamma(
        landscape,
        half_length=5,
        radius=3,
        physical_params=params,
    )
    swapped = compute_p_star_over_gamma(
        landscape,
        half_length=3,
        radius=5,
        physical_params=params,
    )
    assert not np.isclose(correct, swapped)


def test_positive_saturation_parameters_are_monotone():
    x = np.linspace(0, 0.5, 1000)
    y = saturating_response(x, baseline=1.0, amplitude=4.0, scale=0.07, shape=1.3)
    assert np.all(np.diff(y) >= 0)
