import numpy as np

from scripts.chirality_plots.canting_threeway import (
    FormulaGeometry,
    convert_g_to_plot_units,
    fit_flux_derivatives,
    gas_log_flux_g,
    large_casr_g,
)
from scripts.chirality_plots.canting_plotter import K_ALPHA_MM


def test_local_polynomial_fit_recovers_pole_derivatives():
    radius = 3.0
    count = 80
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    s = radius * np.arctan2(np.sin(angles), np.cos(angles))
    flux = 2.0 + 0.3 * s - 0.4 * s**2 + 0.05 * s**3 + 0.02 * s**4

    fitted = fit_flux_derivatives(s, flux, degree=4, half_width_nodes=5)

    assert np.isclose(fitted.J0, 2.0)
    assert np.isclose(fitted.Js0, 0.3)
    assert np.isclose(fitted.Jss0, -0.8)
    assert fitted.sample_count == 11


def test_two_g_formulas_use_the_specified_terms():
    geometry = FormulaGeometry(
        b=0.2,
        h0=0.3,
        D0=0.8,
        B0=-5.0,
        a1=-0.1,
        c1=1.2,
        c2=1.4,
        Dss=0.04,
    )
    J0, Js0, Jss0 = 2.0, 0.5, -0.3

    expected_flux = Jss0 / J0 - (Js0 / J0) ** 2
    expected_full = geometry.c2 * (
        Jss0 / J0 - geometry.Dss / geometry.D0
    ) - geometry.c1**2 * (Js0 / J0) ** 2

    assert np.isclose(gas_log_flux_g(J0, Js0, Jss0), expected_flux)
    assert np.isclose(large_casr_g(geometry, J0, Js0, Jss0), expected_full)


def test_plot_conversion_is_the_cleaned_legacy_curvature_factor():
    geometric_g = np.array([-0.5, 0.0, 0.75])
    expected = np.rad2deg(K_ALPHA_MM * geometric_g)
    assert np.allclose(convert_g_to_plot_units(geometric_g), expected)
