"""Checks of the harmonic thermodynamics that do not need any potential.

The tests use limits and identities that must hold whatever the frequencies
are, so a mistake in a formula or a unit shows up as a failed physical law.
"""

import math

import numpy as np
import pytest

from forces2free.thermo import H, KB, R, THZ, dulong_petit, harmonic_thermo, mode_functions


@pytest.fixture
def spectrum():
    """A made-up spectrum: 7 q-points x 6 bands between 0.5 and 15 THz."""
    rng = np.random.default_rng(0)
    frequencies = rng.uniform(0.5, 15.0, size=(7, 6))
    weights = np.array([1, 6, 8, 12, 24, 6, 7])
    return frequencies, weights


def test_single_mode_at_x_equal_one():
    """At h nu = k_B T the Einstein formulas give e/(e-1)^2 and friends."""
    nu = 1.0  # THz
    temperature = H * nu * THZ / KB  # x = 1
    free, internal, entropy, cv = mode_functions(nu, temperature)
    e = math.e
    assert cv / KB == pytest.approx(e / (e - 1) ** 2, rel=1e-12)
    assert entropy / KB == pytest.approx(1 / (e - 1) - math.log(1 - 1 / e), rel=1e-12)
    assert internal / (KB * temperature) == pytest.approx(0.5 + 1 / (e - 1), rel=1e-12)
    assert free / (KB * temperature) == pytest.approx(0.5 + math.log(1 - 1 / e), rel=1e-12)


def test_zero_temperature_leaves_only_zero_point_energy():
    free, internal, entropy, cv = mode_functions([1.0, 10.0], 0.0)
    np.testing.assert_allclose(free, H * np.array([1.0, 10.0]) * THZ / 2)
    np.testing.assert_allclose(internal, free)
    assert not entropy.any() and not cv.any()


def test_extreme_temperatures_stay_finite():
    """x = 7000 (10 THz at 0.07 K) must not overflow; x ~ 1e-6 must not lose precision."""
    for temperature in (0.07, 1e7):
        values = mode_functions([10.0], temperature)
        assert all(np.isfinite(v).all() for v in values)
    cv_cold = mode_functions([10.0], 0.07)[3]
    cv_hot = mode_functions([10.0], 1e7)[3]
    assert cv_cold[0] == 0.0
    assert cv_hot[0] / KB == pytest.approx(1.0, rel=1e-9)


def test_internal_energy_equals_f_plus_ts(spectrum):
    frequencies, weights = spectrum
    t = harmonic_thermo(frequencies, weights, np.linspace(10, 2000, 50))
    u_from_f = t.free_energy + t.temperatures * t.entropy / 1000
    np.testing.assert_allclose(t.internal_energy, u_from_f, rtol=1e-12)


def test_entropy_is_minus_df_dt_and_cv_is_du_dt(spectrum):
    """Central differences of F and U reproduce S and Cv."""
    frequencies, weights = spectrum
    temps = np.array([30.0, 100.0, 298.15, 700.0, 1500.0])
    dt = 1e-3
    t = harmonic_thermo(frequencies, weights, temps)
    up = harmonic_thermo(frequencies, weights, temps + dt)
    down = harmonic_thermo(frequencies, weights, temps - dt)
    entropy = -(up.free_energy - down.free_energy) / (2 * dt) * 1000
    cv = (up.internal_energy - down.internal_energy) / (2 * dt) * 1000
    np.testing.assert_allclose(t.entropy, entropy, rtol=1e-6)
    np.testing.assert_allclose(t.heat_capacity, cv, rtol=1e-6)


def test_high_temperature_limit_is_dulong_petit(spectrum):
    """6 modes per primitive cell = 2 atoms; with Z = 2 formula units Cv -> 3R."""
    frequencies, weights = spectrum
    t = harmonic_thermo(frequencies, weights, [2e4], formula_units=2)
    assert t.heat_capacity[0] == pytest.approx(dulong_petit(1), rel=1e-4)
    assert dulong_petit(1) == pytest.approx(24.943, abs=1e-3)
    assert R == pytest.approx(8.314462618, rel=1e-9)


def test_low_temperature_limit(spectrum):
    frequencies, weights = spectrum
    t = harmonic_thermo(frequencies, weights, [0.0, 0.5])
    assert t.entropy[0] == 0.0 and t.heat_capacity[0] == 0.0
    assert t.free_energy[0] == pytest.approx(t.zero_point_energy, rel=1e-12)
    assert t.heat_capacity[1] < 1e-10


def test_values_scale_with_formula_units(spectrum):
    frequencies, weights = spectrum
    temps = [0.0, 300.0, 1000.0]
    one = harmonic_thermo(frequencies, weights, temps, formula_units=1)
    two = harmonic_thermo(frequencies, weights, temps, formula_units=2)
    np.testing.assert_allclose(two.entropy, one.entropy / 2)
    np.testing.assert_allclose(two.free_energy, one.free_energy / 2)
    assert two.zero_point_energy == pytest.approx(one.zero_point_energy / 2)


def test_zero_and_imaginary_modes_are_left_out(spectrum):
    frequencies, weights = spectrum
    padded = np.hstack([frequencies, np.zeros((7, 1)), -np.full((7, 1), 0.3)])
    temps = [0.0, 300.0]
    reference = harmonic_thermo(frequencies, weights, temps)
    result = harmonic_thermo(padded, weights, temps)
    np.testing.assert_allclose(result.entropy, reference.entropy)
    np.testing.assert_allclose(result.free_energy, reference.free_energy)
