"""Harmonic-oscillator thermodynamics of a crystal from its phonon frequencies.

Every phonon mode with frequency nu is an independent quantum harmonic
oscillator with levels E_n = (n + 1/2) h nu. With x = h nu / (k_B T), one mode
contributes

    F  = h nu / 2 + k_B T ln(1 - e^-x)
    U  = h nu / 2 + h nu / (e^x - 1)
    S  = k_B [x / (e^x - 1) - ln(1 - e^-x)]
    Cv = k_B x^2 e^x / (e^x - 1)^2

and the crystal values are sums over all modes. The derivation is written out
in docs/derivation.md.

Units: frequencies are ordinary frequencies nu in THz (the phonopy convention),
not angular frequencies omega = 2 pi nu. Results are per mole of formula units:
F and U in kJ/mol, S and Cv in J/(K mol).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import constants

H = constants.h  # Planck constant, J s
KB = constants.k  # Boltzmann constant, J/K
NA = constants.N_A  # Avogadro constant, 1/mol
R = NA * KB  # gas constant, J/(K mol)
THZ = 1e12  # Hz per THz


@dataclass(frozen=True)
class HarmonicThermo:
    """Thermodynamic functions at constant volume, per mole of formula units."""

    temperatures: NDArray[np.float64]  # K
    free_energy: NDArray[np.float64]  # kJ/mol, includes the zero-point energy
    internal_energy: NDArray[np.float64]  # kJ/mol, includes the zero-point energy
    entropy: NDArray[np.float64]  # J/(K mol)
    heat_capacity: NDArray[np.float64]  # J/(K mol), this is Cv, not Cp
    zero_point_energy: float  # kJ/mol


def mode_functions(
    frequencies_thz: ArrayLike, temperature: float
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return F, U, S, Cv of single harmonic modes in J, J, J/K, J/K.

    Frequencies must be positive. The formulas are written with e^-x so they
    neither overflow at low temperature (large x) nor lose precision at high
    temperature (small x).
    """
    energy = H * np.asarray(frequencies_thz, dtype=float) * THZ  # h nu in J
    if temperature <= 0:
        zeros = np.zeros_like(energy)
        return energy / 2, energy / 2, zeros, zeros

    x = energy / (KB * temperature)
    boltzmann = np.exp(-x)  # e^-x
    one_minus = -np.expm1(-x)  # 1 - e^-x
    occupation = boltzmann / one_minus  # Bose-Einstein n = 1 / (e^x - 1)

    free = energy / 2 + KB * temperature * np.log(one_minus)
    internal = energy / 2 + energy * occupation
    entropy = KB * (x * occupation - np.log(one_minus))
    heat_capacity = KB * x**2 * boltzmann / one_minus**2
    return free, internal, entropy, heat_capacity


def harmonic_thermo(
    frequencies_thz: ArrayLike,
    weights: ArrayLike,
    temperatures: ArrayLike,
    formula_units: int = 1,
    cutoff_thz: float = 0.0,
) -> HarmonicThermo:
    """Sum the mode contributions over a q-point mesh.

    Parameters
    ----------
    frequencies_thz
        Phonon frequencies of the primitive cell on the mesh, shape
        (n_qpoints, n_bands), in THz. Imaginary modes are negative numbers
        (phonopy convention).
    weights
        Multiplicity of each irreducible q-point, shape (n_qpoints,).
    temperatures
        Temperatures in K.
    formula_units
        Number of formula units Z in the primitive cell (2 for Si, whose
        primitive cell holds two atoms).
    cutoff_thz
        Modes with frequency <= cutoff are left out. The default 0 removes
        zero-frequency and imaginary modes. Set the three acoustic modes at
        Gamma to zero beforehand so that they are removed as well.

    The weighted mesh average sum_q w_q sum_j f(nu_qj) / sum_q w_q is a value
    per primitive cell. Multiplying by N_A gives a value per mole of
    primitive cells, and dividing by Z a value per mole of formula units.
    """
    freqs = np.asarray(frequencies_thz, dtype=float)
    w = np.asarray(weights, dtype=float)
    temps = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if freqs.ndim != 2 or freqs.shape[0] != w.size:
        raise ValueError("frequencies must have shape (n_qpoints, n_bands) matching weights")

    mask = freqs > cutoff_thz
    nu = freqs[mask]
    mode_weights = np.broadcast_to(w[:, None], freqs.shape)[mask] / w.sum()
    to_molar = NA / formula_units

    totals = np.empty((4, temps.size))
    for i, temperature in enumerate(temps):
        for k, values in enumerate(mode_functions(nu, temperature)):
            totals[k, i] = values @ mode_weights * to_molar
    zero_point = (H * nu * THZ / 2) @ mode_weights * to_molar

    return HarmonicThermo(
        temperatures=temps,
        free_energy=totals[0] / 1000,
        internal_energy=totals[1] / 1000,
        entropy=totals[2],
        heat_capacity=totals[3],
        zero_point_energy=float(zero_point) / 1000,
    )


def dulong_petit(atoms_per_formula: int) -> float:
    """High-temperature limit of Cv, 3 n R, in J/(K mol)."""
    return 3 * atoms_per_formula * R
