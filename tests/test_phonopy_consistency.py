"""Our thermodynamics against phonopy's implementation on real phonon spectra.

EMT, a cheap classical potential in ASE, stands in for the ML potential so the
tests run in seconds without a GPU. hcp Cu has two atoms (Z = 2 formula units)
in its primitive cell, which exercises the per-formula-unit conversion.
"""

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from phonopy.physical_units import get_physical_units

from forces2free import thermo
from forces2free.phonons import build_force_constants, compute_forces, formula_units, mesh_frequencies
from forces2free.relax import relax
from forces2free.thermo import R, harmonic_thermo

TEMPERATURES = np.arange(0.0, 3001.0, 50.0)
MESH = (12, 12, 12)


@pytest.fixture(scope="module", params=["Al-fcc", "Cu-hcp"])
def phonon(request):
    atoms, supercell = {
        "Al-fcc": (bulk("Al", "fcc", a=4.05, cubic=True), (2, 2, 2)),
        "Cu-hcp": (bulk("Cu", "hcp", a=2.55, c=4.16), (3, 3, 2)),
    }[request.param]
    result = relax(atoms, EMT())
    assert result.converged
    phonon, residual = compute_forces(result.atoms, EMT(), supercell)
    assert residual < 1e-3
    build_force_constants(phonon)
    return phonon


def _both(phonon):
    mesh = mesh_frequencies(phonon, MESH)
    assert mesh.is_stable
    z = formula_units(phonon, atoms_per_formula=1)
    ours = harmonic_thermo(mesh.frequencies, mesh.weights, TEMPERATURES, formula_units=z)
    theirs = phonon.run_thermal_properties(temperatures=TEMPERATURES, exclude_gamma_acoustic=True)
    return ours, theirs, z


def test_phonopy_reports_per_mole_of_primitive_cells(phonon):
    """Cv -> 3R per atom: 3R x (atoms in primitive cell) in phonopy, 3R in ours."""
    ours, theirs, z = _both(phonon)
    assert theirs.heat_capacity[-1] == pytest.approx(3 * R * len(phonon.primitive), rel=2e-3)
    assert ours.heat_capacity[-1] == pytest.approx(3 * R, rel=2e-3)


def test_agrees_with_phonopy(phonon):
    """phonopy still uses CODATA 2006 constants; ours are the exact SI 2019
    values. The constants differ by ~6e-7, which moves results by a few 1e-6.
    F crosses zero at some temperature, where only an absolute tolerance
    (1e-5 kJ/mol) makes sense."""
    ours, theirs, z = _both(phonon)
    np.testing.assert_allclose(ours.free_energy, theirs.free_energy / z, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(ours.entropy, theirs.entropy / z, rtol=2e-5)
    np.testing.assert_allclose(ours.heat_capacity, theirs.heat_capacity / z, rtol=2e-5)
    assert ours.zero_point_energy == pytest.approx(theirs.zero_point_energy / z, rel=2e-5)


def test_agrees_exactly_with_phonopy_constants(phonon, monkeypatch):
    """With phonopy's constants swapped in, the only remaining difference is rounding."""
    units = get_physical_units()
    monkeypatch.setattr(thermo, "H", units.PlanckConstant * units.EV)
    monkeypatch.setattr(thermo, "KB", units.KB_J)
    monkeypatch.setattr(thermo, "NA", units.Avogadro)
    ours, theirs, z = _both(phonon)
    np.testing.assert_allclose(ours.free_energy, theirs.free_energy / z, rtol=1e-10)
    np.testing.assert_allclose(ours.entropy, theirs.entropy / z, rtol=1e-10)
    np.testing.assert_allclose(ours.heat_capacity, theirs.heat_capacity / z, rtol=1e-10)
