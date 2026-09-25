"""Every configured starting structure has the intended symmetry and formula units."""

import warnings

import numpy as np
import pytest
from phonopy import Phonopy

from forces2free.config import load_materials
from forces2free.phonons import formula_units, to_phonopy_atoms
from forces2free.relax import spacegroup

# space group, atoms in the built cell, formula units Z in the primitive cell
EXPECTED = {
    "Si": ("Fd-3m (227)", 8, 2),
    "SiC": ("F-43m (216)", 8, 1),
    "AlN": ("P6_3mc (186)", 4, 2),
    "Al": ("Fm-3m (225)", 4, 1),
    "Mg": ("P6_3/mmc (194)", 2, 2),
    "NaCl": ("Fm-3m (225)", 8, 1),
    "NaF": ("Fm-3m (225)", 8, 1),
    "MgO": ("Fm-3m (225)", 8, 1),
    "Al2O3": ("R-3c (167)", 30, 2),
}


def test_every_material_is_checked():
    assert set(load_materials()) == set(EXPECTED)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_structure(key):
    material = load_materials()[key]
    atoms = material.build()
    group, n_atoms, z = EXPECTED[key]
    assert spacegroup(atoms) == group
    assert len(atoms) == n_atoms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phonon = Phonopy(to_phonopy_atoms(atoms), supercell_matrix=np.eye(3, dtype=int))
    assert formula_units(phonon, material.atoms_per_formula) == z


def test_comparison_range():
    materials = load_materials()
    assert materials["Si"].compare_tmax == pytest.approx(0.7 * 1685)
    assert materials["SiC"].compare_tmax == 1500  # no melting in JANAF, explicit cap
    assert materials["SiC"].curve_tmax >= materials["SiC"].compare_tmax
    assert materials["Al"].curve_tmax == 1000


def test_static_lattice_build():
    materials = load_materials()
    with_lattice = {k for k, m in materials.items() if m.static_lattice}
    assert with_lattice == {"Si", "SiC", "Al", "NaCl", "NaF", "MgO"}
    for key in with_lattice:
        material = materials[key]
        atoms = material.build(a=material.static_lattice["a"])
        assert atoms.cell.cellpar()[0] == pytest.approx(material.static_lattice["a"])
        assert spacegroup(atoms) == EXPECTED[key][0]
        assert "Hao" in material.static_lattice["source"]
    with pytest.raises(ValueError):
        materials["Mg"].build(a=3.2)  # hexagonal: one lattice constant is not enough
    with pytest.raises(ValueError):
        materials["Al2O3"].build(a=4.8)


def test_electronic_gamma_only_for_metals():
    materials = load_materials()
    assert {k for k, m in materials.items() if m.gamma_electronic} == {"Al", "Mg"}
    assert materials["Al"].gamma_electronic == pytest.approx(1.35e-3)  # J/(mol K^2)
