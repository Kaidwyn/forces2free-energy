"""Finite-displacement phonons: phonopy builds displaced supercells, an ASE
calculator (here an ML potential) supplies the forces on them.

Written against phonopy 4.6. Its Python API changed substantially in 4.x, so
check the installed source before copying older examples.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import phonopy
from ase import Atoms
from ase.calculators.calculator import Calculator
from numpy.typing import NDArray
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import PrimitiveMatrixAutoDefaultWarning

DISPLACEMENT = 0.01  # A, phonopy default
IMAGINARY_TOLERANCE = 0.05  # THz; smaller negative frequencies are numerical noise


@contextmanager
def _auto_primitive() -> Iterator[None]:
    """We want the primitive cell phonopy 4 finds by default ("auto"); its
    warning only says that phonopy 3 defaulted to the input cell instead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrimitiveMatrixAutoDefaultWarning)
        yield


def to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def to_ase_atoms(cell: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True
    )


def _forces(cell: PhonopyAtoms, calc: Calculator) -> NDArray[np.float64]:
    atoms = to_ase_atoms(cell)
    atoms.calc = calc
    return atoms.get_forces()


def compute_forces(
    atoms: Atoms,
    calc: Calculator,
    supercell: tuple[int, int, int],
    distance: float = DISPLACEMENT,
) -> tuple[Phonopy, float]:
    """Displace atoms in a supercell of `atoms` and store the resulting forces.

    Forces on the undisplaced supercell are subtracted from every displaced
    one, which cancels whatever residual force the relaxation left behind.
    Returns the Phonopy object and the largest residual force in eV/A.
    """
    with _auto_primitive():
        phonon = Phonopy(to_phonopy_atoms(atoms), supercell_matrix=np.diag(supercell))
    phonon.generate_displacements(distance=distance)
    residual = _forces(phonon.supercell, calc)
    phonon.forces = np.array(
        [_forces(cell, calc) - residual for cell in phonon.supercells_with_displacements]
    )
    return phonon, float(np.abs(residual).max())


def build_force_constants(phonon: Phonopy) -> None:
    """Fit force constants to the stored forces and impose all symmetries.

    The symfc projector enforces space-group, translational and permutation
    symmetry at once, so the acoustic modes at Gamma come out at zero.
    """
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants(use_symfc_projector=True, show_drift=False)


def load(path: str | Path) -> Phonopy:
    """Load displacements and forces saved by Phonopy.save and rebuild force constants."""
    with _auto_primitive():
        phonon = phonopy.load(path, produce_fc=False, log_level=0)
    build_force_constants(phonon)
    return phonon


@dataclass(frozen=True)
class MeshFrequencies:
    frequencies: NDArray[np.float64]  # THz, (n_qpoints, n_bands); acoustic modes at Gamma set to 0
    weights: NDArray[np.int64]  # multiplicity of each irreducible q-point
    min_frequency: float  # THz, most negative frequency (0 if none is imaginary)
    n_imaginary: int  # modes below -IMAGINARY_TOLERANCE, counted with q-point weights

    @property
    def is_stable(self) -> bool:
        return self.n_imaginary == 0


def mesh_frequencies(phonon: Phonopy, mesh: tuple[int, int, int]) -> MeshFrequencies:
    """Phonon frequencies on a Gamma-centred q-point mesh of the primitive cell.

    Gamma-centred because phonopy 4.6 shifts even meshes by half a grid step
    by default, which breaks the point-group symmetry reduction.

    The three acoustic modes at Gamma should be exactly zero but come out as
    tiny numbers of either sign; they are set to zero here so that the
    thermodynamic sums skip them consistently (phonopy: exclude_gamma_acoustic).
    """
    result = phonon.run_mesh(list(mesh), is_gamma_center=True)
    frequencies = np.array(result.frequencies, dtype=float)
    weights = np.array(result.weights)
    at_gamma = np.flatnonzero(np.all(np.abs(result.qpoints) < 1e-10, axis=1))
    if at_gamma.size:
        g = at_gamma[0]
        acoustic = np.argsort(np.abs(frequencies[g]))[:3]
        frequencies[g, acoustic] = 0.0

    imaginary = frequencies < -IMAGINARY_TOLERANCE
    return MeshFrequencies(
        frequencies=frequencies,
        weights=weights,
        min_frequency=float(min(frequencies.min(), 0.0)),
        n_imaginary=int((imaginary * weights[:, None]).sum()),
    )


def formula_units(phonon: Phonopy, atoms_per_formula: int) -> int:
    """Number of formula units Z in the primitive cell used for the mesh."""
    n_atoms = len(phonon.primitive)
    if n_atoms % atoms_per_formula:
        raise ValueError(f"{n_atoms} atoms is not a whole number of formula units")
    return n_atoms // atoms_per_formula
