"""Supercell-size and q-mesh convergence of the harmonic thermodynamics.

The structure is relaxed once; forces are recomputed for each supercell and
the q-mesh is varied on top of each set of force constants.

    uv run python scripts/convergence.py --model mace-matpes-pbe-0 --material Si

Output: results/<model>/<material>/convergence.csv
"""

import argparse

import pandas as pd

from forces2free import phonons
from forces2free.config import load_materials, load_models, results_dir
from forces2free.models import load_calculator
from forces2free.relax import relax
from forces2free.thermo import harmonic_thermo

SUPERCELLS = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
MESHES = [8, 12, 16, 20, 24, 32, 40]
TEMPERATURES = [100.0, 298.15, 1000.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mace-matpes-pbe-0")
    parser.add_argument("--material", default="Si")
    args = parser.parse_args()

    material = load_materials()[args.material]
    calc = load_calculator(load_models()[args.model])
    relaxed = relax(material.build(), calc).atoms

    rows = []
    for supercell in SUPERCELLS:
        phonon, _ = phonons.compute_forces(relaxed, calc, supercell)
        phonons.build_force_constants(phonon)
        z = phonons.formula_units(phonon, material.atoms_per_formula)
        for n in MESHES:
            freqs = phonons.mesh_frequencies(phonon, (n, n, n))
            t = harmonic_thermo(freqs.frequencies, freqs.weights, TEMPERATURES, formula_units=z)
            row = {"supercell_atoms": len(phonon.supercell), "mesh": n, "min_frequency_THz": freqs.min_frequency}
            for temperature, s, cv in zip(TEMPERATURES, t.entropy, t.heat_capacity):
                row[f"S_{temperature:g}"] = s
                row[f"Cv_{temperature:g}"] = cv
            rows.append(row)

    table = pd.DataFrame(rows)
    path = results_dir(args.model, args.material) / "convergence.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.6g")
    print(table.pivot(index="mesh", columns="supercell_atoms", values="S_298.15").round(4).to_string())
    print(f"\nS(298.15 K) in J/(K mol) by mesh (rows) and supercell atoms (columns); saved {path}")


if __name__ == "__main__":
    main()
