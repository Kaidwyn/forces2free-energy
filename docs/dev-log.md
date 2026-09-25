# Development log

Problems found while building the project: what happened, how each one was found and what was
done about it. Every entry happened as written; nothing is added afterwards or polished.

The entries dated 2026-09-25 cover stages 0 and 1 (environment setup and silicon end to end).
Continue below in the same format.

---

## 2026-09-25: stages 0 and 1

### 1. An outdated search result: "PyTorch has no GPU build for Python 3.14"

- **What happened**: a web search said PyTorch offers no CUDA build for Python 3.14, and that
  almost became the stated reason for changing the Python version.
- **How it was found**: the official PyTorch wheel index shows that `torch-2.14.0+cu126` does have
  cp314 wheels; the search result was stale.
- **What was done**: the switch to Python 3.12 stayed, but with the real reasons: matscipy, a MACE
  dependency, has no Windows wheel for 3.14; dm-tree 0.1.8, needed by ORB, stops at 3.12; and
  MatterSim and ORB require at least 3.12. Only 3.12 satisfies all three.
- **Lesson**: settle environment questions from primary sources (PyPI metadata, the official wheel
  index); treat search summaries as leads only.

### 2. A dependency conflict found before installing

- **What happened**: mace-torch pins `e3nn==0.4.4`, while MatterSim and SevenNet require
  `e3nn>=0.5`, so they cannot share one environment.
- **How it was found**: reading each package's dependency declaration on PyPI before installing.
- **What was done**: the workflow was split into `compute` (forces) and `analyze` (analysis), which
  communicate only through the file `phonopy_params.yaml`. Conflicting models will run only
  `compute`, in their own environments.

### 3. The phonopy 4.x API differs from older examples

- **What happened**: most examples use the phonopy 3.x API (for example
  `get_thermal_properties_dict` and `PhonopyQHA`). The installed version was 4.6.0, released the
  day before (2026-09-24): `run_*` methods now return result objects, `QHAResult` replaces
  `PhonopyQHA`, and QHA results are now in eV, Å and K.
- **How it was found**: the v4.6.0 source was read before writing any code.
- **What was done**: the code follows the source. Reading it also confirmed three things that are
  easy to miss in the documentation: thermodynamic quantities are per mole of primitive cells; the
  default cutoff frequency is 0; and the acoustic modes at Γ enter the sums unless
  `exclude_gamma_acoustic=True`.
- **Lesson**: when a library is newer than the examples at hand, read the installed source first.
  This rule is now in `CONTRIBUTING.md`.

### 4. MACE defaults to single precision

- **What happened**: `mace_mp()` defaults to `default_dtype="float32"`. Phonons come from force
  differences of order 1e-3 eV/Å between supercells displaced by 0.01 Å; single-precision noise is
  too large for that and easily produces spurious imaginary modes.
- **How it was found**: the default was noticed while reading the MACE source.
- **What was done**: `models.load_calculator` always passes `float64`, and `CONTRIBUTING.md` lists
  this as a mandatory rule.

### 5. Our code and phonopy differed by 3×10⁻⁶: not a bug, different physical constants

- **What happened**: our partition-function code and phonopy differed by about 3×10⁻⁶ in relative
  terms, far more than floating-point rounding.
- **How it was found**: in the cross-check on Al and Cu with the EMT potential.
- **What was done**: the phonopy source showed that it still uses CODATA 2006 constants (Planck
  constant 4.13566733×10⁻¹⁵ eV·s), while this project uses the exact values fixed by the 2019 SI
  redefinition; the two differ by about 6×10⁻⁷. With phonopy's constants substituted into our code
  the two agree to 10⁻¹⁰, and that check became a test.
- **Lesson**: trace a numerical difference to its cause. Show where it comes from first, then set
  the tolerance; do not loosen the tolerance just to make the test pass.

### 6. A spurious relative error where the free energy crosses zero

- **What happened**: one temperature failed the test above with a relative error of 4×10⁻⁵, although
  the absolute error was only 4.4×10⁻⁶ kJ/mol.
- **How it was found**: by inspecting the values after the test failed.
- **What was done**: F changes sign near that temperature, and close to zero a relative error is
  meaningless. An absolute tolerance of 1×10⁻⁵ kJ/mol was added for F only, with the reason in the
  test's docstring. The same point agrees to 10⁻¹⁰ in the swapped-constants test, which justifies
  the change.

### 7. phonopy 4.6 shifts even meshes by half a step

- **What happened**: a `MeshSymmetryFallbackWarning` appeared: even meshes are shifted by half a
  grid step by default, which breaks the point-group symmetry, so phonopy turned off the symmetry
  reduction.
- **How it was found**: from the runtime warning.
- **What was done**: switched to Γ-centred meshes (`is_gamma_center=True`), which restores the
  symmetry reduction and includes Γ in the mesh.

### 8. The pandas `.T` trap (a bug caught by tests)

- **What happened**: the temperature column was named `T` and filtered with
  `data[data.T == 298.15]`. In pandas, `DataFrame.T` is the transpose, so that line compared the
  whole transposed table.
- **How it was found**: two unit tests of the JANAF parser failed.
- **What was done**: the column was renamed to `temperature`.
- **Lesson**: write tests before trusting code. This bug looks perfectly reasonable when reading
  the code; only a test catches it.

### 9. A comment stated a result before it was checked

- **What happened**: while writing the configuration, the supercell size was annotated with
  "chosen from `scripts/convergence.py`", although that convergence script had not been written,
  let alone run.
- **How it was found**: in a review of the changes, before anything was run.
- **What was done**: the comment was changed to "to be checked"; after the convergence study had run
  (216 and 512 atoms give S(298 K) within 0.001 J/(K·mol)), the verified statement was written.
- **Lesson**: it is easy to write down what "should" hold before checking it. `CONTRIBUTING.md`
  now says that comments state only verified facts.

### 10. MACE-MP-0's entropy is 28 % too high: a bug or the model?

- **What happened**: MACE-MP-0 gave S(298 K) = 24.08 J/(K·mol) for silicon against the experimental
  18.82, 28 % too high. Its optical phonon at Γ is at 11.18 THz against about 15.6 THz measured.
- **How it was found**: in the very first end-to-end run. A deviation this large first suggested a
  program error, such as a factor of 2 in the units.
- **What was done**: a completely independent method was used as a check: move the two Si atoms in
  opposite directions by d and fit the energy E(d) to get the Γ optical frequency. This uses energies
  only and never goes through phonopy. It gave 11.190 THz against phonopy's 11.184 THz, a difference
  of 0.05 %. So the program is right and this is the model's "systematic softening", as reported by
  Deng et al. (2025). The check became a permanent test. The other three models all have entropy
  errors within ±0.6 J/(K·mol) at 298 K, which also supports the pipeline.
- **Lesson**: when a result disagrees with expectations, rule out program errors with an independent
  method before drawing physical conclusions.

### 11. Experimental data needs a traceable source

- **What happened**: for the phonon-dispersion figure, neutron-scattering frequencies of silicon at
  the X and L points (for example TA(X) ≈ 4.5 THz) were available only as remembered values; no
  accessible original table could be found to check them against.
- **What was done**: the figure shows only the experimental value that could be confirmed, the Raman
  line of silicon at 520 cm⁻¹ (15.6 THz). The X and L neutron data will be added after checking the
  original paper, G. Nilsson and G. Nelin, Phys. Rev. B 6, 3777 (1972).
- **Lesson**: remembered numbers may be right or slightly off; data without a source does not go into
  a figure or a table.

### 12. Small mistakes

- A wrong matplotlib parameter name: `axes.titleloc` does not exist, the correct one is
  `axes.titlelocation`. Fixed after the error at run time.
- `import mace` sat outside the block that silences deprecation warnings, so the warnings still
  appeared; found when running the tests.
- A context manager was entered by calling `__enter__()` by hand, an awkward pattern; rewritten
  with `@contextmanager` before running.

---

## 2026-09-25: stage 2 (nine crystals, five potentials)

### 13. A dependency conflict that did not exist

- **What happened**: the plan for stage 2 said that ORB-v3 conflicts with MACE and needs its own
  environment. That was a confusion with MatterSim and SevenNet, which need e3nn ≥ 0.5.
- **How it was found**: reading orb-models' dependency declaration on PyPI before installing:
  it does not depend on e3nn at all.
- **What was done**: orb-models 0.7.0 was added to the main environment; the resolver kept torch
  2.14, mace-torch 0.3.16 and e3nn 0.4.4 unchanged.
- **Lesson**: check a dependency claim against the package metadata before designing around it.

### 14. The orb-models 0.7 API differs from older examples

- **What happened**: in orb-models 0.7 a pretrained loader returns a pair (model, atoms adapter),
  and `ORBCalculator` needs both; older examples pass a single model. The loaders also compile the
  model with `torch.compile` by default, which needs Triton, and Triton is not available on Windows.
- **How it was found**: reading `orb_models/forcefield/pretrained.py` and
  `inference/calculator.py` before writing the loader.
- **What was done**: `models.load_calculator` unpacks the pair and passes `compile=False` and
  `precision="float64"`.

### 15. ORB-v3 predicts forces on a perfect crystal where symmetry requires none

- **What happened**: on a perfect 216-atom silicon supercell, where every force must vanish by
  symmetry, ORB-v3 gave forces up to 7.6×10⁻⁴ eV/Å; the MACE models give about 10⁻¹⁴ eV/Å. Force
  differences of about 10⁻³ eV/Å are what phonons are computed from.
- **How it was found**: a single-point test before the production runs.
- **What was done**: the neighbour search was ruled out first (the exact brute-force search gives
  the same 7.6×10⁻⁴). A rotation test then showed the cause: rotating a rattled supercell changes
  the rotated-back ORB forces by 1.8×10⁻³ eV/Å and the energy by 3 meV, while for MACE the changes
  are 6×10⁻¹⁴ eV/Å and zero. ORB learns rotational symmetry from data instead of building it in.
  The workflow already limits the effect: relaxation runs under `FixSymmetry`, the residual forces of
  the perfect supercell are subtracted, and the force constants are symmetrized with the symfc
  projector.
- **Lesson**: architectural symmetry is a property to test, not assume, before using forces from a
  model for finite differences.

### 16. Two materials have no melting point in JANAF

- **What happened**: the comparison range is 0.7 × the melting point, but the JANAF tables of SiC
  and AlN list no melting; both decompose first.
- **What was done**: instead of filling in a decomposition temperature from memory, the
  configuration sets an explicit comparison limit of 1500 K for these two, with the reason stated.
  All other melting points now come from the JANAF tables themselves (Si: 1685 K instead of the
  1687 K used before; the compared temperatures did not change).

### 17. MACE-MATPES-PBE-0 gives MgO an entropy 54 % too high: a workflow error or the model?

- **What happened**: for MgO, MACE-MATPES-PBE-0 gave S(298 K) = 41.6 J/(K·mol) against the
  experimental 26.9, with a lattice constant of 4.163 Å, smaller than experiment (4.212 Å), yet the
  softest phonons of all models. A smaller lattice with softer phonons is physically suspicious.
- **How it was found**: in the table of entropy errors after the production runs.
- **What was done**: two explanations were tested. First, a fine-tuned MACE model can carry several
  output heads, and the calculator might use the wrong one; all three MACE-OMAT-0-based models turned
  out to have a single head. Second, the relaxation could have stopped at a wrong point; a scan of the
  energy against the lattice constant puts the minimum at 4.165 Å, matching the relaxation, with a
  bulk modulus of 106 GPa against 153 GPa (MACE-OMAT-0) and 163 GPa (MACE-MATPES-r2SCAN-0). The
  deviation therefore comes from the model's energy surface and is reported as a result.
- **Lesson**: before reporting a large deviation as a property of a model, check the two things the
  workflow could have got wrong: which model output was used, and whether the structure is really at
  the model's minimum.

### 18. A timing distorted by a concurrent job

- **What happened**: `compute.json` recorded 1966 s for the ORB-v3 forces of MgO, against 3–55 s for
  every other run.
- **How it was found**: while reading the run log.
- **What was done**: the energy scan of entry 17 had been running on the same GPU at the same time.
  The MgO calculation was repeated alone: 15 s, with identical results, and the record was replaced.
- **Lesson**: do not run timing-sensitive jobs side by side on one GPU.

### 19. Overwriting a figure failed on Windows

- **What happened**: `savefig` failed with `OSError: [Errno 22] Invalid argument` for a PNG that
  already existed, and the failure moved from one file to another between attempts.
- **How it was found**: the figure script stopped with the error.
- **What was done**: Windows lets another process (an image viewer, the thumbnail cache, a virus
  scanner) hold a file open for a moment. Figures are now written to a temporary file and swapped in
  with `os.replace`, retried for up to 10 s.

---

## Later entries

(Continue here: what happened, how it was found, what was done, lesson.)
