# Development conventions for forces2free-energy

## What the project does

Computes phonons and thermodynamic properties of crystals with universal machine-learning
interatomic potentials (uMLIPs) and compares them with NIST-JANAF experimental data.
Pipeline: relax → phonopy finite displacements → force constants → frequencies on a q-mesh →
partition function → F, S, Cv → comparison with experiment.
The roadmap is in the Status section of README.md; stages 1 (Si) and 2 (nine crystals, five potentials) are done.

Language: English for everything in the repository: code, comments, docs, commit messages and
script output.

## Environment

- uv manages the environment and pins Python to 3.12: `uv sync`, `uv run ...`. Python 3.14 does
  not work: matscipy, a MACE dependency, has no 3.14 wheel for Windows.
- The CUDA build of PyTorch is about 3 GB. If the system drive is short on space, point
  `UV_CACHE_DIR` to another drive.
- MACE and ORB models run in the main environment. MatterSim, SevenNet and others need e3nn ≥ 0.5,
  which conflicts with MACE's e3nn 0.4.4. They will get their own environment that only runs
  `pipeline.compute` and hands `phonopy_params.yaml` to the main environment for analysis.
- `uv.lock` is the source of truth for versions. **Read the installed phonopy source
  (`.venv/Lib/site-packages/phonopy/`) before writing phonopy code.** The 4.x API differs a lot from
  older versions and tutorials; do not write it from memory.

## Physics and numerics (must follow)

- Phonon calculations run in double precision: MACE with `default_dtype="float64"` (MACE defaults
  to float32), ORB with `precision="float64"` and `compile=False` (torch.compile needs Triton, which
  is not available on Windows).
- ORB-v3 is not exactly rotation-equivariant, so it gives small forces (up to ~2e-3 eV/Å) where
  symmetry requires none. Keep `FixSymmetry` in relaxations, the subtraction of the undisplaced
  supercell's forces, and the symfc symmetrization of force constants.
- Relaxation: `FixSymmetry` + `FrechetCellFilter`, fmax ≤ 1e-3 eV/Å. The space group must be the
  same before and after.
- Units: phonopy frequencies are ordinary frequencies ν in THz, not angular frequencies ω = 2πν;
  the energy of one phonon is E = hν.
- phonopy reports thermodynamic quantities per mole of **primitive cells**. This project always
  reports per mole of **formula units**, i.e. divided by the number of formula units Z in the
  primitive cell (Si: 2 atoms, Z = 2). F and U in kJ/mol; S, Cv and Cp in J/(K·mol).
- The three acoustic modes at Γ are zero in theory and tiny numbers of either sign in practice.
  Set them to zero before summing (phonopy: `exclude_gamma_acoustic=True`).
- Use Γ-centred q-meshes (`is_gamma_center=True`). phonopy 4.6 shifts even meshes by half a step
  by default, which disables the symmetry reduction.
- phonopy uses CODATA 2006 constants and this project the exact SI 2019 values, so differences of
  about 1e-6 between the two are expected.
- The harmonic approximation gives Cv; JANAF gives Cp. Say so in every comparison. Stage 3 adds the
  quasi-harmonic approximation and compares Cp with Cp. In phonopy 4.6, `QHAResult` replaces
  `PhonopyQHA`, and QHA results are in eV, Å and K.
- Comparisons with experiment stop at 0.7 × the melting point, taken from the JANAF table. For
  materials that decompose before melting (SiC, AlN), `compare_up_to` sets an explicit limit.

## Workflow

- Run `uv run pytest` after every change; the slow tests that load a model run with
  `uv run pytest -m mlip`.
- **Never edit a test or loosen a tolerance just to make it pass.** If a change is really needed,
  find and verify the cause first (for example, prove a constants mismatch by recomputing with
  phonopy's constants), then explain it in the commit message and in `docs/dev-log.md`.
- Comments and configuration notes state only verified facts; write "to be checked" for anything
  that has not been run yet.
- Use experimental data only with a traceable source, cited in the code; never fill it in from memory.
- Small commits, one change each, with English messages that say what changed and why.
- Record every problem that gets caught in `docs/dev-log.md`: what happened, how it was found,
  what was done, and the lesson. Do not invent entries.
- Open text files with an explicit `encoding="utf-8"`; the Windows default is not UTF-8.
- Figure colours are fixed per model by its position in `configs/models.yaml`, and experiment is
  always drawn in black. The shared style is in `src/forces2free/plotting.py`.
- Record timings only for runs that had the GPU to themselves.
