"""Universal ML interatomic potentials as ASE calculators."""

from __future__ import annotations

import warnings

from ase.calculators.calculator import Calculator

from forces2free.config import Model


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_calculator(model: Model, device: str | None = None) -> Calculator:
    """Return the model as an ASE calculator in double precision.

    Phonons come from force differences of about 1e-3 eV/A between supercells
    displaced by 0.01 A; single precision (the MACE default) is too noisy for
    that and produces spurious imaginary modes.
    """
    device = device or default_device()
    if model.family == "mace":
        # torch 2.14 flags the TorchScript calls inside e3nn/MACE as deprecated;
        # they still work, so keep the log readable while importing and loading.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
            from mace.calculators import mace_mp

            return mace_mp(
                model=model.checkpoint, device=device, default_dtype="float64", dispersion=model.dispersion
            )
    if model.dispersion:
        raise ValueError(f"dispersion is only implemented for MACE models, not {model.key}")
    if model.family == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        # orb-models 0.7 returns (model, atoms adapter). torch.compile is off
        # because it needs Triton, which is not available on Windows.
        loader = getattr(pretrained, model.checkpoint)
        network, adapter = loader(device=device, precision="float64", compile=False)
        return ORBCalculator(network, adapter, device=device)
    raise ValueError(f"unknown model family {model.family!r} for {model.key}")
