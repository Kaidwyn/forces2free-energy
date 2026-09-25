"""Run compute + analyze for every (model, material) pair listed in configs/.

    uv run python scripts/run_all.py                                 # everything missing
    uv run python scripts/run_all.py --models mace-mp-0 --materials Si
    uv run python scripts/run_all.py --force                         # recompute forces
    uv run python scripts/run_all.py --analyze-only                  # reuse saved forces
"""

import argparse

from forces2free import pipeline
from forces2free.config import load_materials, load_models, results_dir
from forces2free.models import default_device, load_calculator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help="model keys from configs/models.yaml")
    parser.add_argument("--materials", nargs="+", help="material keys from configs/materials.yaml")
    parser.add_argument("--force", action="store_true", help="recompute even if forces exist")
    parser.add_argument("--analyze-only", action="store_true", help="skip the force calculation")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    materials = load_materials()
    models = load_models()
    device = args.device or default_device()
    for model_key in args.models or list(models):
        model = models[model_key]
        calc = None
        for material_key in args.materials or list(materials):
            material = materials[material_key]
            saved = results_dir(model.key, material.key) / "phonopy_params.yaml"
            if not args.analyze_only and (args.force or not saved.exists()):
                calc = calc or load_calculator(model, device)
                info = pipeline.compute(material, model, calc, device=device)
                a = info["relax"]["cell_lengths_A"][0]
                t = info["time_s"]
                print(f"[{model.key}/{material.key}] a = {a:.4f} A, relax {t['relax']:.1f} s, forces {t['forces']:.1f} s")
            summary = pipeline.analyze(material, model.key)
            line = f"[{model.key}/{material.key}] S(298) = {summary['S_298_J_K_mol']:.3f} J/(K mol)"
            if "janaf" in summary:
                j = summary["janaf"]
                line += f" (exp {j['S_298_exp']:.3f}), S MAE {j['S_mae_J_K_mol']:.2f}, Cp MAPE {j['Cp_mape_pct']:.1f}%"
            if summary["imaginary_modes"]:
                line += f"  WARNING: {summary['imaginary_modes']} imaginary modes"
            print(line)


if __name__ == "__main__":
    main()
