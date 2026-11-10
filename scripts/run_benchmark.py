"""Run GMTKN55 benchmark — one subset at a time, saves incremental results.

Usage:
    PYTHONPATH=src python3 scripts/run_benchmark.py --subset SIE4x4 [--basis def2-TZVP]
    PYTHONPATH=src python3 scripts/run_benchmark.py --subset BH76
    ...
    # After all subsets done:
    PYTHONPATH=src python3 scripts/run_benchmark.py --summarize
"""

import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from pyscf import gto, dft
from xc.benchmark.gmtkn55 import parse_res_file, load_molecule_geometry

HARTREE_TO_KCAL = 627.509474
RESULTS_DIR = Path("benchmark/results")
FUNCTIONALS = [
    # Rung 1-2: LDA/GGA
    "PBE",
    # Rung 3: meta-GGA
    "SCAN", "R2SCAN", "B97M-V",
    # Rung 4: hybrid GGA
    "PBE0", "B3LYP",
    # Rung 4: hybrid meta-GGA
    "SCAN0", "r2SCAN0", "TPSSh", "M06", "M06-2X",
    # Rung 4: range-separated
    "CAM-B3LYP", "wB97X-V", "wB97M-V",
]


def run_subset(subset: str, functionals: list[str], basis: str, gmtkn55_dir: Path) -> dict:
    """Evaluate all functionals on one GMTKN55 subset."""
    subset_dir = gmtkn55_dir / subset
    rxns = parse_res_file(subset_dir / ".res")
    print(f"{subset}: {len(rxns)} reactions")

    mol_cache = {}

    def get_energy(mol_name, xc_str):
        key = (mol_name, xc_str)
        if key in mol_cache:
            return mol_cache[key]
        mol_dir = subset_dir / mol_name
        if not mol_dir.is_dir():
            mol_cache[key] = None
            return None
        charge = 0
        spin = 0
        chrg_file = mol_dir / ".CHRG"
        if chrg_file.exists():
            try:
                charge = int(chrg_file.read_text().strip())
            except (ValueError, IndexError):
                pass
        uhf_file = mol_dir / ".UHF"
        if uhf_file.exists():
            try:
                spin = int(uhf_file.read_text().strip())
            except (ValueError, IndexError):
                pass
        geom = load_molecule_geometry(subset_dir, mol_name)
        if geom is None:
            mol_cache[key] = None
            return None
        atoms, coords = geom
        try:
            mol = gto.M(
                atom=list(zip(atoms, coords)), basis=basis,
                charge=charge, spin=spin, verbose=0, unit="Angstrom",
            )
            if spin > 0:
                mf = dft.UKS(mol)
            else:
                mf = dft.RKS(mol)
            mf.xc = xc_str
            mf.conv_tol = 1e-8
            mf.max_cycle = 200
            mf.kernel()
            if not mf.converged:
                mol_cache[key] = None
                return None
            mol_cache[key] = mf.e_tot
            return mf.e_tot
        except Exception:
            mol_cache[key] = None
            return None

    results = {}
    for xc_str in functionals:
        errors = []
        n_fail = 0
        t0 = time.time()
        for rxn in rxns:
            rxn_energy = 0.0
            failed = False
            for group, stoich in zip(rxn["species_groups"], rxn["stoich_coeffs"]):
                for mol_name in group:
                    e = get_energy(mol_name, xc_str)
                    if e is None:
                        failed = True
                        break
                    rxn_energy += stoich * e
                if failed:
                    break
            if failed:
                n_fail += 1
                continue
            rxn_kcal = rxn_energy * HARTREE_TO_KCAL
            ref_kcal = rxn["ref_value"]
            errors.append(abs(rxn_kcal - ref_kcal))

        mae = float(np.mean(errors)) if errors else None
        rms = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else None
        max_err = float(np.max(errors)) if errors else None
        dt = time.time() - t0
        results[xc_str] = {
            "mae_kcal": round(mae, 3) if mae else None,
            "rms_kcal": round(rms, 3) if rms else None,
            "max_error_kcal": round(max_err, 3) if max_err else None,
            "n_reactions": len(errors),
            "n_failures": n_fail,
            "walltime_s": round(dt, 1),
        }
        if mae:
            print(f"  {xc_str:12s} MAE={mae:.3f} ({len(errors)} rxns, {n_fail} fail, {dt:.0f}s)")
        else:
            print(f"  {xc_str:12s} ALL FAILED ({n_fail} failures)")

    return results


def summarize(all_data: dict, subsets: list[str]) -> None:
    """Print comparison table and optimization space analysis."""
    print(f"\n{'='*110}")
    print(f"  COMPARISON TABLE (MAE in kcal/mol)")
    print(f"{'='*110}")
    header = f"{'Functional':<14}"
    for s in subsets:
        header += f" {s:>8}"
    header += f" {'Mean':>8}"
    print(header)
    print("-" * 110)

    for xc_str in FUNCTIONALS:
        row = f"{xc_str:<14}"
        subset_maes = []
        for s in subsets:
            r = all_data.get(xc_str, {}).get(s, {})
            mae = r.get("mae_kcal")
            if mae is not None:
                row += f" {mae:>8.2f}"
                subset_maes.append(mae)
            else:
                row += f" {'—':>8}"
        if subset_maes:
            overall = np.mean(subset_maes)
            row += f" {overall:>8.2f}"
        else:
            row += f" {'—':>8}"
        print(row)

    # Best per subset
    print(f"\n{'='*110}")
    print(f"  BEST FUNCTIONAL PER SUBSET")
    print(f"{'='*110}")
    for s in subsets:
        best_mae = float("inf")
        best_func = ""
        for xc_str in FUNCTIONALS:
            mae = all_data.get(xc_str, {}).get(s, {}).get("mae_kcal")
            if mae is not None and mae < best_mae:
                best_mae = mae
                best_func = xc_str
        if best_func:
            print(f"  {s:<12}: {best_func} ({best_mae:.3f} kcal/mol)")

    # Gap: PBE0 vs best
    print(f"\n{'='*110}")
    print(f"  PBE0 vs BEST-IN-CLASS (optimization gap)")
    print(f"{'='*110}")
    pbe0_data = all_data.get("PBE0", {})
    for s in subsets:
        pbe0_mae = pbe0_data.get(s, {}).get("mae_kcal")
        best_mae = float("inf")
        best_func = ""
        for xc_str in FUNCTIONALS:
            mae = all_data.get(xc_str, {}).get(s, {}).get("mae_kcal")
            if mae is not None and mae < best_mae:
                best_mae = mae
                best_func = xc_str
        if pbe0_mae is not None and best_func:
            gap = pbe0_mae - best_mae
            print(f"  {s:<12}: PBE0={pbe0_mae:.3f}, best={best_func}={best_mae:.3f}, "
                  f"gap={gap:+.3f} kcal/mol")


def main():
    parser = argparse.ArgumentParser(description="GMTKN55 benchmark runner")
    parser.add_argument("--subset", type=str, help="Single subset to evaluate")
    parser.add_argument("--subsets", nargs="+",
                        default=["SIE4x4", "BH76", "G21IP", "G21EA", "W4-11", "S66", "ISO34"],
                        help="All subsets to evaluate")
    parser.add_argument("--basis", default="def2-TZVP")
    parser.add_argument("--gmtkn55-dir", default="benchmark/GMTKN55")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize all saved results")
    parser.add_argument("--functionals", nargs="+",
                        help="Override FUNCTIONALS list (e.g. --functionals wB97M-V M06-2X)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gmtkn55_dir = Path(args.gmtkn55_dir)
    func_list = args.functionals if args.functionals else FUNCTIONALS

    if args.summarize:
        # Load all saved results and print summary
        all_data = {}
        for xc_str in FUNCTIONALS:
            fpath = RESULTS_DIR / f"{xc_str}.json"
            if fpath.exists():
                all_data[xc_str] = json.loads(fpath.read_text())
        summarize(all_data, args.subsets)
        return

    if args.subset:
        # Run single subset for all functionals
        subset = args.subset
        print(f"Running {subset} with basis={args.basis}, functionals={func_list}")
        results = run_subset(subset, func_list, args.basis, gmtkn55_dir)

        # Save per-functional results
        for xc_str, r in results.items():
            fpath = RESULTS_DIR / f"{xc_str}.json"
            if fpath.exists():
                data = json.loads(fpath.read_text())
            else:
                data = {}
            data[subset] = r
            fpath.write_text(json.dumps(data, indent=2))

        # Quick summary
        print(f"\n  {subset} summary:")
        for xc_str in func_list:
            mae = results.get(xc_str, {}).get("mae_kcal")
            if mae:
                print(f"    {xc_str:14s}: {mae:.3f} kcal/mol")
        return

    # Run all subsets sequentially
    all_data = {}
    for xc_str in func_list:
        fpath = RESULTS_DIR / f"{xc_str}.json"
        if fpath.exists():
            all_data[xc_str] = json.loads(fpath.read_text())

    for subset in args.subsets:
        if all(xc_str in all_data and subset in all_data[xc_str]
               for xc_str in func_list):
            print(f"  {subset}: already computed, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  {subset}")
        print(f"{'='*60}")
        results = run_subset(subset, func_list, args.basis, gmtkn55_dir)

        for xc_str, r in results.items():
            if xc_str not in all_data:
                all_data[xc_str] = {}
            all_data[xc_str][subset] = r
            fpath = RESULTS_DIR / f"{xc_str}.json"
            fpath.write_text(json.dumps(all_data[xc_str], indent=2))

    summarize(all_data, args.subsets)


if __name__ == "__main__":
    main()