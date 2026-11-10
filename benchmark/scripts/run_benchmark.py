"""
PySCF XC Functional Benchmark Runner for GMTKN55.

Reads GMTKN55 subset directories, computes DFT energies with specified
XC functionals, and compares against reference reaction energies.

Usage:
    python run_benchmark.py --xc "PBE,PBE" --subset W4-11 --basis def2-TZVP
    python run_benchmark.py --xc "PBE,PBE" --all --basis def2-TZVP
"""

import argparse
import os
import re
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# PySCF imports
try:
    from pyscf import gto, dft, scf
except ImportError:
    print("PySCF not installed. Run: pip install pyscf")
    sys.exit(1)


GMTKN55_DIR = Path(__file__).resolve().parent.parent / "GMTKN55"

# GMTKN55 categories and their subsets (from the paper)
CATEGORIES = {
    "small_reactions": [
        "G21EA", "G21IP", "G2RC", "HEAVY28", "HEAVYSB11",
        "HAL59", "W4-11", "ALK8", "ALKBDE10",
    ],
    "larger_reactions": [
        "MB16-43", "DARC", "ISO34", "ISOL24", "INV24",
        "BH76", "BHDIV10", "BHPERI", "BHROT27", "BHDIV10",
    ],
    "barrier_heights": [
        "BH76", "BHPERI", "BHROT27", "BHDIV10",
    ],
    "intermolecular_nci": [
        "S22", "S66", "RG18", "AHB21", "IL16",
        "ADIM6", "CDIE20", "NBPRC", "HAL59",
    ],
    "intramolecular_nci": [
        "IDISP", "ICONF", "ACONF", "SCONF", "PCONF21",
        "MCONF", "BUT14DIOL",
    ],
}


def parse_xyz(xyz_path):
    """Parse an XYZ file and return element symbols and coordinates."""
    with open(xyz_path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    atoms = []
    coords = []
    for line in lines[2:2 + n_atoms]:
        parts = line.strip().split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, coords


def build_mol(atoms, coords, basis, charge=0, spin=0, verbose=0):
    """Build a PySCF Mole object from atoms and coordinates."""
    mol = gto.Mole()
    mol.atom = list(zip(atoms, coords))
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = verbose
    mol.build()
    return mol


def compute_energy(mol, xc_code, ext_params=None):
    """Compute DFT energy for a molecule with given XC functional."""
    if mol.spin > 0:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = xc_code

    # Apply ext_params if provided (for REWIRE operations)
    if ext_params:
        from pyscf.dft import libxc
        libxc.register_custom_functional_(
            f'CUSTOM_{xc_code.replace(",", "_")}',
            xc_code,
            ext_params=ext_params,
        )
        mf.xc = f'CUSTOM_{xc_code.replace(",", "_")}'

    mf.conv_tol = 1e-8
    mf.max_cycle = 200
    mf.kernel()
    return mf.e_tot


def parse_res_file(res_path):
    """Parse a GMTKN55 .res file and extract reaction definitions.

    Returns list of dicts with keys:
        species: list of (name, stoich_coeff, element_counts)
        ref_value: reference energy in kcal/mol
    """
    reactions = []
    with open(res_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('if') or line.startswith('then') or line.startswith('else') or line.startswith('fi') or 'tmer' in line.split()[0:2] or line.startswith('f=') or line.startswith('w='):
            continue

        # Parse reaction line: species groups, x separator, stoich coefficients, ref value
        # Format: $tmer {mol1,mol2,...}/$f {mol3,...}/$f ... x coeff1 coeff2 ... $w ref_value
        # We need to extract molecule names and reference values
        try:
            # Extract reference value (last number on the line)
            parts = line.split()
            ref_val = None
            for p in reversed(parts):
                try:
                    ref_val = float(p)
                    break
                except ValueError:
                    continue
            if ref_val is None:
                continue

            # Extract molecule names from curly braces
            mol_names = re.findall(r'\{([^}]+)\}', line)
            species_list = []
            for group in mol_names:
                names = [n.strip() for n in group.split(',')]
                species_list.append(names)

            # Extract stoichiometry coefficients (numbers after 'x')
            stoich_start = line.index('x')
            stoich_part = line[stoich_start + 1:]
            stoich_nums = []
            for p in stoich_part.split():
                try:
                    stoich_nums.append(int(p))
                except ValueError:
                    if p == '$w':
                        break

            reactions.append({
                'species': species_list,
                'stoich': stoich_nums,
                'ref_value': ref_val,
            })
        except (ValueError, IndexError):
            continue

    return reactions


def compute_reaction_energy(subset_dir, reaction, xc_code, basis, ext_params=None):
    """Compute reaction energy for a GMTKN55 reaction definition.

    reaction_energy = sum(stoich_i * E(molecule_i))
    """
    energy = 0.0
    for group_idx, (species_group, stoich) in enumerate(
        zip(reaction['species'], reaction['stoich'])
    ):
        for mol_name in species_group:
            mol_dir = subset_dir / mol_name
            xyz_path = mol_dir / "struc.xyz"
            if not xyz_path.exists():
                # Try coord file (TurboMole format)
                coord_path = mol_dir / "coord"
                if coord_path.exists():
                    # Convert coord to xyz-like format
                    atoms, coords = parse_turbomole_coord(coord_path)
                else:
                    print(f"  Warning: geometry not found for {mol_name}")
                    return None
            else:
                atoms, coords = parse_xyz(xyz_path)

            # Detect spin from directory name or charge
            charge = 0
            spin = 0
            # Check for charge/spin info in directory
            info_path = mol_dir / "info"
            if info_path.exists():
                with open(info_path) as f:
                    for l in f:
                        if 'charge' in l.lower():
                            charge = int(l.split()[-1])
                        if 'spin' in l.lower() or 'uhf' in l.lower():
                            spin = int(l.split()[-1])

            try:
                mol = build_mol(atoms, coords, basis, charge=charge, spin=spin, verbose=0)
                e = compute_energy(mol, xc_code, ext_params)
                energy += stoich * e
                mol.close()
            except Exception as ex:
                print(f"  Error computing {mol_name}: {ex}")
                return None

    # Convert from Hartree to kcal/mol
    energy_kcal = energy * 627.509474
    return energy_kcal


def parse_turbomole_coord(coord_path):
    """Parse TurboMole coord file."""
    atoms = []
    coords = []
    with open(coord_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('$') or not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[-1])
            coords.append([float(x) for x in parts[0:3]])
    return atoms, coords


def run_subset(subset_name, xc_code, basis, ext_params=None, verbose=1):
    """Run a single GMTKN55 subset and compute errors."""
    subset_dir = GMTKN55_DIR / subset_name
    if not subset_dir.exists():
        print(f"Subset {subset_name} not found in {GMTKN55_DIR}")
        return None

    res_path = subset_dir / ".res"
    if not res_path.exists():
        print(f"No .res file for {subset_name}")
        return None

    reactions = parse_res_file(res_path)
    if not reactions:
        print(f"No reactions parsed from {subset_name}")
        return None

    errors = []
    computed = []
    references = []
    for rxn in reactions:
        e_calc = compute_reaction_energy(subset_dir, rxn, xc_code, basis, ext_params)
        if e_calc is None:
            continue
        e_ref = rxn['ref_value']
        error = abs(e_calc - e_ref)
        errors.append(error)
        computed.append(e_calc)
        references.append(e_ref)
        if verbose >= 2:
            print(f"  ref={e_ref:.3f} calc={e_calc:.3f} err={error:.3f} kcal/mol")

    if not errors:
        return None

    mae = np.mean(errors)
    max_err = np.max(errors)
    rms = np.sqrt(np.mean(np.array(errors) ** 2))

    result = {
        'subset': subset_name,
        'xc': xc_code,
        'basis': basis,
        'n_reactions': len(errors),
        'mae_kcal': mae,
        'max_error_kcal': max_err,
        'rms_kcal': rms,
        'errors': errors,
        'computed': computed,
        'references': references,
    }

    if verbose >= 1:
        print(f"{subset_name}: MAE={mae:.3f} kcal/mol, MaxErr={max_err:.3f}, "
              f"RMS={rms:.3f}, N={len(errors)}")

    return result


def run_all(xc_code, basis, ext_params=None, subsets=None, verbose=1):
    """Run multiple subsets and aggregate results."""
    if subsets is None:
        # Use key subsets covering all categories
        subsets = [
            "W4-11",     # atomization energies
            "G21EA",     # electron affinities
            "G21IP",     # ionization potentials
            "BH76",      # barrier heights
            "S22",       # noncovalent interactions (small)
            "S66",       # noncovalent interactions (medium)
            "RSE43",     # radical stabilization
            "ISO34",     # isomerization energies
            "DC13",      # difficult cases
        ]

    results = {}
    for subset in subsets:
        r = run_subset(subset, xc_code, basis, ext_params, verbose)
        if r:
            results[subset] = r

    # Aggregate
    all_errors = []
    for r in results.values():
        all_errors.extend(r['errors'])

    if all_errors:
        overall_mae = np.mean(all_errors)
        overall_rms = np.sqrt(np.mean(np.array(all_errors) ** 2))
        print(f"\n=== Overall: MAE={overall_mae:.3f} kcal/mol, "
              f"RMS={overall_rms:.3f}, N={len(all_errors)} ===")

    return results


def main():
    parser = argparse.ArgumentParser(description="PySCF XC Functional Benchmark Runner")
    parser.add_argument("--xc", required=True, help="XC functional code (e.g. 'PBE,PBE', 'B3LYP')")
    parser.add_argument("--basis", default="def2-TZVP", help="Basis set (default: def2-TZVP)")
    parser.add_argument("--subset", default=None, help="Single subset name (e.g. W4-11)")
    parser.add_argument("--all", action="store_true", help="Run all key subsets")
    parser.add_argument("--ext-params", default=None, help="JSON string of ext_params for REWIRE")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0-2)")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    ext_params = None
    if args.ext_params:
        ext_params = json.loads(args.ext_params)

    if args.subset:
        result = run_subset(args.subset, args.xc, args.basis, ext_params, args.verbose)
        results = {args.subset: result} if result else {}
    elif args.all:
        results = run_all(args.xc, args.basis, ext_params, verbose=args.verbose)
    else:
        # Default: run key subsets
        results = run_all(args.xc, args.basis, ext_params, verbose=args.verbose)

    if args.output:
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for k, v in results.items():
            d = dict(v)
            d['errors'] = list(d['errors'])
            d['computed'] = list(d['computed'])
            d['references'] = list(d['references'])
            serializable[k] = d
        with open(args.output, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()