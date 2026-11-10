"""GMTKN55 dataset parser — reads XYZ geometries and .res reference energies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def parse_xyz(xyz_path: Path) -> tuple[list[str], list[list[float]]]:
    """Parse XYZ file → (element_symbols, coordinates)."""
    lines = xyz_path.read_text().strip().splitlines()
    n_atoms = int(lines[0].strip())
    atoms, coords = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.strip().split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, coords


def parse_turbomole_coord(coord_path: Path) -> tuple[list[str], list[list[float]]]:
    """Parse TurboMole coord file → (element_symbols, coordinates)."""
    atoms, coords = [], []
    lines = coord_path.read_text().strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4 and not line.strip().startswith('$'):
            atoms.append(parts[-1])
            coords.append([float(x) for x in parts[0:3]])
    return atoms, coords


def _parse_res_line(line: str) -> dict | None:
    """Parse a single reaction line from a .res file.

    Two formats:
    1. Brace format: $tmer h{,2+_1.0}/$f x 1 -1 $w 64.4
       Species inside braces are comma-separated, each gets its own stoich coeff.
    2. Space format: $tmer h/$f n2o/$f n2ohts/$f x -1 -1 1 $w 17.7
       Species are space-separated, each ending with /$f.
    """
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('if') or \
       line.startswith('then') or line.startswith('else') or \
       line.startswith('fi') or line.startswith('f=') or line.startswith('w='):
        return None

    # Remove $tmer prefix
    if '$tmer' in line:
        line = line[line.index('$tmer') + 5:]

    # Extract reference value (last number before end of line, after $w)
    ref_val = None
    w_match = re.search(r'\$w\s+([-+]?\d*\.?\d+)', line)
    if w_match:
        ref_val = float(w_match.group(1))

    # Remove $w and everything after
    if '$w' in line:
        line = line[:line.index('$w')]

    # Extract stoichiometry coefficients (numbers after 'x')
    stoich_coeffs = []
    x_match = re.search(r'\bx\s+(.*?)$', line)
    if x_match:
        stoich_str = x_match.group(1).strip()
        for token in stoich_str.split():
            try:
                stoich_coeffs.append(int(token))
            except ValueError:
                break
        line = line[:line.index('x')]

    if not stoich_coeffs or ref_val is None:
        return None

    # Extract species
    species_groups = []

    # Check for brace format: name{spec1,spec2,...}
    # The prefix is prepended to each comma-separated entry inside braces
    brace_match = re.search(r'(\w*)\{([^}]+)\}', line)
    if brace_match:
        prefix = brace_match.group(1)
        inner = brace_match.group(2)
        # Split by comma, keeping empty entries (they represent the prefix itself)
        entries = inner.split(',')
        names = []
        for entry in entries:
            entry = entry.strip()
            if entry:
                names.append(prefix + entry)
            else:
                names.append(prefix)
        species_groups = [[n] for n in names]
    else:
        # Space-separated format: species/$f
        parts = line.strip().split()
        for part in parts:
            part = part.strip()
            if not part or part.startswith('$'):
                continue
            # Remove /$f suffix
            if '/$f' in part:
                mol_name = part[:part.index('/$f')].strip()
            else:
                mol_name = part.strip()
            if mol_name:
                species_groups.append([mol_name])

    if not species_groups:
        return None

    # For brace format, species_groups already has one group per species
    # For space format, same
    # Stoich coeffs should match number of species groups
    if len(species_groups) != len(stoich_coeffs):
        return None

    return {
        'species_groups': species_groups,
        'stoich_coeffs': stoich_coeffs,
        'ref_value': ref_val,
    }


def parse_res_file(res_path: Path) -> list[dict]:
    """Parse a GMTKN55 .res file → list of reaction definitions."""
    reactions = []
    text = res_path.read_text()

    for line in text.splitlines():
        rxn = _parse_res_line(line)
        if rxn is not None:
            reactions.append(rxn)

    return reactions


def load_molecule_geometry(subset_dir: Path, mol_name: str) -> Optional[tuple[list[str], list[list[float]]]]:
    """Load geometry for a molecule from a GMTKN55 subset directory."""
    mol_dir = subset_dir / mol_name
    xyz_path = mol_dir / "struc.xyz"
    if xyz_path.exists():
        return parse_xyz(xyz_path)
    coord_path = mol_dir / "coord"
    if coord_path.exists():
        return parse_turbomole_coord(coord_path)
    return None