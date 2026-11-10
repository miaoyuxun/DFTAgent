"""Band-gap diagnostics shared by ZnO evaluators and search feedback."""

from __future__ import annotations

from typing import Any

HARTREE_TO_EV = 27.211386245988


def analyze_kpoint_gap(
    mo_energy: Any,
    mo_occ: Any,
    *,
    kpoints_cartesian: Any | None = None,
    occupation_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Return sampled, direct, and Gamma-point gaps from final SCF orbitals."""
    import numpy as np

    energies = np.asarray(mo_energy, dtype=float)
    occupations = np.asarray(mo_occ, dtype=float)
    if energies.shape != occupations.shape:
        raise ValueError("MO energies and occupations must have matching shapes")
    if energies.ndim == 1:
        energies = energies[None, :]
        occupations = occupations[None, :]
    if energies.ndim != 2:
        raise ValueError(f"Expected (nkpoint, norbital) arrays, got shape={energies.shape}")

    fractional_mask = (
        (occupations > occupation_tolerance)
        & (occupations < 2.0 - occupation_tolerance)
    )
    has_fractional = bool(np.any(fractional_mask))
    per_kpoint: list[dict[str, Any]] = []
    for index, (energy, occupation) in enumerate(zip(energies, occupations)):
        occupied = energy[occupation > occupation_tolerance]
        unoccupied = energy[occupation <= occupation_tolerance]
        if occupied.size == 0 or unoccupied.size == 0:
            continue
        vbm = float(np.max(occupied))
        cbm = float(np.min(unoccupied))
        per_kpoint.append(
            {
                "kpoint_index": index,
                "vbm_hartree": vbm,
                "cbm_hartree": cbm,
                "direct_gap_ev": max(0.0, cbm - vbm) * HARTREE_TO_EV,
            }
        )
    if not per_kpoint:
        raise ValueError("Could not identify occupied and unoccupied orbitals at any k-point")

    vbm_record = max(per_kpoint, key=lambda item: item["vbm_hartree"])
    cbm_record = min(per_kpoint, key=lambda item: item["cbm_hartree"])
    sampled_gap_ev = max(
        0.0,
        float(cbm_record["cbm_hartree"]) - float(vbm_record["vbm_hartree"]),
    ) * HARTREE_TO_EV
    minimum_direct = min(per_kpoint, key=lambda item: item["direct_gap_ev"])

    gamma_index = 0
    if kpoints_cartesian is not None:
        kpoints = np.asarray(kpoints_cartesian, dtype=float)
        if kpoints.ndim == 2 and len(kpoints) == len(energies):
            gamma_index = int(np.argmin(np.linalg.norm(kpoints, axis=1)))
    gamma_record = next(
        (item for item in per_kpoint if item["kpoint_index"] == gamma_index),
        per_kpoint[0],
    )
    gamma_gap_ev = float(gamma_record["direct_gap_ev"])
    vbm_index = int(vbm_record["kpoint_index"])
    cbm_index = int(cbm_record["kpoint_index"])
    is_direct = vbm_index == cbm_index
    return {
        "gap_ev": sampled_gap_ev,
        "sampled_gap_ev": sampled_gap_ev,
        "gamma_gap_ev": gamma_gap_ev,
        "minimum_direct_gap_ev": float(minimum_direct["direct_gap_ev"]),
        "minimum_direct_gap_kpoint_index": int(minimum_direct["kpoint_index"]),
        "directness_delta_ev": gamma_gap_ev - sampled_gap_ev,
        "vbm_hartree": float(vbm_record["vbm_hartree"]),
        "cbm_hartree": float(cbm_record["cbm_hartree"]),
        "vbm_kpoint_index": vbm_index,
        "cbm_kpoint_index": cbm_index,
        "gamma_kpoint_index": int(gamma_record["kpoint_index"]),
        "is_direct_gap": is_direct,
        "is_gamma_direct_gap": bool(is_direct and vbm_index == int(gamma_record["kpoint_index"])),
        "is_metallic": bool(sampled_gap_ev <= 0.0),
        "has_fractional_occupations": has_fractional,
        "per_kpoint": per_kpoint,
    }
