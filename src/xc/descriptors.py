"""Restricted descriptor corrections for custom XC candidates.

This module intentionally exposes a small template DSL instead of executing
LLM-written Python. Descriptor corrections are additive, GGA-level terms that
can be combined with ordinary libxc components and HF exchange.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


DESCRIPTOR_ROLE = "descriptor_correction"
ALLOWED_DESCRIPTOR_TEMPLATES = {
    "bounded_gradient",
    "tanh_gradient",
    "low_density_gradient",
    "density_saturation",
}
DESCRIPTOR_PARAM_BOUNDS = {
    "scale": (-0.05, 0.05),
    "gradient_sensitivity": (0.05, 10.0),
    "density_decay": (0.0, 10.0),
    "rho_power": (0.0, 1.0),
}


def normalize_descriptor_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical, bounded descriptor spec."""
    spec = dict(raw or {})
    nested = spec.get("descriptor")
    if isinstance(nested, dict):
        spec = {**spec, **nested}
        spec.pop("descriptor", None)

    template = str(spec.get("template") or spec.get("family") or "bounded_gradient").strip()
    params = dict(spec.get("params") or {})
    for key in ("scale", "gradient_sensitivity", "density_decay", "rho_power"):
        if key in spec and key not in params:
            params[key] = spec[key]

    canonical = {
        "name": str(spec.get("name") or f"DESC_{template}").strip()[:64],
        "template": template,
        "xctype": "GGA",
        "role": DESCRIPTOR_ROLE,
        "params": {
            "scale": float(params.get("scale", 0.0)),
            "gradient_sensitivity": float(params.get("gradient_sensitivity", params.get("a", 1.0))),
            "density_decay": float(params.get("density_decay", 0.0)),
            "rho_power": float(params.get("rho_power", 1.0 / 3.0)),
        },
        "intended_effect": str(spec.get("intended_effect") or spec.get("rationale") or "").strip(),
    }
    return canonical


def validate_descriptor_spec(spec: dict[str, Any]) -> list[str]:
    """Validate descriptor schema and cheap numerical stability."""
    violations: list[str] = []
    template = str(spec.get("template", ""))
    if template not in ALLOWED_DESCRIPTOR_TEMPLATES:
        violations.append(f"HARD: unsupported descriptor template {template!r}")

    params = spec.get("params", {})
    if not isinstance(params, dict):
        violations.append("HARD: descriptor params must be a dict")
        return violations
    for key, (lo, hi) in DESCRIPTOR_PARAM_BOUNDS.items():
        value = params.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            violations.append(f"HARD: descriptor parameter {key} must be finite numeric")
            continue
        if float(value) < lo or float(value) > hi:
            violations.append(f"HARD: descriptor parameter {key}={value} outside [{lo}, {hi}]")

    if not violations:
        try:
            preflight_descriptor_spec(spec)
        except Exception as exc:
            violations.append(f"HARD: descriptor preflight failed: {exc}")
    return violations


def preflight_descriptor_spec(spec: dict[str, Any]) -> None:
    """Check finite values and finite-difference derivatives on toy densities."""
    rho = np.array([1e-6, 1e-4, 1e-2, 0.1, 1.0, 5.0], dtype=float)
    s2 = np.array([0.0, 0.1, 1.0, 10.0, 50.0, 100.0], dtype=float)
    sigma = s2 * np.power(rho, 8.0 / 3.0)
    exc, vrho, vsigma = evaluate_descriptor(spec, rho, sigma, deriv=1)
    arrays = [exc, vrho, vsigma]
    if not all(np.all(np.isfinite(item)) for item in arrays):
        raise ValueError("non-finite descriptor value or derivative")
    if np.max(np.abs(exc)) > 1.0:
        raise ValueError("descriptor correction magnitude is too large on toy densities")


def evaluate_descriptor(
    spec: dict[str, Any],
    rho: np.ndarray,
    sigma: np.ndarray,
    *,
    deriv: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Evaluate descriptor exc and first derivatives by finite difference."""
    rho = np.asarray(rho, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    exc = _descriptor_exc(spec, rho, sigma)
    if deriv <= 0:
        return exc, None, None

    eps_rho = np.maximum(1e-6, np.abs(rho) * 1e-5)
    eps_sigma = np.maximum(1e-10, np.abs(sigma) * 1e-5)
    rho_minus = np.maximum(rho - eps_rho, 1e-12)
    sigma_minus = np.maximum(sigma - eps_sigma, 0.0)
    vrho = (
        _descriptor_exc(spec, rho + eps_rho, sigma)
        - _descriptor_exc(spec, rho_minus, sigma)
    ) / ((rho + eps_rho) - rho_minus)
    vsigma = (
        _descriptor_exc(spec, rho, sigma + eps_sigma)
        - _descriptor_exc(spec, rho, sigma_minus)
    ) / ((sigma + eps_sigma) - sigma_minus)
    return exc, vrho, vsigma


def _descriptor_exc(spec: dict[str, Any], rho: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    params = spec.get("params", {})
    scale = float(params.get("scale", 0.0))
    a = float(params.get("gradient_sensitivity", 1.0))
    density_decay = float(params.get("density_decay", 0.0))
    rho_power = float(params.get("rho_power", 1.0 / 3.0))
    template = str(spec.get("template", "bounded_gradient"))

    rho_safe = np.maximum(np.asarray(rho, dtype=float), 1e-12)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 0.0)
    rho_factor = np.power(rho_safe, rho_power)
    s2 = np.minimum(sigma_safe / (np.power(rho_safe, 8.0 / 3.0) + 1e-18), 1e6)

    if template == "bounded_gradient":
        shape = s2 / (1.0 + a * s2)
    elif template == "tanh_gradient":
        shape = np.tanh(a * s2)
    elif template == "low_density_gradient":
        shape = np.exp(-density_decay * np.power(rho_safe, 1.0 / 3.0)) * s2 / (1.0 + a * s2)
    elif template == "density_saturation":
        shape = 1.0 / (1.0 + density_decay * np.power(rho_safe, 1.0 / 3.0))
    else:
        raise ValueError(f"unsupported descriptor template: {template}")
    return scale * rho_factor * shape


def descriptor_components(xc) -> list[Any]:
    return [
        comp for comp in getattr(xc, "components", [])
        if getattr(comp, "role", "") == DESCRIPTOR_ROLE or getattr(comp, "descriptor_spec", None)
    ]


def base_xc_without_descriptors(xc):
    """Copy an XCFunctional without descriptor components and without HF terms."""
    from .xc_functional import HFExchange, XCFunctional

    base = copy.deepcopy(xc)
    base.components = [
        comp for comp in base.components
        if comp not in descriptor_components(base)
    ]
    base.hf_exchange = HFExchange()
    base.metadata.pop("native_xc_code", None)
    return XCFunctional(name=f"{xc.name}_base", components=base.components, hf_exchange=base.hf_exchange)


def compile_descriptor_eval_xc(xc):
    """Build a PySCF eval_xc callable for base libxc + descriptor corrections."""
    from pyscf.dft import libxc

    descriptors = descriptor_components(xc)
    if not descriptors:
        raise ValueError(f"No descriptor components found in {xc.name}")
    base = base_xc_without_descriptors(xc)
    base_xc_code = base.to_xc_code()
    specs = [normalize_descriptor_spec(comp.descriptor_spec) for comp in descriptors]

    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        if base_xc_code and base_xc_code != ",":
            exc, vxc, _fxc, _kxc = libxc.eval_xc(
                base_xc_code,
                rho,
                spin=spin,
                relativity=relativity,
                deriv=min(deriv, 1),
                omega=omega,
            )
        else:
            rho0 = _total_rho(rho, spin)
            exc = np.zeros_like(rho0)
            vxc = _zero_vxc(rho0, spin)

        total_rho, total_sigma = _total_rho_sigma(rho, spin)
        desc_exc = np.zeros_like(total_rho)
        if deriv > 0:
            desc_vrho = np.zeros_like(total_rho)
            desc_vsigma = np.zeros_like(total_rho)
        else:
            desc_vrho = desc_vsigma = None
        for spec in specs:
            add_exc, add_vrho, add_vsigma = evaluate_descriptor(spec, total_rho, total_sigma, deriv=deriv)
            desc_exc += add_exc
            if deriv > 0:
                desc_vrho += add_vrho
                desc_vsigma += add_vsigma

        exc = np.asarray(exc) + desc_exc
        if deriv <= 0:
            return exc, None, None, None
        return exc, _add_descriptor_vxc(vxc, desc_vrho, desc_vsigma, spin), None, None

    return eval_xc


def _total_rho(rho, spin: int) -> np.ndarray:
    arr = np.asarray(rho)
    if spin == 0:
        return np.maximum(arr[0], 1e-12)
    return np.maximum(arr[0, 0] + arr[1, 0], 1e-12)


def _total_rho_sigma(rho, spin: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(rho)
    if spin == 0:
        total_rho = np.maximum(arr[0], 1e-12)
        grad = arr[1:4]
        sigma = np.sum(grad * grad, axis=0)
        return total_rho, np.maximum(sigma, 0.0)
    total_rho = np.maximum(arr[0, 0] + arr[1, 0], 1e-12)
    grad = arr[0, 1:4] + arr[1, 1:4]
    sigma = np.sum(grad * grad, axis=0)
    return total_rho, np.maximum(sigma, 0.0)


def _zero_vxc(rho0: np.ndarray, spin: int):
    if spin == 0:
        return (np.zeros_like(rho0), np.zeros_like(rho0))
    return (np.zeros((rho0.size, 2)), np.zeros((rho0.size, 3)))


def _add_descriptor_vxc(vxc, vrho, vsigma, spin: int):
    if spin == 0:
        base_vrho = np.asarray(vxc[0]) if vxc else np.zeros_like(vrho)
        base_vsigma = np.asarray(vxc[1]) if vxc and len(vxc) > 1 else np.zeros_like(vsigma)
        rest = tuple(vxc[2:]) if vxc and len(vxc) > 2 else ()
        return (base_vrho + vrho, base_vsigma + vsigma, *rest)

    base_vrho = np.asarray(vxc[0]) if vxc else np.zeros((vrho.size, 2))
    base_vsigma = np.asarray(vxc[1]) if vxc and len(vxc) > 1 else np.zeros((vrho.size, 3))
    desc_vrho = np.column_stack([vrho, vrho])
    desc_vsigma = np.column_stack([vsigma, 2.0 * vsigma, vsigma])
    rest = tuple(vxc[2:]) if vxc and len(vxc) > 2 else ()
    return (base_vrho + desc_vrho, base_vsigma + desc_vsigma, *rest)
