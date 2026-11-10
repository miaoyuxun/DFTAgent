"""Validation for XCFunctional: physical constraints, numerical derivative checks."""

from __future__ import annotations

import numpy as np
from .xc_functional import XCFunctional


def check_constraints(xc: XCFunctional) -> list[str]:
    """Return list of constraint violations (empty = valid)."""
    return xc.validate()


def numerical_derivative_check(
    eval_xc_fn,
    rho: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-4,
) -> dict[str, float]:
    """Check analytical vxc against finite-difference derivative of exc.

    Parameters
    ----------
    eval_xc_fn : callable
        Function with signature eval_xc(rho) -> (exc, vxc, fxc, kxc)
        where exc is energy density, vxc is potential.
    rho : ndarray
        Density array of shape (N,) or (4, N) for GGA (rho, grad_x, grad_y, grad_z).
    eps : float
        Finite difference step size.
    tol : float
        Relative tolerance for passing.

    Returns
    -------
    dict with keys 'max_rel_error', 'mean_rel_error', 'passed'
    """
    exc, vxc, _, _ = eval_xc_fn(rho)

    # Finite difference on exc
    rho_plus = rho.copy()
    if rho.ndim == 1:
        rho_plus = rho_plus + eps
        exc_plus, _, _, _ = eval_xc_fn(rho_plus)
        fd_vxc = (exc_plus - exc) / eps
        analytical = vxc
    else:
        # GGA: perturb rho[0] only
        rho_plus[0] = rho_plus[0] + eps
        exc_plus, _, _, _ = eval_xc_fn(rho_plus)
        fd_vxc = (exc_plus - exc) / eps
        analytical = vxc[0] if vxc.ndim > 1 else vxc

    # Avoid division by zero
    mask = np.abs(analytical) > 1e-12
    if not np.any(mask):
        return {"max_rel_error": 0.0, "mean_rel_error": 0.0, "passed": True}

    rel_error = np.abs((fd_vxc[mask] - analytical[mask]) / analytical[mask])
    max_rel = float(np.max(rel_error))
    mean_rel = float(np.mean(rel_error))

    return {
        "max_rel_error": max_rel,
        "mean_rel_error": mean_rel,
        "passed": max_rel < tol,
    }
