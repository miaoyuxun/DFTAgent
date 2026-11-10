"""Register XCFunctional with PySCF — two paths: REWIRE and custom eval_xc.

For range-separated hybrids (omega > 0), the xc_code string must NOT contain
LR_HF — PySCF handles long-range HF exchange through the hyb tuple
(alpha_sr, alpha_sr + beta_lr, omega). The LR_HF token in xc_code is a
human-readable annotation only; it is stripped before registration.

PySCF bug: libxc._rsh_coeff calls LIBXC_is_cam_rsh(xc) with a single
c_void_p argument, but the C function expects (int, c_void_p). This causes
TypeError for any xc_code containing RSH components (GGA_X_HJS_PBE, etc).
We patch _rsh_coeff at module import to skip the broken check.
"""

from __future__ import annotations

import ctypes
import logging
import numpy as np

from pyscf.dft import libxc

from .xc_functional import XCFunctional, NAME_TO_ID, LIBXC_PARAMS
from .descriptors import compile_descriptor_eval_xc, descriptor_components

logger = logging.getLogger(__name__)
_warned_missing_named_ext_param_api = False

# ---------------------------------------------------------------------------
# Patch PySCF bug: LIBXC_is_cam_rsh called with wrong argument count
# ---------------------------------------------------------------------------
# The C function signature is (int functional_id, void* pointer) but
# _rsh_coeff passes just the c_void_p pointer. We replace _rsh_coeff
# with a version that skips the broken LIBXC_is_cam_rsh check entirely,
# since all libxc RSH functionals use the CAM kernel anyway.

_original_rsh_coeff = libxc._rsh_coeff


def _patched_rsh_coeff(xc_objs, hyb, facs, xc_code):
    """Patched _rsh_coeff that skips the broken LIBXC_is_cam_rsh check."""
    check_omega = True
    if isinstance(xc_code, str) and ',' in xc_code:
        xc_code = libxc.format_xc_code(xc_code)
        xc_code = xc_code.split(',')[0] + ','
        if 'SR_HF' in xc_code or 'LR_HF' in xc_code or 'RSH(' in xc_code:
            check_omega = False

    hyb_val, alpha, omega = hyb
    if omega == 0:
        assert hyb_val == alpha
        beta = 0.
    else:
        beta = hyb_val - alpha
    rsh_pars = [omega, alpha, beta]
    rsh_tmp = (ctypes.c_double * 3)()
    for fac, xc in zip(facs, xc_objs):
        libxc._itrf.LIBXC_rsh_coeff(xc, rsh_tmp)
        if rsh_pars[0] == 0:
            rsh_pars[0] = rsh_tmp[0]
        elif check_omega:
            # Skip LIBXC_is_cam_rsh check (broken ctypes binding).
            # All libxc RSH functionals use the CAM kernel.
            if rsh_tmp[0] != 0 and rsh_pars[0] != rsh_tmp[0]:
                raise ValueError(
                    'Different values of omega found for RSH functionals')
        rsh_pars[1] += rsh_tmp[1] * fac
        rsh_pars[2] += rsh_tmp[2] * fac
    return tuple(rsh_pars)


libxc._rsh_coeff = _patched_rsh_coeff


def _strip_lr_hf(xc_code: str) -> str:
    """Remove LR_HF terms from xc_code before passing to PySCF.

    PySCF handles long-range HF exchange through the hyb tuple, not through
    the xc_code string. LR_HF in xc_code is a human-readable annotation
    that must be stripped before registration.

    Example: "0.75*PBE+0.25*HF+0.75*LR_HF,PBE" -> "0.75*PBE+0.25*HF,PBE"
    """
    parts = xc_code.split(",")
    if len(parts) >= 1:
        exch_terms = parts[0].split("+")
        exch_terms = [t for t in exch_terms if "LR_HF" not in t]
        parts[0] = "+".join(exch_terms)
    return ",".join(parts)


def register_with_pyscf(mf, xc: XCFunctional) -> None:
    """Register an XCFunctional with a PySCF mean-field object.

    Three paths:
    - Custom: at least one component has eval_xc_code -> define_xc_()
    - Compound RSH native: compound RSH with no ext_params -> use native name
    - REWIRE: all components are libxc builtins -> register_custom_functional_()
    """
    if xc.is_custom:
        _register_custom(mf, xc)
    else:
        _register_rewire(mf, xc)


def _native_xc_code(xc: XCFunctional) -> str:
    code = xc.metadata.get("native_xc_code", "")
    return str(code).strip()


_EXT_PARAM_RUNTIME_ALIASES: dict[tuple[int, str], tuple[str, ...]] = {
    # Libxc exposes the historical third GGA_C_PBE parameter as _B.  The
    # project keeps the _d key for compatibility with existing trajectories.
    (130, "_d"): ("_B",),
}


def _runtime_ext_param_name(xid: int, project_name: str) -> str | None:
    """Resolve a project parameter key against the linked Libxc version."""
    global _warned_missing_named_ext_param_api
    try:
        find_by_name = libxc._itrf.LIBXC_xc_func_find_ext_params_name
    except AttributeError:
        if not _warned_missing_named_ext_param_api:
            logger.warning(
                "PySCF libxc_itrf lacks named external-parameter lookup; "
                "using validated positional Libxc parameters"
            )
            _warned_missing_named_ext_param_api = True
        return None
    cache = libxc.XCFunctionalCache(xid)
    func = cache.obj_by_id()[xid]
    candidates = (project_name,) + _EXT_PARAM_RUNTIME_ALIASES.get(
        (xid, project_name), ()
    )
    for candidate in candidates:
        index = find_by_name(func, candidate.encode())
        if index >= 0:
            return candidate
    return None


def _build_ext_params(
    xc: XCFunctional,
) -> dict[int, dict[str, float] | np.ndarray]:
    """Build ext_params dict for register_custom_functional_().

    Prefer Libxc's named-parameter API so that newly added runtime parameters
    retain their native defaults.  This is important for GGA_C_PBE: Libxc
    7.1 added ``_tscale``, while existing DFTAgent trajectories intentionally
    retain the original three-dimensional search space.
    """
    ext_params: dict[int, dict[str, float] | np.ndarray] = {}
    for comp in xc.components:
        # With no explicit override, let the installed Libxc version use its
        # own defaults.  Passing our curated defaults unnecessarily couples a
        # fixed functional to one Libxc parameter-array layout.
        if not comp.ext_params:
            continue
        xid = NAME_TO_ID.get(comp.name)
        if xid is None:
            continue
        param_spec = LIBXC_PARAMS.get(xid, [])
        if not param_spec:
            continue
        named_values: dict[str, float] = {}
        values: list[float] = []
        names_resolved = True
        for pname, _default, _bounds in param_spec:
            v = comp.ext_params.get(pname, _default)
            values.append(float(v))
            runtime_name = _runtime_ext_param_name(xid, pname)
            if runtime_name is None:
                names_resolved = False
            else:
                named_values[runtime_name] = float(v)

        if names_resolved:
            ext_params[xid] = named_values
            continue

        # Compatibility fallback for older Libxc builds whose parameter names
        # are unavailable through the named API.  A positional array is safe
        # only when its complete runtime length matches the curated catalogue.
        cache = libxc.XCFunctionalCache(xid)
        func = cache.obj_by_id()[xid]
        info = libxc._itrf.xc_func_get_info(func)
        runtime_count = libxc._itrf.xc_func_info_get_n_ext_params(info)
        if runtime_count != len(values):
            raise ValueError(
                f"Cannot safely register external parameters for Libxc "
                f"functional {xid}: runtime expects {runtime_count}, project "
                f"provides {len(values)}, and named lookup was incomplete"
            )
        ext_params[xid] = np.asarray(values, dtype=float)
    return ext_params


def _is_compound_rsh(xc: XCFunctional) -> bool:
    """Check if functional uses a compound RSH functional (e.g. CAM-B3LYP, wB97X)."""
    for comp in xc.components:
        if comp.role == "exchange_correlation" and xc.hf_exchange.omega > 0:
            return True
    return False


def _has_rsh_component(xc: XCFunctional) -> bool:
    """Check if any exchange component is a range-separated (short-range) functional."""
    available = None
    for comp in xc.components:
        if comp.role in ("exchange", "exchange_correlation"):
            xid = NAME_TO_ID.get(comp.name)
            if xid is None:
                if available is None:
                    available = libxc.available_libxc_functionals()
                xid = available.get(comp.name)
            if xid is not None:
                try:
                    rsh = libxc.rsh_coeff(xid)
                    if rsh and len(rsh) >= 3 and rsh[0] > 0:
                        return True
                except Exception:
                    pass
    return False


def _registration_name(xc: XCFunctional) -> str:
    """Return a PySCF custom-functional name that cannot be parsed as built-in XC.

    Names derived from candidate names can contain tokens like PBE0, HF, or
    LR_HF. PySCF attempts to parse the custom name before registering it and
    warns if it looks like a valid built-in xc_code. A hash-only suffix avoids
    that parser path while remaining deterministic for unregister/re-register.
    """
    return f"dftagent_custom_{xc.structural_signature()}"


def _register_rewire(mf, xc: XCFunctional) -> None:
    """Register using libxc.register_custom_functional_() with ext_params.

    For compound RSH functionals (CAM-B3LYP, wB97X), we use the native PySCF
    name directly because register_custom_functional_() triggers a PySCF bug
    where LIBXC_is_cam_rsh is called with wrong argument count.

    For functionals with RSH exchange components (GGA_X_HJS_PBE etc.), we
    pass the omega parameter to override component-level omega values.
    """
    native_xc_code = _native_xc_code(xc)
    if native_xc_code:
        mf.xc = native_xc_code
        return

    # Compound RSH: use native name directly to avoid PySCF ctypes bug
    if _is_compound_rsh(xc):
        xc_code = xc.to_xc_code()
        mf.xc = xc_code
        return

    reg_name = _registration_name(xc)

    # Unregister previous if exists
    try:
        libxc.unregister_custom_functional_(reg_name)
    except Exception:
        pass

    # Strip LR_HF from xc_code — PySCF handles LR HF via hyb tuple
    xc_code = _strip_lr_hf(xc.to_xc_code())
    ext_params = _build_ext_params(xc)

    alpha = xc.hf_exchange.alpha
    beta = xc.hf_exchange.beta
    omega = xc.hf_exchange.omega

    # hyb tuple: (alpha_sr, alpha_sr + beta_lr, omega)
    hyb = (
        (alpha, alpha + beta, omega)
        if (alpha > 0 or beta > 0 or omega > 0)
        else (0.0, 0.0, 0.0)
    )

    # For RSH functionals, pass omega to override component-level values
    # This is needed when xc_code contains short-range functionals (GGA_X_HJS_*)
    # whose built-in omega differs from our desired omega
    kwargs = {
        "ext_params": ext_params if ext_params else None,
        "hyb": hyb,
    }
    if omega > 0 and _has_rsh_component(xc):
        # Count components in xc_code to set omega for each
        parsed = libxc.parse_xc(xc_code)
        n_comps = len(parsed[1])
        kwargs["omega"] = (omega,) * n_comps

    libxc.register_custom_functional_(
        reg_name,
        xc_code,
        **kwargs,
    )

    mf.xc = reg_name


def _register_custom(mf, xc: XCFunctional) -> None:
    """Register using mf.define_xc_() with compiled eval_xc callable."""
    callable_obj = compile_eval_xc(xc)
    highest_rung = xc.highest_rung

    alpha = xc.hf_exchange.alpha
    beta = xc.hf_exchange.beta
    omega = xc.hf_exchange.omega

    # define_xc_ parameters
    hyb = alpha
    rsh = (omega, alpha, beta) if omega > 0 else None

    mf.xc = "LDA,VWN"  # bookkeeping string
    mf.define_xc_(callable_obj, xctype=highest_rung, hyb=hyb, rsh=rsh)


def compile_eval_xc(xc: XCFunctional) -> callable:
    """Compile eval_xc_code from components into a single callable.

    The callable signature matches PySCF's eval_xc interface:
        eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None)
    Returns: (exc, vxc, fxc, kxc)
    """
    if descriptor_components(xc):
        return compile_descriptor_eval_xc(xc)

    code_blocks = []
    for comp in xc.components:
        if comp.eval_xc_code:
            code_blocks.append(comp.eval_xc_code)

    if not code_blocks:
        raise ValueError(f"No eval_xc_code found in {xc.name}")

    combined_code = "\n".join(code_blocks)
    namespace: dict = {}
    exec(combined_code, namespace)

    if "eval_xc" not in namespace:
        raise ValueError(f"No eval_xc function found in compiled code for {xc.name}")

    return namespace["eval_xc"]
