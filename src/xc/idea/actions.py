"""Actions for XC functional optimization — 5 atomic operations."""

from __future__ import annotations

import copy
import logging
from enum import Enum
from dataclasses import dataclass, field

from ..xc_functional import (
    XCFunctional,
    ComponentSpec,
    HFExchange,
    NAME_TO_ID,
    LIBXC_PARAMS,
    PURE_COMPONENT_ROLES,
    COMPONENT_XCTYPES,
)
from ..descriptors import normalize_descriptor_spec

logger = logging.getLogger(__name__)

SPLIT_ADD_DEFAULT_MIXING = 0.10
SPLIT_ADD_MAX_MIXING = 0.20
ZERO_MIXING_TOLERANCE = 1e-12


class ActionType(Enum):
    REWIRE = "rewire"              # tune parameters (alpha, omega, mu, kappa, mixing_factor)
    ADD_COMPONENT = "add"          # add a libxc component (including dispersion)
    ADD_DESCRIPTOR = "add_descriptor"  # add a restricted custom descriptor correction
    REMOVE_COMPONENT = "remove"    # remove an existing component
    REPLACE_COMPONENT = "replace"  # swap one component for another
    ENFORCE_CONSTRAINT = "enforce" # apply a physical constraint


@dataclass
class Action:
    action_type: ActionType
    changes: dict = field(default_factory=dict)
    rationale: str = ""
    edit_plan: 'EditPlan | None' = None


# ---------------------------------------------------------------------------
# EditPlan structures (mirrors Xcientist's mcts_runtime.py)
# ---------------------------------------------------------------------------

@dataclass
class ValidationProtocol:
    """What to check after applying a ComponentEdit."""
    check_sum_rules: bool = True
    check_parameter_bounds: bool = True
    check_lieb_oxford: bool = False
    check_sie: bool = False
    check_lda_limit: bool = False
    custom_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "check_sum_rules": self.check_sum_rules,
            "check_parameter_bounds": self.check_parameter_bounds,
            "check_lieb_oxford": self.check_lieb_oxford,
            "check_sie": self.check_sie,
            "check_lda_limit": self.check_lda_limit,
            "custom_checks": self.custom_checks,
        }


@dataclass
class ComponentEdit:
    """A single atomic edit targeting one component or parameter.

    Mirrors Xcientist's ComponentEdit with op, component, target,
    condition, details, reason, validation_protocol.
    """
    op: ActionType
    component: str = ""
    target: dict = field(default_factory=dict)
    condition: str = ""
    details: str = ""
    reason: str = ""
    validation: ValidationProtocol = field(default_factory=ValidationProtocol)

    def to_dict(self) -> dict:
        return {
            "op": self.op.value,
            "component": self.component,
            "target": self.target,
            "condition": self.condition,
            "details": self.details,
            "reason": self.reason,
            "validation": self.validation.to_dict(),
        }


@dataclass
class EditPlan:
    """Structured plan for modifying an XC functional.

    Mirrors Xcientist's EditPlan with skill_name, objective,
    target_defects, component_edits, validation, guardrails,
    memory_refs, compile_notes.
    """
    skill_name: str = ""
    objective: str = ""
    target_defects: list[str] = field(default_factory=list)
    component_edits: list[ComponentEdit] = field(default_factory=list)
    validation: ValidationProtocol = field(default_factory=ValidationProtocol)
    guardrails: list[str] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)
    compile_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "objective": self.objective,
            "target_defects": self.target_defects,
            "component_edits": [ce.to_dict() for ce in self.component_edits],
            "validation": self.validation.to_dict(),
            "guardrails": self.guardrails,
            "memory_refs": self.memory_refs,
            "compile_notes": self.compile_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EditPlan':
        """Parse from LLM JSON output."""
        edits = []
        for ce_data in data.get("component_edits", []):
            try:
                op = ActionType(ce_data.get("op", "rewire"))
            except ValueError:
                op = ActionType.REWIRE
            val_data = ce_data.get("validation", {})
            validation = ValidationProtocol(
                check_sum_rules=val_data.get("check_sum_rules", True),
                check_parameter_bounds=val_data.get("check_parameter_bounds", True),
                check_lieb_oxford=val_data.get("check_lieb_oxford", False),
                check_sie=val_data.get("check_sie", False),
                check_lda_limit=val_data.get("check_lda_limit", False),
                custom_checks=val_data.get("custom_checks", []),
            )
            edits.append(ComponentEdit(
                op=op,
                component=ce_data.get("component", ""),
                target=ce_data.get("target", {}),
                condition=ce_data.get("condition", ""),
                details=ce_data.get("details", ""),
                reason=ce_data.get("reason", ""),
                validation=validation,
            ))
        val_data = data.get("validation", {})
        plan_validation = ValidationProtocol(
            check_sum_rules=val_data.get("check_sum_rules", True),
            check_parameter_bounds=val_data.get("check_parameter_bounds", True),
            check_lieb_oxford=val_data.get("check_lieb_oxford", False),
            check_sie=val_data.get("check_sie", False),
            check_lda_limit=val_data.get("check_lda_limit", False),
            custom_checks=val_data.get("custom_checks", []),
        )
        return cls(
            skill_name=data.get("skill_name", ""),
            objective=data.get("objective", ""),
            target_defects=data.get("target_defects", []),
            component_edits=edits,
            validation=plan_validation,
            guardrails=data.get("guardrails", []),
            memory_refs=data.get("memory_refs", []),
            compile_notes=data.get("compile_notes", []),
        )


# ---------------------------------------------------------------------------
# Dispersion correction catalogue
# ---------------------------------------------------------------------------
DISPERSION_CATALOGUE = {
    "D3": {
        "xc_code_suffix": ",D3",
        "params": {"s6": 1.0, "s8": 0.0, "a1": 0.0, "a2": 0.0},
        "param_bounds": {"s6": (0.0, 2.0), "s8": (0.0, 2.0), "a1": (0.0, 1.0), "a2": (0.0, 10.0)},
        "description": "Grimme D3 with zero-damping",
    },
    "D3BJ": {
        "xc_code_suffix": ",D3BJ",
        "params": {"s6": 1.0, "s8": 0.0, "a1": 0.0, "a2": 0.0},
        "param_bounds": {"s6": (0.0, 2.0), "s8": (0.0, 2.0), "a1": (0.0, 1.0), "a2": (0.0, 10.0)},
        "description": "Grimme D3 with Becke-Johnson damping",
    },
}

# ---------------------------------------------------------------------------
# Physical constraint catalogue
# ---------------------------------------------------------------------------
CONSTRAINT_CATALOGUE = {
    "lieb_oxford": {
        "description": "Lieb-Oxford bound: exchange energy ≤ -1.679 * ∫ρ^(4/3) dr",
        "check_code": "E_x / (-1.679 * ∫ρ^(4/3)) ≤ 1.0",
    },
    "one_electron_sie": {
        "description": "One-electron self-interaction: E_x[ρ] = -E_H[ρ] for one-electron density",
        "check_code": "E_x(ρ_1e) + E_H(ρ_1e) = 0",
    },
    "lda_limit": {
        "description": "Uniform electron gas limit: F_x(s→0) → 1, ε_c(r_s→∞) → 0",
        "check_code": "F_x(0) = 1, ε_c(∞) = 0",
    },
}


# Keys that belong to the rewire operation — these should be processed
# regardless of the primary action_type, because composite intents
# (e.g. replace_component + set omega) are common.
_REWIRE_KEYS = {"alpha", "beta", "omega", "s6", "s8", "a1", "a2"}

# Keys that belong to the mixing operation (rewire sub-category)
_MIXING_PREFIX = "mixing:"
_ROLE_SCALE_PREFIX = "scale:"


def _set_role_scale(xc: XCFunctional, role: str, value: float) -> None:
    if role not in {"exchange", "correlation"}:
        raise ValueError(f"Unknown role scale {role!r}; expected exchange or correlation")
    xc.role_scales[role] = min(1.2, max(0.8, float(value)))


def apply_action(xc: XCFunctional, action: Action) -> XCFunctional:
    """Apply an action to an XCFunctional and return a new one.

    Processes the primary action_type operation first, then applies any
    rewire keys (alpha, beta, omega, etc.) found in changes. This ensures
    composite intents like "replace component + set omega" are fully
    compiled without discarding parameters.
    """
    new = copy.deepcopy(xc)
    new.metadata["parent"] = xc.name
    new.metadata["action"] = action.action_type.value
    new.metadata["rationale"] = action.rationale

    # 1. Apply the primary operation
    if action.action_type == ActionType.REWIRE:
        _apply_rewire(new, action)
    elif action.action_type == ActionType.ADD_COMPONENT:
        _apply_add_component(new, action)
    elif action.action_type == ActionType.ADD_DESCRIPTOR:
        _apply_add_descriptor(new, action)
    elif action.action_type == ActionType.REMOVE_COMPONENT:
        _apply_remove_component(new, action)
    elif action.action_type == ActionType.REPLACE_COMPONENT:
        _apply_replace_component(new, action)
    elif action.action_type == ActionType.ENFORCE_CONSTRAINT:
        _apply_enforce_constraint(new, action)

    # 2. Apply any rewire keys that were NOT consumed by the primary operation.
    # This handles composite intents where the LLM provides both structural
    # changes (replace, add) AND parameter changes (alpha, beta, omega).
    # Rewire itself already processed these keys, so skip for REWIRE type.
    if action.action_type != ActionType.REWIRE:
        _apply_cross_action_rewire(new, action)

    # A zero-weight term is physically absent. Canonicalize it before
    # signatures, novelty distances, and deduplication can observe the result.
    removed_components = _drop_zero_mixing_components(new)
    if removed_components:
        previous = new.metadata.get("auto_removed_zero_mixing_components", [])
        recorded = list(previous) if isinstance(previous, list) else []
        for component_name in removed_components:
            if component_name not in recorded:
                recorded.append(component_name)
        new.metadata["auto_removed_zero_mixing_components"] = recorded
        logger.info(
            "Treating zero mixing as remove_component: removed=%s action=%s",
            removed_components,
            action.action_type.value,
        )

    # Validate: candidate must differ from parent
    if new.signature() == xc.signature():
        logger.warning(f"Action produced identical functional — applying default change for {action.action_type.value}")
        _apply_default_change(new, xc, action)

    # Keep the return value canonical even if a fallback edit introduced a
    # dormant component in a future action implementation.
    _drop_zero_mixing_components(new)

    return new


def _drop_zero_mixing_components(xc: XCFunctional) -> list[str]:
    """Remove components whose numerical contribution is exactly zero."""
    removed = [
        component.name
        for component in xc.components
        if abs(float(component.mixing_factor)) <= ZERO_MIXING_TOLERANCE
    ]
    if removed:
        xc.components = [
            component
            for component in xc.components
            if abs(float(component.mixing_factor)) > ZERO_MIXING_TOLERANCE
        ]
    return removed


def _apply_cross_action_rewire(xc: XCFunctional, action: Action) -> None:
    """Apply rewire keys from a non-rewire action's changes dict.

    This ensures that parameters like alpha, beta, omega are not silently
    discarded when the primary action_type is replace_component, add_component,
    etc. The logic mirrors _apply_rewire but only processes keys that belong
    to the rewire domain.
    """
    for key, value in action.changes.items():
        if key == "alpha" and isinstance(value, (int, float)):
            xc.hf_exchange.alpha = float(value)
            exch_sum = sum(c.mixing_factor for c in xc.components if c.role == "exchange")
            target_dft_exch = 1.0 - xc.hf_exchange.alpha
            if exch_sum > 0 and target_dft_exch > 0:
                scale = target_dft_exch / exch_sum
                for c in xc.components:
                    if c.role == "exchange":
                        c.mixing_factor *= scale
        elif key == "beta" and isinstance(value, (int, float)):
            xc.hf_exchange.beta = float(value)
        elif key == "omega" and isinstance(value, (int, float)):
            xc.hf_exchange.omega = float(value)
        elif key.startswith(_MIXING_PREFIX) and isinstance(value, (int, float)):
            comp_name = key[len(_MIXING_PREFIX):]
            for c in xc.components:
                if c.name == comp_name:
                    c.mixing_factor = float(value)
        elif key.startswith(_ROLE_SCALE_PREFIX) and isinstance(value, (int, float)):
            _set_role_scale(xc, key[len(_ROLE_SCALE_PREFIX):], float(value))
        elif "." in key and isinstance(value, (int, float)):
            comp_name, param_name = key.split(".", 1)
            # HF_EXCHANGE variants (GLOBAL, RANGE_SEPARATED, etc.)
            if comp_name.startswith("HF_EXCHANGE") and param_name in ("alpha", "beta", "omega"):
                setattr(xc.hf_exchange, param_name, float(value))
            else:
                for c in xc.components:
                    if c.name == comp_name and c.descriptor_spec:
                        params = c.descriptor_spec.setdefault("params", {})
                        if param_name in params:
                            params[param_name] = float(value)
                            c.ext_params[param_name] = float(value)
                    elif c.name == comp_name and param_name in c.ext_params:
                        c.ext_params[param_name] = float(value)


def _apply_default_change(new: XCFunctional, parent: XCFunctional, action: Action) -> None:
    """Apply a minimal default change when the LLM-provided action had no effect."""
    if action.action_type == ActionType.REWIRE:
        # Increase alpha by 0.05 (bounded at 0.50)
        new.hf_exchange.alpha = min(parent.hf_exchange.alpha + 0.05, 0.50)
        exch_sum = sum(c.mixing_factor for c in new.components if c.role == "exchange")
        target_dft_exch = 1.0 - new.hf_exchange.alpha
        if exch_sum > 0 and target_dft_exch > 0:
            scale = target_dft_exch / exch_sum
            for c in new.components:
                if c.role == "exchange":
                    c.mixing_factor *= scale
    elif action.action_type == ActionType.ADD_COMPONENT:
        new.components.append(ComponentSpec(
            name="DISP_D3BJ", xctype="GGA", mixing_factor=1.0, role="dispersion",
            ext_params={"s6": 1.0, "s8": 0.0, "a1": 0.0, "a2": 0.0},
            ext_param_bounds={"s6": (0.0, 2.0), "s8": (0.0, 2.0), "a1": (0.0, 1.0), "a2": (0.0, 10.0)},
        ))
    elif action.action_type == ActionType.REPLACE_COMPONENT:
        for i, c in enumerate(new.components):
            if c.role == "exchange":
                new.components[i] = ComponentSpec(
                    name="GGA_X_PBE_SOL", xctype="GGA", mixing_factor=c.mixing_factor, role="exchange",
                    ext_params={"_kappa": 0.123, "_mu": 0.10},
                    ext_param_bounds={"_kappa": (0.05, 0.30), "_mu": (0.05, 0.20)},
                )
                break
    elif action.action_type == ActionType.ADD_DESCRIPTOR:
        _apply_add_descriptor(
            new,
            Action(
                ActionType.ADD_DESCRIPTOR,
                {
                    "descriptor": {
                        "template": "bounded_gradient",
                        "scale": -0.005,
                        "gradient_sensitivity": 1.0,
                        "density_decay": 1.0,
                        "rho_power": 1.0 / 3.0,
                        "intended_effect": "small bounded gradient correction",
                    }
                },
                action.rationale,
            ),
        )
    elif action.action_type == ActionType.ENFORCE_CONSTRAINT:
        constraints = new.metadata.get("enforced_constraints", [])
        if "one_electron_sie" not in constraints:
            constraints.append("one_electron_sie")
        new.metadata["enforced_constraints"] = constraints


def _apply_rewire(xc: XCFunctional, action: Action) -> None:
    """Tune parameters based on action.changes keys.

    Key formats:
    - "alpha" → change HF exchange alpha
    - "beta" → change HF exchange beta
    - "omega" → change HF exchange omega
    - "mixing:COMP_NAME" → change mixing_factor for COMP_NAME
    - "COMP_NAME.PARAM" → change ext_params for COMP_NAME.PARAM (e.g. "GGA_X_PBE._mu")
    """
    for key, value in action.changes.items():
        if key == "alpha":
            xc.hf_exchange.alpha = value
            exch_sum = sum(c.mixing_factor for c in xc.components if c.role == "exchange")
            target_dft_exch = 1.0 - xc.hf_exchange.alpha
            if exch_sum > 0 and target_dft_exch > 0:
                scale = target_dft_exch / exch_sum
                for c in xc.components:
                    if c.role == "exchange":
                        c.mixing_factor *= scale
        elif key == "beta":
            xc.hf_exchange.beta = value
        elif key == "omega":
            xc.hf_exchange.omega = value
        elif key.startswith("mixing:"):
            comp_name = key[len("mixing:"):]
            matched = False
            for c in xc.components:
                if c.name == comp_name:
                    c.mixing_factor = float(value)
                    matched = True
            if not matched:
                logger.warning("Rewire requested unknown mixing component %r", comp_name)
        elif key.startswith(_ROLE_SCALE_PREFIX):
            _set_role_scale(xc, key[len(_ROLE_SCALE_PREFIX):], float(value))
        elif "." in key:
            comp_name, param_name = key.split(".", 1)
            # HF_EXCHANGE variants (GLOBAL, RANGE_SEPARATED, etc.)
            if comp_name.startswith("HF_EXCHANGE") and param_name in ("alpha", "beta", "omega"):
                setattr(xc.hf_exchange, param_name, value)
            else:
                for c in xc.components:
                    if c.name == comp_name and c.descriptor_spec:
                        params = c.descriptor_spec.setdefault("params", {})
                        if param_name in params:
                            params[param_name] = float(value)
                            c.ext_params[param_name] = float(value)
                    elif c.name == comp_name and param_name in c.ext_params:
                        c.ext_params[param_name] = value


def _apply_add_component(xc: XCFunctional, action: Action) -> None:
    """Add a component. Handles both regular libxc components and dispersion.

    For regular components: changes = {"name": ..., "xctype": ..., "role": ..., ...}
    For dispersion: changes = {"type": "D3BJ", "params": {...}}
    """
    # Check if this is a dispersion addition
    if "type" in action.changes and action.changes["type"] in DISPERSION_CATALOGUE:
        disp_type = action.changes["type"]
        disp_params = action.changes.get("params", {})
        catalogue_entry = DISPERSION_CATALOGUE[disp_type]
        merged_params = {**catalogue_entry["params"], **disp_params}
        merged_bounds = catalogue_entry["param_bounds"]

        comp = ComponentSpec(
            name=f"DISP_{disp_type}",
            xctype="GGA",
            mixing_factor=1.0,
            role="dispersion",
            ext_params=merged_params,
            ext_param_bounds=merged_bounds,
            formula=f"E_disp = {disp_type} correction",
        )
        xc.components.append(comp)
        xc.metadata["dispersion"] = disp_type
        return

    # Regular component addition
    spec = action.changes
    name = spec.get("name", "")
    xctype = spec.get("xctype", "GGA")
    role = spec.get("role", "exchange")
    requested_mixing = spec.get("mixing_factor", spec.get("mixing"))
    mixing_factor = float(requested_mixing) if isinstance(requested_mixing, (int, float)) else 1.0
    ext_params = spec.get("ext_params", {})
    ext_param_bounds = spec.get("ext_param_bounds", {})
    if not name or name not in NAME_TO_ID:
        raise ValueError(
            f"Cannot add unknown XC component {name!r}; use a libxc component "
            "or express HF/range-separation edits through alpha/beta/omega rewiring."
        )
    canonical_role = PURE_COMPONENT_ROLES.get(name)
    if canonical_role is not None:
        if role not in {"exchange", "correlation", canonical_role}:
            raise ValueError(
                f"Cannot add pure {canonical_role} component {name!r} with role={role!r}"
            )
        role = canonical_role
        mixing_factor = _split_add_mixing_factor(
            xc,
            role=role,
            new_component=name,
            requested_mixing=requested_mixing,
        )
    xctype = COMPONENT_XCTYPES.get(name, xctype)

    if not ext_params and name in NAME_TO_ID:
        libxc_id = NAME_TO_ID[name]
        if libxc_id in LIBXC_PARAMS and LIBXC_PARAMS[libxc_id]:
            ext_params = {pname: default for pname, default, _ in LIBXC_PARAMS[libxc_id]}
            ext_param_bounds = {pname: bounds for pname, _, bounds in LIBXC_PARAMS[libxc_id]}

    comp = ComponentSpec(
        name=name, xctype=xctype, mixing_factor=mixing_factor, role=role,
        ext_params=ext_params, ext_param_bounds=ext_param_bounds,
    )
    xc.components.append(comp)


def _split_add_mixing_factor(
    xc: XCFunctional,
    *,
    role: str,
    new_component: str,
    requested_mixing: object,
) -> float:
    """Add pure X/C components by splitting an existing same-role slot.

    Naked add_component with mixing=1.0 almost always violates XC sum rules on
    normal functionals.  For pure exchange/correlation components, treat add as
    a conservative mixture-model edit: carve a small weight from the largest
    existing same-role component and give it to the new component.
    """
    if any(c.name == new_component and c.role == role for c in xc.components):
        raise ValueError(f"Cannot add duplicate {role} component {new_component!r}")

    same_role = [c for c in xc.components if c.role == role and c.mixing_factor > 1e-10]
    requested = (
        float(requested_mixing)
        if isinstance(requested_mixing, (int, float))
        else SPLIT_ADD_DEFAULT_MIXING
    )
    requested = max(0.0, min(SPLIT_ADD_MAX_MIXING, requested))

    if same_role:
        donor = max(same_role, key=lambda c: c.mixing_factor)
        new_mixing = min(requested, donor.mixing_factor)
        if new_mixing <= 1e-12:
            raise ValueError(f"Cannot split-add {new_component!r}; donor {donor.name} has no mixing weight")
        donor.mixing_factor = max(0.0, donor.mixing_factor - new_mixing)
        split_log = {
            "new_component": new_component,
            "role": role,
            "new_mixing": new_mixing,
            "donor_component": donor.name,
            "donor_remaining_mixing": donor.mixing_factor,
        }
        xc.metadata.setdefault("split_add_component_edits", []).append(split_log)
        return new_mixing

    if role == "exchange":
        current = sum(c.mixing_factor for c in xc.components if c.role == "exchange")
        room = max(0.0, 1.0 - xc.hf_exchange.alpha - current)
    else:
        current = sum(c.mixing_factor for c in xc.components if c.role == "correlation")
        room = max(0.0, 1.0 - current)
    new_mixing = min(room, max(requested, room if requested_mixing is None else 0.0))
    if new_mixing <= 1e-12:
        raise ValueError(f"Cannot add {new_component!r}; no available {role} mixing budget")
    xc.metadata.setdefault("split_add_component_edits", []).append({
        "new_component": new_component,
        "role": role,
        "new_mixing": new_mixing,
        "donor_component": "",
        "donor_remaining_mixing": None,
    })
    return new_mixing


def _apply_add_descriptor(xc: XCFunctional, action: Action) -> None:
    """Add one restricted descriptor correction component."""
    if any(c.descriptor_spec or c.role == "descriptor_correction" for c in xc.components):
        raise ValueError("Cannot add more than one descriptor correction component")
    raw = action.changes.get("descriptor", action.changes)
    if not isinstance(raw, dict):
        raise ValueError("add_descriptor requires changes.descriptor to be a dict")
    spec = normalize_descriptor_spec(raw)
    name = spec["name"]
    if not name.startswith("DESC_"):
        name = "DESC_" + name
        spec["name"] = name
    params = dict(spec.get("params", {}))
    comp = ComponentSpec(
        name=name,
        xctype="GGA",
        mixing_factor=1.0,
        role="descriptor_correction",
        formula=f"{spec['template']} descriptor correction",
        descriptor_spec=spec,
        ext_params=params,
        ext_param_bounds={
            "scale": (-0.05, 0.05),
            "gradient_sensitivity": (0.05, 10.0),
            "density_decay": (0.0, 10.0),
            "rho_power": (0.0, 1.0),
        },
    )
    xc.components.append(comp)


def _apply_remove_component(xc: XCFunctional, action: Action) -> None:
    name = action.changes.get("name")
    idx = action.changes.get("index")

    if name is not None:
        target = next((c for c in xc.components if c.name == name), None)
        _validate_component_removal(xc, target)
        xc.components = [c for c in xc.components if c.name != name]
    elif idx is not None:
        index = int(idx)
        target = xc.components[index]
        _validate_component_removal(xc, target)
        xc.components.pop(index)


def _validate_component_removal(xc: XCFunctional, target: ComponentSpec | None) -> None:
    if target is None:
        return
    if target.role not in {"exchange", "correlation"}:
        return
    active_same_role = [
        c for c in xc.components
        if c.role == target.role and c.mixing_factor > 1e-10
    ]
    if len(active_same_role) <= 1:
        raise ValueError(
            f"Cannot remove the only active {target.role} component {target.name!r}; "
            "use replace_component or split-and-add another same-role component first"
        )


def _apply_replace_component(xc: XCFunctional, action: Action) -> None:
    changes = _normalize_replace_changes(action.changes)
    old_name = changes.get("old_name", "")
    new_name = changes.get("new_name", "")
    new_xctype = changes.get("new_xctype")
    new_ext_params = changes.get("new_ext_params")
    new_ext_param_bounds = changes.get("new_ext_param_bounds")
    new_role = changes.get("role")
    new_mixing_factor = changes.get("mixing_factor")
    if not new_name or new_name not in NAME_TO_ID:
        raise ValueError(f"Cannot replace with unknown XC component {new_name!r}")

    for i, c in enumerate(xc.components):
        if c.name == old_name:
            canonical_role = PURE_COMPONENT_ROLES.get(new_name)
            if canonical_role is not None and c.role != canonical_role:
                raise ValueError(
                    f"Cannot replace {c.role} component {old_name!r} with pure "
                    f"{canonical_role} component {new_name!r}"
                )
            canonical_xctype = COMPONENT_XCTYPES.get(new_name)
            # ext_params must be a dict; coerce non-dict values
            if isinstance(new_ext_params, dict):
                ext_params = new_ext_params
            elif isinstance(new_ext_params, (int, float)):
                # LLM returned a single value (e.g. omega=0.2) — try to map to _omega
                ext_params = {"_omega": float(new_ext_params)}
            else:
                ext_params = {}
            ext_param_bounds = new_ext_param_bounds if isinstance(new_ext_param_bounds, dict) else {}
            if not ext_params and new_name in NAME_TO_ID:
                libxc_id = NAME_TO_ID[new_name]
                if libxc_id in LIBXC_PARAMS and LIBXC_PARAMS[libxc_id]:
                    ext_params = {pname: default for pname, default, _ in LIBXC_PARAMS[libxc_id]}
                    ext_param_bounds = {pname: bounds for pname, _, bounds in LIBXC_PARAMS[libxc_id]}

            xc.components[i] = ComponentSpec(
                name=new_name,
                xctype=canonical_xctype or new_xctype or c.xctype,
                mixing_factor=float(new_mixing_factor) if isinstance(new_mixing_factor, (int, float)) else c.mixing_factor,
                role=canonical_role or (new_role if isinstance(new_role, str) and new_role else c.role),
                ext_params=ext_params,
                ext_param_bounds=ext_param_bounds,
            )
            break


def _normalize_replace_changes(changes: dict) -> dict:
    """Accept common nested replace schemas and return flat executable keys."""
    normalized = dict(changes or {})
    nested = normalized.pop("replace", None) or normalized.pop("replace_component", None)
    if isinstance(nested, dict):
        normalized.setdefault(
            "old_name",
            _component_name(nested.get("old_name") or nested.get("from") or nested.get("old_component")),
        )
        _merge_component_spec(
            normalized,
            nested.get("new_name") or nested.get("to") or nested.get("new_component"),
            prefix="new_",
        )
        for key in ("role", "mixing_factor", "new_xctype"):
            if key in nested and key not in normalized:
                normalized[key] = nested[key]
        if "ext_params" in nested and isinstance(nested["ext_params"], dict):
            normalized.setdefault("new_ext_params", nested["ext_params"])

    if "old_component" in normalized and "old_name" not in normalized:
        normalized["old_name"] = _component_name(normalized.pop("old_component"))
    if "new_component" in normalized and "new_name" not in normalized:
        _merge_component_spec(normalized, normalized.pop("new_component"), prefix="new_")
    if isinstance(normalized.get("old_name"), dict):
        normalized["old_name"] = _component_name(normalized["old_name"])
    if isinstance(normalized.get("new_name"), dict):
        spec = normalized.pop("new_name")
        _merge_component_spec(normalized, spec, prefix="new_")
    return normalized


def _component_name(value) -> str:
    if isinstance(value, dict):
        for key in ("name", "component", "id"):
            if isinstance(value.get(key), str):
                return value[key]
        return ""
    return str(value) if value is not None else ""


def _merge_component_spec(target: dict, value, *, prefix: str) -> None:
    if isinstance(value, dict):
        name = _component_name(value)
        if name:
            target[f"{prefix}name"] = name
        if isinstance(value.get("xctype"), str):
            target[f"{prefix}xctype"] = value["xctype"]
        if isinstance(value.get("role"), str) and "role" not in target:
            target["role"] = value["role"]
        if isinstance(value.get("mixing_factor"), (int, float)) and "mixing_factor" not in target:
            target["mixing_factor"] = float(value["mixing_factor"])
        if isinstance(value.get("ext_params"), dict) and f"{prefix}ext_params" not in target:
            target[f"{prefix}ext_params"] = value["ext_params"]
        if isinstance(value.get("ext_param_bounds"), dict) and f"{prefix}ext_param_bounds" not in target:
            target[f"{prefix}ext_param_bounds"] = value["ext_param_bounds"]
        return
    name = _component_name(value)
    if name:
        target[f"{prefix}name"] = name


def _apply_enforce_constraint(xc: XCFunctional, action: Action) -> None:
    constraint_name = action.changes.get("constraint", "")
    catalogue_entry = CONSTRAINT_CATALOGUE.get(constraint_name)

    if catalogue_entry is None:
        logger.warning(f"Unknown constraint: {constraint_name}")
        return

    constraints = xc.metadata.get("enforced_constraints", [])
    if constraint_name not in constraints:
        constraints.append(constraint_name)
    xc.metadata["enforced_constraints"] = constraints

    # Actually modify parameters to satisfy the constraint
    if constraint_name == "lieb_oxford":
        for c in xc.components:
            if c.role == "exchange" and "_kappa" in c.ext_params:
                if c.ext_params["_kappa"] > 1.679:
                    c.ext_params["_kappa"] = min(c.ext_params["_kappa"], 0.804)

    elif constraint_name == "one_electron_sie":
        target_alpha = min(xc.hf_exchange.alpha + 0.05, 0.50)
        xc.hf_exchange.alpha = target_alpha
        exch_sum = sum(c.mixing_factor for c in xc.components if c.role == "exchange")
        target_dft = 1.0 - target_alpha
        if exch_sum > 0 and target_dft > 0:
            scale = target_dft / exch_sum
            for c in xc.components:
                if c.role == "exchange":
                    c.mixing_factor *= scale

    elif constraint_name == "lda_limit":
        for c in xc.components:
            if c.name == "GGA_X_PBE" and "_mu" in c.ext_params:
                if c.ext_params["_mu"] < 0.10:
                    c.ext_params["_mu"] = 0.1234


def parse_action_from_json(json_str: str) -> Action:
    """Parse an LLM response JSON into an Action object."""
    import json
    data = json.loads(json_str)
    action_type = ActionType(data["action_type"])
    changes = data.get("changes", {})
    for k, v in changes.items():
        if isinstance(v, (int, float)):
            changes[k] = float(v)
    rationale = data.get("rationale", "")
    return Action(action_type=action_type, changes=changes, rationale=rationale)
