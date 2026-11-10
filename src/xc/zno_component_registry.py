"""Load the audited component pool for the semilocal bulk-ZnO gap search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ZNO_COMPONENT_REGISTRY = (
    Path(__file__).resolve().parents[2] / "knowledge" / "zno_gap_component_registry.json"
)
DEFAULT_ZNO_COMPONENT_PROMPT_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "knowledge"
    / "zno_gap_component_prompt_registry.json"
)
SUPPORTED_COMPONENT_POOLS = frozenset({"expanded", "legacy", "zno_gap"})
VIRTUAL_ZNO_COMPONENT_ROLES = {
    "GGA_X_HCTH_407_HLE16": "exchange",
    "GGA_C_HCTH_407_HLE16": "correlation",
}


def load_zno_component_registry(path: Path | str | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else DEFAULT_ZNO_COMPONENT_REGISTRY
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "zno-gap-component-registry-v1":
        raise ValueError(f"Unsupported ZnO component registry schema in {registry_path}")
    if not isinstance(payload.get("components"), list):
        raise ValueError(f"ZnO component registry has no component records: {registry_path}")
    return payload


def load_zno_component_prompt_registry(path: Path | str | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else DEFAULT_ZNO_COMPONENT_PROMPT_REGISTRY
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "zno-gap-component-prompt-v1":
        raise ValueError(f"Unsupported ZnO prompt-registry schema in {registry_path}")
    if not isinstance(payload.get("priority_components"), list):
        raise ValueError(f"ZnO prompt registry has no priority component records: {registry_path}")
    return payload


def zno_gap_component_roles(path: Path | str | None = None) -> dict[str, str]:
    registry = load_zno_component_registry(path)
    roles = {
        str(item["name"]): str(item["role"])
        for item in registry["components"]
        if item.get("eligible") is True
    }
    # These are exact repository-defined decompositions of native HLE16, not
    # standalone Libxc names, so they are intentionally absent from the
    # generated Libxc audit registry above.
    roles.update(VIRTUAL_ZNO_COMPONENT_ROLES)
    return roles


def component_roles_for_pool(component_pool: str) -> dict[str, str]:
    """Resolve a named component pool without duplicating pool logic."""
    from .xc_functional import LEGACY_PURE_COMPONENT_ROLES, PURE_COMPONENT_ROLES

    if component_pool == "legacy":
        return dict(LEGACY_PURE_COMPONENT_ROLES)
    if component_pool == "expanded":
        return dict(PURE_COMPONENT_ROLES)
    if component_pool == "zno_gap":
        return zno_gap_component_roles()
    raise ValueError(f"Unsupported component_pool: {component_pool!r}")


def component_pool_signature(component_pool: str) -> dict[str, Any]:
    """Return a stable signature used to freeze an experimental search space."""
    roles = component_roles_for_pool(component_pool)
    canonical = json.dumps(
        sorted(roles.items()),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        "component_pool": component_pool,
        "component_count": len(roles),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def validate_component_pool_lock(
    component_pool: str,
    lock: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail before evaluation if a tracked experiment's component pool drifted."""
    actual = component_pool_signature(component_pool)
    if not lock:
        return actual

    expected_pool = str(lock.get("component_pool", "")).strip()
    if expected_pool and expected_pool != component_pool:
        raise ValueError(
            f"Component-pool lock targets {expected_pool!r}, but the run uses "
            f"{component_pool!r}"
        )

    expected_count = lock.get("component_count")
    if expected_count is not None and int(expected_count) != actual["component_count"]:
        raise ValueError(
            "Frozen component pool changed: "
            f"expected {int(expected_count)} components, found {actual['component_count']}"
        )

    expected_sha256 = str(lock.get("sha256", "")).strip().lower()
    if expected_sha256 and expected_sha256 != actual["sha256"]:
        raise ValueError(
            "Frozen component pool changed: "
            f"expected sha256={expected_sha256}, found sha256={actual['sha256']}"
        )
    return actual


def validate_component_for_pool(
    component_name: str,
    *,
    component_pool: str,
    roles: Mapping[str, str] | None = None,
) -> None:
    """Reject add/replace targets that are outside a selected audited pool."""
    if component_pool != "zno_gap":
        return
    allowed = dict(roles) if roles is not None else zno_gap_component_roles()
    if component_name not in allowed:
        registry = load_zno_component_registry()
        record = next(
            (item for item in registry["components"] if item.get("name") == component_name),
            None,
        )
        reasons = (record or {}).get("hard_exclusion_reasons", [])
        detail = "; ".join(str(item.get("reason", "")) for item in reasons) or "not in audited pool"
        raise ValueError(
            f"Component {component_name!r} is not allowed by the zno_gap pool: {detail}"
        )


def validate_functional_for_pool(xc: Any, *, component_pool: str) -> None:
    """Validate active exchange/correlation slots before starting a search."""
    if component_pool != "zno_gap":
        return
    allowed = zno_gap_component_roles()
    for component in getattr(xc, "components", []):
        if getattr(component, "role", "") not in {"exchange", "correlation"}:
            continue
        validate_component_for_pool(
            str(getattr(component, "name", "")),
            component_pool=component_pool,
            roles=allowed,
        )


def format_zno_gap_design_evidence(path: Path | str | None = None) -> str:
    """Return compact, scope-aware evidence for an LLM proposal prompt."""
    registry = load_zno_component_registry(path)
    records = []
    for item in registry["components"]:
        if not item.get("eligible") or not item.get("design_evidence"):
            continue
        for evidence in item["design_evidence"]:
            records.append((item["name"], evidence))

    order = {"direct": 0, "indirect": 1, "benchmark_only": 2, "context_only": 3}
    records.sort(key=lambda value: (order.get(value[1].get("relevance_to_bulk_zno_gap"), 9), value[0]))
    lines = [
        "Evidence is about original design or complete-functional benchmarks, not a causal guarantee for an isolated mixed component.",
        "Only entries marked direct were explicitly designed for band-gap-related behavior; context_only entries must not be presented as gap priors.",
    ]
    for name, evidence in records:
        sources = ",".join(evidence.get("source_ids", [])) or "unreferenced"
        lines.append(
            f"- {name}: relevance={evidence.get('relevance_to_bulk_zno_gap')}; "
            f"scope={evidence.get('evidence_scope')}; confidence={evidence.get('confidence')}; "
            f"claim={evidence.get('claim')} pairing={evidence.get('recommended_pairing')} "
            f"sources={sources}"
        )
    return "\n".join(lines)


def format_zno_gap_prompt_registry(path: Path | str | None = None) -> str:
    """Return the compact audited component catalogue used by proposal prompts."""
    return json.dumps(
        load_zno_component_prompt_registry(path),
        indent=2,
        ensure_ascii=False,
    )
