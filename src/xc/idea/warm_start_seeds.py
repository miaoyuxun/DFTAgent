"""Utilities for turning LLM warm-start results into EA initial seeds."""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import Any

from ..xc_functional import ComponentSpec, XCFunctional
from .evolutionary import EvolutionarySearch, EvolutionConfig

logger = logging.getLogger(__name__)

CORRELATION_MIX_COMPONENTS = ("GGA_C_PBE", "GGA_C_APBE", "MGGA_C_SCAN")


def load_llm_initial_structure_scaffolds(
    warm_start_dir: Path | str,
    *,
    max_structures: int = 5,
) -> list[tuple[str, XCFunctional]]:
    """Restore the raw LLM structures before any warm-start parameter tuning.

    The warm-start artifact records ``structure_initial`` once per scaffold.
    This loader deliberately ignores every later ``tuning`` record so an
    immediate-EA control tests LLM structural initialization alone.
    """
    warm_start_dir = Path(warm_start_dir)
    records = _load_records(warm_start_dir)
    scaffolds: list[tuple[str, XCFunctional]] = []
    seen: set[str] = set()
    for record in records:
        if record.get("stage") != "structure_initial":
            continue
        if int(record.get("scaffold_index", 0) or 0) <= 0:
            # The unchanged root baseline is supplied by MixedEvolutionarySearch.
            continue
        raw_xc = record.get("xc_dict")
        if not isinstance(raw_xc, dict):
            logger.warning("Skipping raw LLM scaffold without xc_dict: %s", record.get("name", "unknown"))
            continue
        try:
            xc = XCFunctional.from_dict(raw_xc)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping unreadable raw LLM scaffold %s: %s", record.get("name", "unknown"), exc)
            continue
        signature = xc.effective_signature()
        if signature in seen:
            continue
        seen.add(signature)
        label = str(record.get("scaffold_name") or f"llm_structure_{len(scaffolds) + 1}")
        xc.metadata.update({
            "llm_structure_source": str(warm_start_dir),
            "llm_structure_label": label,
            "llm_structure_original_action": copy.deepcopy(record.get("action_detail") or {}),
            "llm_structure_original_rationale": str(record.get("rationale") or ""),
        })
        scaffolds.append((label, xc))
        if len(scaffolds) >= max(0, int(max_structures)):
            break
    logger.info(
        "Loaded %d raw pre-tuning LLM structure scaffolds from %s",
        len(scaffolds),
        warm_start_dir,
    )
    return scaffolds


def build_correlation_mixture_scaffold(root_xc: XCFunctional) -> XCFunctional:
    """Build a shared EA scaffold for PBE/APBE/SCAN correlation mixtures.

    Warm-start can produce incompatible structures (replacement vs split-add).
    The EA handles them cleanly by optimizing a single superset scaffold where
    correlation mixing chooses among PBE, APBE, and SCAN-C.
    """
    source = copy.deepcopy(root_xc)
    components_by_name = {component.name: component for component in source.components}
    hjs_exchange = copy.deepcopy(components_by_name.get("GGA_X_HJS_PBE"))
    if hjs_exchange is None:
        exchanges = [component for component in source.components if component.role == "exchange"]
        if not exchanges:
            raise ValueError("Warm-start EA scaffold requires at least one exchange component")
        hjs_exchange = copy.deepcopy(exchanges[0])
    hjs_exchange.role = "exchange"

    pbe_c = copy.deepcopy(components_by_name.get("GGA_C_PBE") or XCFunctional._pbe_c())
    pbe_c.role = "correlation"
    pbe_c.mixing_factor = float(pbe_c.mixing_factor)

    apbe_c = ComponentSpec(
        name="GGA_C_APBE",
        xctype="GGA",
        mixing_factor=0.0,
        role="correlation",
    )
    scan_c = ComponentSpec(
        name="MGGA_C_SCAN",
        xctype="MGGA",
        mixing_factor=0.0,
        role="correlation",
    )

    scaffold = XCFunctional(
        name=f"{source.name}__corrmix_pbe_apbe_scan",
        components=[hjs_exchange, pbe_c, apbe_c, scan_c],
        hf_exchange=copy.deepcopy(source.hf_exchange),
        metadata={
            **copy.deepcopy(source.metadata),
            "action": "evolutionary_warm_start_scaffold",
            "warm_start_scaffold": "correlation_mixture_pbe_apbe_scan",
            "warm_start_root": source.name,
        },
    )
    return scaffold



def build_llm_component_union_scaffold(
    root_xc: XCFunctional,
    scaffolds: list[tuple[str, XCFunctional]],
) -> XCFunctional:
    """Build a zero-weight union scaffold from raw LLM structure hypotheses.

    Existing root components retain their original coefficients.  Every pure
    exchange/correlation component introduced by an LLM structure is added at
    zero mixing, so a Latin-hypercube EA initializes all candidates from the
    same common parameter space rather than from scaffold-default parameters.
    """
    union = copy.deepcopy(root_xc)
    known = {(component.role, component.name) for component in union.components}
    added: list[dict[str, str]] = []
    for label, scaffold in scaffolds:
        for component in scaffold.components:
            if component.role not in {"exchange", "correlation"}:
                continue
            key = (component.role, component.name)
            if key in known:
                continue
            new_component = copy.deepcopy(component)
            new_component.mixing_factor = 0.0
            union.components.append(new_component)
            known.add(key)
            added.append({"label": label, "name": component.name, "role": component.role})
    union.name = f"{root_xc.name}__llm_component_union"
    union.metadata.update({
        "action": "llm_component_union_scaffold",
        "search_algorithm": "mixed_evolutionary",
        "llm_component_union": added,
        "llm_component_union_source": [label for label, _ in scaffolds],
    })
    logger.info(
        "Built LLM component-union scaffold %s: added=%s",
        union.name,
        added,
    )
    return union


def load_warm_start_seed_vectors(
    warm_start_dir: Path | str,
    scaffold: XCFunctional,
    *,
    min_reward: float = 0.3,
    max_seeds: int | None = None,
    scorer_config: EvolutionConfig | None = None,
) -> list[dict[str, float]]:
    """Load warm-start candidates and map them into ``scaffold`` parameters."""
    warm_start_dir = Path(warm_start_dir)
    records = _load_records(warm_start_dir)
    if not records:
        logger.warning("No warm-start records found in %s", warm_start_dir)
        return []

    scorer = EvolutionarySearch(scorer_config or EvolutionConfig())
    parameters = scorer._parameter_space(scaffold)
    baseline_vector = scorer._read_parameters(scaffold, parameters)
    parameter_keys = {parameter.key for parameter in parameters}

    eligible = [
        record for record in records
        if _finite_float(record.get("fitness")) >= min_reward
        and int(record.get("search_failures") or 0) == 0
    ]
    if not eligible:
        logger.warning(
            "No warm-start candidates passed min_reward=%.3f in %s",
            min_reward,
            warm_start_dir,
        )
        return []

    ordered = _best_per_cluster_first(eligible)
    vectors: list[dict[str, float]] = []
    selected_records: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for record in ordered:
        vector = _record_to_vector(record, baseline_vector, parameter_keys)
        repaired = scorer._repair_vector(vector, parameters)
        key = tuple(round(repaired[parameter.key], 10) for parameter in parameters)
        if key in seen:
            continue
        seen.add(key)
        vectors.append(repaired)
        selected_records.append(record)
        if max_seeds is not None and len(vectors) >= max_seeds:
            break

    logger.info(
        "Loaded %d EA warm-start seed vectors from %s (eligible=%d, min_reward=%.3f, scaffold=%s)",
        len(vectors),
        warm_start_dir,
        len(eligible),
        min_reward,
        scaffold.name,
    )
    for index, (record, vector) in enumerate(zip(selected_records, vectors, strict=False), start=1):
        logger.info(
            "EA warm-start seed %02d cluster=%s scaffold=%s fitness=%.6f vector=%s",
            index,
            _cluster_label(record),
            record.get("scaffold_name") or record.get("scaffold_index") or "unknown",
            _finite_float(record.get("fitness")),
            json.dumps(vector, sort_keys=True),
        )
    return vectors


def _load_records(warm_start_dir: Path) -> list[dict[str, Any]]:
    for name in ("warm_start_all_candidates.json", "warm_start_population.json"):
        path = warm_start_dir / name
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read warm-start records from %s: %s", path, exc)
            continue
        if isinstance(raw, list):
            logger.info("Loaded %d warm-start records from %s", len(raw), path)
            return [item for item in raw if isinstance(item, dict)]
    return []


def _best_per_cluster_first(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=lambda item: _finite_float(item.get("fitness")), reverse=True)
    best_by_cluster: dict[str, dict[str, Any]] = {}
    for record in sorted_records:
        best_by_cluster.setdefault(_cluster_label(record), record)
    selected_ids = {id(record) for record in best_by_cluster.values()}
    return list(best_by_cluster.values()) + [
        record for record in sorted_records
        if id(record) not in selected_ids
    ]


def _cluster_label(record: dict[str, Any]) -> str:
    parameters = dict(record.get("parameters") or {})
    pbe = _finite_float(parameters.get("mixing:GGA_C_PBE"))
    apbe = _finite_float(parameters.get("mixing:GGA_C_APBE"))
    scan = _finite_float(parameters.get("mixing:MGGA_C_SCAN"))
    if not math.isfinite(pbe):
        pbe = _component_mixing(record, "GGA_C_PBE")
    if not math.isfinite(apbe):
        apbe = _component_mixing(record, "GGA_C_APBE")
    if not math.isfinite(scan):
        scan = _component_mixing(record, "MGGA_C_SCAN")
    pbe = 0.0 if not math.isfinite(pbe) else pbe
    apbe = 0.0 if not math.isfinite(apbe) else apbe
    scan = 0.0 if not math.isfinite(scan) else scan
    if apbe > 0.5:
        return "APBE-dominant"
    if scan > 0.5:
        return "SCAN-dominant"
    if scan > 0.02:
        return "PBE+small-SCAN"
    return "PBE-only"


def _component_mixing(record: dict[str, Any], name: str) -> float:
    for component in dict(record.get("xc_dict") or {}).get("components", []):
        if isinstance(component, dict) and component.get("name") == name:
            return _finite_float(component.get("mixing_factor"))
    return float("-inf")


def _record_to_vector(
    record: dict[str, Any],
    baseline_vector: dict[str, float],
    parameter_keys: set[str],
) -> dict[str, float]:
    vector = dict(baseline_vector)
    record_parameters = {
        str(key): float(value)
        for key, value in dict(record.get("parameters") or {}).items()
        if _is_number(value)
    }

    for key, value in record_parameters.items():
        if key in parameter_keys and not key.startswith("mixing:GGA_C_") and key != "mixing:MGGA_C_SCAN":
            vector[key] = value

    for key in (f"mixing:{name}" for name in CORRELATION_MIX_COMPONENTS):
        if key in parameter_keys:
            vector[key] = 0.0

    correlation_seen = False
    for component in dict(record.get("xc_dict") or {}).get("components", []):
        if not isinstance(component, dict):
            continue
        name = component.get("name")
        key = f"mixing:{name}"
        if name in CORRELATION_MIX_COMPONENTS and key in parameter_keys:
            vector[key] = float(component.get("mixing_factor") or 0.0)
            correlation_seen = True

    for name in CORRELATION_MIX_COMPONENTS:
        key = f"mixing:{name}"
        if key in parameter_keys and key in record_parameters:
            vector[key] = record_parameters[key]
            correlation_seen = True

    if not correlation_seen and "mixing:GGA_C_PBE" in parameter_keys:
        vector["mixing:GGA_C_PBE"] = 1.0

    for key, value in record_parameters.items():
        if key in parameter_keys and not key.startswith("mixing:"):
            vector[key] = value
    return vector


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return number if math.isfinite(number) else float("-inf")


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
