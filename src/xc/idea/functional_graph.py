"""Functional graph construction from benchmark result collections.

Phase 1 builds canonical nodes from mass sweeps and benchmark caches. Phase 2
adds directed, replay-validated Agent transitions between those nodes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from .actions import Action, ActionType, apply_action
from ..xc_functional import (
    COMPONENT_XCTYPES,
    LIBXC_PARAMS,
    NAME_TO_ID,
    ComponentSpec,
    HFExchange,
    XCFunctional,
)


TARGET_SUBSETS = ("SIE4x4",)
GUARD_SUBSETS = ("BH76", "G21IP", "G21EA")
EXPECTED_SUBSETS = TARGET_SUBSETS + GUARD_SUBSETS

CONVERSION_OK = "ok"
CONVERSION_PARTIAL = "partial"

BASIN_GRAPH_MAX_GUARD_RATIO = 1.50

_NUMBERED_TERM = re.compile(
    r"^(?P<factor>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*\*\s*(?P<name>.+)$"
)
_KNOWN_LIBXC_ALIASES = {
    "HYB_GGA_XC_B3LYP": "B3LYP",
    "HYB_GGA_XC_CAM_B3LYP": "CAM-B3LYP",
    "HYB_GGA_XC_PBEH": "PBE0",
    "HYB_GGA_XC_PBE0": "PBE0",
    "HYB_MGGA_XC_WB97X": "WB97X",
}


@dataclass
class FunctionalNode:
    """Serializable node produced from one or more benchmark result files."""

    id: str
    name: str
    source_files: list[str]
    conversion: dict[str, Any]
    state: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    semantic: dict[str, Any]
    _build: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def conversion_status(self) -> str:
        """Compatibility accessor for build-time status checks."""

        return str(self.conversion.get("status", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source_files": list(self.source_files),
            "conversion": copy.deepcopy(self.conversion),
            "state": copy.deepcopy(self.state),
            "metrics": copy.deepcopy(self.metrics),
            "semantic": copy.deepcopy(self.semantic),
        }


@dataclass
class FunctionalEdge:
    """Directed, evaluated transition between two functional nodes."""

    id: str
    source: str
    target: str
    operation: dict[str, Any]
    metric_delta: dict[str, dict[str, Any]]
    semantic: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "operation": copy.deepcopy(self.operation),
            "metric_delta": copy.deepcopy(self.metric_delta),
            "semantic": copy.deepcopy(self.semantic),
        }


@dataclass
class NodeBuildReport:
    """Result of scanning and merging one or more benchmark result sources."""

    source_dirs: dict[str, str]
    source_file_counts: dict[str, int]
    scanned_file_counts: dict[str, int]
    input_files: int
    nodes: list[FunctionalNode]
    invalid_files: list[dict[str, str]]
    skipped_files: list[dict[str, str]]

    @property
    def mass_results_dir(self) -> str:
        return self.source_dirs.get("mass", "")

    @property
    def benchmark_results_dir(self) -> str:
        return self.source_dirs.get("benchmark_cache", "")

    @property
    def ok_nodes(self) -> int:
        return sum(node.conversion_status == CONVERSION_OK for node in self.nodes)

    @property
    def partial_nodes(self) -> int:
        return sum(node.conversion_status == CONVERSION_PARTIAL for node in self.nodes)

    def source_summary(self) -> dict[str, Any]:
        return {
            "input_dirs": copy.deepcopy(self.source_dirs),
            "n_input_files": self.input_files,
            "n_nodes": len(self.nodes),
            "n_partial_nodes": self.partial_nodes,
        }


@dataclass
class EdgeBuildReport:
    """Result of Phase 2 candidate generation and action replay."""

    edges: list[FunctionalEdge]
    eligible_nodes: int
    rewire_candidate_pairs: int
    replacement_candidate_pairs: int
    add_remove_candidate_pairs: int
    rejected_directions: int

    def summary(self) -> dict[str, int]:
        counts = Counter(edge.operation["action_type"] for edge in self.edges)
        return {
            "n_edges": len(self.edges),
            "n_eligible_nodes": self.eligible_nodes,
            "n_rewire_edges": counts.get("rewire", 0),
            "n_replace_edges": counts.get("replace", 0),
            "n_add_edges": counts.get("add", 0),
            "n_remove_edges": counts.get("remove", 0),
            "n_rewire_candidate_pairs": self.rewire_candidate_pairs,
            "n_replacement_candidate_pairs": self.replacement_candidate_pairs,
            "n_add_remove_candidate_pairs": self.add_remove_candidate_pairs,
            "n_rejected_directions": self.rejected_directions,
        }


@dataclass
class FunctionalGraph:
    """Loaded graph with deterministic indexes and compact query methods."""

    generated_at: str
    source: dict[str, Any]
    nodes: list[FunctionalNode]
    edges: list[FunctionalEdge]
    build_warnings: list[str]
    node_by_id: dict[str, FunctionalNode] = field(init=False, repr=False)
    node_by_structural_signature: dict[str, FunctionalNode] = field(
        init=False, repr=False
    )
    nodes_by_structural_signature: dict[str, list[FunctionalNode]] = field(
        init=False, repr=False
    )
    outgoing_edges: dict[str, list[FunctionalEdge]] = field(init=False, repr=False)
    incoming_edges: dict[str, list[FunctionalEdge]] = field(init=False, repr=False)
    _node_profiles: dict[str, dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.nodes.sort(key=lambda node: node.id)
        self.edges.sort(key=lambda edge: edge.id)
        self._build_indexes()

    def _build_indexes(self) -> None:
        self.node_by_id = {}
        self.node_by_structural_signature = {}
        self.nodes_by_structural_signature = defaultdict(list)
        self.outgoing_edges = defaultdict(list)
        self.incoming_edges = defaultdict(list)
        self._node_profiles = {}

        for node in self.nodes:
            _validate_loaded_node(node)
            if node.id in self.node_by_id:
                raise ValueError(f"duplicate functional graph node id: {node.id}")
            self.node_by_id[node.id] = node
            structural = _text(node.state.get("structural_signature"))
            if structural:
                self.nodes_by_structural_signature[structural].append(node)
            self._node_profiles[node.id] = _node_profile(node)

        seen_edge_ids: set[str] = set()
        for edge in self.edges:
            if edge.id in seen_edge_ids:
                raise ValueError(f"duplicate functional graph edge id: {edge.id}")
            seen_edge_ids.add(edge.id)
            if edge.source not in self.node_by_id:
                raise ValueError(
                    f"edge {edge.id} references unknown source node {edge.source}"
                )
            if edge.target not in self.node_by_id:
                raise ValueError(
                    f"edge {edge.id} references unknown target node {edge.target}"
                )
            _validate_loaded_edge(edge)
            self.outgoing_edges[edge.source].append(edge)
            self.incoming_edges[edge.target].append(edge)

        for node_id in self.node_by_id:
            self.outgoing_edges.setdefault(node_id, [])
            self.incoming_edges.setdefault(node_id, [])
        for signature, nodes in self.nodes_by_structural_signature.items():
            nodes.sort(key=lambda node: (not node.id.startswith("sig_"), node.id))
            self.node_by_structural_signature[signature] = nodes[0]
        for index in (
            self.outgoing_edges,
            self.incoming_edges,
        ):
            for values in index.values():
                values.sort(key=lambda item: item.id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "source": copy.deepcopy(self.source),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "build_warnings": list(self.build_warnings),
        }

    def query_neighbors(
        self,
        reference: str | FunctionalNode | XCFunctional | Mapping[str, Any],
        *,
        top_k: int = 5,
        include_self: bool = False,
        operability: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return structurally similar nodes as compact summaries."""

        limit = _query_limit(top_k)
        exact = self._resolve_node(reference)
        profile = self._reference_profile(reference, exact)
        if limit == 0 or profile is None:
            return []

        ranked: list[tuple[tuple[Any, ...], FunctionalNode, float]] = []
        for node in self.nodes:
            if not include_self and exact is not None and node.id == exact.id:
                continue
            if operability is not None and node.state.get("operability") != operability:
                continue
            match = _state_match(profile, self._node_profiles[node.id])
            ranked.append((_node_query_rank(node, match), node, match))
        ranked.sort(key=lambda item: item[0])
        return [
            _compact_node_summary(node, match)
            for _rank, node, match in ranked[:limit]
        ]

    def _resolve_node(
        self,
        reference: str | FunctionalNode | XCFunctional | Mapping[str, Any],
    ) -> FunctionalNode | None:
        if isinstance(reference, FunctionalNode):
            return self.node_by_id.get(reference.id)
        if isinstance(reference, XCFunctional):
            return self.node_by_structural_signature.get(
                reference.structural_signature()
            )
        if isinstance(reference, str):
            if reference in self.node_by_id:
                return self.node_by_id[reference]
            signature = reference.removeprefix("sig_")
            return self.node_by_structural_signature.get(signature)
        if isinstance(reference, Mapping):
            node_id = _text(reference.get("id"))
            if node_id in self.node_by_id:
                return self.node_by_id[node_id]
            state = reference.get("state", reference)
            if isinstance(state, Mapping):
                signature = _text(state.get("structural_signature"))
                if signature:
                    return self.node_by_structural_signature.get(signature)
                xc_dict = state.get("xc_dict", state)
                if isinstance(xc_dict, Mapping) and "components" in xc_dict:
                    try:
                        signature = functional_from_graph_xc_dict(
                            xc_dict
                        ).structural_signature()
                    except (KeyError, TypeError, ValueError):
                        return None
                    return self.node_by_structural_signature.get(signature)
        return None

    def _reference_profile(
        self,
        reference: str | FunctionalNode | XCFunctional | Mapping[str, Any],
        exact: FunctionalNode | None,
    ) -> dict[str, Any] | None:
        if exact is not None:
            return self._node_profiles[exact.id]
        return _profile_from_reference(reference)


def load_functional_graph(path: str | Path) -> FunctionalGraph:
    """Load and validate a Functional Graph artifact."""

    graph_path = Path(path)
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid functional graph JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("functional graph top-level value must be an object")

    expected_keys = {
        "generated_at",
        "source",
        "nodes",
        "edges",
        "build_warnings",
    }
    _require_exact_keys(payload, expected_keys, "functional graph")
    nodes_payload = payload["nodes"]
    edges_payload = payload["edges"]
    source = payload["source"]
    warnings = payload["build_warnings"]
    if not isinstance(nodes_payload, list):
        raise ValueError("functional graph nodes must be an array")
    if not isinstance(edges_payload, list):
        raise ValueError("functional graph edges must be an array")
    if not isinstance(source, Mapping):
        raise ValueError("functional graph source must be an object")
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        raise ValueError("functional graph build_warnings must be a string array")

    return FunctionalGraph(
        generated_at=str(payload["generated_at"]),
        source=dict(source),
        nodes=[
            _loaded_node(item, index)
            for index, item in enumerate(nodes_payload)
        ],
        edges=[
            _loaded_edge(item, index)
            for index, item in enumerate(edges_payload)
        ],
        build_warnings=list(warnings),
    )


def query_neighbors(
    graph: FunctionalGraph,
    reference: str | FunctionalNode | XCFunctional | Mapping[str, Any],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return graph.query_neighbors(reference, **kwargs)


def append_basin_agent_artifact(
    graph_path: str | Path,
    artifact_path: str | Path,
) -> dict[str, int]:
    """Project admitted Basin nodes and their lineage transitions into a graph."""

    graph_file = Path(graph_path)
    artifact_file = Path(artifact_path)
    graph = load_functional_graph(graph_file)
    try:
        records = json.loads(artifact_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Basin agent JSON: {exc}") from exc
    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise ValueError("Basin agent artifact must be an object array")

    raw_graph = json.loads(graph_file.read_text(encoding="utf-8"))
    if not isinstance(raw_graph, dict):
        raise ValueError("functional graph top-level value must be an object")
    raw_nodes = raw_graph.get("nodes")
    raw_edges = raw_graph.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError("functional graph nodes and edges must be arrays")

    source_root = graph_file.resolve().parent.parent
    source_label = _source_label(artifact_file, source_root)
    source_prefix = f"{source_label}#"
    records_by_id = {
        _text(record.get("node_id")): record
        for record in records
        if _text(record.get("node_id"))
    }

    current_node_ids = {
        _text(node.get("id"))
        for node in raw_nodes
        if isinstance(node, Mapping)
        and any(
            isinstance(source_file, str) and source_file.startswith(source_prefix)
            for source_file in node.get("source_files", [])
        )
    }
    current_edge_ids = {
        _text(edge.get("id"))
        for edge in raw_edges
        if isinstance(edge, Mapping)
        and (
            _text(edge.get("source")) in current_node_ids
            or _text(edge.get("target")) in current_node_ids
        )
    }
    base_nodes = [
        node for node in graph.nodes
        if node.id not in current_node_ids
    ]
    base_node_payloads = [
        node for node in raw_nodes
        if not isinstance(node, Mapping) or _text(node.get("id")) not in current_node_ids
    ]
    base_edge_payloads = [
        edge for edge in raw_edges
        if not isinstance(edge, Mapping) or _text(edge.get("id")) not in current_edge_ids
    ]

    existing_ids = {node.id for node in base_nodes}
    structural_to_graph = {
        signature: node.id
        for signature, nodes in _nodes_by_structural_signature(base_nodes).items()
        for node in nodes[:1]
    }
    selected_records = _select_basin_graph_records(
        records,
        records_by_id,
        existing_graph_signatures=_basin_graph_signatures(base_nodes),
    )

    candidate_nodes: dict[str, FunctionalNode] = {}
    skipped_invalid = 0
    for record in records:
        basin_node_id = _text(record.get("node_id"))
        if basin_node_id not in selected_records:
            continue
        try:
            node = _basin_record_to_graph_node(
                record,
                source_label=source_label,
                occupied_ids=existing_ids,
            )
            _validate_loaded_node(node)
        except (KeyError, TypeError, ValueError):
            skipped_invalid += 1
            continue
        candidate_nodes[basin_node_id] = node
        existing_ids.add(node.id)

    new_nodes = sorted(candidate_nodes.values(), key=lambda node: node.id)
    new_edges: list[FunctionalEdge] = []
    skipped_edges = 0
    basin_to_graph = {
        basin_node_id: node.id
        for basin_node_id, node in candidate_nodes.items()
    }
    ordered_records = sorted(
        (
            records_by_id[node_id]
            for node_id in candidate_nodes
        ),
        key=lambda record: (
            _basin_lineage_depth(record, records_by_id),
            _text(record.get("node_id")),
        ),
    )
    for record in ordered_records:
        basin_node_id = _text(record.get("node_id"))
        target_node = candidate_nodes[basin_node_id]
        source = _basin_graph_source_and_path(
            record,
            records_by_id=records_by_id,
            basin_to_graph=basin_to_graph,
            structural_to_graph=structural_to_graph,
        )
        if source is None:
            skipped_edges += 1
            continue
        source_id, source_record, path_records = source
        steps: list[dict[str, Any]] = []
        for path_record in path_records:
            parent_record = records_by_id.get(_text(path_record.get("parent_id")))
            for step in _basin_operation_steps(
                path_record,
                parent_record=parent_record,
            ):
                step["index"] = len(steps) + 1
                steps.append(step)
        if not steps:
            skipped_edges += 1
            continue
        source_basin_id = _text(source_record.get("node_id"))
        operation = {
            "action_type": "composite",
            "composite": True,
            "source": "basin_agent",
            "basin_parent_id": source_basin_id,
            "basin_child_id": basin_node_id,
            "primitive_action_count": len(steps),
            "steps": steps,
        }
        identity = {
            "source": source_id,
            "target": target_node.id,
            "operation": operation,
        }
        edge_hash = hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()[:16]
        edge_id = f"edge_basin_{edge_hash}"
        source_metrics = _basin_record_metrics(source_record)
        target_metrics = _basin_record_metrics(record)
        edge = FunctionalEdge(
            id=edge_id,
            source=source_id,
            target=target_node.id,
            operation=operation,
            metric_delta=_metric_delta_from_payloads(source_metrics, target_metrics),
            semantic={"llm_analysis": {"status": "pending"}},
        )
        _validate_loaded_edge(edge)
        new_edges.append(edge)

    new_edges.extend(build_functional_edges(new_nodes).edges)

    written_nodes = [
        *base_node_payloads,
        *(node.to_dict() for node in new_nodes),
    ]
    written_edges = [
        *base_edge_payloads,
        *(edge.to_dict() for edge in new_edges),
    ]
    written_nodes.sort(key=lambda item: str(item.get("id", "")))
    written_edges.sort(key=lambda item: str(item.get("id", "")))
    raw_graph["nodes"] = written_nodes
    raw_graph["edges"] = written_edges
    source = raw_graph.get("source")
    if isinstance(source, dict):
        source["n_nodes"] = len(written_nodes)
        source["n_edges"] = len(written_edges)
    _write_graph_payload_atomic(raw_graph, graph_file)

    new_node_ids = {node.id for node in new_nodes}
    new_edge_ids = {edge.id for edge in new_edges}

    return {
        "nodes_added": len(new_node_ids - current_node_ids),
        "edges_added": len(new_edge_ids - current_edge_ids),
        "skipped_invalid_nodes": skipped_invalid,
        "skipped_edges": skipped_edges,
    }


def _select_basin_graph_records(
    records: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    existing_graph_signatures: set[str] | None = None,
) -> dict[str, str]:
    """Select one champion, parent-gain, and novelty node per cycle."""

    root_reward = _basin_root_reward(records)
    if root_reward is None:
        return {}
    existing_signatures = existing_graph_signatures or set()
    eligible_by_cycle: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        node_id = _text(record.get("node_id"))
        if not node_id:
            continue
        if not _basin_base_graph_eligible(record, root_reward=root_reward):
            continue
        signatures = _basin_record_signatures(record)
        if not signatures or signatures & existing_signatures:
            continue
        eligible_by_cycle[int(record.get("cycle", 0) or 0)].append(record)

    selected: dict[str, str] = {}
    selected_signatures: set[str] = set()
    for cycle in sorted(eligible_by_cycle):
        cycle_records = eligible_by_cycle[cycle]
        selected_lineages: set[str] = set()
        role_orders = (
            ("champion", _basin_champion_order),
            (
                "parent_delta",
                lambda record: _basin_parent_delta_order(record, records_by_id),
            ),
            ("diversity", _basin_diversity_order),
        )
        for role, order_key in role_orders:
            for record in sorted(cycle_records, key=order_key):
                node_id = _text(record.get("node_id"))
                lineage_id = _basin_lineage_id(record)
                signatures = _basin_record_signatures(record)
                if node_id in selected or lineage_id in selected_lineages:
                    continue
                if signatures & selected_signatures:
                    continue
                selected[node_id] = role
                selected_lineages.add(lineage_id)
                selected_signatures.update(signatures)
                break
    return selected


def _basin_champion_order(record: Mapping[str, Any]) -> tuple[float, float, str]:
    return (
        -float(record["raw_reward"]),
        float(record["worst_validation_ratio"]),
        _text(record.get("node_id")),
    )


def _basin_parent_delta_order(
    record: Mapping[str, Any],
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, float, str]:
    return (
        -_basin_parent_delta(record, records_by_id),
        -float(record["raw_reward"]),
        float(record["worst_validation_ratio"]),
        _text(record.get("node_id")),
    )


def _basin_diversity_order(record: Mapping[str, Any]) -> tuple[float, float, str]:
    novelty = record.get("nearest_delta")
    return (
        -float(novelty) if _finite_number(novelty) else 0.0,
        -float(record["raw_reward"]),
        _text(record.get("node_id")),
    )


def _basin_root_reward(records: Sequence[Mapping[str, Any]]) -> float | None:
    for record in records:
        if _text(record.get("stage")) == "root" and _finite_number(
            record.get("raw_reward")
        ):
            return float(record["raw_reward"])
    return None


def _basin_base_graph_eligible(
    record: Mapping[str, Any],
    *,
    root_reward: float,
) -> bool:
    if not _text(record.get("node_id")) or _text(record.get("stage")) == "root":
        return False
    if not _basin_record_complete(record):
        return False
    if record.get("promotable") is not True:
        return False
    if record.get("success_vs_parent") is not True:
        return False
    if _text(record.get("failure_reason")):
        return False
    reward = record.get("raw_reward")
    guard_ratio = record.get("worst_validation_ratio")
    if not _finite_number(reward) or not _finite_number(guard_ratio):
        return False
    if float(reward) < root_reward:
        return False
    if float(guard_ratio) > BASIN_GRAPH_MAX_GUARD_RATIO:
        return False
    search = record.get("search")
    guard = record.get("guard")
    if not _basin_metrics_have_no_failures(search):
        return False
    if not _basin_metrics_have_no_failures(guard):
        return False
    return True


def _basin_record_complete(record: Mapping[str, Any]) -> bool:
    if not _finite_number(record.get("raw_reward")):
        return False
    search = record.get("search")
    guard = record.get("guard")
    if not isinstance(search, Mapping) or not search:
        return False
    if not isinstance(guard, Mapping) or not guard:
        return False
    return all(isinstance(payload, Mapping) for payload in search.values()) and all(
        isinstance(payload, Mapping) for payload in guard.values()
    )


def _basin_metrics_have_no_failures(metrics: Any) -> bool:
    if not isinstance(metrics, Mapping) or not metrics:
        return False
    for payload in metrics.values():
        if not isinstance(payload, Mapping):
            return False
        failures = _metric_integer(payload.get("n_failures"))
        if failures is None or failures != 0:
            return False
    return True


def _basin_graph_signatures(nodes: Sequence[FunctionalNode]) -> set[str]:
    return {
        _text(node.state.get("effective_signature"))
        for node in nodes
        if _text(node.state.get("effective_signature"))
    }


def _basin_record_signatures(record: Mapping[str, Any]) -> set[str]:
    effective = _text(record.get("effective_signature"))
    return {effective} if effective else set()


def _basin_lineage_id(record: Mapping[str, Any]) -> str:
    if _text(record.get("stage")) == "local_ea":
        parent_id = _text(record.get("parent_id"))
        if parent_id:
            return parent_id
    return _text(record.get("node_id"))


def _basin_parent_delta(
    record: Mapping[str, Any],
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> float:
    raw_reward = float(record["raw_reward"])
    parent = records_by_id.get(_text(record.get("parent_id")))
    parent_reward = parent.get("raw_reward") if parent is not None else None
    return (
        raw_reward - float(parent_reward)
        if _finite_number(parent_reward)
        else 0.0
    )


def _nodes_by_structural_signature(
    nodes: Sequence[FunctionalNode],
) -> dict[str, list[FunctionalNode]]:
    grouped: dict[str, list[FunctionalNode]] = defaultdict(list)
    for node in nodes:
        signature = _text(node.state.get("structural_signature"))
        if signature:
            grouped[signature].append(node)
    for values in grouped.values():
        values.sort(key=lambda node: node.id)
    return grouped


def _basin_lineage_depth(
    record: Mapping[str, Any],
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> int:
    depth = 0
    parent_id = _text(record.get("parent_id"))
    visited: set[str] = set()
    while parent_id and parent_id not in visited:
        visited.add(parent_id)
        parent = records_by_id.get(parent_id)
        if parent is None:
            break
        depth += 1
        parent_id = _text(parent.get("parent_id"))
    return depth


def _basin_graph_source_and_path(
    record: Mapping[str, Any],
    *,
    records_by_id: Mapping[str, Mapping[str, Any]],
    basin_to_graph: Mapping[str, str],
    structural_to_graph: Mapping[str, str],
) -> tuple[str, Mapping[str, Any], list[Mapping[str, Any]]] | None:
    path = [record]
    target_cycle = int(record.get("cycle", 0) or 0)
    parent_id = _text(record.get("parent_id"))
    visited: set[str] = set()
    while parent_id and parent_id not in visited:
        visited.add(parent_id)
        parent = records_by_id.get(parent_id)
        if parent is None:
            return None
        source_id = basin_to_graph.get(parent_id)
        parent_cycle = int(parent.get("cycle", 0) or 0)
        if source_id is not None and parent_cycle < target_cycle:
            return source_id, parent, list(reversed(path))
        if _text(parent.get("stage")) == "root":
            source_id = structural_to_graph.get(_text(parent.get("signature")))
            if source_id is None:
                return None
            return source_id, parent, list(reversed(path))
        path.append(parent)
        parent_id = _text(parent.get("parent_id"))
    return None


def _basin_record_to_graph_node(
    record: Mapping[str, Any],
    *,
    source_label: str,
    occupied_ids: set[str],
) -> FunctionalNode:
    xc_payload = record.get("xc_dict")
    if not isinstance(xc_payload, Mapping):
        raise ValueError("Basin node has no xc_dict")
    xc = XCFunctional.from_dict(copy.deepcopy(dict(xc_payload)))
    structural = xc.structural_signature()
    effective = xc.effective_signature()
    basin_node_id = _text(record.get("node_id"))
    source_ref = f"{source_label}#{basin_node_id}"
    node_id = f"sig_{structural}"
    if node_id in occupied_ids:
        suffix = hashlib.sha256(
            f"{source_ref}|{effective}".encode("utf-8")
        ).hexdigest()[:16]
        node_id = f"sig_{structural}__basin_{suffix}"
        collision = 2
        while node_id in occupied_ids:
            node_id = f"sig_{structural}__basin_{suffix}_{collision}"
            collision += 1

    return FunctionalNode(
        id=node_id,
        name=xc.name,
        source_files=[source_ref],
        conversion={
            "status": CONVERSION_OK,
            "method": "basin_agent_xc_dict",
            "warnings": [],
        },
        state=_state_from_xc(xc, "full", backend_metadata=xc.metadata),
        metrics=_basin_record_metrics(record),
        semantic={
            "rung": xc.highest_rung,
            "role": _functional_role(xc, xc.metadata),
        },
    )


def _basin_record_metrics(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for group_name in ("search", "guard"):
        group = record.get(group_name)
        if not isinstance(group, Mapping):
            continue
        for subset, payload in group.items():
            if not isinstance(payload, Mapping):
                continue
            metrics[str(subset)] = {
                "mae_kcal": _metric_number(payload.get("mae_kcal")),
                "n_reactions": _metric_integer(payload.get("n_reactions")),
                "n_failures": _metric_integer(payload.get("n_failures")),
            }
    return metrics


def _basin_operation_steps(
    record: Mapping[str, Any],
    *,
    parent_record: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    detail = record.get("action_detail")
    raw_steps: list[Mapping[str, Any]] = []
    if isinstance(detail, Mapping) and isinstance(detail.get("actions"), list):
        raw_steps = [
            step for step in detail["actions"] if isinstance(step, Mapping)
        ]
    else:
        changes = _basin_parameter_changes(parent_record, record)
        if changes:
            raw_steps = [{"action_type": "rewire", "changes": changes}]

    aliases = {
        "add_component": "add",
        "remove_component": "remove",
        "replace_component": "replace",
        "enforce_constraint": "enforce",
    }
    steps: list[dict[str, Any]] = []
    for raw in raw_steps:
        action_type = aliases.get(
            _text(raw.get("action_type")), _text(raw.get("action_type"))
        )
        changes = raw.get("changes")
        if action_type not in {"rewire", "replace", "add", "remove", "enforce"}:
            continue
        if not isinstance(changes, Mapping):
            continue
        steps.append(
            {
                "index": len(steps) + 1,
                "action_type": action_type,
                "changes": copy.deepcopy(dict(changes)),
            }
        )
    return steps


def _basin_parameter_changes(
    parent_record: Mapping[str, Any] | None,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if parent_record is None:
        return {}
    source = parent_record.get("parameters")
    target = record.get("parameters")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        return {}
    changes: dict[str, Any] = {}
    for key in sorted(set(source) | set(target), key=str):
        source_value = source.get(key)
        target_value = target.get(key)
        if source_value != target_value:
            changes[str(key)] = copy.deepcopy(target_value)
    return changes


def _metric_delta_from_payloads(
    source_metrics: Mapping[str, Mapping[str, Any]],
    target_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    subsets = sorted(set(EXPECTED_SUBSETS) | set(source_metrics) | set(target_metrics))
    result: dict[str, dict[str, Any]] = {}
    for subset in subsets:
        source_payload = source_metrics.get(subset, {})
        target_payload = target_metrics.get(subset, {})
        source_mae = _metric_number(source_payload.get("mae_kcal"))
        target_mae = _metric_number(target_payload.get("mae_kcal"))
        source_failures = _metric_integer(source_payload.get("n_failures"))
        target_failures = _metric_integer(target_payload.get("n_failures"))
        delta_mae = None
        if source_mae is not None and target_mae is not None:
            delta_mae = round(target_mae - source_mae, 12)
        failure_delta = None
        if source_failures is not None and target_failures is not None:
            failure_delta = target_failures - source_failures
        result[subset] = {
            "delta_mae": delta_mae,
            "failure_delta": failure_delta,
        }
    return result


def _write_graph_payload_atomic(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        load_functional_graph(temp_path)
        os.replace(temp_path, output)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _loaded_node(payload: Any, index: int) -> FunctionalNode:
    context = f"functional graph node[{index}]"
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    required_fields = {
        "id", "name", "source_files", "conversion", "state", "metrics", "semantic"
    }
    _require_exact_keys(payload, required_fields, context)
    source_files = payload["source_files"]
    conversion = payload["conversion"]
    state = payload["state"]
    metrics = payload["metrics"]
    semantic = payload["semantic"]
    if not isinstance(payload["id"], str) or not payload["id"]:
        raise ValueError(f"{context}.id must be a non-empty string")
    if not isinstance(payload["name"], str):
        raise ValueError(f"{context}.name must be a string")
    if not isinstance(source_files, list) or not all(
        isinstance(item, str) for item in source_files
    ):
        raise ValueError(f"{context}.source_files must be a string array")
    for field_name, value in (
        ("conversion", conversion),
        ("state", state),
        ("metrics", metrics),
        ("semantic", semantic),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{context}.{field_name} must be an object")
    _require_exact_keys(
        conversion,
        {"status", "method", "warnings"},
        f"{context}.conversion",
    )
    if conversion["status"] not in {CONVERSION_OK, CONVERSION_PARTIAL}:
        raise ValueError(f"{context}.conversion.status is invalid")
    if not isinstance(conversion["method"], str):
        raise ValueError(f"{context}.conversion.method must be a string")
    conversion_warnings = conversion["warnings"]
    if not isinstance(conversion_warnings, list) or not all(
        isinstance(item, str) for item in conversion_warnings
    ):
        raise ValueError(f"{context}.conversion.warnings must be a string array")
    if state.get("operability") not in {
        "full",
        "restricted_native",
        "unavailable",
        "invalid_state",
    }:
        raise ValueError(f"{context}.state.operability is invalid")
    if not all(isinstance(value, Mapping) for value in metrics.values()):
        raise ValueError(f"{context}.metrics values must be objects")
    return FunctionalNode(
        id=payload["id"],
        name=payload["name"],
        source_files=list(source_files),
        conversion=dict(conversion),
        state=dict(state),
        metrics={str(key): dict(value) for key, value in metrics.items()},
        semantic=dict(semantic),
    )


def _loaded_edge(payload: Any, index: int) -> FunctionalEdge:
    context = f"functional graph edge[{index}]"
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    required_fields = {"id", "source", "target", "operation", "metric_delta", "semantic"}
    _require_exact_keys(payload, required_fields, context)
    for field_name in ("id", "source", "target"):
        if not isinstance(payload[field_name], str) or not payload[field_name]:
            raise ValueError(f"{context}.{field_name} must be a non-empty string")
    for field_name in ("operation", "metric_delta", "semantic"):
        if not isinstance(payload[field_name], Mapping):
            raise ValueError(f"{context}.{field_name} must be an object")
    metric_delta = payload["metric_delta"]
    if not all(isinstance(value, Mapping) for value in metric_delta.values()):
        raise ValueError(f"{context}.metric_delta values must be objects")
    return FunctionalEdge(
        id=payload["id"],
        source=payload["source"],
        target=payload["target"],
        operation=dict(payload["operation"]),
        metric_delta={str(key): dict(value) for key, value in metric_delta.items()},
        semantic=dict(payload["semantic"]),
    )


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(payload)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise ValueError(f"{context} fields are invalid: {'; '.join(details)}")


@dataclass
class _Conversion:
    xc: XCFunctional | None
    status: str
    method: str
    operability: str
    warnings: list[str]


@dataclass
class _NormalizedPayload:
    functional: str
    xc_string: str
    metadata: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    embedded_xc_dict: dict[str, Any] | None
    source_kind: str
    benchmark_protocol: dict[str, Any]


@dataclass
class _SourceScan:
    source_kind: str
    directory: str
    scanned_files: int
    input_files: int
    nodes: list[FunctionalNode]
    invalid_files: list[dict[str, str]]
    skipped_files: list[dict[str, str]]


def build_nodes_from_mass_results(
    mass_results_dir: str | Path,
    *,
    source_root: str | Path | None = None,
) -> NodeBuildReport:
    """Read, convert, and deduplicate all JSON files in ``mass_results_dir``."""

    mass_dir = Path(mass_results_dir)
    root = Path(source_root).resolve() if source_root is not None else mass_dir.parent.resolve()
    scan = _scan_result_source(mass_dir, source_kind="mass", source_root=root)
    return _node_report_from_scans([scan])


def build_nodes_from_benchmark_results(
    benchmark_results_dir: str | Path,
    *,
    source_root: str | Path | None = None,
) -> NodeBuildReport:
    """Read functional benchmark caches while skipping reports and other JSON artifacts."""

    benchmark_dir = Path(benchmark_results_dir)
    root = (
        Path(source_root).resolve()
        if source_root is not None
        else benchmark_dir.parent.resolve()
    )
    scan = _scan_result_source(
        benchmark_dir,
        source_kind="benchmark_cache",
        source_root=root,
    )
    return _node_report_from_scans([scan])


def build_nodes_from_result_sources(
    mass_results_dir: str | Path,
    benchmark_results_dir: str | Path | None = None,
    *,
    source_root: str | Path | None = None,
) -> NodeBuildReport:
    """Build one deduplicated node set from all configured result directories."""

    mass_dir = Path(mass_results_dir)
    root = Path(source_root).resolve() if source_root is not None else mass_dir.parent.resolve()
    scans = [_scan_result_source(mass_dir, source_kind="mass", source_root=root)]
    if benchmark_results_dir is not None:
        scans.append(_scan_result_source(
            Path(benchmark_results_dir),
            source_kind="benchmark_cache",
            source_root=root,
        ))
    return _node_report_from_scans(scans)


def _scan_result_source(
    directory: Path,
    *,
    source_kind: str,
    source_root: Path,
) -> _SourceScan:
    if not directory.is_dir():
        raise FileNotFoundError(f"{source_kind} results directory does not exist: {directory}")

    paths = sorted(directory.glob("*.json"), key=lambda path: path.name)
    nodes: list[FunctionalNode] = []
    invalid_files: list[dict[str, str]] = []
    skipped_files: list[dict[str, str]] = []
    input_files = 0
    for path in paths:
        source_file = _source_label(path, source_root)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise TypeError("top-level JSON value must be an object")
        except Exception as exc:
            input_files += 1
            invalid_files.append({
                "source_file": source_file,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        if source_kind == "benchmark_cache" and not _is_functional_benchmark_cache(raw):
            skipped_files.append({
                "source_file": source_file,
                "reason": "JSON object is not a functional benchmark cache",
            })
            continue

        input_files += 1
        try:
            nodes.append(build_functional_node(
                raw,
                source_file=source_file,
                source_kind=source_kind,
            ))
        except Exception as exc:
            invalid_files.append({
                "source_file": source_file,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return _SourceScan(
        source_kind=source_kind,
        directory=_source_label(directory, source_root),
        scanned_files=len(paths),
        input_files=input_files,
        nodes=nodes,
        invalid_files=invalid_files,
        skipped_files=skipped_files,
    )


def _node_report_from_scans(scans: Sequence[_SourceScan]) -> NodeBuildReport:
    return NodeBuildReport(
        source_dirs={scan.source_kind: scan.directory for scan in scans},
        source_file_counts={scan.source_kind: scan.input_files for scan in scans},
        scanned_file_counts={scan.source_kind: scan.scanned_files for scan in scans},
        input_files=sum(scan.input_files for scan in scans),
        nodes=merge_functional_nodes([
            node
            for scan in scans
            for node in scan.nodes
        ]),
        invalid_files=[
            item
            for scan in scans
            for item in scan.invalid_files
        ],
        skipped_files=[
            item
            for scan in scans
            for item in scan.skipped_files
        ],
    )


def _is_functional_benchmark_cache(raw: Mapping[str, Any]) -> bool:
    meta = raw.get("__meta__")
    if not isinstance(meta, Mapping):
        return False
    xc_dict = meta.get("xc_dict")
    effective_spec = meta.get("effective_spec")
    functional = _text(meta.get("xc_name"))
    if not functional and isinstance(xc_dict, Mapping):
        functional = _text(xc_dict.get("name"))
    has_state = isinstance(xc_dict, Mapping) or (
        isinstance(effective_spec, Mapping)
        and bool(_text(effective_spec.get("xc_code")))
    )
    return bool(functional and has_state)


def _normalize_input_payload(
    raw: Mapping[str, Any],
    *,
    source_file: str,
    source_kind: str | None,
    warnings: list[str],
) -> _NormalizedPayload:
    inferred_kind = source_kind or (
        "benchmark_cache" if _is_functional_benchmark_cache(raw) else "mass"
    )
    if inferred_kind == "benchmark_cache":
        return _normalize_benchmark_cache_payload(raw, source_file=source_file, warnings=warnings)
    if inferred_kind != "mass":
        raise ValueError(f"unsupported graph result source kind: {inferred_kind!r}")

    metadata = _mapping_copy(raw.get("metadata"), warnings, "metadata")
    functional = _text(raw.get("functional")) or Path(source_file).stem
    xc_string = _text(raw.get("xc_string")) or _expression_from_metadata(functional, metadata)
    metrics = _normalize_metrics(raw.get("results"), warnings)
    benchmark_config = raw.get("benchmark_config", metadata.get("benchmark_config", {}))
    if not isinstance(benchmark_config, Mapping):
        benchmark_config = {}
    protocol = {
        "source_format": "mass_result",
        "cache_schema": _text(raw.get("cache_schema")) or _text(metadata.get("cache_schema")),
        "benchmark_config": copy.deepcopy(dict(benchmark_config)),
    }
    return _NormalizedPayload(
        functional=functional,
        xc_string=xc_string,
        metadata=metadata,
        metrics=metrics,
        embedded_xc_dict=None,
        source_kind="mass",
        benchmark_protocol=protocol,
    )


def _normalize_benchmark_cache_payload(
    raw: Mapping[str, Any],
    *,
    source_file: str,
    warnings: list[str],
) -> _NormalizedPayload:
    if not _is_functional_benchmark_cache(raw):
        raise ValueError("JSON object is not a functional benchmark cache")

    meta = _mapping_copy(raw.get("__meta__"), warnings, "__meta__")
    embedded = meta.get("xc_dict")
    embedded_xc_dict = copy.deepcopy(dict(embedded)) if isinstance(embedded, Mapping) else None
    effective_spec = meta.get("effective_spec")
    effective_spec = copy.deepcopy(dict(effective_spec)) if isinstance(effective_spec, Mapping) else {}
    benchmark_config = meta.get("benchmark_config")
    benchmark_config = (
        copy.deepcopy(dict(benchmark_config))
        if isinstance(benchmark_config, Mapping)
        else {}
    )

    functional = _text(meta.get("xc_name"))
    if not functional and embedded_xc_dict is not None:
        functional = _text(embedded_xc_dict.get("name"))
    functional = functional or Path(source_file).stem

    xc_string = _text(effective_spec.get("xc_code"))
    if not xc_string and embedded_xc_dict is not None:
        embedded_metadata = embedded_xc_dict.get("metadata")
        if isinstance(embedded_metadata, Mapping):
            xc_string = _text(embedded_metadata.get("native_xc_code"))
    xc_string = xc_string or functional

    metric_payload = {
        str(key): value
        for key, value in raw.items()
        if key != "__meta__" and _looks_like_subset_metric(value)
    }
    metrics = _normalize_metrics(metric_payload, warnings)
    ignored_keys = sorted(
        str(key)
        for key in raw
        if key != "__meta__" and key not in metric_payload
    )
    if ignored_keys:
        warnings.append(
            "ignored non-metric benchmark cache fields: " + ", ".join(ignored_keys)
        )

    metadata: dict[str, Any] = {
        "source_kind": "benchmark_cache",
        "cache_schema": _text(meta.get("cache_schema")),
        "declared_structural_signature": _text(meta.get("structural_signature")),
        "declared_effective_signature": _text(meta.get("effective_signature")),
        "effective_spec": effective_spec,
        "benchmark_config": benchmark_config,
    }
    recovery = meta.get("recovery")
    if isinstance(recovery, Mapping):
        metadata["recovery"] = copy.deepcopy(dict(recovery))
    for key in ("completed_at", "generated_at", "timestamp", "updated_at"):
        if meta.get(key):
            metadata[key] = meta[key]

    if embedded_xc_dict is not None:
        hf_exchange = embedded_xc_dict.get("hf_exchange")
        if isinstance(hf_exchange, Mapping):
            alpha = _float(hf_exchange.get("alpha"))
            beta = _float(hf_exchange.get("beta"))
            omega = _float(hf_exchange.get("omega"))
            metadata["hybrid_coeff"] = alpha + beta if abs(omega) > 1e-12 else alpha
            metadata["rsh_coeff"] = [omega, alpha + beta, -beta]
            metadata["is_hybrid"] = abs(alpha) > 1e-12 or abs(beta) > 1e-12
            metadata["is_rsh"] = abs(omega) > 1e-12
        embedded_metadata = embedded_xc_dict.get("metadata")
        if isinstance(embedded_metadata, Mapping):
            if _finite_number(embedded_metadata.get("hybrid_coeff")):
                metadata["hybrid_coeff"] = float(embedded_metadata["hybrid_coeff"])
            embedded_rsh = embedded_metadata.get("rsh_coeff")
            if (
                isinstance(embedded_rsh, (list, tuple))
                and len(embedded_rsh) >= 3
                and all(_finite_number(value) for value in embedded_rsh[:3])
            ):
                metadata["rsh_coeff"] = [float(value) for value in embedded_rsh[:3]]

    protocol = {
        "source_format": "benchmark_cache",
        "cache_schema": _text(meta.get("cache_schema")),
        "benchmark_config": benchmark_config,
    }
    return _NormalizedPayload(
        functional=functional,
        xc_string=xc_string,
        metadata=metadata,
        metrics=metrics,
        embedded_xc_dict=embedded_xc_dict,
        source_kind="benchmark_cache",
        benchmark_protocol=protocol,
    )


def _looks_like_subset_metric(value: Any) -> bool:
    return isinstance(value, Mapping) and any(
        key in value
        for key in (
            "mae_kcal",
            "rms_kcal",
            "max_error_kcal",
            "n_reactions",
            "n_failures",
            "walltime_s",
        )
    )


def build_functional_node(
    raw: Mapping[str, Any],
    *,
    source_file: str,
    source_kind: str | None = None,
) -> FunctionalNode:
    """Convert one mass result or benchmark cache payload into a graph node."""

    warnings: list[str] = []
    normalized = _normalize_input_payload(
        raw,
        source_file=source_file,
        source_kind=source_kind,
        warnings=warnings,
    )
    functional = normalized.functional
    metadata = normalized.metadata
    xc_string = normalized.xc_string
    metrics = normalized.metrics

    conversion = _convert_functional(
        functional,
        xc_string,
        metadata,
        embedded_xc_dict=normalized.embedded_xc_dict,
    )
    warnings.extend(conversion.warnings)
    state = _state_from_xc(
        conversion.xc,
        conversion.operability,
        backend_metadata=metadata,
    )

    if conversion.status == CONVERSION_OK and conversion.xc is not None:
        node_id = f"sig_{conversion.xc.structural_signature()}"
    else:
        identity = xc_string or functional
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        node_id = f"raw_{digest}"

    source_timestamp = _source_timestamp(raw, metadata)
    metric_sources = {subset: source_file for subset in metrics}
    build_info = {
        "has_any_metric": _has_any_metric(metrics),
        "total_failures": _total_failures(metrics),
        "missing_subsets": [subset for subset in EXPECTED_SUBSETS if subset not in metrics],
        "active_record_source": source_file,
        "metric_sources": metric_sources,
        "source_timestamps": {source_file: source_timestamp},
        "source_kinds": {source_file: normalized.source_kind},
        "benchmark_protocols": {
            source_file: copy.deepcopy(normalized.benchmark_protocol),
        },
        "metric_observations": {
            subset: [{
                "source": source_file,
                "protocol": copy.deepcopy(normalized.benchmark_protocol),
                "metrics": copy.deepcopy(payload),
            }]
            for subset, payload in metrics.items()
        },
        "metric_conflicts": [],
    }

    semantic = {
        "rung": conversion.xc.highest_rung if conversion.xc is not None else _normalize_xctype(metadata.get("xc_type")),
        "role": _functional_role(conversion.xc, metadata),
    }

    return FunctionalNode(
        id=node_id,
        name=functional,
        source_files=[source_file],
        conversion={
            "status": conversion.status,
            "method": conversion.method,
            "warnings": _stable_unique(warnings),
        },
        state=state,
        metrics=metrics,
        semantic=semantic,
        _build=build_info,
    )


def merge_functional_nodes(nodes: Sequence[FunctionalNode]) -> list[FunctionalNode]:
    """Merge nodes with the same canonical ID without discarding provenance."""

    merged: dict[str, FunctionalNode] = {}
    ordered = sorted(nodes, key=lambda node: (node.id, node.source_files[0]))
    for node in ordered:
        current = merged.get(node.id)
        if current is None:
            merged[node.id] = copy.deepcopy(node)
        else:
            _merge_node(current, node)
    return [merged[node_id] for node_id in sorted(merged)]


def build_functional_edges(nodes: Sequence[FunctionalNode]) -> EdgeBuildReport:
    """Build all conservative one-action edges between eligible nodes."""

    node_by_id = {node.id: node for node in nodes}
    xc_by_id: dict[str, XCFunctional] = {}
    for node in sorted(nodes, key=lambda item: item.id):
        xc = _eligible_xc(node)
        if xc is not None:
            xc_by_id[node.id] = xc

    rewire_buckets: dict[str, set[str]] = defaultdict(set)
    replacement_buckets: dict[str, set[str]] = defaultdict(set)
    component_structure_buckets: dict[str, set[str]] = defaultdict(set)
    one_component_removed_buckets: dict[str, set[str]] = defaultdict(set)
    for node_id, xc in xc_by_id.items():
        rewire_buckets[_rewire_bucket_key(xc)].add(node_id)
        components = sorted(xc.components, key=_component_sort_key)
        component_structure_buckets[_component_structure_key(components)].add(node_id)
        for index, component in enumerate(components):
            remaining = [
                _component_static_shape(item, include_name=True)
                for offset, item in enumerate(components)
                if offset != index
            ]
            key = _canonical_json({
                "component_count": len(components),
                "replacement_role": component.role,
                "remaining": remaining,
            })
            replacement_buckets[key].add(node_id)
            one_component_removed_buckets[
                _component_structure_key([
                    item
                    for offset, item in enumerate(components)
                    if offset != index
                ])
            ].add(node_id)

    rewire_pairs = _pairs_from_buckets(rewire_buckets)
    replacement_pairs: set[tuple[str, str]] = set()
    for pair in _pairs_from_buckets(replacement_buckets):
        left, right = pair
        if _replacement_difference(xc_by_id[left], xc_by_id[right]) is not None:
            replacement_pairs.add(pair)

    add_remove_pairs: set[tuple[str, str]] = set()
    for key in sorted(set(component_structure_buckets) & set(one_component_removed_buckets)):
        for smaller_id in sorted(component_structure_buckets[key]):
            for larger_id in sorted(one_component_removed_buckets[key]):
                if smaller_id == larger_id:
                    continue
                if _addition_difference(
                    xc_by_id[smaller_id],
                    xc_by_id[larger_id],
                ) is not None:
                    add_remove_pairs.add((smaller_id, larger_id))

    edges: dict[str, FunctionalEdge] = {}
    rejected_directions = 0
    for left, right in sorted(rewire_pairs):
        for source_id, target_id in ((left, right), (right, left)):
            edge = _build_rewire_edge(
                node_by_id[source_id],
                xc_by_id[source_id],
                node_by_id[target_id],
                xc_by_id[target_id],
            )
            if edge is None:
                rejected_directions += 1
            else:
                edges[edge.id] = edge

    for left, right in sorted(replacement_pairs):
        for source_id, target_id in ((left, right), (right, left)):
            edge = _build_replace_edge(
                node_by_id[source_id],
                xc_by_id[source_id],
                node_by_id[target_id],
                xc_by_id[target_id],
            )
            if edge is None:
                rejected_directions += 1
            else:
                edges[edge.id] = edge

    for smaller_id, larger_id in sorted(add_remove_pairs):
        add_edge = _build_add_edge(
            node_by_id[smaller_id],
            xc_by_id[smaller_id],
            node_by_id[larger_id],
            xc_by_id[larger_id],
        )
        if add_edge is None:
            rejected_directions += 1
        else:
            edges[add_edge.id] = add_edge

        remove_edge = _build_remove_edge(
            node_by_id[larger_id],
            xc_by_id[larger_id],
            node_by_id[smaller_id],
            xc_by_id[smaller_id],
        )
        if remove_edge is None:
            rejected_directions += 1
        else:
            edges[remove_edge.id] = remove_edge

    return EdgeBuildReport(
        edges=[edges[edge_id] for edge_id in sorted(edges)],
        eligible_nodes=len(xc_by_id),
        rewire_candidate_pairs=len(rewire_pairs),
        replacement_candidate_pairs=len(replacement_pairs),
        add_remove_candidate_pairs=len(add_remove_pairs),
        rejected_directions=rejected_directions,
    )


def _eligible_xc(node: FunctionalNode) -> XCFunctional | None:
    if node.conversion_status != CONVERSION_OK:
        return None
    if node.state.get("operability") != "full":
        return None
    xc_dict = node.state.get("xc_dict")
    if not isinstance(xc_dict, Mapping):
        return None
    try:
        xc = functional_from_graph_xc_dict(xc_dict)
    except (KeyError, TypeError, ValueError):
        return None
    if xc.is_custom or any(component.descriptor_spec for component in xc.components):
        return None
    if any(item.startswith("HARD:") for item in xc.validate()):
        return None
    canonical_id = f"sig_{xc.structural_signature()}"
    if node.id != canonical_id and not node.id.startswith(f"{canonical_id}__basin_"):
        return None
    return xc


def _pairs_from_buckets(buckets: Mapping[str, set[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        for left, right in combinations(sorted(members), 2):
            pairs.add((left, right))
    return pairs


def _rewire_bucket_key(xc: XCFunctional) -> str:
    return _canonical_json({
        "component_count": len(xc.components),
        "components": [
            _component_static_shape(component, include_name=True)
            for component in sorted(xc.components, key=_component_sort_key)
        ],
    })


def _component_structure_key(components: Sequence[ComponentSpec]) -> str:
    return _canonical_json([
        _component_static_shape(component, include_name=True)
        for component in sorted(components, key=_component_sort_key)
    ])


def _component_static_shape(
    component: ComponentSpec,
    *,
    include_name: bool,
) -> dict[str, Any]:
    shape = {
        "xctype": component.xctype,
        "role": component.role,
        "formula": component.formula,
        "eval_xc_code": component.eval_xc_code,
        "ext_param_keys": sorted(component.ext_params),
        "descriptor_spec": copy.deepcopy(component.descriptor_spec),
    }
    if include_name:
        shape["name"] = component.name
    return shape


def _component_sort_key(component: ComponentSpec) -> tuple[str, str, str]:
    return component.role, component.name, _canonical_json(
        _component_static_shape(component, include_name=False)
    )


def _functional_numeric_shape(xc: XCFunctional) -> str:
    return _canonical_json([
        _component_static_shape(component, include_name=True)
        for component in sorted(xc.components, key=_component_sort_key)
    ])


def _replacement_difference(
    source: XCFunctional,
    target: XCFunctional,
) -> tuple[ComponentSpec, ComponentSpec] | None:
    if len(source.components) != len(target.components):
        return None
    source_counts = Counter((component.role, component.name) for component in source.components)
    target_counts = Counter((component.role, component.name) for component in target.components)
    removed = list((source_counts - target_counts).elements())
    added = list((target_counts - source_counts).elements())
    if len(removed) != 1 or len(added) != 1 or removed[0][0] != added[0][0]:
        return None

    old_role, old_name = removed[0]
    new_role, new_name = added[0]
    old_matches = [
        component for component in source.components
        if component.role == old_role and component.name == old_name
    ]
    new_matches = [
        component for component in target.components
        if component.role == new_role and component.name == new_name
    ]
    if len(old_matches) != 1 or len(new_matches) != 1:
        return None
    if any(component.name == new_name for component in source.components):
        return None
    return old_matches[0], new_matches[0]


def _addition_difference(
    source: XCFunctional,
    target: XCFunctional,
) -> ComponentSpec | None:
    """Return the one component added by ``source -> target``, if unambiguous."""

    if len(target.components) != len(source.components) + 1:
        return None
    if len({component.name for component in source.components}) != len(source.components):
        return None
    if len({component.name for component in target.components}) != len(target.components):
        return None

    source_by_key = {
        (component.role, component.name): component
        for component in source.components
    }
    target_by_key = {
        (component.role, component.name): component
        for component in target.components
    }
    if not set(source_by_key).issubset(target_by_key):
        return None
    added_keys = set(target_by_key) - set(source_by_key)
    if len(added_keys) != 1:
        return None

    for key, source_component in source_by_key.items():
        if _component_static_shape(
            source_component,
            include_name=True,
        ) != _component_static_shape(
            target_by_key[key],
            include_name=True,
        ):
            return None
    return target_by_key[next(iter(added_keys))]


def _build_rewire_edge(
    source_node: FunctionalNode,
    source_xc: XCFunctional,
    target_node: FunctionalNode,
    target_xc: XCFunctional,
) -> FunctionalEdge | None:
    if _functional_numeric_shape(source_xc) != _functional_numeric_shape(target_xc):
        return None
    changes = _numeric_changes(source_xc, target_xc)
    if not changes:
        return None
    action = Action(
        ActionType.REWIRE,
        _agent_rewire_changes(changes),
        "Replay historical functional-graph transition.",
    )
    if not _action_replays_target(source_xc, target_xc, action):
        return None
    operation = {
        "action_type": ActionType.REWIRE.value,
        "composite": False,
        "rewire_changes": copy.deepcopy(changes),
    }
    return _make_edge(source_node, target_node, operation)


def _build_replace_edge(
    source_node: FunctionalNode,
    source_xc: XCFunctional,
    target_node: FunctionalNode,
    target_xc: XCFunctional,
) -> FunctionalEdge | None:
    difference = _replacement_difference(source_xc, target_xc)
    if difference is None:
        return None
    old_component, new_component = difference
    if new_component.name not in NAME_TO_ID:
        return None

    base_changes: dict[str, Any] = {
        "old_name": old_component.name,
        "new_name": new_component.name,
        "role": old_component.role,
    }
    try:
        base_candidate = apply_action(
            source_xc,
            Action(
                ActionType.REPLACE_COMPONENT,
                copy.deepcopy(base_changes),
                "Replay historical functional-graph replacement.",
            ),
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return None

    if _functional_numeric_shape(base_candidate) != _functional_numeric_shape(target_xc):
        return None
    numeric_changes = _numeric_changes(base_candidate, target_xc)
    if numeric_changes is None:
        return None
    executable_changes = {
        **base_changes,
        **_agent_rewire_changes(numeric_changes),
    }
    action = Action(
        ActionType.REPLACE_COMPONENT,
        executable_changes,
        "Replay historical functional-graph replacement.",
    )
    if not _action_replays_target(source_xc, target_xc, action):
        return None
    operation = {
        "action_type": ActionType.REPLACE_COMPONENT.value,
        "composite": bool(numeric_changes),
        "structural_change": copy.deepcopy(base_changes),
    }
    if numeric_changes:
        operation["rewire_changes"] = copy.deepcopy(numeric_changes)
    return _make_edge(source_node, target_node, operation)


def _build_add_edge(
    source_node: FunctionalNode,
    source_xc: XCFunctional,
    target_node: FunctionalNode,
    target_xc: XCFunctional,
) -> FunctionalEdge | None:
    added = _addition_difference(source_xc, target_xc)
    if added is None or added.name not in NAME_TO_ID:
        return None

    structural_change: dict[str, Any] = {
        "name": added.name,
        "xctype": added.xctype,
        "role": added.role,
        "mixing_factor": float(added.mixing_factor),
        "ext_params": copy.deepcopy(added.ext_params),
        "ext_param_bounds": copy.deepcopy(added.ext_param_bounds),
    }
    try:
        base_candidate = apply_action(
            source_xc,
            Action(
                ActionType.ADD_COMPONENT,
                copy.deepcopy(structural_change),
                "Replay historical functional-graph component addition.",
            ),
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return None

    if _functional_numeric_shape(base_candidate) != _functional_numeric_shape(target_xc):
        return None
    rewire_changes = _numeric_changes(base_candidate, target_xc)
    if rewire_changes is None:
        return None
    executable_changes = {
        **structural_change,
        **_agent_rewire_changes(rewire_changes),
    }
    action = Action(
        ActionType.ADD_COMPONENT,
        executable_changes,
        "Replay historical functional-graph component addition.",
    )
    if not _action_replays_target(source_xc, target_xc, action):
        return None

    operation = {
        "action_type": ActionType.ADD_COMPONENT.value,
        "composite": bool(rewire_changes),
        "structural_change": copy.deepcopy(structural_change),
    }
    if rewire_changes:
        operation["rewire_changes"] = copy.deepcopy(rewire_changes)
    return _make_edge(source_node, target_node, operation)


def _build_remove_edge(
    source_node: FunctionalNode,
    source_xc: XCFunctional,
    target_node: FunctionalNode,
    target_xc: XCFunctional,
) -> FunctionalEdge | None:
    removed = _addition_difference(target_xc, source_xc)
    if removed is None:
        return None

    executable_structural_change = {"name": removed.name}
    try:
        base_candidate = apply_action(
            source_xc,
            Action(
                ActionType.REMOVE_COMPONENT,
                copy.deepcopy(executable_structural_change),
                "Replay historical functional-graph component removal.",
            ),
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return None

    if _functional_numeric_shape(base_candidate) != _functional_numeric_shape(target_xc):
        return None
    rewire_changes = _numeric_changes(base_candidate, target_xc)
    if rewire_changes is None:
        return None
    executable_changes = {
        **executable_structural_change,
        **_agent_rewire_changes(rewire_changes),
    }
    action = Action(
        ActionType.REMOVE_COMPONENT,
        executable_changes,
        "Replay historical functional-graph component removal.",
    )
    if not _action_replays_target(source_xc, target_xc, action):
        return None

    structural_change = {
        "name": removed.name,
        "xctype": removed.xctype,
        "role": removed.role,
    }
    operation = {
        "action_type": ActionType.REMOVE_COMPONENT.value,
        "composite": bool(rewire_changes),
        "structural_change": structural_change,
    }
    if rewire_changes:
        operation["rewire_changes"] = copy.deepcopy(rewire_changes)
    return _make_edge(source_node, target_node, operation)


def _numeric_changes(
    source: XCFunctional,
    target: XCFunctional,
) -> dict[str, float] | None:
    source_keys = _action_space_keys(source)
    target_keys = _action_space_keys(target)
    if source_keys.keys() != target_keys.keys():
        return None
    if not all(_finite_number(value) for value in (*source_keys.values(), *target_keys.values())):
        return None

    changes: dict[str, float] = {}
    hf_keys = (
        "hybrid.hf_exchange.alpha",
        "hybrid.hf_exchange.beta",
        "hybrid.hf_exchange.omega",
    )
    for key in hf_keys:
        if source_keys[key] != target_keys[key]:
            changes[key] = target_keys[key]

    alpha_changed = "hybrid.hf_exchange.alpha" in changes
    for key in sorted(target_keys):
        if key in hf_keys:
            continue
        if source_keys[key] != target_keys[key] or (alpha_changed and key.startswith("mixing:")):
            changes[key] = target_keys[key]
    return changes


def _action_replays_target(
    source: XCFunctional,
    target: XCFunctional,
    action: Action,
) -> bool:
    try:
        replayed = apply_action(source, action)
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    if any(item.startswith("HARD:") for item in replayed.validate()):
        return False
    return replayed.structural_signature() == target.structural_signature()


def functional_from_graph_xc_dict(payload: Mapping[str, Any]) -> XCFunctional:
    """Restore an ``XCFunctional`` from graph state."""

    restored = copy.deepcopy(dict(payload))
    hybrid = restored.get("hybrid")
    if isinstance(hybrid, Mapping):
        hf_exchange = hybrid.get("hf_exchange")
        if not isinstance(hf_exchange, Mapping):
            raise TypeError("graph xc_dict.hybrid.hf_exchange must be an object")
        restored["hf_exchange"] = copy.deepcopy(dict(hf_exchange))
    return XCFunctional.from_dict(restored)


def _agent_rewire_changes(changes: Mapping[str, Any]) -> dict[str, Any]:
    """Translate explicit graph paths to the existing Agent action vocabulary."""

    executable: dict[str, Any] = {}
    prefix = "hybrid.hf_exchange."
    for key, value in changes.items():
        action_key = key
        if key.startswith(prefix) and key[len(prefix):] in {"alpha", "beta", "omega"}:
            action_key = key[len(prefix):]
        executable[action_key] = copy.deepcopy(value)
    return executable


def operation_to_action(
    operation: Mapping[str, Any],
    *,
    rationale: str = "Replay functional-graph transition.",
) -> Action:
    """Reconstruct the executable Agent action represented by a graph edge."""

    if operation.get("action_type") == "composite":
        raise ValueError(
            "Basin composite operations contain ordered steps and cannot be "
            "collapsed into one Action"
        )
    action_type = ActionType(str(operation["action_type"]))
    structural = operation.get("structural_change", {})
    rewire = operation.get("rewire_changes", {})
    if not isinstance(structural, Mapping) or not isinstance(rewire, Mapping):
        raise TypeError("graph operation changes must be objects")
    return Action(
        action_type,
        {
            **copy.deepcopy(dict(structural)),
            **_agent_rewire_changes(rewire),
        },
        rationale,
    )


def _make_edge(
    source: FunctionalNode,
    target: FunctionalNode,
    operation: Mapping[str, Any],
) -> FunctionalEdge:
    operation_dict = copy.deepcopy(dict(operation))
    operation_identity = {
        "source": source.id,
        "target": target.id,
        "operation": operation_dict,
    }
    operation_hash = hashlib.sha256(
        _canonical_json(operation_identity).encode("utf-8")
    ).hexdigest()[:24]
    edge_id = f"edge_{operation_hash}"
    metric_delta = _metric_delta(source, target)
    return FunctionalEdge(
        id=edge_id,
        source=source.id,
        target=target.id,
        operation=operation_dict,
        metric_delta=metric_delta,
        semantic={"llm_analysis": {"status": "pending"}},
    )


def _metric_delta(
    source: FunctionalNode,
    target: FunctionalNode,
) -> dict[str, dict[str, Any]]:
    subsets = sorted(set(EXPECTED_SUBSETS) | set(source.metrics) | set(target.metrics))
    result: dict[str, dict[str, Any]] = {}
    for subset in subsets:
        source_payload = source.metrics.get(subset, {})
        target_payload = target.metrics.get(subset, {})
        source_mae = _metric_number(source_payload.get("mae_kcal"))
        target_mae = _metric_number(target_payload.get("mae_kcal"))
        source_failures = _metric_integer(source_payload.get("n_failures"))
        target_failures = _metric_integer(target_payload.get("n_failures"))

        delta_mae: float | None = None
        if source_mae is not None and target_mae is not None:
            delta_mae = round(target_mae - source_mae, 12)

        failure_delta: int | None = None
        if source_failures is not None and target_failures is not None:
            failure_delta = target_failures - source_failures
        result[subset] = {
            "delta_mae": delta_mae,
            "failure_delta": failure_delta,
        }
    return result


def _metric_number(value: Any) -> float | None:
    if not _finite_number(value):
        return None
    return float(value)


def _metric_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return int(value)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _convert_functional(
    functional: str,
    xc_string: str,
    metadata: Mapping[str, Any],
    *,
    embedded_xc_dict: Mapping[str, Any] | None = None,
) -> _Conversion:
    embedded: _Conversion | None = None
    embedded_error = ""
    if embedded_xc_dict is not None:
        try:
            embedded_xc = XCFunctional.from_dict(copy.deepcopy(dict(embedded_xc_dict)))
            is_native = bool(embedded_xc.metadata.get("native_xc_code"))
            embedded = _validated_conversion(
                embedded_xc,
                status=CONVERSION_PARTIAL if is_native else CONVERSION_OK,
                method="benchmark_xc_dict_native" if is_native else "benchmark_xc_dict",
                operability="restricted_native" if is_native else "full",
                warnings=(
                    ["benchmark cache contains an opaque native XC state"]
                    if is_native
                    else []
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            embedded_error = f"cannot restore benchmark __meta__.xc_dict: {exc}"
        if embedded is not None and embedded.status == CONVERSION_OK:
            return embedded

    known = _try_known_functional(functional, xc_string)
    if known is not None:
        if known.metadata.get("native_xc_code"):
            known_conversion = _validated_conversion(
                known,
                status=CONVERSION_PARTIAL,
                method="known_native",
                operability="restricted_native",
                warnings=["known functional is represented by an opaque native XC expression"],
            )
        else:
            return _validated_conversion(
                known,
                status=CONVERSION_OK,
                method="known_functional",
                operability="full",
            )
    else:
        known_conversion = None

    if not xc_string:
        if embedded is not None:
            return embedded
        if known_conversion is not None:
            return known_conversion
        return _Conversion(
            xc=None,
            status=CONVERSION_PARTIAL,
            method="unresolved",
            operability="unavailable",
            warnings=_stable_unique([
                embedded_error,
                "missing xc_string and no paired component metadata",
            ]),
        )

    parse_error = ""
    if not _requires_native_representation(functional, xc_string, metadata):
        try:
            parsed = _parse_component_expression(functional, xc_string, metadata)
        except ValueError as exc:
            parse_error = f"cannot parse xc_string: {exc}"
        else:
            return _validated_conversion(
                parsed,
                status=CONVERSION_OK,
                method=(
                    "benchmark_xc_string"
                    if embedded_xc_dict is not None
                    else "mass_xc_string"
                ),
                operability="full",
            )

    if embedded is not None:
        embedded.warnings = _stable_unique(embedded.warnings + [parse_error])
        return embedded
    if known_conversion is not None:
        known_conversion.warnings = _stable_unique(known_conversion.warnings + [parse_error])
        return known_conversion

    if _requires_native_representation(functional, xc_string, metadata):
        native = _native_functional(functional, xc_string, metadata)
        return _validated_conversion(
            native,
            status=CONVERSION_PARTIAL,
            method="native_expression",
            operability="restricted_native",
            warnings=_stable_unique([
                embedded_error,
                "native, hybrid, NLC, or compound expression could not be safely decomposed",
            ]),
        )

    return _Conversion(
        xc=None,
        status=CONVERSION_PARTIAL,
        method="unresolved",
        operability="unavailable",
        warnings=_stable_unique([embedded_error, parse_error]),
    )


def _try_known_functional(functional: str, xc_string: str) -> XCFunctional | None:
    candidates: list[str] = [functional]
    upper = functional.upper()
    if upper in _KNOWN_LIBXC_ALIASES:
        candidates.append(_KNOWN_LIBXC_ALIASES[upper])
    if xc_string and all(token not in xc_string for token in (",", "+", "*")):
        candidates.append(xc_string)

    for candidate in _stable_unique(candidates):
        try:
            return XCFunctional.from_known(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _validated_conversion(
    xc: XCFunctional,
    *,
    status: str,
    method: str,
    operability: str,
    warnings: Sequence[str] = (),
) -> _Conversion:
    conversion_warnings = list(warnings)
    violations = xc.validate()
    hard_violations = [item for item in violations if item.startswith("HARD:")]
    conversion_warnings.extend(item for item in violations if item not in hard_violations)
    if hard_violations:
        conversion_warnings.extend(hard_violations)
        status = CONVERSION_PARTIAL
        operability = "invalid_state"
    return _Conversion(
        xc=xc,
        status=status,
        method=method,
        operability=operability,
        warnings=conversion_warnings,
    )


def _requires_native_representation(
    functional: str,
    xc_string: str,
    metadata: Mapping[str, Any],
) -> bool:
    role = _text(metadata.get("role")) or _infer_role(functional)
    if role == "exchange_correlation":
        return True
    if bool(metadata.get("is_hybrid")) or bool(metadata.get("is_rsh")):
        return True
    if abs(_float(metadata.get("hybrid_coeff"))) > 1e-12:
        return True
    rsh = metadata.get("rsh_coeff")
    if isinstance(rsh, (list, tuple)) and rsh and abs(_float(rsh[0])) > 1e-12:
        return True
    if xc_string.count(",") > 1:
        return True
    if bool(metadata.get("is_nlc")) and not _is_decomposable_component_pair(
        functional,
        xc_string,
        metadata,
    ):
        return True
    return False


def _is_decomposable_component_pair(
    functional: str,
    xc_string: str,
    metadata: Mapping[str, Any],
) -> bool:
    if xc_string.count(",") != 1:
        return False
    exchange_side, correlation_side = xc_string.split(",", 1)
    if not exchange_side.strip() or not correlation_side.strip():
        return False
    try:
        parsed = _parse_component_expression(functional, xc_string, metadata)
    except ValueError:
        return False
    roles = {component.role for component in parsed.components}
    return (
        "exchange" in roles
        and "correlation" in roles
        and "exchange_correlation" not in roles
    )


def _native_functional(
    functional: str,
    xc_string: str,
    metadata: Mapping[str, Any],
) -> XCFunctional:
    return XCFunctional(
        name=functional,
        components=[ComponentSpec(
            name=functional,
            xctype=_normalize_xctype(metadata.get("xc_type")),
            mixing_factor=1.0,
            role="exchange_correlation",
        )],
        hf_exchange=_hf_from_metadata(metadata),
        metadata={
            "native_xc_code": xc_string,
            "graph_operability": "restricted_native",
        },
    )


def _parse_component_expression(
    functional: str,
    xc_string: str,
    metadata: Mapping[str, Any],
) -> XCFunctional:
    if xc_string.count(",") == 1:
        exchange_side, correlation_side = xc_string.split(",", 1)
        components = _parse_terms(exchange_side, "exchange", metadata)
        components.extend(_parse_terms(correlation_side, "correlation", metadata))
    elif "," not in xc_string:
        role = _text(metadata.get("role")) or _infer_role(functional)
        if role not in {"exchange", "correlation"}:
            raise ValueError(f"cannot infer editable role for {functional!r}")
        components = _parse_terms(xc_string, role, metadata)
    else:
        raise ValueError("expected at most one exchange/correlation separator")

    if not components:
        raise ValueError("expression contains no LibXC components")
    if any(component.role == "exchange_correlation" for component in components):
        raise ValueError("compound component is not safely decomposable")

    return XCFunctional(
        name=functional,
        components=components,
        hf_exchange=HFExchange(),
    )


def _parse_terms(
    expression_side: str,
    role: str,
    metadata: Mapping[str, Any],
) -> list[ComponentSpec]:
    side = expression_side.strip()
    if not side:
        return []

    components: list[ComponentSpec] = []
    for raw_term in side.split("+"):
        term = raw_term.strip()
        if not term:
            raise ValueError(f"empty term in {expression_side!r}")
        factor = 1.0
        name = term
        match = _NUMBERED_TERM.match(term)
        if match:
            factor = float(match.group("factor"))
            name = match.group("name").strip()
        if not name or name.upper() in {"HF", "LR_HF", "SR_HF"}:
            raise ValueError(f"unsupported explicit HF term {name!r}")
        inferred_role = _infer_role(name)
        if inferred_role == "exchange_correlation":
            raise ValueError(f"compound component {name!r} cannot be assigned to one side")
        if inferred_role in {"exchange", "correlation"} and inferred_role != role:
            raise ValueError(f"component {name!r} has role {inferred_role}, expected {role}")
        components.append(_component_spec(name, role, factor, metadata))
    return components


def _component_spec(
    name: str,
    role: str,
    mixing_factor: float,
    metadata: Mapping[str, Any],
) -> ComponentSpec:
    libxc_id = NAME_TO_ID.get(name)
    param_spec = LIBXC_PARAMS.get(libxc_id, []) if libxc_id is not None else []
    ext_params = {param: default for param, default, _bounds in param_spec}
    ext_param_bounds = {param: bounds for param, _default, bounds in param_spec}
    return ComponentSpec(
        name=name,
        xctype=COMPONENT_XCTYPES.get(name, _infer_xctype(name, metadata.get("xc_type"))),
        mixing_factor=float(mixing_factor),
        role=role,
        ext_params=ext_params,
        ext_param_bounds=ext_param_bounds,
    )


def _state_from_xc(
    xc: XCFunctional | None,
    operability: str,
    *,
    backend_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if xc is None:
        return {"operability": operability}
    xc_dict = _graph_xc_dict(xc, backend_metadata)
    return {
        "xc_dict": xc_dict,
        "structural_signature": xc.structural_signature(),
        "effective_signature": xc.effective_signature(),
        "operability": operability,
    }


def _graph_xc_dict(
    xc: XCFunctional,
    backend_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    for component in xc.components:
        payload: dict[str, Any] = {
            "name": component.name,
            "xctype": component.xctype,
            "mixing_factor": float(component.mixing_factor),
            "role": component.role,
        }
        if component.ext_params:
            payload["ext_params"] = {
                key: float(value)
                for key, value in sorted(component.ext_params.items())
            }
        if component.ext_param_bounds:
            payload["ext_param_bounds"] = {
                key: [float(bounds[0]), float(bounds[1])]
                for key, bounds in sorted(component.ext_param_bounds.items())
            }
        if component.formula:
            payload["formula"] = component.formula
        if component.eval_xc_code:
            payload["eval_xc_code"] = component.eval_xc_code
        if component.descriptor_spec:
            payload["descriptor_spec"] = copy.deepcopy(component.descriptor_spec)
        components.append(payload)

    return {
        "name": xc.name,
        "components": components,
        "hybrid": {
            "backend_coefficients": _backend_coefficients(xc, backend_metadata),
            "hf_exchange": {
                "alpha": float(xc.hf_exchange.alpha),
                "beta": float(xc.hf_exchange.beta),
                "omega": float(xc.hf_exchange.omega),
            },
        },
    }


def _backend_coefficients(
    xc: XCFunctional,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    alpha = float(xc.hf_exchange.alpha)
    beta = float(xc.hf_exchange.beta)
    omega = float(xc.hf_exchange.omega)
    long_range_hf = alpha + beta

    raw_hybrid = metadata.get("hybrid_coeff")
    hybrid_coeff = (
        float(raw_hybrid)
        if _finite_number(raw_hybrid)
        else (long_range_hf if abs(omega) > 1e-12 else alpha)
    )
    raw_rsh = metadata.get("rsh_coeff")
    if (
        isinstance(raw_rsh, (list, tuple))
        and len(raw_rsh) >= 3
        and all(_finite_number(value) for value in raw_rsh[:3])
    ):
        rsh_coeff = [float(value) for value in raw_rsh[:3]]
    else:
        rsh_coeff = [omega, long_range_hf, -beta]
    return {
        "hybrid_coeff": hybrid_coeff,
        "rsh_coeff": rsh_coeff,
    }


def _action_space_keys(xc: XCFunctional) -> dict[str, float]:
    keys: dict[str, float] = {
        "hybrid.hf_exchange.alpha": float(xc.hf_exchange.alpha),
        "hybrid.hf_exchange.beta": float(xc.hf_exchange.beta),
        "hybrid.hf_exchange.omega": float(xc.hf_exchange.omega),
    }
    for component in sorted(xc.components, key=lambda item: (item.name, item.role)):
        keys[f"mixing:{component.name}"] = float(component.mixing_factor)
        for param, value in sorted(component.ext_params.items()):
            keys[f"{component.name}.{param}"] = float(value)
    return dict(sorted(keys.items()))


def _merge_node(current: FunctionalNode, incoming: FunctionalNode) -> None:
    current.source_files = sorted(set(current.source_files + incoming.source_files))
    merged_warnings = _stable_unique(
        list(current.conversion.get("warnings", []))
        + list(incoming.conversion.get("warnings", []))
    )
    current._build.setdefault("source_timestamps", {}).update(
        incoming._build.get("source_timestamps", {})
    )
    current._build.setdefault("source_kinds", {}).update(
        incoming._build.get("source_kinds", {})
    )
    current._build.setdefault("benchmark_protocols", {}).update(
        copy.deepcopy(incoming._build.get("benchmark_protocols", {}))
    )

    if _record_rank(incoming) > _record_rank(current):
        current.name = incoming.name
        current.conversion = copy.deepcopy(incoming.conversion)
        current.state = copy.deepcopy(incoming.state)
        current.semantic = copy.deepcopy(incoming.semantic)
        current._build["active_record_source"] = incoming._build.get(
            "active_record_source", ""
        )
    current.conversion["warnings"] = merged_warnings

    for subset, payload in sorted(incoming.metrics.items()):
        _merge_metric_observations(current, incoming, subset)
        incoming_source = incoming._build.get("metric_sources", {}).get(
            subset, incoming.source_files[0]
        )
        if subset not in current.metrics:
            current.metrics[subset] = copy.deepcopy(payload)
            current._build.setdefault("metric_sources", {})[subset] = incoming_source
            continue
        if _canonical_json(current.metrics[subset]) == _canonical_json(payload):
            continue

        current_source = current._build.get("metric_sources", {}).get(
            subset, current.source_files[0]
        )
        current_payload = copy.deepcopy(current.metrics[subset])
        if _metric_rank(payload, incoming_source, current) > _metric_rank(
            current.metrics[subset], current_source, current
        ):
            current.metrics[subset] = copy.deepcopy(payload)
            current._build.setdefault("metric_sources", {})[subset] = incoming_source
        _record_metric_conflict(
            current,
            subset,
            current_source,
            current_payload,
            incoming_source,
            payload,
        )

    current._build["has_any_metric"] = _has_any_metric(current.metrics)
    current._build["total_failures"] = _total_failures(current.metrics)
    current._build["missing_subsets"] = [
        subset for subset in EXPECTED_SUBSETS if subset not in current.metrics
    ]


def _merge_metric_observations(
    current: FunctionalNode,
    incoming: FunctionalNode,
    subset: str,
) -> None:
    observations = current._build.setdefault("metric_observations", {}).setdefault(subset, [])
    incoming_observations = incoming._build.get("metric_observations", {}).get(subset, [])
    known = {_canonical_json(item) for item in observations}
    for observation in incoming_observations:
        identity = _canonical_json(observation)
        if identity in known:
            continue
        observations.append(copy.deepcopy(observation))
        known.add(identity)
    observations.sort(key=lambda item: (
        _text(item.get("source")),
        _canonical_json(item.get("protocol", {})),
    ))


def _record_metric_conflict(
    node: FunctionalNode,
    subset: str,
    current_source: str,
    current_payload: Mapping[str, Any],
    incoming_source: str,
    incoming_payload: Mapping[str, Any],
) -> None:
    conflicts = node._build.setdefault("metric_conflicts", [])
    conflict = next((item for item in conflicts if item.get("subset") == subset), None)
    if conflict is None:
        conflict = {
            "subset": subset,
            "records": [{
                "source": current_source,
                "protocol": copy.deepcopy(
                    node._build.get("benchmark_protocols", {}).get(current_source, {})
                ),
                "metrics": copy.deepcopy(dict(current_payload)),
            }],
            "active_source": node._build.get("metric_sources", {}).get(subset, current_source),
        }
        conflicts.append(conflict)
    known_sources = {item.get("source") for item in conflict["records"]}
    if incoming_source not in known_sources:
        conflict["records"].append({
            "source": incoming_source,
            "protocol": copy.deepcopy(
                node._build.get("benchmark_protocols", {}).get(incoming_source, {})
            ),
            "metrics": copy.deepcopy(dict(incoming_payload)),
        })
        conflict["records"].sort(key=lambda item: item["source"])
    conflict["active_source"] = node._build.get("metric_sources", {}).get(subset, current_source)
    conflict["values_differ"] = True


def _record_rank(node: FunctionalNode) -> tuple[int, str, str]:
    completeness = sum(_metric_completeness(payload) for payload in node.metrics.values())
    source = node._build.get("active_record_source", node.source_files[0])
    timestamp = node._build.get("source_timestamps", {}).get(source, "")
    return completeness, timestamp, source


def _metric_rank(
    payload: Mapping[str, Any],
    source: str,
    node: FunctionalNode,
) -> tuple[int, str, str]:
    timestamp = node._build.get("source_timestamps", {}).get(source, "")
    return _metric_completeness(payload), timestamp, source


def _metric_completeness(payload: Mapping[str, Any]) -> int:
    return sum(value is not None for value in payload.values())


def _normalize_metrics(value: Any, warnings: list[str]) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        warnings.append("results is not an object")
        return {}
    metrics: dict[str, dict[str, Any]] = {}
    for subset, payload in sorted(value.items(), key=lambda item: str(item[0])):
        if not isinstance(payload, Mapping):
            warnings.append(f"results.{subset} is not an object")
            continue
        normalized_payload = copy.deepcopy(dict(payload))
        normalized_payload.pop("walltime_s", None)
        metrics[str(subset)] = normalized_payload
    return metrics


def _mapping_copy(value: Any, warnings: list[str], field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        warnings.append(f"{field_name} is not an object")
        return {}
    return copy.deepcopy(dict(value))


def _expression_from_metadata(functional: str, metadata: Mapping[str, Any]) -> str:
    role = _text(metadata.get("role")) or _infer_role(functional)
    if role == "exchange" and _text(metadata.get("paired_correlation")):
        return f"{functional},{_text(metadata.get('paired_correlation'))}"
    if role == "correlation" and _text(metadata.get("paired_exchange")):
        return f"{_text(metadata.get('paired_exchange'))},{functional}"
    return functional


def _hf_from_metadata(metadata: Mapping[str, Any]) -> HFExchange:
    hybrid = max(0.0, _float(metadata.get("hybrid_coeff")))
    rsh = metadata.get("rsh_coeff")
    if isinstance(rsh, (list, tuple)) and len(rsh) >= 3 and _float(rsh[0]) > 0:
        omega = max(0.0, _float(rsh[0]))
        long_range = max(0.0, _float(rsh[1]))
        short_range = max(0.0, long_range + _float(rsh[2]))
        return HFExchange(
            alpha=short_range,
            beta=max(0.0, long_range - short_range),
            omega=omega,
        )
    return HFExchange(alpha=hybrid)


def _functional_role(
    xc: XCFunctional | None,
    metadata: Mapping[str, Any],
) -> str:
    if xc is not None:
        roles = {component.role for component in xc.components}
        if "exchange_correlation" in roles or {"exchange", "correlation"}.issubset(roles):
            return "exchange_correlation"
        physical_roles = roles - {"descriptor_correction", "dispersion"}
        if len(physical_roles) == 1:
            return next(iter(physical_roles))
        if roles:
            return sorted(roles)[0]
    return _text(metadata.get("role")) or "exchange_correlation"


def _infer_role(name: str) -> str:
    upper = name.upper()
    if "_XC_" in upper:
        return "exchange_correlation"
    if "_X_" in upper or upper == "LDA_X":
        return "exchange"
    if "_C_" in upper:
        return "correlation"
    return ""


def _infer_xctype(name: str, fallback: Any = "GGA") -> str:
    upper = name.upper()
    if upper.startswith(("HYB_MGGA_", "MGGA_")):
        return "MGGA"
    if upper.startswith(("HYB_GGA_", "GGA_")):
        return "GGA"
    if upper.startswith("LDA_"):
        return "LDA"
    return _normalize_xctype(fallback)


def _normalize_xctype(value: Any) -> str:
    text = _text(value).upper()
    if "MGGA" in text:
        return "MGGA"
    if "LDA" in text:
        return "LDA"
    return "GGA"


def _source_timestamp(raw: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    for key in ("completed_at", "generated_at", "timestamp", "updated_at"):
        value = _text(raw.get(key)) or _text(metadata.get(key))
        if value:
            return value
    return ""


def _source_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _has_any_metric(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(any(value is not None for value in payload.values()) for payload in metrics.values())


def _total_failures(metrics: Mapping[str, Mapping[str, Any]]) -> int:
    total = 0
    for payload in metrics.values():
        try:
            total += int(payload.get("n_failures", 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _validate_loaded_node(node: FunctionalNode) -> None:
    allowed_state = {
        "xc_dict",
        "structural_signature",
        "effective_signature",
        "operability",
    }
    unexpected_state = set(node.state) - allowed_state
    if unexpected_state:
        raise ValueError(
            f"node {node.id} has unexpected state fields: "
            + ", ".join(sorted(unexpected_state))
        )
    xc_dict = node.state.get("xc_dict")
    if not isinstance(xc_dict, Mapping):
        raise ValueError(f"node {node.id} state.xc_dict must be an object")
    _require_exact_keys(xc_dict, {"name", "components", "hybrid"}, f"node {node.id}.xc_dict")
    components = xc_dict["components"]
    hybrid = xc_dict["hybrid"]
    if not isinstance(components, list) or not all(
        isinstance(component, Mapping) for component in components
    ):
        raise ValueError(f"node {node.id} xc_dict.components must be an object array")
    if not isinstance(hybrid, Mapping):
        raise ValueError(f"node {node.id} xc_dict.hybrid must be an object")
    _require_exact_keys(
        hybrid,
        {"backend_coefficients", "hf_exchange"},
        f"node {node.id}.xc_dict.hybrid",
    )
    backend = hybrid["backend_coefficients"]
    hf_exchange = hybrid["hf_exchange"]
    if not isinstance(backend, Mapping) or not isinstance(hf_exchange, Mapping):
        raise ValueError(f"node {node.id} hybrid entries must be objects")
    _require_exact_keys(
        backend,
        {"hybrid_coeff", "rsh_coeff"},
        f"node {node.id}.backend_coefficients",
    )
    _require_exact_keys(
        hf_exchange,
        {"alpha", "beta", "omega"},
        f"node {node.id}.hf_exchange",
    )
    if not all(_finite_number(hf_exchange.get(key)) for key in ("alpha", "beta", "omega")):
        raise ValueError(f"node {node.id} hf_exchange values must be finite numbers")
    signature = _text(node.state.get("structural_signature"))
    effective = _text(node.state.get("effective_signature"))
    if not signature or not effective:
        raise ValueError(f"node {node.id} must contain both signatures")
    expected_prefix = "sig_" if node.conversion_status == CONVERSION_OK else "raw_"
    if not node.id.startswith(expected_prefix):
        raise ValueError(
            f"node {node.id} does not match conversion status {node.conversion_status}"
        )
    _require_exact_keys(node.semantic, {"rung", "role"}, f"node {node.id}.semantic")


def _validate_loaded_edge(edge: FunctionalEdge) -> None:
    operation = edge.operation
    if operation.get("action_type") == "composite":
        _validate_basin_composite_operation(edge)
    else:
        _validate_legacy_operation(edge)

    _require_exact_keys(edge.semantic, {"llm_analysis"}, f"edge {edge.id}.semantic")
    analysis = edge.semantic["llm_analysis"]
    if not isinstance(analysis, Mapping):
        raise ValueError(f"edge {edge.id} llm_analysis must be an object")
    status = analysis.get("status")
    if status not in {"pending", "complete", "failed"}:
        raise ValueError(f"edge {edge.id} has invalid llm_analysis status")
    if status == "pending" and set(analysis) != {"status"}:
        raise ValueError(f"edge {edge.id} pending llm_analysis has placeholder fields")
    if status == "failed" and not _text(analysis.get("error")):
        raise ValueError(f"edge {edge.id} failed llm_analysis has no error")

    metric_fields = {
        "delta_mae",
        "failure_delta",
    }
    for subset, metric in edge.metric_delta.items():
        _require_exact_keys(metric, metric_fields, f"edge {edge.id}.metric_delta.{subset}")
        delta_mae = metric["delta_mae"]
        failure_delta = metric["failure_delta"]
        if delta_mae is not None and _metric_number(delta_mae) is None:
            raise ValueError(f"edge {edge.id} has invalid {subset} MAE delta")
        if failure_delta is not None and _metric_integer(failure_delta) is None:
            raise ValueError(f"edge {edge.id} has invalid {subset} failure delta")


def _validate_legacy_operation(edge: FunctionalEdge) -> None:
    operation = edge.operation
    allowed = {"action_type", "composite", "structural_change", "rewire_changes"}
    required = {"action_type", "composite"}
    missing = required - set(operation)
    unexpected = set(operation) - allowed
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected " + ", ".join(sorted(unexpected)))
        raise ValueError(f"edge {edge.id} operation is invalid: {'; '.join(details)}")
    action_type = operation["action_type"]
    if action_type not in {"rewire", "replace", "add", "remove"}:
        raise ValueError(f"edge {edge.id} has invalid action type {action_type!r}")
    if not isinstance(operation["composite"], bool):
        raise ValueError(f"edge {edge.id} operation.composite must be boolean")
    structural = operation.get("structural_change")
    rewire = operation.get("rewire_changes")
    if structural is not None and (
        not isinstance(structural, Mapping) or not structural
    ):
        raise ValueError(f"edge {edge.id} structural_change must be a non-empty object")
    if rewire is not None and (not isinstance(rewire, Mapping) or not rewire):
        raise ValueError(f"edge {edge.id} rewire_changes must be a non-empty object")
    has_rewire = isinstance(rewire, Mapping) and bool(rewire)
    if action_type == "rewire":
        if structural is not None or not has_rewire or operation["composite"]:
            raise ValueError(f"edge {edge.id} has an invalid pure rewire operation")
    else:
        if not isinstance(structural, Mapping):
            raise ValueError(f"edge {edge.id} structural action has no structural_change")
        if operation["composite"] != has_rewire:
            raise ValueError(f"edge {edge.id} composite flag disagrees with rewire_changes")


def _validate_basin_composite_operation(edge: FunctionalEdge) -> None:
    operation = edge.operation
    fields = {
        "action_type",
        "composite",
        "source",
        "basin_parent_id",
        "basin_child_id",
        "primitive_action_count",
        "steps",
    }
    _require_exact_keys(operation, fields, f"edge {edge.id}.operation")
    if operation["composite"] is not True:
        raise ValueError(f"edge {edge.id} composite operation must set composite=true")
    if operation["source"] != "basin_agent":
        raise ValueError(f"edge {edge.id} has unsupported operation source")
    for field_name in ("basin_parent_id", "basin_child_id"):
        if not _text(operation[field_name]):
            raise ValueError(f"edge {edge.id} operation.{field_name} is empty")
    steps = operation["steps"]
    count = operation["primitive_action_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"edge {edge.id} primitive_action_count must be positive")
    if not isinstance(steps, list) or len(steps) != count:
        raise ValueError(f"edge {edge.id} primitive_action_count disagrees with steps")
    allowed_actions = {"rewire", "replace", "add", "remove", "enforce"}
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            raise ValueError(f"edge {edge.id} operation step[{index}] must be an object")
        _require_exact_keys(
            step,
            {"index", "action_type", "changes"},
            f"edge {edge.id}.operation.steps[{index}]",
        )
        if step["index"] != index:
            raise ValueError(f"edge {edge.id} operation step indexes are not sequential")
        if step["action_type"] not in allowed_actions:
            raise ValueError(f"edge {edge.id} has invalid primitive action type")
        if not isinstance(step["changes"], Mapping):
            raise ValueError(f"edge {edge.id} operation step changes must be an object")


def _node_component_names(node: FunctionalNode) -> list[str]:
    xc_dict = node.state.get("xc_dict", {})
    components = xc_dict.get("components", []) if isinstance(xc_dict, Mapping) else []
    return sorted({
        _text(component.get("name"))
        for component in components
        if isinstance(component, Mapping) and _text(component.get("name"))
    })


def _node_profile(node: FunctionalNode) -> dict[str, Any]:
    xc_dict = node.state.get("xc_dict", {})
    return _profile_from_graph_xc_dict(
        xc_dict if isinstance(xc_dict, Mapping) else {},
        structural_signature=_text(node.state.get("structural_signature")),
        rung=_text(node.semantic.get("rung")),
        role=_text(node.semantic.get("role")),
    )


def _profile_from_reference(reference: Any) -> dict[str, Any] | None:
    if isinstance(reference, FunctionalNode):
        return _node_profile(reference)
    if isinstance(reference, XCFunctional):
        return {
            "structural_signature": reference.structural_signature(),
            "components": Counter(
                (component.role, component.name) for component in reference.components
            ),
            "roles": Counter(component.role for component in reference.components),
            "rung": reference.highest_rung,
            "role": _functional_role(reference, {}),
            "hf_mode": _hf_mode(
                reference.hf_exchange.alpha,
                reference.hf_exchange.beta,
                reference.hf_exchange.omega,
            ),
        }
    if not isinstance(reference, Mapping):
        return None
    state = reference.get("state", reference)
    if not isinstance(state, Mapping):
        return None
    xc_dict = state.get("xc_dict", state)
    if not isinstance(xc_dict, Mapping) or "components" not in xc_dict:
        return None
    semantic = reference.get("semantic", {})
    if not isinstance(semantic, Mapping):
        semantic = {}
    return _profile_from_graph_xc_dict(
        xc_dict,
        structural_signature=_text(state.get("structural_signature")),
        rung=_text(semantic.get("rung")),
        role=_text(semantic.get("role")),
    )


def _profile_from_graph_xc_dict(
    xc_dict: Mapping[str, Any],
    *,
    structural_signature: str,
    rung: str,
    role: str,
) -> dict[str, Any]:
    components = xc_dict.get("components", [])
    component_pairs: list[tuple[str, str]] = []
    component_roles: list[str] = []
    xctypes: list[str] = []
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, Mapping):
                continue
            name = _text(component.get("name"))
            component_role = _text(component.get("role"))
            if name:
                component_pairs.append((component_role, name))
            if component_role:
                component_roles.append(component_role)
            xctype = _text(component.get("xctype"))
            if xctype:
                xctypes.append(xctype)
    hybrid = xc_dict.get("hybrid", {})
    hf_exchange = hybrid.get("hf_exchange", {}) if isinstance(hybrid, Mapping) else {}
    if not rung and xctypes:
        rung = max(xctypes, key=lambda item: {"LDA": 0, "GGA": 1, "MGGA": 2}.get(item, 1))
    return {
        "structural_signature": structural_signature,
        "components": Counter(component_pairs),
        "roles": Counter(component_roles),
        "rung": rung,
        "role": role,
        "hf_mode": _hf_mode(
            _float(hf_exchange.get("alpha")),
            _float(hf_exchange.get("beta")),
            _float(hf_exchange.get("omega")),
        ),
    }


def _hf_mode(alpha: float, beta: float, omega: float) -> str:
    if abs(omega) > 1e-12:
        return "range_separated"
    if abs(alpha) > 1e-12 or abs(beta) > 1e-12:
        return "global_hybrid"
    return "semilocal"


def _state_match(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    reference_signature = _text(reference.get("structural_signature"))
    if reference_signature and reference_signature == candidate.get("structural_signature"):
        return 1.0
    component_match = _counter_similarity(
        reference.get("components", Counter()),
        candidate.get("components", Counter()),
    )
    rung_match = float(
        bool(reference.get("rung"))
        and reference.get("rung") == candidate.get("rung")
    )
    role_match = float(
        bool(reference.get("role"))
        and reference.get("role") == candidate.get("role")
    )
    hf_match = 1.0 if reference.get("hf_mode") == candidate.get("hf_mode") else 0.0
    return round(
        0.75 * component_match
        + 0.10 * rung_match
        + 0.10 * role_match
        + 0.05 * hf_match,
        6,
    )


def _counter_similarity(left: Any, right: Any) -> float:
    if not isinstance(left, Counter) or not isinstance(right, Counter):
        return 0.0
    keys = set(left) | set(right)
    union = sum(max(left[key], right[key]) for key in keys)
    if union == 0:
        return 0.0
    intersection = sum(min(left[key], right[key]) for key in keys)
    return intersection / union


def _query_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("query limits must be non-negative integers")
    return value


def _node_data_completeness(node: FunctionalNode) -> float:
    available = sum(
        _metric_number(node.metrics.get(subset, {}).get("mae_kcal")) is not None
        for subset in EXPECTED_SUBSETS
    )
    return available / len(EXPECTED_SUBSETS)


def _node_query_rank(node: FunctionalNode, match: float) -> tuple[Any, ...]:
    target_mae = _metric_number(
        node.metrics.get(TARGET_SUBSETS[0], {}).get("mae_kcal")
    )
    return (
        -match,
        target_mae is None,
        target_mae if target_mae is not None else math.inf,
        _total_failures(node.metrics),
        -_node_data_completeness(node),
        node.id,
    )


def _compact_node_summary(node: FunctionalNode, match: float) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for subset in EXPECTED_SUBSETS:
        payload = node.metrics.get(subset)
        if not isinstance(payload, Mapping):
            continue
        metrics[subset] = {
            "mae_kcal": _metric_number(payload.get("mae_kcal")),
            "n_failures": _metric_integer(payload.get("n_failures")),
        }
    return {
        "node_id": node.id,
        "name": node.name,
        "match_score": match,
        "structural_signature": node.state.get("structural_signature"),
        "effective_signature": node.state.get("effective_signature"),
        "operability": node.state.get("operability"),
        "components": _node_component_names(node),
        "semantic": copy.deepcopy(node.semantic),
        "metrics": metrics,
        "data_completeness": round(_node_data_completeness(node), 6),
        "source_count": len(node.source_files),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
