"""Diversity-aware mixed-variable BO basin search.

This is a non-LLM baseline for the basin-agent workflow.  It uses the same
outer shape as the LLM basin agent: propose a small batch, evaluate it, update
history, then repeat.  The proposal policy is a lightweight surrogate plus a
diversity bonus, so it can run without optional Bayesian-optimization
dependencies on the PySCF machines.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..benchmark.result import BenchmarkResult
from ..search_tasks.base import SearchTaskEvaluation
from ..xc_functional import XCFunctional
from ..zno_component_registry import SUPPORTED_COMPONENT_POOLS, component_roles_for_pool
from .actions import Action, ActionType, apply_action
from .basin_agent import (
    BasinAgentConfig,
    BasinAgentResult,
    BasinNode,
    LocalEvolutionarySearch,
    _benchmark_result_from_node_metrics,
    _deterministic_digest,
    _node_ref,
    _novelty_delta,
)
from .evolutionary import (
    EvolutionCandidate,
    EvolutionConfig,
    EvolutionParameter,
    EvolutionarySearch,
    MixedEvolutionConfig,
    MixedEvolutionarySearch,
)

logger = logging.getLogger(__name__)

_EXACT_DUPLICATE_TOLERANCE = 1e-12
_MIN_POSITIVE_RSH_OMEGA = 1e-4
_MIN_POSITIVE_RSH_BETA = 1e-6
_ACTION_FAMILIES = {
    "structure_add",
    "structure_remove",
    "structure_replace",
    "hybrid",
    "mixing",
    "scale",
    "ext_param",
}


def _passes_novelty_threshold(distance: float | None, threshold: float) -> bool:
    """Allow arbitrarily small moves while always rejecting exact duplicates."""
    if distance is None:
        return True
    return distance > _EXACT_DUPLICATE_TOLERANCE and distance >= threshold


def _normalize_action_map(
    raw: Any,
    *,
    value_type: type[float] | type[int],
    label: str,
    allow_structure_group: bool = False,
) -> dict[str, Any]:
    """Normalize YAML mappings and compact CLI ``family=value`` strings."""
    if raw in (None, ""):
        return {}
    if isinstance(raw, str):
        entries: dict[str, str] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"{label} entry must be family=value, got {item!r}")
            key, value = item.split("=", 1)
            entries[key.strip()] = value.strip()
        raw = entries
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping or comma-separated family=value string")

    allowed = set(_ACTION_FAMILIES)
    if allow_structure_group:
        allowed.add("structure")
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        family = str(key).strip()
        if family not in allowed:
            raise ValueError(f"Unsupported {label} family {family!r}; allowed={sorted(allowed)}")
        converted = value_type(value)
        if converted < 0:
            raise ValueError(f"{label}[{family!r}] must be non-negative")
        if converted > 0:
            normalized[family] = converted
    return normalized


def _action_family(operations: list[dict[str, Any]]) -> str:
    if not operations:
        return "unknown"
    operation = operations[0]
    step_type = str(operation.get("step_type") or "")
    if step_type == "structure":
        op = str(operation.get("op") or "")
        family = f"structure_{op}"
        return family if family in _ACTION_FAMILIES else "unknown"
    kind = str(operation.get("kind") or "")
    if kind in {
        "range_hybrid_split",
        "range_hybrid_activation",
        "range_hybrid_deactivation",
        "global_hf",
    }:
        return "hybrid"
    if kind == "mixing_transfer":
        return "mixing"
    if kind == "role_scale":
        return "scale"
    if kind == "ext_param":
        return "ext_param"
    return "unknown"


def _family_matches(requested: str, actual: str) -> bool:
    if requested == "structure":
        return actual.startswith("structure_")
    return requested == actual


def _weighted_family_targets(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Use largest remainders so integer quotas always sum to ``total``."""
    positive = {key: float(value) for key, value in weights.items() if float(value) > 0}
    if total <= 0 or not positive:
        return {}
    weight_sum = sum(positive.values())
    exact = {key: total * value / weight_sum for key, value in positive.items()}
    targets = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(targets.values())
    order = sorted(positive, key=lambda key: (exact[key] - targets[key], key), reverse=True)
    for key in order[:remaining]:
        targets[key] += 1
    return targets


@dataclass
class DiverseBOBasinConfig(BasinAgentConfig):
    """Configuration for the non-LLM diverse BO basin baseline."""

    candidate_pool_size: int = 512
    diversity_weight: float = 0.0
    uncertainty_weight: float = 0.25
    kernel_scale: float = 0.35
    structural_mutation_probability: float = 0.70
    max_structure_edits: int = 2
    parameter_global_sample_probability: float = 0.35
    single_step_proposals: bool = False
    enumerate_single_step_candidates: bool = False
    single_step_parameter_sigma_fraction: float = 0.12
    single_step_mixing_transfer: float = 0.10
    stratified_action_pool: bool = False
    action_pool_weights: dict[str, float] = field(default_factory=dict)
    batch_action_minimums: dict[str, int] = field(default_factory=dict)
    component_pool: str = "expanded"

    def __post_init__(self) -> None:
        raw_ea_seed_count = self.ea_seed_count
        super().__post_init__()
        # The LLM basin agent always runs a local-EA phase, but the BO baseline
        # also needs a pure-proposal mode for budget-matched ablations.
        self.ea_seed_count = max(0, int(raw_ea_seed_count))
        self.candidate_pool_size = max(16, int(self.candidate_pool_size))
        self.diversity_weight = max(0.0, float(self.diversity_weight))
        self.uncertainty_weight = max(0.0, float(self.uncertainty_weight))
        self.kernel_scale = max(1e-6, float(self.kernel_scale))
        self.structural_mutation_probability = min(
            1.0,
            max(0.0, float(self.structural_mutation_probability)),
        )
        self.max_structure_edits = max(0, int(self.max_structure_edits))
        self.parameter_global_sample_probability = min(
            1.0,
            max(0.0, float(self.parameter_global_sample_probability)),
        )
        self.single_step_proposals = bool(self.single_step_proposals)
        self.enumerate_single_step_candidates = bool(self.enumerate_single_step_candidates)
        self.single_step_parameter_sigma_fraction = max(
            1e-6,
            float(self.single_step_parameter_sigma_fraction),
        )
        self.single_step_mixing_transfer = max(1e-6, float(self.single_step_mixing_transfer))
        self.stratified_action_pool = bool(self.stratified_action_pool)
        self.action_pool_weights = _normalize_action_map(
            self.action_pool_weights,
            value_type=float,
            label="action_pool_weights",
        )
        self.batch_action_minimums = _normalize_action_map(
            self.batch_action_minimums,
            value_type=int,
            label="batch_action_minimums",
            allow_structure_group=True,
        )
        if self.stratified_action_pool:
            if not self.single_step_proposals:
                raise ValueError("stratified_action_pool requires single_step_proposals=true")
            if not self.action_pool_weights:
                raise ValueError("stratified_action_pool requires non-empty action_pool_weights")
            if sum(self.batch_action_minimums.values()) > self.proposal_batch_size:
                raise ValueError(
                    "batch_action_minimums cannot exceed proposal_batch_size: "
                    f"{sum(self.batch_action_minimums.values())} > {self.proposal_batch_size}"
                )
        if self.component_pool not in SUPPORTED_COMPONENT_POOLS:
            raise ValueError(f"Unsupported BO component_pool: {self.component_pool}")


@dataclass
class _BOPoolItem:
    parent: BasinNode
    xc: XCFunctional
    parameters: dict[str, float]
    operations: list[dict[str, Any]]
    predicted_mean: float = 0.0
    predicted_std: float = 1.0
    novelty_to_history: float = 0.0
    score: float = 0.0
    nearest_node_id: str = ""
    nearest_delta: float | None = None
    action_family: str = "unknown"


class DiverseBOBasinSearch:
    """Sequential batch diverse-BO basin search plus local EA refinement."""

    def __init__(
        self,
        *,
        config: DiverseBOBasinConfig | None = None,
        scorer_config: EvolutionConfig | None = None,
    ):
        self.config = config or DiverseBOBasinConfig()
        self.scorer = EvolutionarySearch(scorer_config or EvolutionConfig(seed=self.config.seed))
        self.random = random.Random(self.config.seed)
        self.nodes: list[BasinNode] = []
        self._nodes_by_id: dict[str, BasinNode] = {}
        self._effective_signatures: set[str] = set()
        self._next_id = 0
        self.success_digest = "No successful diverse-BO basin lessons yet."
        self.failure_digest = "No failed diverse-BO basin lessons yet."
        self._mixed_helper = MixedEvolutionarySearch(
            MixedEvolutionConfig(
                seed=self.config.seed,
                exact_mixing_sum=self.scorer.config.exact_mixing_sum,
                max_components_per_role=4,
                initial_structure_edits=max(1, self.config.max_structure_edits),
                component_pool=self.config.component_pool,
            )
        )

    def search(
        self,
        *,
        root_xc: XCFunctional,
        root_result: BenchmarkResult,
        baseline_validation_result: BenchmarkResult,
        runner,
        validation_runner,
        output_dir: Path | str,
    ) -> BasinAgentResult:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DIVERSE_BO_RESUME_CONFIG resume_nodes_path=%r resume_through_cycle=%r "
            "cycles=%d output_dir=%s",
            self.config.resume_nodes_path,
            self.config.resume_through_cycle,
            self.config.cycles,
            output_path,
        )
        if self.config.resume_nodes_path:
            resume_path = Path(self.config.resume_nodes_path)
            logger.info(
                "DIVERSE_BO_RESUME_PATH path=%s absolute=%s exists=%s",
                resume_path,
                resume_path.resolve(),
                resume_path.exists(),
            )
            loaded_cycle = self._load_resume_nodes(
                resume_path,
                through_cycle=self.config.resume_through_cycle,
            )
            start_cycle = loaded_cycle + 1
            end_cycle = loaded_cycle + self.config.cycles
            logger.info(
                "DIVERSE_BO_RESUME loaded_nodes=%d through_cycle=%d next_cycle=%d end_cycle=%d source=%s",
                len(self.nodes),
                loaded_cycle,
                start_cycle,
                end_cycle,
                self.config.resume_nodes_path,
            )
        else:
            logger.warning("DIVERSE_BO_NO_RESUME: resume_nodes_path is empty; starting from root.")
            root_candidate = EvolutionCandidate(
                xc=copy.deepcopy(root_xc),
                parameters=self._read_parameters(root_xc),
                generation=0,
                search_result=root_result,
                validation_result=baseline_validation_result,
            )
            self.scorer._score(root_candidate, root_result, baseline_validation_result)
            root_node = self._register_node(
                cycle=0,
                stage="root",
                parent_id="",
                xc=root_candidate.xc,
                candidate=root_candidate,
                rationale="bootstrap root for diverse-BO basin search",
                action_detail={"action": "root"},
            )
            root_node.candidate.xc.metadata["basin_parent_reward"] = root_node.raw_reward
            start_cycle = 1
            end_cycle = self.config.cycles

        cycle_summaries: list[dict[str, Any]] = []
        for cycle in range(start_cycle, end_cycle + 1):
            evaluated: list[BasinNode] = []
            batch_index = 0
            while (
                len(evaluated) < self.config.proposals_per_cycle
                and batch_index < self.config.max_proposal_batches
            ):
                batch_index += 1
                count = min(
                    self.config.proposal_batch_size,
                    self.config.proposals_per_cycle - len(evaluated),
                )
                logger.info(
                    "DIVERSE_BO_BATCH_START cycle=%d batch=%d evaluated_total=%d/%d count=%d",
                    cycle,
                    batch_index,
                    len(evaluated),
                    self.config.proposals_per_cycle,
                    count,
                )
                proposed = self._propose_batch(cycle=cycle, batch_index=batch_index, count=count)
                if not proposed:
                    logger.warning("Diverse BO cycle %d batch %d proposed no valid points", cycle, batch_index)
                    continue
                batch_evaluated = self._evaluate_nodes(
                    proposed,
                    root_result=root_result,
                    baseline_validation=baseline_validation_result,
                    runner=runner,
                    validation_runner=validation_runner,
                )
                evaluated.extend(batch_evaluated)
                self._write_artifacts(output_path, cycle_summaries)
                logger.info(
                    "DIVERSE_BO_BATCH_DONE cycle=%d batch=%d evaluated_batch=%d evaluated_total=%d/%d successes_total=%d",
                    cycle,
                    batch_index,
                    len(batch_evaluated),
                    len(evaluated),
                    self.config.proposals_per_cycle,
                    sum(node.is_success for node in evaluated),
                )

            selected = self._select_ea_seeds(evaluated)
            ea_nodes = self._run_local_ea(
                cycle=cycle,
                seeds=selected,
                root_result=root_result,
                baseline_validation=baseline_validation_result,
                runner=runner,
                validation_runner=validation_runner,
            )
            cycle_nodes = [*evaluated, *ea_nodes]
            self.success_digest = _deterministic_digest(
                "success",
                [node for node in cycle_nodes if node.is_success],
                previous=self.success_digest,
            )
            self.failure_digest = _deterministic_digest(
                "failure",
                [node for node in cycle_nodes if not node.is_success],
                previous=self.failure_digest,
            )
            summary = self._cycle_summary(cycle, proposed=evaluated, ea_nodes=ea_nodes, selected=selected)
            summary["proposal_batches"] = batch_index
            cycle_summaries.append(summary)
            self._write_artifacts(output_path, cycle_summaries)

        result = BasinAgentResult(
            nodes=self.nodes,
            cycle_summaries=cycle_summaries,
            success_digest=self.success_digest,
            failure_digest=self.failure_digest,
        )
        self._write_artifacts(output_path, cycle_summaries)
        return result

    def _propose_batch(self, *, cycle: int, batch_index: int, count: int) -> list[BasinNode]:
        pool = self._candidate_pool(cycle=cycle, batch_index=batch_index)
        selected_items: list[_BOPoolItem] = []
        history_xcs = [node.xc for node in self.nodes]
        if self.config.stratified_action_pool:
            for family, minimum in self.config.batch_action_minimums.items():
                for _ in range(min(int(minimum), count - len(selected_items))):
                    item = self._pop_best_pool_item(
                        pool,
                        selected_items=selected_items,
                        history_xcs=history_xcs,
                        required_family=family,
                    )
                    if item is None:
                        break
                    selected_items.append(item)

        while len(selected_items) < count and pool:
            item = self._pop_best_pool_item(
                pool,
                selected_items=selected_items,
                history_xcs=history_xcs,
            )
            if item is None:
                break
            selected_items.append(item)

        nodes: list[BasinNode] = []
        for item in selected_items:
            node = self._build_node_from_item(cycle, batch_index, item)
            nodes.append(node)
            self._effective_signatures.add(node.xc.effective_signature())
        logger.info(
            "DIVERSE_BO_PROPOSED cycle=%d batch=%d ids=%s parents=%s families=%s scores=%s",
            cycle,
            batch_index,
            [node.node_id for node in nodes],
            [node.parent_id for node in nodes],
            [item.action_family for item in selected_items],
            [round(node.action_detail.get("bo_score", 0.0), 4) for node in nodes],
        )
        return nodes

    def _pop_best_pool_item(
        self,
        pool: list[_BOPoolItem],
        *,
        selected_items: list[_BOPoolItem],
        history_xcs: list[XCFunctional],
        required_family: str | None = None,
    ) -> _BOPoolItem | None:
        while pool:
            best_index: int | None = None
            best_score = float("-inf")
            for index, item in enumerate(pool):
                if required_family and not _family_matches(required_family, item.action_family):
                    continue
                diversity_archive = [*history_xcs, *(selected.xc for selected in selected_items)]
                novelty = min((_mixed_distance(item.xc, other) for other in diversity_archive), default=1.0)
                score = (
                    item.predicted_mean
                    + self.config.uncertainty_weight * item.predicted_std
                    + self.config.diversity_weight * novelty
                )
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is None:
                return None
            item = pool.pop(best_index)
            if not _passes_novelty_threshold(item.nearest_delta, self.config.novelty_threshold):
                continue
            if any(
                not _passes_novelty_threshold(
                    _novelty_delta(item.xc, selected.xc),
                    self.config.novelty_threshold,
                )
                for selected in selected_items
            ):
                continue
            if item.xc.effective_signature() in self._effective_signatures:
                continue
            item.score = best_score
            return item
        return None

    def _candidate_pool(self, *, cycle: int, batch_index: int) -> list[_BOPoolItem]:
        if self.config.single_step_proposals and self.config.enumerate_single_step_candidates:
            return self._enumerated_single_step_candidate_pool(cycle=cycle, batch_index=batch_index)

        if self.config.stratified_action_pool:
            return self._stratified_candidate_pool(cycle=cycle, batch_index=batch_index)

        parents = self._success_parent_pool()
        if not parents:
            parents = self.nodes[:1]
        pool: list[_BOPoolItem] = []
        attempts = 0
        max_attempts = self.config.candidate_pool_size * 20
        seen: set[str] = set()
        while len(pool) < self.config.candidate_pool_size and attempts < max_attempts:
            attempts += 1
            parent = self._sample_parent(parents)
            if self.config.single_step_proposals:
                xc, parameters, operations = self._sample_single_step_candidate(parent.xc)
            else:
                xc, operations = self._sample_structure(parent.xc)
                parameters = self._sample_parameters(xc)
                parameters = self.scorer._repair_vector(parameters, self.scorer._parameter_space(xc))
            candidate_xc = self.scorer._build_candidate(xc, parameters, cycle)
            self._drop_dormant_components(candidate_xc)
            candidate_xc.name = (
                f"{parent.xc.name}__dbo_{cycle:02d}_{batch_index:02d}_{candidate_xc.effective_signature()}"
            )
            hard = [item for item in candidate_xc.validate() if item.startswith("HARD:")]
            if hard:
                continue
            sig = candidate_xc.effective_signature()
            if sig in seen or sig in self._effective_signatures:
                continue
            seen.add(sig)
            nearest_id, nearest_delta = self._nearest_existing(candidate_xc)
            mean, std = self._predict_reward(candidate_xc)
            novelty = min((_mixed_distance(candidate_xc, node.xc) for node in self.nodes), default=1.0)
            pool.append(
                _BOPoolItem(
                    parent=parent,
                    xc=candidate_xc,
                    parameters=self._read_parameters(candidate_xc),
                    operations=operations,
                    predicted_mean=mean,
                    predicted_std=std,
                    novelty_to_history=novelty,
                    score=mean + self.config.uncertainty_weight * std + self.config.diversity_weight * novelty,
                    nearest_node_id=nearest_id,
                    nearest_delta=nearest_delta,
                    action_family=_action_family(operations),
                )
            )
        return pool

    def _stratified_candidate_pool(self, *, cycle: int, batch_index: int) -> list[_BOPoolItem]:
        parents = self._success_parent_pool()
        if not parents:
            parents = self.nodes[:1]
        targets = _weighted_family_targets(self.config.candidate_pool_size, self.config.action_pool_weights)
        counts = {family: 0 for family in targets}
        pool: list[_BOPoolItem] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = self.config.candidate_pool_size * 80
        last_accept_attempt = 0
        stall_limit = self.config.candidate_pool_size * 8

        while len(pool) < self.config.candidate_pool_size and attempts < max_attempts:
            attempts += 1
            if pool and attempts - last_accept_attempt >= stall_limit:
                break
            parent = self._sample_parent(parents)
            xc, parameters, operations = self._sample_single_step_candidate(parent.xc)
            family = _action_family(operations)
            if family not in targets or counts[family] >= targets[family]:
                continue
            candidate_xc = self.scorer._build_candidate(xc, parameters, cycle)
            self._drop_dormant_components(candidate_xc)
            candidate_xc.name = (
                f"{parent.xc.name}__dbo_{cycle:02d}_{batch_index:02d}_{candidate_xc.effective_signature()}"
            )
            hard = [item for item in candidate_xc.validate() if item.startswith("HARD:")]
            if hard or not self._pool_candidate_allowed(candidate_xc):
                continue
            sig = candidate_xc.effective_signature()
            if sig in seen or sig in self._effective_signatures:
                continue
            seen.add(sig)
            nearest_id, nearest_delta = self._nearest_existing(candidate_xc)
            mean, std = self._predict_reward(candidate_xc)
            novelty = min((_mixed_distance(candidate_xc, node.xc) for node in self.nodes), default=1.0)
            pool.append(
                _BOPoolItem(
                    parent=parent,
                    xc=candidate_xc,
                    parameters=self._read_parameters(candidate_xc),
                    operations=operations,
                    predicted_mean=mean,
                    predicted_std=std,
                    novelty_to_history=novelty,
                    score=mean + self.config.uncertainty_weight * std + self.config.diversity_weight * novelty,
                    nearest_node_id=nearest_id,
                    nearest_delta=nearest_delta,
                    action_family=family,
                )
            )
            counts[family] += 1
            last_accept_attempt = attempts

        logger.info(
            "DIVERSE_BO_STRATIFIED_POOL cycle=%d batch=%d requested=%s actual=%s total=%d attempts=%d",
            cycle,
            batch_index,
            targets,
            counts,
            len(pool),
            attempts,
        )
        return pool

    def _pool_candidate_allowed(self, xc: XCFunctional) -> bool:
        """Task-aware subclasses can reject candidates before quotas are consumed."""
        return True

    @staticmethod
    def _drop_dormant_components(xc: XCFunctional) -> None:
        """Remove zero-weight terms before deduplication and parent reuse."""
        xc.components = [
            component
            for component in xc.components
            if abs(float(component.mixing_factor)) > _EXACT_DUPLICATE_TOLERANCE
        ]

    def _enumerated_single_step_candidate_pool(self, *, cycle: int, batch_index: int) -> list[_BOPoolItem]:
        parents = self._success_parent_pool()
        if not parents:
            parents = self.nodes[:1]
        pool: list[_BOPoolItem] = []
        seen: set[str] = set()
        raw_count = 0
        rejected_hard = 0
        duplicate_count = 0
        for parent in parents:
            for xc, parameters, operations in self._enumerate_single_step_candidates(parent.xc):
                raw_count += 1
                candidate_xc = self.scorer._build_candidate(xc, parameters, cycle)
                self._drop_dormant_components(candidate_xc)
                candidate_xc.name = (
                    f"{parent.xc.name}__dbo_{cycle:02d}_{batch_index:02d}_{candidate_xc.effective_signature()}"
                )
                hard = [item for item in candidate_xc.validate() if item.startswith("HARD:")]
                if hard:
                    rejected_hard += 1
                    continue
                sig = candidate_xc.effective_signature()
                if sig in seen or sig in self._effective_signatures:
                    duplicate_count += 1
                    continue
                seen.add(sig)
                nearest_id, nearest_delta = self._nearest_existing(candidate_xc)
                mean, std = self._predict_reward(candidate_xc)
                novelty = min((_mixed_distance(candidate_xc, node.xc) for node in self.nodes), default=1.0)
                pool.append(
                    _BOPoolItem(
                        parent=parent,
                        xc=candidate_xc,
                        parameters=self._read_parameters(candidate_xc),
                        operations=operations,
                        predicted_mean=mean,
                        predicted_std=std,
                        novelty_to_history=novelty,
                        score=mean + self.config.uncertainty_weight * std + self.config.diversity_weight * novelty,
                        nearest_node_id=nearest_id,
                        nearest_delta=nearest_delta,
                        action_family=_action_family(operations),
                    )
                )
        logger.info(
            "DIVERSE_BO_ENUMERATED_SINGLE_STEP_POOL cycle=%d batch=%d parents=%d raw=%d kept=%d "
            "rejected_hard=%d duplicate=%d",
            cycle,
            batch_index,
            len(parents),
            raw_count,
            len(pool),
            rejected_hard,
            duplicate_count,
        )
        return pool

    def _enumerate_single_step_candidates(
        self,
        parent_xc: XCFunctional,
    ) -> list[tuple[XCFunctional, dict[str, float], list[dict[str, Any]]]]:
        candidates: list[tuple[XCFunctional, dict[str, float], list[dict[str, Any]]]] = []
        candidates.extend(self._enumerate_single_structure_steps(parent_xc))
        candidates.extend(self._enumerate_single_parameter_steps(parent_xc))
        return candidates

    def _enumerate_single_structure_steps(
        self,
        parent_xc: XCFunctional,
    ) -> list[tuple[XCFunctional, dict[str, float], list[dict[str, Any]]]]:
        result: list[tuple[XCFunctional, dict[str, float], list[dict[str, Any]]]] = []
        component_pool = component_roles_for_pool(self.config.component_pool)
        # Match the random mutation helper: HJS-PBE is valid outside the ZnO
        # pool even though it is omitted from the proposal-facing pure list.
        if self.config.component_pool != "zno_gap":
            component_pool = dict(component_pool)
            component_pool["GGA_X_HJS_PBE"] = "exchange"

        for role in ("exchange", "correlation"):
            components = [component for component in parent_xc.components if component.role == role]
            names = {component.name for component in components}
            legal_new_names = sorted(
                name
                for name, component_role in component_pool.items()
                if component_role == role and name not in names
            )
            actions: list[Action] = []
            for component in components:
                for new_name in legal_new_names:
                    actions.append(
                        Action(
                            ActionType.REPLACE_COMPONENT,
                            changes={"old_name": component.name, "new_name": new_name},
                        )
                    )
            if len(components) < self._mixed_helper.config.max_components_per_role:
                for new_name in legal_new_names:
                    actions.append(
                        Action(
                            ActionType.ADD_COMPONENT,
                            changes={"name": new_name, "role": role},
                        )
                    )
            if len(components) > 1:
                for component in components:
                    actions.append(
                        Action(
                            ActionType.REMOVE_COMPONENT,
                            changes={"name": component.name},
                        )
                    )

            for action in actions:
                try:
                    candidate = apply_action(parent_xc, action)
                    if action.action_type == ActionType.REMOVE_COMPONENT:
                        before = next(
                            (component for component in parent_xc.components if component.name == action.changes.get("name")),
                            None,
                        )
                        removed_weight = before.mixing_factor if before is not None else 0.0
                        if removed_weight > 0:
                            recipients = [
                                component for component in candidate.components if component.role == (before.role if before else "")
                            ]
                            if recipients:
                                max(recipients, key=lambda component: component.mixing_factor).mixing_factor += removed_weight
                    if any(item.startswith("HARD:") for item in candidate.validate()):
                        continue
                except (ValueError, KeyError, TypeError):
                    continue
                result.append(
                    (
                        candidate,
                        self._read_parameters(candidate),
                        [{"step_type": "structure", "op": action.action_type.value, "changes": copy.deepcopy(action.changes)}],
                    )
                )
        return result

    def _enumerate_single_parameter_steps(
        self,
        parent_xc: XCFunctional,
    ) -> list[tuple[XCFunctional, dict[str, float], list[dict[str, Any]]]]:
        parameters = self.scorer._parameter_space(parent_xc)
        if not parameters:
            return []
        base = self.scorer._read_parameters(parent_xc, parameters)
        result: list[tuple[XCFunctional, dict[str, float], list[dict[str, Any]]]] = []

        for parameter in parameters:
            # alpha/beta/omega form one physical hybrid state.  Enumerating
            # them independently creates beta>0,omega=0 failures and
            # beta=0 omega-only no-ops around a global hybrid such as PBE0.
            if parameter.kind == "global_hf":
                continue
            if parameter.kind not in {"role_scale", "ext_param", "global_hf"}:
                continue
            effective_lower = self._effective_scalar_lower_bound(
                parameter,
                base,
            )
            for new_value in self._single_scalar_values(
                parameter,
                float(base.get(parameter.key, 0.0)),
                effective_lower=effective_lower,
            ):
                vector = dict(base)
                old_value = float(vector[parameter.key])
                if abs(new_value - old_value) <= 1e-12:
                    continue
                vector[parameter.key] = new_value
                repaired = self.scorer._repair_vector(vector, parameters)
                candidate = self.scorer._build_candidate(parent_xc, repaired, generation=0)
                result.append(
                    (
                        candidate,
                        self._read_parameters(candidate),
                        [
                            {
                                "step_type": "parameter",
                                "kind": parameter.kind,
                                "key": parameter.key,
                                "old": old_value,
                                "new": float(repaired.get(parameter.key, new_value)),
                            }
                        ],
                    )
                )

        result.extend(self._enumerate_single_hybrid_steps(parent_xc, parameters, base))

        mixing_by_role: dict[str, list[EvolutionParameter]] = {}
        for parameter in parameters:
            if parameter.kind == "mixing" and parameter.role in {"exchange", "correlation"}:
                mixing_by_role.setdefault(parameter.role or "", []).append(parameter)
        for role, role_parameters in sorted(mixing_by_role.items()):
            if len(role_parameters) < 2:
                continue
            for donor_param in role_parameters:
                for recipient_param in role_parameters:
                    if recipient_param.key == donor_param.key:
                        continue
                    transfer = self._strict_mixing_transfer(
                        parent_xc,
                        parameters,
                        base,
                        donor_param,
                        recipient_param,
                    )
                    if transfer is not None:
                        result.append(transfer)
        return result

    def _enumerate_single_hybrid_steps(
        self,
        parent_xc: XCFunctional,
        parameters: list[EvolutionParameter],
        base: dict[str, float],
    ) -> list[tuple[XCFunctional, dict[str, float], list[dict[str, Any]]]]:
        """Enumerate coherent global-hybrid and RSH parameter chains."""
        by_key = {parameter.key: parameter for parameter in parameters}
        if not {"alpha", "beta", "omega"} <= set(by_key):
            return []

        alpha_param = by_key["alpha"]
        omega_param = by_key["omega"]
        current_alpha = float(base["alpha"])
        current_beta = float(base["beta"])
        current_omega = float(base["omega"])
        total_hf = min(1.0, max(0.0, current_alpha + current_beta))
        result: list[tuple[XCFunctional, dict[str, float], list[dict[str, Any]]]] = []

        def append(vector: dict[str, float], operation: dict[str, Any]) -> None:
            repaired = self.scorer._repair_vector(vector, parameters)
            beta = float(repaired["beta"])
            omega = float(repaired["omega"])
            if beta >= _MIN_POSITIVE_RSH_BETA and omega < _MIN_POSITIVE_RSH_OMEGA:
                return
            if beta < _MIN_POSITIVE_RSH_BETA and omega > _EXACT_DUPLICATE_TOLERANCE:
                return
            candidate = self.scorer._build_candidate(parent_xc, repaired, generation=0)
            result.append((candidate, self._read_parameters(candidate), [operation]))

        if current_beta < _MIN_POSITIVE_RSH_BETA:
            # A global hybrid has one meaningful HF coordinate: alpha.
            for new_alpha in self._single_scalar_values(alpha_param, current_alpha):
                vector = dict(base)
                vector.update(alpha=new_alpha, beta=0.0, omega=0.0)
                append(
                    vector,
                    {
                        "step_type": "parameter",
                        "kind": "global_hf",
                        "key": "alpha",
                        "old": current_alpha,
                        "new": new_alpha,
                    },
                )

            # Activation is one atomic chain: split the existing global HF
            # fraction and turn on a positive range-separation parameter.
            max_alpha = min(alpha_param.upper, total_hf - _MIN_POSITIVE_RSH_BETA)
            if max_alpha >= alpha_param.lower:
                alpha_values = [
                    value
                    for value in self._single_scalar_values(alpha_param, current_alpha)
                    if alpha_param.lower <= value <= max_alpha
                ]
                omega_values = self._single_scalar_values(
                    omega_param,
                    current_omega,
                    effective_lower=_MIN_POSITIVE_RSH_OMEGA,
                )
                for new_alpha in alpha_values:
                    for new_omega in omega_values[:3]:
                        vector = dict(base)
                        vector.update(
                            alpha=new_alpha,
                            beta=total_hf - new_alpha,
                            omega=new_omega,
                        )
                        append(
                            vector,
                            {
                                "step_type": "parameter_chain",
                                "kind": "range_hybrid_activation",
                                "keys": ["alpha", "beta", "omega", "exchange_mixing_sum"],
                                "old": {
                                    "alpha": current_alpha,
                                    "beta": current_beta,
                                    "omega": current_omega,
                                },
                                "new": {
                                    "alpha": new_alpha,
                                    "beta": total_hf - new_alpha,
                                    "omega": new_omega,
                                },
                            },
                        )
            return result

        # Existing RSH: alpha/beta may be repartitioned at fixed long-range
        # HF, and omega may be adjusted independently while beta stays active.
        max_alpha = min(alpha_param.upper, total_hf - _MIN_POSITIVE_RSH_BETA)
        for new_alpha in self._single_scalar_values(alpha_param, current_alpha):
            if new_alpha > max_alpha:
                continue
            vector = dict(base)
            vector.update(alpha=new_alpha, beta=total_hf - new_alpha)
            append(
                vector,
                {
                    "step_type": "parameter_chain",
                    "kind": "range_hybrid_split",
                    "keys": ["alpha", "beta", "exchange_mixing_sum"],
                    "old": {"alpha": current_alpha, "beta": current_beta},
                    "new": {"alpha": new_alpha, "beta": total_hf - new_alpha},
                },
            )
        for new_omega in self._single_scalar_values(
            omega_param,
            current_omega,
            effective_lower=_MIN_POSITIVE_RSH_OMEGA,
        ):
            vector = dict(base)
            vector["omega"] = new_omega
            append(
                vector,
                {
                    "step_type": "parameter",
                    "kind": "global_hf",
                    "key": "omega",
                    "old": current_omega,
                    "new": new_omega,
                },
            )
        if alpha_param.lower <= total_hf <= alpha_param.upper:
            vector = dict(base)
            vector.update(alpha=total_hf, beta=0.0, omega=0.0)
            append(
                vector,
                {
                    "step_type": "parameter_chain",
                    "kind": "range_hybrid_deactivation",
                    "keys": ["alpha", "beta", "omega", "exchange_mixing_sum"],
                    "old": {
                        "alpha": current_alpha,
                        "beta": current_beta,
                        "omega": current_omega,
                    },
                    "new": {"alpha": total_hf, "beta": 0.0, "omega": 0.0},
                },
            )
        return result

    def _strict_mixing_transfer(
        self,
        parent_xc: XCFunctional,
        parameters: list[EvolutionParameter],
        base: dict[str, float],
        donor_param: EvolutionParameter,
        recipient_param: EvolutionParameter,
    ) -> tuple[XCFunctional, dict[str, float], list[dict[str, Any]]] | None:
        """Build one exact same-role transfer without projection side effects."""
        amount = float(self.config.single_step_mixing_transfer)
        if amount <= _EXACT_DUPLICATE_TOLERANCE:
            return None
        donor_value = float(base.get(donor_param.key, donor_param.lower))
        recipient_value = float(base.get(recipient_param.key, recipient_param.lower))
        if donor_value - donor_param.lower < amount - _EXACT_DUPLICATE_TOLERANCE:
            return None
        if recipient_param.upper - recipient_value < amount - _EXACT_DUPLICATE_TOLERANCE:
            return None

        expected = dict(base)
        expected[donor_param.key] = donor_value - amount
        expected[recipient_param.key] = recipient_value + amount
        repaired = self.scorer._repair_vector(expected, parameters)

        # A nominal two-component transfer must not be converted by simplex
        # projection into a three-or-more-component move.
        if any(
            abs(
                float(repaired.get(parameter.key, parameter.lower))
                - float(expected.get(parameter.key, parameter.lower))
            )
            > _EXACT_DUPLICATE_TOLERANCE
            for parameter in parameters
        ):
            return None

        candidate = self.scorer._build_candidate(parent_xc, repaired, generation=0)
        actual = self.scorer._read_parameters(candidate, parameters)
        if any(
            abs(
                float(actual.get(parameter.key, parameter.lower))
                - float(expected.get(parameter.key, parameter.lower))
            )
            > _EXACT_DUPLICATE_TOLERANCE
            for parameter in parameters
        ):
            return None
        return (
            candidate,
            actual,
            [
                {
                    "step_type": "parameter_chain",
                    "kind": "mixing_transfer",
                    "role": donor_param.role,
                    "amount": amount,
                    "donor": donor_param.key,
                    "recipient": recipient_param.key,
                }
            ],
        )

    @staticmethod
    def _effective_scalar_lower_bound(
        parameter: EvolutionParameter,
        values: dict[str, float],
    ) -> float:
        if parameter.key == "omega" and float(values.get("beta", 0.0)) > _EXACT_DUPLICATE_TOLERANCE:
            return max(parameter.lower, _MIN_POSITIVE_RSH_OMEGA)
        return parameter.lower

    def _single_scalar_values(
        self,
        parameter: EvolutionParameter,
        current: float,
        *,
        effective_lower: float | None = None,
    ) -> list[float]:
        lower = parameter.lower if effective_lower is None else max(parameter.lower, effective_lower)
        span = parameter.upper - parameter.lower
        step = self.config.single_step_parameter_sigma_fraction * span
        candidates = [
            current - 2.0 * step,
            current - step,
            current + step,
            current + 2.0 * step,
            lower,
            parameter.upper,
        ]
        values: list[float] = []
        for value in candidates:
            clipped = max(lower, parameter.clip(float(value)))
            if abs(clipped - current) <= 1e-12:
                continue
            if all(abs(clipped - old) > 1e-12 for old in values):
                values.append(clipped)
        return values

    def _sample_parent(self, parents: list[BasinNode]) -> BasinNode:
        ranked = sorted(parents, key=lambda node: node.raw_reward, reverse=True)
        weights = [1.0 / (1.0 + index) for index, _node in enumerate(ranked)]
        total = sum(weights)
        pick = self.random.random() * total
        running = 0.0
        for node, weight in zip(ranked, weights):
            running += weight
            if running >= pick:
                return node
        return ranked[-1]

    def _sample_structure(self, parent_xc: XCFunctional) -> tuple[XCFunctional, list[dict[str, Any]]]:
        candidate = copy.deepcopy(parent_xc)
        operations: list[dict[str, Any]] = []
        if self.config.max_structure_edits <= 0 or self.random.random() > self.config.structural_mutation_probability:
            return candidate, operations
        edits = self.random.randint(1, self.config.max_structure_edits)
        self._mixed_helper.random = self.random
        for _ in range(edits):
            candidate, operation = self._mixed_helper._mutate_structure(candidate)
            if operation is None:
                break
            operations.append(operation)
        return candidate, operations

    def _sample_parameters(self, xc: XCFunctional) -> dict[str, float]:
        parameters = self.scorer._parameter_space(xc)
        base = self.scorer._read_parameters(xc, parameters) if parameters else {}
        vector: dict[str, float] = {}
        global_sample = self.random.random() < self.config.parameter_global_sample_probability
        for parameter in parameters:
            if global_sample:
                vector[parameter.key] = self.random.uniform(parameter.lower, parameter.upper)
                continue
            center = float(base.get(parameter.key, (parameter.lower + parameter.upper) / 2.0))
            span = parameter.upper - parameter.lower
            sigma = 0.28 * span
            if parameter.kind == "mixing":
                sigma = 0.22 * span
            if parameter.key in {"alpha", "beta", "omega"}:
                sigma = 0.18 * span
            vector[parameter.key] = parameter.clip(center + self.random.gauss(0.0, sigma))
        return vector

    def _sample_single_step_candidate(
        self,
        parent_xc: XCFunctional,
    ) -> tuple[XCFunctional, dict[str, float], list[dict[str, Any]]]:
        """Sample exactly one BO basin move.

        A move is atomic at the optimizer level, not necessarily a single scalar:
        alpha/beta/exchange normalization is treated as one range-hybrid split
        move, and moving 0.1 mixing weight between same-role components is one
        component-mixing transfer.
        """
        structure_first = self.random.random() < self.config.structural_mutation_probability
        if structure_first:
            sampled = self._sample_single_structure_step(parent_xc)
            if sampled is not None:
                return sampled
        sampled_parameters = self._sample_single_parameter_step(parent_xc)
        if sampled_parameters is not None:
            return sampled_parameters
        sampled = self._sample_single_structure_step(parent_xc)
        if sampled is not None:
            return sampled
        parameters = self._read_parameters(parent_xc)
        return copy.deepcopy(parent_xc), parameters, [{"step_type": "identity_fallback"}]

    def _sample_single_structure_step(
        self,
        parent_xc: XCFunctional,
    ) -> tuple[XCFunctional, dict[str, float], list[dict[str, Any]]] | None:
        candidate = copy.deepcopy(parent_xc)
        self._mixed_helper.random = self.random
        candidate, operation = self._mixed_helper._mutate_structure(candidate)
        if operation is None:
            return None
        hard = [item for item in candidate.validate() if item.startswith("HARD:")]
        if hard:
            return None
        parameters = self._read_parameters(candidate)
        return candidate, parameters, [{
            "step_type": "structure",
            **operation,
        }]

    def _sample_single_parameter_step(
        self,
        parent_xc: XCFunctional,
    ) -> tuple[XCFunctional, dict[str, float], list[dict[str, Any]]] | None:
        parameters = self.scorer._parameter_space(parent_xc)
        if not parameters:
            return None
        base = self.scorer._read_parameters(parent_xc, parameters)
        parameter_by_key = {parameter.key: parameter for parameter in parameters}
        groups: list[dict[str, Any]] = []
        if {"alpha", "beta", "omega"} <= set(parameter_by_key):
            current_beta = float(base.get("beta", 0.0))
            current_omega = float(base.get("omega", 0.0))
            if current_beta < _MIN_POSITIVE_RSH_BETA:
                groups.append({"kind": "global_hf", "key": "alpha"})
                total_hf = float(base.get("alpha", 0.0)) + current_beta
                if total_hf > parameter_by_key["alpha"].lower + _MIN_POSITIVE_RSH_BETA:
                    groups.append({"kind": "range_hybrid_activation"})
            else:
                groups.append({"kind": "range_hybrid_split", "keys": ["alpha", "beta"]})
                if current_omega >= _MIN_POSITIVE_RSH_OMEGA:
                    groups.append({"kind": "global_hf", "key": "omega"})
                total_hf = float(base.get("alpha", 0.0)) + current_beta
                if parameter_by_key["alpha"].lower <= total_hf <= parameter_by_key["alpha"].upper:
                    groups.append({"kind": "range_hybrid_deactivation"})
        elif {"alpha", "beta"} <= set(parameter_by_key):
            groups.append({"kind": "range_hybrid_split", "keys": ["alpha", "beta"]})
        elif "alpha" in parameter_by_key:
            groups.append({"kind": "global_hf", "key": "alpha"})
        for parameter in parameters:
            if parameter.kind == "role_scale":
                groups.append({"kind": "scalar", "key": parameter.key})
            elif parameter.kind == "ext_param" and self.config.allow_ext_param_rewire:
                groups.append({"kind": "scalar", "key": parameter.key})

        mixing_by_role: dict[str, list[EvolutionParameter]] = {}
        for parameter in parameters:
            if parameter.kind == "mixing" and parameter.role in {"exchange", "correlation"}:
                mixing_by_role.setdefault(parameter.role or "", []).append(parameter)
        for role, role_parameters in sorted(mixing_by_role.items()):
            feasible_pairs = [
                (donor, recipient)
                for donor in role_parameters
                for recipient in role_parameters
                if donor.key != recipient.key
                and self._strict_mixing_transfer(
                    parent_xc,
                    parameters,
                    base,
                    donor,
                    recipient,
                )
                is not None
            ]
            if feasible_pairs:
                groups.append({"kind": "mixing_transfer", "role": role, "pairs": feasible_pairs})

        if not groups:
            return None

        group = self.random.choice(groups)
        vector = dict(base)
        operation: dict[str, Any]
        global_sample = self.random.random() < self.config.parameter_global_sample_probability
        if group["kind"] == "range_hybrid_activation":
            alpha_param = parameter_by_key["alpha"]
            omega_param = parameter_by_key["omega"]
            total_hf = max(0.0, min(1.0, float(base["alpha"]) + float(base["beta"])))
            max_alpha = min(alpha_param.upper, total_hf - _MIN_POSITIVE_RSH_BETA)
            if max_alpha < alpha_param.lower:
                return None
            if global_sample:
                new_alpha = self.random.uniform(alpha_param.lower, max_alpha)
                new_omega = self.random.uniform(
                    max(omega_param.lower, _MIN_POSITIVE_RSH_OMEGA),
                    omega_param.upper,
                )
            else:
                alpha_sigma = self.config.single_step_parameter_sigma_fraction * (
                    alpha_param.upper - alpha_param.lower
                )
                new_alpha = min(
                    max_alpha,
                    max(alpha_param.lower, float(base["alpha"]) + self.random.gauss(0.0, alpha_sigma)),
                )
                omega_sigma = self.config.single_step_parameter_sigma_fraction * (
                    omega_param.upper - omega_param.lower
                )
                new_omega = max(
                    _MIN_POSITIVE_RSH_OMEGA,
                    omega_param.clip(abs(self.random.gauss(max(omega_sigma, 0.1), omega_sigma))),
                )
            new_beta = total_hf - new_alpha
            vector.update(alpha=new_alpha, beta=new_beta, omega=new_omega)
            operation = {
                "step_type": "parameter_chain",
                "kind": "range_hybrid_activation",
                "keys": ["alpha", "beta", "omega", "exchange_mixing_sum"],
                "old": {"alpha": base["alpha"], "beta": base["beta"], "omega": base["omega"]},
                "new": {"alpha": new_alpha, "beta": new_beta, "omega": new_omega},
            }
        elif group["kind"] == "range_hybrid_deactivation":
            total_hf = max(0.0, min(1.0, float(base["alpha"]) + float(base["beta"])))
            vector.update(alpha=total_hf, beta=0.0, omega=0.0)
            operation = {
                "step_type": "parameter_chain",
                "kind": "range_hybrid_deactivation",
                "keys": ["alpha", "beta", "omega", "exchange_mixing_sum"],
                "old": {"alpha": base["alpha"], "beta": base["beta"], "omega": base["omega"]},
                "new": {"alpha": total_hf, "beta": 0.0, "omega": 0.0},
            }
        elif group["kind"] == "range_hybrid_split":
            alpha_param = parameter_by_key["alpha"]
            beta_param = parameter_by_key["beta"]
            current_alpha = float(base["alpha"])
            total_hf = max(0.0, min(1.0, float(base["alpha"]) + float(base["beta"])))
            max_alpha = min(alpha_param.upper, total_hf - _MIN_POSITIVE_RSH_BETA)
            if global_sample:
                new_alpha = self.random.uniform(
                    alpha_param.lower,
                    max_alpha,
                )
            else:
                sigma = self.config.single_step_parameter_sigma_fraction * (
                    alpha_param.upper - alpha_param.lower
                )
                new_alpha = alpha_param.clip(current_alpha + self.random.gauss(0.0, sigma))
                new_alpha = min(new_alpha, max_alpha)
            new_beta = beta_param.clip(max(0.0, total_hf - new_alpha))
            vector["alpha"] = new_alpha
            vector["beta"] = new_beta
            operation = {
                "step_type": "parameter_chain",
                "kind": "range_hybrid_split",
                "keys": ["alpha", "beta", "exchange_mixing_sum"],
                "old": {"alpha": base["alpha"], "beta": base["beta"]},
                "new": {"alpha": new_alpha, "beta": new_beta},
            }
        elif group["kind"] == "mixing_transfer":
            donor_param, recipient_param = self.random.choice(group["pairs"])
            return self._strict_mixing_transfer(
                parent_xc,
                parameters,
                base,
                donor_param,
                recipient_param,
            )
        else:
            key = str(group["key"])
            parameter = parameter_by_key[key]
            old_value = float(vector[key])
            effective_lower = self._effective_scalar_lower_bound(parameter, base)
            if global_sample:
                new_value = self.random.uniform(effective_lower, parameter.upper)
            else:
                sigma = self.config.single_step_parameter_sigma_fraction * (
                    parameter.upper - parameter.lower
                )
                new_value = max(
                    effective_lower,
                    parameter.clip(old_value + self.random.gauss(0.0, sigma)),
                )
            vector[key] = new_value
            operation = {
                "step_type": "parameter",
                "kind": "global_hf" if key in {"alpha", "omega"} else parameter.kind,
                "key": key,
                "old": old_value,
                "new": new_value,
            }

        repaired = self.scorer._repair_vector(vector, parameters)
        repaired_beta = float(repaired.get("beta", 0.0))
        repaired_omega = float(repaired.get("omega", 0.0))
        if repaired_beta >= _MIN_POSITIVE_RSH_BETA and repaired_omega < _MIN_POSITIVE_RSH_OMEGA:
            return None
        if repaired_beta < _MIN_POSITIVE_RSH_BETA:
            repaired["beta"] = 0.0
            repaired["omega"] = 0.0
        candidate = self.scorer._build_candidate(parent_xc, repaired, generation=0)
        return candidate, self._read_parameters(candidate), [operation]

    def _predict_reward(self, xc: XCFunctional) -> tuple[float, float]:
        observed = [node for node in self.nodes if math.isfinite(node.raw_reward)]
        if not observed:
            return 0.5, 1.0
        rewards = [node.raw_reward for node in observed]
        global_mean = sum(rewards) / len(rewards)
        global_var = sum((value - global_mean) ** 2 for value in rewards) / max(1, len(rewards))
        weights: list[tuple[float, float]] = []
        for node in observed:
            distance = _mixed_distance(xc, node.xc)
            weight = math.exp(-distance / self.config.kernel_scale)
            if weight > 1e-8:
                weights.append((weight, node.raw_reward))
        if not weights:
            return global_mean, math.sqrt(global_var + 0.25)
        total = sum(weight for weight, _reward in weights)
        mean = sum(weight * reward for weight, reward in weights) / total
        variance = sum(weight * (reward - mean) ** 2 for weight, reward in weights) / total
        nearest = min((_mixed_distance(xc, node.xc) for node in observed), default=1.0)
        std = math.sqrt(max(variance, 0.0)) + 0.35 * nearest
        return mean, std

    def _build_node_from_item(self, cycle: int, batch_index: int, item: _BOPoolItem) -> BasinNode:
        node_id = self._peek_next_node_id()
        item.xc.name = f"{item.parent.xc.name}__dbo_{cycle:02d}_{node_id}_{item.xc.effective_signature()}"
        item.xc.metadata.update(
            {
                "action": "diverse_bo_basin",
                "search_algorithm": "diverse_bo_basin",
                "basin_cycle": cycle,
                "basin_parent_id": item.parent.node_id,
                "basin_parent_reward": item.parent.raw_reward,
                "action_detail": {
                    "parent_id": item.parent.node_id,
                    "label": f"dbo_cycle{cycle}_batch{batch_index}",
                    "action_family": item.action_family,
                    "structure": copy.deepcopy(item.operations),
                    "continuous": dict(item.parameters),
                    "bo_score": item.score,
                    "predicted_mean": item.predicted_mean,
                    "predicted_std": item.predicted_std,
                    "novelty_to_history": item.novelty_to_history,
                },
            }
        )
        candidate = EvolutionCandidate(
            xc=item.xc,
            parameters=dict(item.parameters),
            generation=cycle,
        )
        node = self._register_node(
            cycle=cycle,
            stage="proposal_pending",
            parent_id=item.parent.node_id,
            xc=item.xc,
            candidate=candidate,
            rationale=(
                "diverse BO proposal: "
                f"mean={item.predicted_mean:.4f}, std={item.predicted_std:.4f}, "
                f"novelty={item.novelty_to_history:.4f}, score={item.score:.4f}"
            ),
            action_detail={
                "parent_id": item.parent.node_id,
                "label": f"dbo_cycle{cycle}_batch{batch_index}",
                "action_family": item.action_family,
                "structure": copy.deepcopy(item.operations),
                "continuous": dict(item.parameters),
                "bo_score": item.score,
                "predicted_mean": item.predicted_mean,
                "predicted_std": item.predicted_std,
                "novelty_to_history": item.novelty_to_history,
            },
            register_signature=False,
        )
        node.nearest_node_id = item.nearest_node_id
        node.nearest_delta = item.nearest_delta
        return node

    def _evaluate_nodes(
        self,
        nodes: list[BasinNode],
        *,
        root_result: BenchmarkResult,
        baseline_validation: BenchmarkResult,
        runner,
        validation_runner,
    ) -> list[BasinNode]:
        # Import lazily to keep this module standalone from LLM dependencies.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        def _run(node: BasinNode) -> BasinNode:
            node.eval_started_at = time.time()
            logger.info(
                "DIVERSE_BO_EVAL_START node=%s stage=proposal parent=%s name=%s",
                node.node_id,
                node.parent_id,
                node.xc.name,
            )
            node.candidate.search_result = runner.evaluate(node.xc)
            node.candidate.validation_result = validation_runner.evaluate(node.xc)
            self.scorer._score(node.candidate, root_result, baseline_validation)
            node.eval_finished_at = time.time()
            node.eval_duration_s = node.eval_finished_at - node.eval_started_at
            node.stage = "proposal"
            logger.info(
                "DIVERSE_BO_EVAL_END node=%s parent=%s raw_reward=%.6f success_vs_parent=%s duration_s=%.1f",
                node.node_id,
                node.parent_id,
                node.raw_reward,
                node.is_success,
                node.eval_duration_s,
            )
            return node

        if self.config.proposal_evaluation_concurrency <= 1 or len(nodes) <= 1:
            return [_run(node) for node in nodes]
        completed: list[BasinNode] = []
        with ThreadPoolExecutor(max_workers=min(self.config.proposal_evaluation_concurrency, len(nodes))) as executor:
            futures = [executor.submit(_run, node) for node in nodes]
            for future in as_completed(futures):
                completed.append(future.result())
        return completed

    def _select_ea_seeds(self, candidates: list[BasinNode]) -> list[BasinNode]:
        if self.config.ea_seed_count <= 0:
            logger.info("DIVERSE_BO_EA_SEED_SELECTED ids=[] disabled=true")
            return []
        eligible = [
            node for node in candidates
            if node.candidate.promotable and math.isfinite(node.raw_reward)
        ]
        ranked = sorted(eligible, key=lambda node: node.raw_reward, reverse=True)
        selected: list[BasinNode] = []
        for node in ranked:
            if all(_mixed_distance(node.xc, other.xc) >= self.config.novelty_threshold for other in selected):
                selected.append(node)
            if len(selected) >= self.config.ea_seed_count:
                break
        for node in ranked:
            if len(selected) >= self.config.ea_seed_count:
                break
            if node not in selected:
                selected.append(node)
        logger.info("DIVERSE_BO_EA_SEED_SELECTED ids=%s", [_node_ref(node) for node in selected])
        return selected

    def _run_local_ea(
        self,
        *,
        cycle: int,
        seeds: list[BasinNode],
        root_result: BenchmarkResult,
        baseline_validation: BenchmarkResult,
        runner,
        validation_runner,
    ) -> list[BasinNode]:
        collected: list[BasinNode] = []
        for seed_index, seed in enumerate(seeds, start=1):
            config = EvolutionConfig(
                generations=self.config.ea_generations,
                population_size=self.config.ea_population_size,
                elite_size=self.config.ea_elite_size,
                mutation_rate=self.scorer.config.mutation_rate,
                mutation_scale=self.scorer.config.mutation_scale,
                exact_mixing_sum=self.scorer.config.exact_mixing_sum,
                evaluation_concurrency=self.scorer.config.evaluation_concurrency,
                seed=self.config.seed + cycle * 100 + seed_index,
            )
            local = LocalEvolutionarySearch(
                config,
                seed_xc=seed.xc,
                seed_parameters=seed.candidate.parameters,
                basin_config=self.config,
                label=f"dbo_cycle{cycle}:seed{seed.node_id}",
            )
            try:
                result = local.search(
                    root_xc=seed.xc,
                    root_result=root_result,
                    validation_runner=validation_runner,
                    runner=runner,
                    baseline_validation_result=baseline_validation,
                    initial_population_vectors=[seed.candidate.parameters],
                )
            except Exception as exc:
                logger.warning("DIVERSE_BO_LOCAL_EA_FAILED cycle=%d seed=%s: %s", cycle, seed.node_id, exc)
                continue
            for candidate in result.evaluated[1:]:
                candidate.xc.metadata["basin_parent_id"] = seed.node_id
                candidate.xc.metadata["basin_parent_reward"] = seed.raw_reward
                node = self._register_node(
                    cycle=cycle,
                    stage="local_ea",
                    parent_id=seed.node_id,
                    xc=candidate.xc,
                    candidate=candidate,
                    rationale=f"local EA refinement from {seed.node_id}",
                    action_detail=dict(candidate.parameters),
                )
                node.eval_duration_s = candidate.xc.metadata.get("basin_eval_duration_s")
                collected.append(node)
        return collected

    def _success_parent_pool(self) -> list[BasinNode]:
        successes = [node for node in self.nodes if node.is_success]
        ranked = sorted(
            successes,
            key=lambda item: (item.raw_reward, -item.candidate.worst_validation_ratio),
            reverse=True,
        )
        selected: list[BasinNode] = []
        for node in ranked:
            if all(_mixed_distance(node.xc, other.xc) >= self.config.novelty_threshold for other in selected):
                selected.append(node)
            if len(selected) >= self.config.parent_pool_size:
                break
        for node in ranked:
            if len(selected) >= self.config.parent_pool_size:
                break
            if node not in selected:
                selected.append(node)
        return selected

    def _nearest_existing(self, xc: XCFunctional) -> tuple[str, float | None]:
        if not self.nodes:
            return "", None
        best_id = ""
        best_delta = float("inf")
        for node in self.nodes:
            delta = _novelty_delta(xc, node.xc)
            if delta < best_delta:
                best_delta = delta
                best_id = node.node_id
        return best_id, best_delta

    def _cycle_summary(
        self,
        cycle: int,
        *,
        proposed: list[BasinNode],
        ea_nodes: list[BasinNode],
        selected: list[BasinNode],
    ) -> dict[str, Any]:
        cycle_nodes = [*proposed, *ea_nodes]
        best = max(cycle_nodes, key=lambda item: EvolutionarySearch._rank_key(item.candidate)) if cycle_nodes else None
        return {
            "cycle": cycle,
            "proposals_evaluated": len(proposed),
            "ea_nodes_evaluated": len(ea_nodes),
            "ea_seed_node_ids": [node.node_id for node in selected],
            "successful_nodes": sum(node.is_success for node in cycle_nodes),
            "best_node_id": best.node_id if best else "",
            "best_raw_reward": best.raw_reward if best else None,
            "best_name": best.xc.name if best else "",
        }

    def _register_node(
        self,
        *,
        cycle: int,
        stage: str,
        parent_id: str,
        xc: XCFunctional,
        candidate: EvolutionCandidate,
        rationale: str,
        action_detail: dict[str, Any],
        register_signature: bool = True,
    ) -> BasinNode:
        node_id = self._allocate_node_id()
        node = BasinNode(
            node_id=node_id,
            cycle=cycle,
            stage=stage,
            parent_id=parent_id,
            xc=xc,
            candidate=candidate,
            rationale=rationale,
            action_detail=action_detail,
        )
        self.nodes.append(node)
        self._nodes_by_id[node_id] = node
        if register_signature:
            self._effective_signatures.add(xc.effective_signature())
        return node

    def _allocate_node_id(self) -> str:
        node_id = f"N{self._next_id:05d}"
        self._next_id += 1
        return node_id

    def _peek_next_node_id(self) -> str:
        return f"N{self._next_id:05d}"

    def _read_parameters(self, xc: XCFunctional) -> dict[str, float]:
        parameters = self.scorer._parameter_space(xc)
        return self.scorer._read_parameters(xc, parameters) if parameters else {}

    def _write_artifacts(self, output_dir: Path, cycle_summaries: list[dict[str, Any]]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "diverse_bo_basin_nodes.json").write_text(
            json.dumps([node.to_dict() for node in self.nodes], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "diverse_bo_basin_summary.json").write_text(
            json.dumps(
                {
                    "cycle_summaries": cycle_summaries,
                    "success_digest": self.success_digest,
                    "failure_digest": self.failure_digest,
                    "best": self.nodes and max(
                        self.nodes,
                        key=lambda item: EvolutionarySearch._rank_key(item.candidate),
                    ).to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._write_csv(output_dir / "diverse_bo_basin_nodes.csv")

    def _write_csv(self, path: Path) -> None:
        fields = [
            "node_id", "cycle", "stage", "parent_id", "name", "raw_reward",
            "tree_reward", "promotable", "success_vs_parent", "failure_reason",
            "guard_penalty", "worst_validation_ratio", "eval_duration_s",
            "nearest_node_id", "nearest_delta", "action_detail", "parameters",
            "search_mae", "validation_mae", "search_failures", "validation_failures",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            import csv

            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for node in self.nodes:
                writer.writerow({
                    "node_id": node.node_id,
                    "cycle": node.cycle,
                    "stage": node.stage,
                    "parent_id": node.parent_id,
                    "name": node.xc.name,
                    "raw_reward": node.raw_reward,
                    "tree_reward": node.candidate.tree_reward,
                    "promotable": node.candidate.promotable,
                    "success_vs_parent": node.is_success,
                    "failure_reason": node.candidate.failure_reason,
                    "guard_penalty": node.candidate.guard_penalty,
                    "worst_validation_ratio": node.candidate.worst_validation_ratio,
                    "eval_duration_s": node.eval_duration_s,
                    "nearest_node_id": node.nearest_node_id,
                    "nearest_delta": node.nearest_delta,
                    "action_detail": json.dumps(node.action_detail, sort_keys=True),
                    "parameters": json.dumps(node.candidate.parameters, sort_keys=True),
                    "search_mae": node.candidate.search_result.overall_mae if node.candidate.search_result else None,
                    "validation_mae": (
                        node.candidate.validation_result.overall_mae
                        if node.candidate.validation_result else None
                    ),
                    "search_failures": (
                        node.candidate.search_result.total_failures
                        if node.candidate.search_result else None
                    ),
                    "validation_failures": (
                        node.candidate.validation_result.total_failures
                        if node.candidate.validation_result else None
                    ),
                })

    def _load_resume_nodes(self, path: Path, *, through_cycle: int | None) -> int:
        if not path.exists():
            raise FileNotFoundError(f"Diverse-BO resume nodes file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records = payload.get("nodes") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"Diverse-BO resume nodes file must contain a list of nodes: {path}")

        self.nodes.clear()
        self._nodes_by_id.clear()
        self._effective_signatures.clear()
        max_cycle = 0
        max_node_number = -1
        for record in records:
            cycle = int(record.get("cycle", 0) or 0)
            if through_cycle is not None and cycle > through_cycle:
                continue
            xc_payload = record.get("xc_dict")
            if not isinstance(xc_payload, dict):
                continue
            xc = XCFunctional.from_dict(xc_payload)
            parameters = {
                str(key): float(value)
                for key, value in (record.get("parameters") or {}).items()
                if value is not None
            }
            if not parameters:
                parameters = self._read_parameters(xc)
            candidate = EvolutionCandidate(
                xc=xc,
                parameters=parameters,
                generation=cycle,
                search_result=_benchmark_result_from_node_metrics(xc.name, record.get("search") or {}),
                validation_result=_benchmark_result_from_node_metrics(xc.name, record.get("guard") or {}),
                fitness=float(record.get("raw_reward", float("-inf"))),
                promotable=bool(record.get("promotable", False)),
                failure_reason=str(record.get("failure_reason") or ""),
                worst_validation_ratio=float(record.get("worst_validation_ratio", float("inf"))),
                guard_penalty=float(record.get("guard_penalty", 0.0)),
                tree_reward=float(record.get("tree_reward", 0.0)),
            )
            node_id = str(record.get("node_id") or f"N{max_node_number + 1:05d}")
            node = BasinNode(
                node_id=node_id,
                cycle=cycle,
                stage=str(record.get("stage") or "resume"),
                parent_id=str(record.get("parent_id") or ""),
                xc=xc,
                candidate=candidate,
                rationale=str(record.get("rationale") or ""),
                action_detail=dict(record.get("action_detail") or {}),
                eval_duration_s=record.get("eval_duration_s"),
                nearest_node_id=str(record.get("nearest_node_id") or ""),
                nearest_delta=record.get("nearest_delta"),
                task_evaluation=SearchTaskEvaluation.from_dict(
                    record.get("task_evaluation")
                ),
            )
            self.nodes.append(node)
            self._nodes_by_id[node.node_id] = node
            self._effective_signatures.add(node.xc.effective_signature())
            max_cycle = max(max_cycle, cycle)
            if node_id.startswith("N") and node_id[1:].isdigit():
                max_node_number = max(max_node_number, int(node_id[1:]))

        for node in self.nodes:
            if node.parent_id and node.parent_id in self._nodes_by_id:
                node.candidate.xc.metadata["basin_parent_reward"] = self._nodes_by_id[node.parent_id].raw_reward
            elif node.stage == "root":
                node.candidate.xc.metadata["basin_parent_reward"] = node.raw_reward
        self._next_id = max_node_number + 1
        successes = [node for node in self.nodes if node.is_success]
        failures = [node for node in self.nodes if not node.is_success]
        self.success_digest = _deterministic_digest(
            "success",
            successes,
            previous="Loaded diverse-BO resume history.",
        )
        self.failure_digest = _deterministic_digest(
            "failure",
            failures,
            previous="Loaded diverse-BO resume history.",
        )
        return through_cycle if through_cycle is not None else max_cycle


def _mixed_distance(left: XCFunctional, right: XCFunctional) -> float:
    left_components = {(component.role, component.name) for component in left.components}
    right_components = {(component.role, component.name) for component in right.components}
    union = left_components | right_components
    if union:
        component_distance = 1.0 - (len(left_components & right_components) / len(union))
    else:
        component_distance = 0.0
    left_values = _parameter_map(left)
    right_values = _parameter_map(right)
    keys = set(left_values) | set(right_values)
    if not keys:
        parameter_distance = 0.0
    else:
        parameter_distance = sum(
            min(1.0, abs(left_values.get(key, 0.0) - right_values.get(key, 0.0)) / _distance_scale(key))
            for key in keys
        ) / len(keys)
    return 0.55 * component_distance + 0.45 * parameter_distance


def _parameter_map(xc: XCFunctional) -> dict[str, float]:
    values = {
        "alpha": float(xc.hf_exchange.alpha),
        "beta": float(xc.hf_exchange.beta),
        "omega": float(xc.hf_exchange.omega),
        "scale:exchange": float(xc.role_scales.get("exchange", 1.0)),
        "scale:correlation": float(xc.role_scales.get("correlation", 1.0)),
    }
    for component in xc.components:
        values[f"mixing:{component.name}"] = float(component.mixing_factor)
        for key, value in component.ext_params.items():
            values[f"{component.name}.{key}"] = float(value)
    return values


def _distance_scale(key: str) -> float:
    if key == "alpha":
        return 0.50
    if key == "beta":
        return 1.0
    if key == "omega":
        return 0.50
    if key.startswith("scale:"):
        return 0.20
    if key.startswith("mixing:"):
        return 1.0
    if key.endswith("._d") or key.endswith(".d"):
        return 1.0
    return 0.10
