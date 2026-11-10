"""LLM-guided basin proposal followed by small local EA refinement."""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..benchmark.result import BenchmarkResult, SubsetResult
from ..logging_utils import log_block, pretty_json_or_text
from ..search_tasks import BenchmarkSearchTask, SearchTask, SearchTaskEvaluation
from ..xc_functional import XCFunctional
from ..zno_component_registry import (
    SUPPORTED_COMPONENT_POOLS,
    component_roles_for_pool,
    format_zno_gap_prompt_registry,
    validate_component_for_pool,
    validate_functional_for_pool,
)
from .actions import ActionType, apply_action
from .agent import IdeaAgent, SYSTEM_PROMPT
from .evolutionary import (
    EvolutionCandidate,
    EvolutionConfig,
    EvolutionParameter,
    EvolutionarySearch,
    EvolutionSearchResult,
)
from .knowledge_engine import KnowledgeEngine
from .warm_start import _extract_json_object, _format_component_catalogue, _format_result, _fmt

logger = logging.getLogger(__name__)


@dataclass
class BasinAgentConfig:
    """Configuration for the LLM basin-search loop."""

    cycles: int = 2
    proposals_per_cycle: int = 32
    parent_pool_size: int = 20
    parent_pool_max_per_seed: int = 4
    resume_nodes_path: str = ""
    resume_through_cycle: int | None = None
    proposal_batch_size: int = 4
    max_proposal_batches: int = 20
    require_full_proposal_budget: bool = False
    novelty_threshold: float = 0.10
    proposal_evaluation_concurrency: int = 4
    ea_seed_count: int = 4
    ea_population_size: int = 8
    ea_generations: int = 4
    ea_elite_size: int = 2
    local_global_radius: float = 0.10
    local_mixing_radius: float = 0.15
    local_role_scale_radius: float = 0.10
    local_ext_relative_radius: float = 0.30
    local_ext_abs_min: float = 0.01
    local_d_radius: float = 0.20
    seed: int = 42
    component_pool: str = "expanded"
    prompt_history_per_ea_group: int = 4
    digest_max_lines: int = 48
    knowledge_component_prior_limit: int = 32
    update_functional_graph: bool = False
    core_output_dir: str = ""
    enable_local_ea: bool = True
    allow_hf_exchange_rewire: bool = True
    allow_ext_param_rewire: bool = True
    use_benchmark_component_priors: bool = True
    use_component_design_priors: bool = True
    anonymize_component_names: bool = False
    enforce_novelty_threshold: bool = True
    freeform_llm_control_prompt: bool = False

    def __post_init__(self) -> None:
        self.cycles = max(1, int(self.cycles))
        self.proposals_per_cycle = max(1, int(self.proposals_per_cycle))
        self.parent_pool_size = max(1, int(self.parent_pool_size))
        self.parent_pool_max_per_seed = max(1, int(self.parent_pool_max_per_seed))
        self.resume_nodes_path = str(self.resume_nodes_path or "")
        if self.resume_through_cycle is not None:
            self.resume_through_cycle = max(0, int(self.resume_through_cycle))
        self.proposal_batch_size = max(1, int(self.proposal_batch_size))
        self.max_proposal_batches = max(1, int(self.max_proposal_batches))
        self.require_full_proposal_budget = bool(self.require_full_proposal_budget)
        self.novelty_threshold = max(0.0, float(self.novelty_threshold))
        self.proposal_evaluation_concurrency = max(1, int(self.proposal_evaluation_concurrency))
        self.ea_seed_count = max(1, int(self.ea_seed_count))
        self.ea_population_size = max(2, int(self.ea_population_size))
        self.ea_generations = max(1, int(self.ea_generations))
        self.ea_elite_size = max(1, min(int(self.ea_elite_size), self.ea_population_size))
        self.local_role_scale_radius = max(0.0, float(self.local_role_scale_radius))
        if self.component_pool not in SUPPORTED_COMPONENT_POOLS:
            raise ValueError(f"Unsupported basin component_pool: {self.component_pool}")
        self.prompt_history_per_ea_group = max(1, int(self.prompt_history_per_ea_group))
        self.digest_max_lines = max(8, int(self.digest_max_lines))
        self.knowledge_component_prior_limit = max(1, int(self.knowledge_component_prior_limit))
        self.update_functional_graph = bool(self.update_functional_graph)
        self.core_output_dir = str(self.core_output_dir or "")
        self.enable_local_ea = bool(self.enable_local_ea)
        self.allow_hf_exchange_rewire = bool(self.allow_hf_exchange_rewire)
        self.allow_ext_param_rewire = bool(self.allow_ext_param_rewire)
        self.use_benchmark_component_priors = bool(self.use_benchmark_component_priors)
        self.use_component_design_priors = bool(self.use_component_design_priors)
        self.anonymize_component_names = bool(self.anonymize_component_names)
        self.enforce_novelty_threshold = bool(self.enforce_novelty_threshold)
        self.freeform_llm_control_prompt = bool(self.freeform_llm_control_prompt)


@dataclass
class BasinNode:
    """One evaluated point in the basin-agent search."""

    node_id: str
    cycle: int
    stage: str
    parent_id: str
    xc: XCFunctional
    candidate: EvolutionCandidate
    task_evaluation: SearchTaskEvaluation | None = None
    rationale: str = ""
    action_detail: dict[str, Any] = field(default_factory=dict)
    prompt_label: str = ""
    eval_started_at: float | None = None
    eval_finished_at: float | None = None
    eval_duration_s: float | None = None
    nearest_node_id: str = ""
    nearest_delta: float | None = None

    @property
    def raw_reward(self) -> float:
        return self.candidate.fitness

    @property
    def is_success(self) -> bool:
        if self.parent_id == "":
            return bool(self.candidate.promotable and math.isfinite(self.candidate.fitness))
        if not self.candidate.promotable or not math.isfinite(self.candidate.fitness):
            return False
        parent_reward = self.candidate.xc.metadata.get("basin_parent_reward")
        if parent_reward is None:
            return False
        return self.candidate.fitness > float(parent_reward) + 1e-12

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cycle": self.cycle,
            "stage": self.stage,
            "parent_id": self.parent_id,
            "name": self.xc.name,
            "signature": self.xc.structural_signature(),
            "effective_signature": self.xc.effective_signature(),
            "raw_reward": self.candidate.fitness,
            "tree_reward": self.candidate.tree_reward,
            "promotable": self.candidate.promotable,
            "success_vs_parent": self.is_success,
            "failure_reason": self.candidate.failure_reason,
            "guard_penalty": self.candidate.guard_penalty,
            "worst_validation_ratio": self.candidate.worst_validation_ratio,
            "parameters": dict(self.candidate.parameters),
            "action_detail": self.action_detail,
            "rationale": self.rationale,
            "eval_duration_s": self.eval_duration_s,
            "nearest_node_id": self.nearest_node_id,
            "nearest_delta": self.nearest_delta,
            "task_evaluation": self.task_evaluation.to_dict() if self.task_evaluation else None,
            "search": _result_metrics(self.candidate.search_result),
            "guard": _result_metrics(self.candidate.validation_result),
            "xc_dict": self.xc.to_dict(),
        }


@dataclass
class BasinAgentResult:
    nodes: list[BasinNode]
    cycle_summaries: list[dict[str, Any]]
    success_digest: str
    failure_digest: str

    @property
    def best(self) -> EvolutionCandidate:
        return max((node.candidate for node in self.nodes), key=EvolutionarySearch._rank_key)

    @property
    def summary(self) -> dict[str, Any]:
        finite = [node.raw_reward for node in self.nodes if math.isfinite(node.raw_reward)]
        best_node = max(self.nodes, key=lambda item: EvolutionarySearch._rank_key(item.candidate))
        return {
            "cycles": len(self.cycle_summaries),
            "evaluated_nodes": len(self.nodes),
            "successful_nodes": sum(node.is_success for node in self.nodes),
            "best": self.best.to_dict(),
            "best_task_evaluation": (
                best_node.task_evaluation.to_dict() if best_node.task_evaluation else None
            ),
            "best_raw_reward": max(finite) if finite else float("-inf"),
            "success_digest": self.success_digest,
            "failure_digest": self.failure_digest,
            "cycle_summaries": self.cycle_summaries,
        }


class LocalEvolutionarySearch(EvolutionarySearch):
    """EA with parameter bounds clipped around one seed point."""

    def __init__(
        self,
        config: EvolutionConfig,
        *,
        seed_xc: XCFunctional,
        seed_parameters: dict[str, float],
        basin_config: BasinAgentConfig,
        label: str,
    ):
        super().__init__(config)
        self.seed_xc = seed_xc
        self.seed_parameters = seed_parameters
        self.basin_config = basin_config
        self.label = label
        self._forced_parameters: list[EvolutionParameter] | None = None

    def _parameter_space(self, xc: XCFunctional) -> list[EvolutionParameter]:
        if self._forced_parameters is not None:
            return self._forced_parameters
        global_parameters = super()._parameter_space(xc)
        local: list[EvolutionParameter] = []
        for parameter in global_parameters:
            center = float(self.seed_parameters.get(parameter.key, parameter.lower))
            radius = self._radius(parameter, center)
            local.append(
                EvolutionParameter(
                    key=parameter.key,
                    lower=max(parameter.lower, center - radius),
                    upper=min(parameter.upper, center + radius),
                    kind=parameter.kind,
                    role=parameter.role,
                )
            )
        self._forced_parameters = local
        return local

    def _radius(self, parameter: EvolutionParameter, center: float) -> float:
        if parameter.key in {"alpha", "beta", "omega"}:
            return self.basin_config.local_global_radius
        if parameter.kind == "mixing":
            return self.basin_config.local_mixing_radius
        if parameter.kind == "role_scale":
            return self.basin_config.local_role_scale_radius
        if parameter.key.endswith("._d") or parameter.key.endswith(".d"):
            return self.basin_config.local_d_radius
        span = max(parameter.upper - parameter.lower, 1e-9)
        return max(self.basin_config.local_ext_abs_min, self.basin_config.local_ext_relative_radius * span)

    def _evaluate_pending_candidates(self, pending, *, runner, validation_runner, root_result, baseline_validation):
        if _runner_uses_docker_queue(runner) and _runner_uses_docker_queue(validation_runner):
            items = list(pending)
            if not items:
                return []
            starts: dict[str, float] = {}
            for effective_sig, candidate in items:
                temp_id = f"{self.label}:{effective_sig[:8]}"
                starts[effective_sig] = time.time()
                logger.info(
                    "BASIN_EVAL_START temp_node=%s stage=local_ea name=%s",
                    temp_id,
                    candidate.xc.name,
                )
            search_results = runner.evaluate_many(
                [candidate.xc for _sig, candidate in items],
                label_prefix=f"{self.label}_local_search",
            )
            validation_results = validation_runner.evaluate_many(
                [candidate.xc for _sig, candidate in items],
                label_prefix=f"{self.label}_local_guard",
            )
            completed = []
            for (effective_sig, candidate), search_result, validation_result in zip(items, search_results, validation_results):
                candidate.search_result = search_result
                candidate.validation_result = validation_result
                self._score(candidate, root_result, baseline_validation)
                duration = time.time() - starts.get(effective_sig, time.time())
                candidate.xc.metadata["basin_eval_duration_s"] = duration
                temp_id = f"{self.label}:{effective_sig[:8]}"
                logger.info(
                    "BASIN_EVAL_END temp_node=%s stage=local_ea raw_reward=%.6f search_mae=%.4f "
                    "validation_mae=%.4f duration_s=%.1f",
                    temp_id,
                    candidate.fitness,
                    candidate.search_result.overall_mae if candidate.search_result else float("inf"),
                    candidate.validation_result.overall_mae if candidate.validation_result else float("inf"),
                    duration,
                )
                completed.append((effective_sig, candidate))
            return completed

        def _run(item):
            effective_sig, candidate = item
            started = time.time()
            temp_id = f"{self.label}:{effective_sig[:8]}"
            logger.info(
                "BASIN_EVAL_START temp_node=%s stage=local_ea name=%s",
                temp_id,
                candidate.xc.name,
            )
            candidate.search_result = runner.evaluate(candidate.xc)
            candidate.validation_result = validation_runner.evaluate(candidate.xc)
            self._score(candidate, root_result, baseline_validation)
            duration = time.time() - started
            candidate.xc.metadata["basin_eval_duration_s"] = duration
            logger.info(
                "BASIN_EVAL_END temp_node=%s stage=local_ea raw_reward=%.6f search_mae=%.4f "
                "validation_mae=%.4f duration_s=%.1f",
                temp_id,
                candidate.fitness,
                candidate.search_result.overall_mae if candidate.search_result else float("inf"),
                candidate.validation_result.overall_mae if candidate.validation_result else float("inf"),
                duration,
            )
            return effective_sig, candidate

        if self.config.evaluation_concurrency <= 1 or len(pending) <= 1:
            return [_run(item) for item in pending]
        completed = []
        with ThreadPoolExecutor(max_workers=min(self.config.evaluation_concurrency, len(pending))) as executor:
            futures = [executor.submit(_run, item) for item in pending]
            for future in as_completed(futures):
                completed.append(future.result())
        return completed


class LLMBasinAgentSearch:
    """Iterative LLM basin proposal with LLM-selected local EA starts."""

    def __init__(
        self,
        *,
        config: BasinAgentConfig | None = None,
        scorer_config: EvolutionConfig | None = None,
    ):
        self.config = config or BasinAgentConfig()
        self.scorer = EvolutionarySearch(scorer_config or EvolutionConfig(seed=self.config.seed))
        self.nodes: list[BasinNode] = []
        self._nodes_by_id: dict[str, BasinNode] = {}
        self._effective_signatures: set[str] = set()
        self._next_id = 0
        self.success_digest = "No successful basin lessons yet."
        self.failure_digest = "No failed basin lessons yet."
        self._task: SearchTask | None = None

    def search(
        self,
        *,
        root_xc: XCFunctional,
        idea_agent: IdeaAgent,
        output_dir: Path | str,
        root_result: BenchmarkResult | None = None,
        baseline_validation_result: BenchmarkResult | None = None,
        runner=None,
        validation_runner=None,
        knowledge: KnowledgeEngine | None = None,
        task: SearchTask | None = None,
        root_evaluation: SearchTaskEvaluation | None = None,
    ) -> BasinAgentResult:
        validate_functional_for_pool(root_xc, component_pool=self.config.component_pool)
        if task is None:
            if root_result is None or baseline_validation_result is None or runner is None or validation_runner is None:
                raise ValueError("Benchmark basin search requires root/guard results and both runners")
            task = BenchmarkSearchTask(
                scorer=self.scorer,
                root_result=root_result,
                baseline_validation=baseline_validation_result,
                runner=runner,
                validation_runner=validation_runner,
                evaluation_concurrency=self.config.proposal_evaluation_concurrency,
            )
        self._task = task
        self._task.validate_functional(root_xc)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cycle_summaries: list[dict[str, Any]] = []

        resume_cycle_evaluated: dict[int, list[BasinNode]] = {}
        if self.config.resume_nodes_path:
            loaded_cycle = self._load_resume_nodes(
                Path(self.config.resume_nodes_path),
                through_cycle=self.config.resume_through_cycle,
            )
            if self.config.resume_through_cycle is None:
                start_cycle = loaded_cycle + 1
                end_cycle = self.config.cycles
                if loaded_cycle > 0:
                    loaded_proposals = [
                        node for node in self.nodes
                        if node.cycle == loaded_cycle and node.stage == "proposal"
                    ]
                    loaded_local_ea = [
                        node for node in self.nodes
                        if node.cycle == loaded_cycle and node.stage == "local_ea"
                    ]
                    if loaded_proposals and not loaded_local_ea:
                        start_cycle = loaded_cycle
                        resume_cycle_evaluated[loaded_cycle] = loaded_proposals
                        logger.info(
                            "BASIN_RESUME_UNFINISHED_CYCLE cycle=%d proposal_nodes=%d/%d; continuing same cycle",
                            loaded_cycle,
                            len(loaded_proposals),
                            self.config.proposals_per_cycle,
                        )
            else:
                start_cycle = loaded_cycle + 1
                end_cycle = loaded_cycle + self.config.cycles
            logger.info(
                "BASIN_RESUME loaded_nodes=%d through_cycle=%s next_cycle=%d end_cycle=%d source=%s",
                len(self.nodes),
                self.config.resume_through_cycle if self.config.resume_through_cycle is not None else "auto",
                start_cycle,
                end_cycle,
                self.config.resume_nodes_path,
            )
            self._write_artifacts(
                output_path,
                cycle_summaries,
                knowledge=knowledge,
            )
        else:
            root_candidate = EvolutionCandidate(
                xc=copy.deepcopy(root_xc),
                parameters=self._read_parameters(root_xc),
                generation=0,
            )
            if root_evaluation is None:
                if isinstance(self._task, BenchmarkSearchTask):
                    root_evaluation = self._task.from_existing(
                        root_result,
                        baseline_validation_result,
                        root_candidate,
                    )
                else:
                    root_evaluation = self._task.evaluate_many(
                        [root_candidate.xc],
                        labels=["root"],
                        stage="root",
                    )[0]
            self._task.apply_to_candidate(root_candidate, root_evaluation)
            if not root_evaluation.valid:
                raise RuntimeError(
                    f"Root functional is invalid for task {root_evaluation.task_name}: "
                    f"{root_evaluation.failure_reason}"
                )
            root_node = self._register_node(
                cycle=0,
                stage="root",
                parent_id="",
                xc=root_candidate.xc,
                candidate=root_candidate,
                rationale="bootstrap root; eligible as the first successful parent anchor",
                action_detail={"action": "root"},
                task_evaluation=root_evaluation,
            )
            root_node.candidate.xc.metadata["basin_parent_reward"] = root_node.raw_reward
            start_cycle = 1
            end_cycle = self.config.cycles

        for cycle in range(start_cycle, end_cycle + 1):
            evaluated: list[BasinNode] = list(resume_cycle_evaluated.get(cycle, []))
            feedback: list[str] = []
            batch_index = int(math.ceil(len(evaluated) / max(1, self.config.proposal_batch_size)))
            if evaluated:
                logger.info(
                    "BASIN_RESUME_CYCLE_STATE cycle=%d preloaded_proposals=%d/%d next_batch=%d",
                    cycle,
                    len(evaluated),
                    self.config.proposals_per_cycle,
                    batch_index + 1,
                )
            include_underexplored_next = False
            while (
                len(evaluated) < self.config.proposals_per_cycle
                and batch_index < self.config.max_proposal_batches
            ):
                batch_index += 1
                logger.info(
                    "BASIN_BATCH_START cycle=%d batch=%d evaluated_total=%d/%d",
                    cycle,
                    batch_index,
                    len(evaluated),
                    self.config.proposals_per_cycle,
                )
                parents = self._success_parent_pool()
                count = min(
                    self.config.proposal_batch_size,
                    self.config.proposals_per_cycle - len(evaluated),
                )
                proposed = self._propose_batch_points(
                    cycle=cycle,
                    batch_index=batch_index,
                    count=count,
                    parents=parents,
                    root_result=root_result,
                    idea_agent=idea_agent,
                    knowledge=knowledge,
                    feedback=feedback,
                    cycle_evaluated=evaluated,
                    include_underexplored=include_underexplored_next,
                )
                include_underexplored_next = len(proposed) < 2
                if include_underexplored_next:
                    logger.info(
                        "BASIN_UNDEREXPLORED_TRIGGER cycle=%d batch=%d accepted=%d threshold=2",
                        cycle,
                        batch_index,
                        len(proposed),
                    )
                if not proposed:
                    logger.warning(
                        "Basin cycle %d batch %d accepted 0/%d proposals before evaluation",
                        cycle,
                        batch_index,
                        count,
                    )
                    continue
                batch_evaluated = self._evaluate_nodes(
                    proposed,
                    root_result=root_result,
                    baseline_validation=baseline_validation_result,
                    runner=runner,
                    validation_runner=validation_runner,
                )
                evaluated.extend(batch_evaluated)
                self._write_artifacts(output_path, cycle_summaries, knowledge=knowledge)
                logger.info(
                    "BASIN_BATCH_DONE cycle=%d batch=%d evaluated_batch=%d evaluated_total=%d/%d successes_total=%d continue=%s",
                    cycle,
                    batch_index,
                    len(batch_evaluated),
                    len(evaluated),
                    self.config.proposals_per_cycle,
                    sum(node.is_success for node in evaluated),
                    len(evaluated) < self.config.proposals_per_cycle and batch_index < self.config.max_proposal_batches,
                )
            if len(evaluated) < self.config.proposals_per_cycle:
                message = (
                    "Basin cycle %d evaluated only %d/%d proposal nodes after %d batches"
                    % (
                        cycle,
                        len(evaluated),
                        self.config.proposals_per_cycle,
                        batch_index,
                    )
                )
                logger.warning("%s", message)
                if self.config.require_full_proposal_budget:
                    raise RuntimeError(
                        message
                        + "; require_full_proposal_budget=true prevents an under-budget control run"
                    )
            selected: list[BasinNode] = []
            ea_nodes: list[BasinNode] = []
            if self.config.enable_local_ea:
                selected = self._select_ea_seeds(
                    cycle=cycle,
                    candidates=evaluated,
                    idea_agent=idea_agent,
                )
                if isinstance(self._task, BenchmarkSearchTask):
                    ea_nodes = self._run_local_ea(
                        cycle=cycle,
                        seeds=selected,
                        root_result=root_result,
                        baseline_validation=baseline_validation_result,
                        runner=runner,
                        validation_runner=validation_runner,
                    )
                else:
                    ea_nodes = self._run_task_local_ea(cycle=cycle, seeds=selected)
            cycle_nodes = [*evaluated, *ea_nodes]
            self._update_digests(cycle=cycle, cycle_nodes=cycle_nodes, idea_agent=idea_agent)
            summary = self._cycle_summary(cycle, proposed=evaluated, ea_nodes=ea_nodes, selected=selected)
            summary["proposal_batches"] = batch_index
            cycle_summaries.append(summary)
            self._write_artifacts(
                output_path,
                cycle_summaries,
                knowledge=knowledge,
                update_graph=True,
            )
        result = BasinAgentResult(
            nodes=self.nodes,
            cycle_summaries=cycle_summaries,
            success_digest=self.success_digest,
            failure_digest=self.failure_digest,
        )
        self._write_artifacts(
            output_path,
            cycle_summaries,
            knowledge=knowledge,
        )
        return result

    def _propose_batch_points(
        self,
        *,
        cycle: int,
        batch_index: int,
        count: int,
        parents: list[BasinNode],
        root_result: BenchmarkResult,
        idea_agent: IdeaAgent,
        knowledge: KnowledgeEngine | None,
        feedback: list[str],
        cycle_evaluated: list[BasinNode],
        include_underexplored: bool = False,
    ) -> list[BasinNode]:
        accepted: list[BasinNode] = []
        prompt = self._proposal_prompt(
            cycle=cycle,
            batch_index=batch_index,
            count=count,
            parents=parents,
            root_result=root_result,
            feedback=feedback,
            knowledge=knowledge,
            cycle_evaluated=cycle_evaluated,
            include_underexplored=include_underexplored,
        )
        logger.info(
            "%s",
            log_block(
                f"BASIN_PROPOSE_CONTEXT cycle={cycle} batch={batch_index}",
                self._proposal_log_summary(
                    cycle=cycle,
                    batch_index=batch_index,
                    count=count,
                    parents=parents,
                    feedback=feedback,
                    cycle_evaluated=cycle_evaluated,
                    include_underexplored=include_underexplored,
                ),
                color="blue",
            ),
        )
        payload = self._llm_json(
            idea_agent,
            prompt,
            f"basin propose cycle={cycle} batch={batch_index}",
            temperature=0.65,
        )
        for index, spec in enumerate(payload.get("points", []) or [], start=1):
            if len(accepted) >= count:
                break
            try:
                node = self._build_proposed_node(cycle, spec, parents, idea_agent)
            except Exception as exc:
                msg = f"cycle {cycle} batch {batch_index} point {index} rejected: {exc}"
                logger.warning("BASIN_PROPOSE_REJECT %s", msg)
                feedback.append(msg)
                continue
            nearest_id, nearest_delta = self._nearest_existing(node.xc, exclude_node_id=node.node_id)
            node.nearest_node_id = nearest_id
            node.nearest_delta = nearest_delta
            if (
                self.config.enforce_novelty_threshold
                and nearest_delta is not None
                and nearest_delta < self.config.novelty_threshold
            ):
                existing = self._nodes_by_id.get(nearest_id)
                msg = (
                    f"{node.node_id} too similar to evaluated {nearest_id}: "
                    f"max_abs_delta={nearest_delta:.4f} < {self.config.novelty_threshold:.4f}; "
                    f"known_result={_node_result_line(existing) if existing else 'unknown'}"
                )
                logger.warning("BASIN_PROPOSE_REJECT %s", msg)
                feedback.append(msg)
                self._discard_node(node)
                continue
            sig = node.xc.effective_signature()
            if sig in self._effective_signatures:
                msg = f"{node.node_id} duplicate effective_signature={sig}"
                logger.warning("BASIN_PROPOSE_REJECT %s", msg)
                feedback.append(msg)
                self._discard_node(node)
                continue
            self._effective_signatures.add(sig)
            accepted.append(node)
        if len(accepted) < count:
            logger.warning(
                "Basin cycle %d batch %d accepted only %d/%d proposals",
                cycle,
                batch_index,
                len(accepted),
                count,
            )
        return accepted

    def _proposal_log_summary(
        self,
        *,
        cycle: int,
        batch_index: int,
        count: int,
        parents: list[BasinNode],
        feedback: list[str],
        cycle_evaluated: list[BasinNode],
        include_underexplored: bool,
    ) -> str:
        context_nodes = self._proposal_context_nodes(cycle=cycle, cycle_evaluated=cycle_evaluated)
        proposal_kind = "new candidate points" if self.config.freeform_llm_control_prompt else "new basin seed points"
        lines = [
            f"Cycle {cycle}, batch {batch_index}: request {count} {proposal_kind}.",
            "Full prompt is sent to the LLM but omitted from logs; this is the compact visible context.",
            "",
            "## Successful Parent Pool",
            _format_node_refs(parents),
            "",
            "## Proposal Context Nodes",
            _format_node_refs(context_nodes),
            "",
            "## Proposal Context Breakdown",
            json.dumps(self._proposal_context_breakdown(cycle=cycle, cycle_evaluated=cycle_evaluated), ensure_ascii=False),
            "",
            "## Rejection Feedback From This Cycle",
            "\n".join(f"- {item}" for item in feedback[-12:]) or "None.",
        ]
        if include_underexplored:
            lines.extend(["", "## Underexplored Directions", self._underexplored_directions()])
        else:
            lines.extend(["", "## Underexplored Directions", "Not triggered for this batch."])
        return "\n".join(lines)

    def _proposal_prompt(
        self,
        *,
        cycle: int,
        batch_index: int,
        count: int,
        parents: list[BasinNode],
        root_result: BenchmarkResult | None,
        feedback: list[str],
        knowledge: KnowledgeEngine | None,
        cycle_evaluated: list[BasinNode],
        include_underexplored: bool = False,
    ) -> str:
        knowledge_text = ""
        use_benchmark_priors = bool(
            self.config.use_benchmark_component_priors
            and not self.config.anonymize_component_names
            and self._task is not None
            and self._task.use_benchmark_component_priors
        )
        if knowledge is not None and parents and root_result is not None and use_benchmark_priors:
            component_priors = knowledge.component_priors_for(
                current_xc=parents[0].xc,
                current_result=parents[0].candidate.search_result or root_result,
                target_subsets=list(root_result.subsets),
                limit=self.config.knowledge_component_prior_limit,
            )
            knowledge_text = _format_component_prior_text(
                component_priors,
                limit=self.config.knowledge_component_prior_limit,
            )
        context_nodes = self._proposal_context_nodes(cycle=cycle, cycle_evaluated=cycle_evaluated)
        component_aliases, alias_to_component = _component_alias_maps(self.config.component_pool)
        anonymize_components = bool(self.config.anonymize_component_names)
        underexplored_text = self._underexplored_directions() if include_underexplored else "Not triggered for this batch."
        if anonymize_components:
            underexplored_text = _mask_component_text(underexplored_text, component_aliases)
        if anonymize_components:
            approved_component_context = _format_anonymized_component_catalogue(component_aliases)
        elif self.config.component_pool == "zno_gap" and self.config.use_component_design_priors:
            approved_component_context = format_zno_gap_prompt_registry()
        else:
            approved_component_context = _format_component_catalogue(self.config.component_pool)
        task_instructions = self._task.prompt_instructions() if self._task is not None else ""
        if anonymize_components:
            task_instructions = _mask_component_text(task_instructions, component_aliases)
        rejection_feedback = "\n".join(f"- {item}" for item in feedback[-12:]) or "None."
        if anonymize_components:
            rejection_feedback = _mask_component_text(rejection_feedback, component_aliases)
        action_rules = [
            "You may change multiple numerical parameters at once and may add/remove/replace components.",
            "Setting mixing:<COMPONENT> to 0 is treated exactly as remove_component: the component is "
            "deleted before validation, signatures, novelty measurement, and evaluation.",
        ]
        if self.config.allow_hf_exchange_rewire:
            action_rules.append(
                "Mixing sum is a hard constraint: correlation component mixing factors must sum to exactly 1.0, "
                "and exchange component mixing factors must sum to exactly 1.0 - alpha. Any final proposal that "
                "violates either sum is discarded before evaluation."
            )
        else:
            action_rules.append(
                "Component mixing is a hard constraint: exchange component mixing factors must sum to exactly "
                "1.0 and correlation component mixing factors must sum to exactly 1.0. Any final proposal that "
                "violates either sum is discarded before evaluation."
            )
        if not self.config.allow_hf_exchange_rewire:
            action_rules.append("HF exchange is forbidden: do not set or change alpha, beta, or omega.")
        if not self.config.allow_ext_param_rewire:
            action_rules.append(
                "Component internal/ext parameters are fixed: rewire may change mixing:<COMPONENT>, "
                "scale:exchange, and scale:correlation only."
            )
        if not self.config.allow_hf_exchange_rewire:
            current_exchange = _component_prompt_name(
                next((component.name for node in parents for component in node.xc.components if component.role == "exchange"), ""),
                component_aliases if anonymize_components else None,
            )
            sample_exchange = (
                _first_alias_for_role(component_aliases, "exchange", exclude=current_exchange)
                if anonymize_components
                else "GGA_X_AK13"
            )
            current_exchange = current_exchange or sample_exchange
            example_actions = [
                {"action_type": "add_component", "changes": {"name": sample_exchange, "role": "exchange"}},
                {
                    "action_type": "rewire",
                    "changes": {
                        f"mixing:{current_exchange}": 0.75,
                        f"mixing:{sample_exchange}": 0.25,
                        "scale:exchange": 1.05,
                    },
                },
            ]
        else:
            current_correlation = _component_prompt_name(
                next((component.name for node in parents for component in node.xc.components if component.role == "correlation"), ""),
                component_aliases if anonymize_components else None,
            )
            sample_correlation = (
                _first_alias_for_role(component_aliases, "correlation", exclude=current_correlation)
                if anonymize_components
                else "GGA_C_APBE"
            )
            current_correlation = current_correlation or sample_correlation
            example_actions = [
                {"action_type": "rewire", "changes": {"alpha": 0.35, "beta": 0.55, "omega": 0.28}},
                {"action_type": "replace_component", "changes": {"old_name": current_correlation, "new_name": sample_correlation}},
                {"action_type": "add_component", "changes": {"name": sample_correlation, "role": "correlation"}},
            ]
        required_example = json.dumps(
            {
                "points": [
                    {
                        "parent_id": "N00000",
                        "label": "short_basin_name",
                        "rationale": "why this basin is worth a full SCF test",
                        "actions": example_actions,
                    }
                ]
            },
            indent=2,
        )
        if self.config.freeform_llm_control_prompt:
            task_header_lines = [
                "## Pure LLM Proposal Control Task",
                f"Cycle {cycle}, batch {batch_index}: propose {count} new candidate points.",
                "This control disables local EA and distance-based novelty filtering; every improvement must come from LLM proposals plus cycle reflections.",
                "You may choose local refinement, conservative parameter/mixing/scale edits, component add/remove/replace moves, or larger structural changes based on observed results.",
                "Do not intentionally repeat an already evaluated effective configuration; exact duplicates are still rejected before evaluation.",
                "Use Proposal Context Nodes below as the single evaluated-history block for this proposal call.",
                "Each proposal MUST choose parent_id from Successful Parent Pool below.",
                "A successful parent means the point improved raw_reward over its own parent; root is a bootstrap anchor.",
            ]
            novelty_lines: list[str] = [
                "There is no minimum-distance novelty rule in this control. Small local tuning is allowed when justified by the observed trajectory.",
            ]
        else:
            task_header_lines = [
                "## Basin Proposal Task",
                f"Cycle {cycle}, batch {batch_index}: propose {count} new basin seed points.",
                "This is a closed-loop batch. Previous batches in this same cycle have already been benchmarked.",
                "Use Proposal Context Nodes below as the single evaluated-history block for this proposal call.",
                "Each proposal MUST choose parent_id from Successful Parent Pool below.",
                "The parent pool is direction-filtered: one proposal seed and its local-EA descendants share a lineage group, and each group may contribute multiple distinct improvement directions but not duplicate same-direction refinements.",
                "A successful parent means the point improved raw_reward over its own parent; root is a bootstrap anchor.",
            ]
            novelty_lines = [
                "Small local tuning is discouraged: each point should seek a meaningfully different basin.",
                f"Novelty rule: compared with every evaluated node, at least one numeric parameter must differ by >= {self.config.novelty_threshold:.3f}, or the component set must change.",
            ]
        return "\n".join(
            [
                *task_header_lines,
                *action_rules,
                *novelty_lines,
                "Do not propose descriptors. Do not remove the only active exchange or correlation component.",
                "For add_component, the compiler splits about 0.10 mixing from the largest same-role component; put any later mixing/alpha/beta/omega edits in a separate rewire action.",
                (
                    "For replace_component, use flat anonymous IDs ONLY: {\"old_name\": \"C001\", \"new_name\": \"C002\"}. Do not put role/mixing_factor/ext_params inside new_name/new_component."
                    if anonymize_components
                    else "For replace_component, use flat changes ONLY: {\"old_name\": \"GGA_C_PBE\", \"new_name\": \"GGA_C_APBE\"}. Do not put role/mixing_factor/ext_params inside new_name/new_component."
                ),
                (
                    "Component names are intentionally blinded for this control. Use only anonymous IDs from Approved Component Context, e.g. X001 or C001; do not guess real Libxc names."
                    if anonymize_components
                    else "Approved components include many entries that have no reliable prior. You may explore unknown components, but only by using exact names from Approved Component Context."
                ),
                (
                    "Component Benchmark Priors are associative whole-functional priors, not causal component effects. Treat them as weak hints only."
                    if use_benchmark_priors
                    else (
                        "All component priors are disabled for this control/task. Use only the approved anonymous component IDs and observed rewards."
                        if anonymize_components
                        else "All component priors are disabled for this control/task. Use only exact names from Approved Component Context and observed rewards."
                    )
                ),
                "",
                "## Scientific Task Contract",
                task_instructions,
                "",
                "## Successful Parent Pool (node IDs are mandatory)",
                json.dumps(
                    [
                        _node_prompt_brief(
                            node,
                            component_aliases=component_aliases if anonymize_components else None,
                        )
                        for node in parents
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
                "## Global Success Digest",
                _mask_component_text(self.success_digest, component_aliases) if anonymize_components else self.success_digest,
                "",
                "## Global Failure Digest",
                _mask_component_text(self.failure_digest, component_aliases) if anonymize_components else self.failure_digest,
                "",
                "## Proposal Context Nodes",
                json.dumps(
                    [
                        _node_prompt_brief(
                            node,
                            component_aliases=component_aliases if anonymize_components else None,
                        )
                        for node in context_nodes
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
                "## Component Benchmark Priors",
                knowledge_text or "Disabled for this task; no molecular benchmark prior is supplied.",
                "",
                "## Approved Component Context",
                approved_component_context,
                "",
                "## Underexplored Directions",
                underexplored_text,
                "",
                "## Rejection Feedback From This Cycle",
                rejection_feedback,
                "",
                "## Required JSON",
                "Return only JSON:",
                required_example,
            ]
        )

    def _build_proposed_node(
        self,
        cycle: int,
        spec: dict[str, Any],
        parents: list[BasinNode],
        idea_agent: IdeaAgent,
    ) -> BasinNode:
        parent_id = str(spec.get("parent_id") or "")
        parent = {node.node_id: node for node in parents}.get(parent_id)
        if parent is None:
            raise ValueError(f"parent_id={parent_id!r} is not in the successful parent pool")
        actions_payload = spec.get("actions")
        if not isinstance(actions_payload, list) or not actions_payload:
            raise ValueError("point spec requires non-empty actions")
        candidate = copy.deepcopy(parent.xc)
        normalized: list[dict[str, Any]] = []
        contract = {
            "action_family": "basin_proposal",
            "allowed_action_types": [
                "rewire",
                "add_component",
                "remove_component",
                "replace_component",
                "enforce_constraint",
            ],
        }
        _component_aliases, alias_to_component = _component_alias_maps(self.config.component_pool)
        for payload in actions_payload:
            payload = _normalize_basin_action_payload(payload)
            if self.config.anonymize_component_names:
                payload = _translate_anonymized_action_payload(payload, alias_to_component)
            skill_name = "parameter-control" if payload.get("action_type") == "rewire" else "component-replacement"
            skill = idea_agent.skill_catalog.get(skill_name)
            action, _semantic = idea_agent._parse_response(
                json.dumps(payload),
                skill,
                action_contract=contract,
            )
            self._validate_action_parameters(action)
            for resumed_node in self.nodes:
                validate_functional_for_pool(
                    resumed_node.xc,
                    component_pool=self.config.component_pool,
                )
            if action.action_type == ActionType.ADD_COMPONENT:
                validate_component_for_pool(
                    str(action.changes.get("name") or ""),
                    component_pool=self.config.component_pool,
                )
            elif action.action_type == ActionType.REPLACE_COMPONENT:
                validate_component_for_pool(
                    str(action.changes.get("new_name") or ""),
                    component_pool=self.config.component_pool,
                )
            if action.action_type == ActionType.ENFORCE_CONSTRAINT:
                normalized.append(
                    {
                        "action_type": action.action_type.value,
                        "changes": action.changes,
                        "rationale": action.rationale,
                    }
                )
                continue
            before_action = candidate
            candidate = apply_action(candidate, action)
            auto_removed_components = sorted(
                {
                    component.name for component in before_action.components
                }
                - {
                    component.name for component in candidate.components
                }
            )
            recorded_auto_removals = set(
                candidate.metadata.get("auto_removed_zero_mixing_components", [])
            )
            auto_removed_components = [
                name for name in auto_removed_components if name in recorded_auto_removals
            ]
            if self._task is not None:
                self._task.validate_functional(candidate)
            action_delta = _novelty_delta(candidate, before_action)
            pending_id = self._peek_next_node_id()
            logger.info(
                "BASIN_ACTION_APPLY pending=%s parent=%s action=%s delta_vs_before=%.6f changes=%s",
                pending_id,
                parent_id,
                action.action_type.value,
                action_delta,
                json.dumps(action.changes, sort_keys=True, default=str),
            )
            if action.changes and action_delta <= 1e-12:
                logger.warning(
                    "BASIN_ACTION_NO_EFFECT pending=%s parent=%s action=%s changes=%s",
                    pending_id,
                    parent_id,
                    action.action_type.value,
                    json.dumps(action.changes, sort_keys=True, default=str),
                )
            normalized_action = {
                "action_type": action.action_type.value,
                "changes": action.changes,
                "rationale": action.rationale,
            }
            if auto_removed_components:
                normalized_action["auto_removed_components"] = auto_removed_components
            normalized.append(normalized_action)
        self._validate_component_mixing_sums(candidate)
        hard = [item for item in candidate.validate() if item.startswith("HARD:")]
        if hard:
            raise ValueError("; ".join(hard))
        node_id = self._peek_next_node_id()
        label = str(spec.get("label") or f"cycle{cycle}_proposal")
        candidate.name = f"{parent.xc.name}__basin_{cycle:02d}_{node_id}_{candidate.effective_signature()}"
        candidate.metadata.update(
            {
                "action": "basin_proposal",
                "action_detail": {"parent_id": parent_id, "label": label, "actions": normalized},
                "search_algorithm": "basin_agent",
                "basin_cycle": cycle,
                "basin_parent_id": parent_id,
                "basin_parent_reward": parent.raw_reward,
            }
        )
        empty = EvolutionCandidate(
            xc=candidate,
            parameters=self._read_parameters(candidate),
            generation=cycle,
        )
        return self._register_node(
            cycle=cycle,
            stage="proposal_pending",
            parent_id=parent_id,
            xc=candidate,
            candidate=empty,
            rationale=str(spec.get("rationale") or ""),
            action_detail={"parent_id": parent_id, "label": label, "actions": normalized},
            register_signature=False,
        )

    @staticmethod
    def _validate_component_mixing_sums(candidate: XCFunctional, *, tolerance: float = 1e-8) -> None:
        exchange_sum = sum(
            float(component.mixing_factor)
            for component in candidate.components
            if component.role == "exchange"
        )
        correlation_sum = sum(
            float(component.mixing_factor)
            for component in candidate.components
            if component.role == "correlation"
        )
        exchange_target = max(0.0, 1.0 - float(candidate.hf_exchange.alpha))
        errors: list[str] = []
        if abs(exchange_sum - exchange_target) > tolerance:
            errors.append(
                f"exchange component mixing sum={exchange_sum:.12g}, required={exchange_target:.12g}"
            )
        if abs(correlation_sum - 1.0) > tolerance:
            errors.append(
                f"correlation component mixing sum={correlation_sum:.12g}, required=1"
            )
        if errors:
            raise ValueError(
                "Invalid final component mixing; proposal will not be evaluated: " + "; ".join(errors)
            )

    def _validate_rewire_action(self, changes: dict[str, Any]) -> None:
        keys = {str(key) for key in changes}
        hf_keys = keys.intersection({"alpha", "beta", "omega"})
        if hf_keys and not self.config.allow_hf_exchange_rewire:
            raise ValueError(f"HF exchange rewiring is disabled for this task: {sorted(hf_keys)}")
        ext_keys = {
            key
            for key in keys
            if key not in {"alpha", "beta", "omega"}
            and not key.startswith("mixing:")
            and not key.startswith("scale:")
        }
        if ext_keys and not self.config.allow_ext_param_rewire:
            raise ValueError(f"Component internal/ext parameter rewiring is disabled: {sorted(ext_keys)}")

    def _validate_action_parameters(self, action: Any) -> None:
        changes = dict(action.changes or {})
        if not self.config.allow_ext_param_rewire:
            structural_ext_keys = {
                "ext_params",
                "ext_param_bounds",
                "new_ext_params",
                "new_ext_param_bounds",
            }
            supplied = structural_ext_keys.intersection(changes)
            if supplied:
                raise ValueError(
                    f"Explicit component internal/ext parameters are disabled: {sorted(supplied)}"
                )
        if action.action_type == ActionType.REWIRE:
            self._validate_rewire_action(changes)
            return
        cross_rewire = {
            key: value
            for key, value in changes.items()
            if key in {"alpha", "beta", "omega"}
            or key.startswith("mixing:")
            or key.startswith("scale:")
            or "." in key
        }
        if cross_rewire:
            self._validate_rewire_action(cross_rewire)

    def _evaluate_nodes(
        self,
        nodes: list[BasinNode],
        *,
        root_result: BenchmarkResult,
        baseline_validation: BenchmarkResult,
        runner,
        validation_runner,
    ) -> list[BasinNode]:
        if not nodes:
            return []
        if self._task is None:
            raise RuntimeError("No active SearchTask configured")
        started = time.time()
        for node in nodes:
            node.eval_started_at = started
            logger.info(
                "BASIN_EVAL_START node=%s stage=proposal parent=%s name=%s",
                node.node_id,
                node.parent_id,
                node.xc.name,
            )
        evaluations = self._task.evaluate_many(
            [node.xc for node in nodes],
            labels=[node.node_id for node in nodes],
            stage=f"cycle_{nodes[0].cycle:02d}_proposal",
        )
        if len(evaluations) != len(nodes):
            raise RuntimeError(f"Task returned {len(evaluations)} evaluations for {len(nodes)} nodes")
        for node, evaluation in zip(nodes, evaluations):
            parent = self._nodes_by_id.get(node.parent_id)
            parent_evaluation = parent.task_evaluation if parent is not None else None
            relative = self._task.relative_feedback(evaluation, parent_evaluation)
            if relative:
                evaluation.proposal_feedback["relative_to_parent"] = relative
            self._task.apply_to_candidate(node.candidate, evaluation)
            node.task_evaluation = evaluation
            node.eval_finished_at = time.time()
            node.eval_duration_s = evaluation.walltime_s
            if node.eval_duration_s is None:
                node.eval_duration_s = node.eval_finished_at - (node.eval_started_at or node.eval_finished_at)
            node.stage = "proposal"
            logger.info(
                "BASIN_EVAL_END node=%s stage=proposal parent=%s raw_reward=%.6f "
                "success_vs_parent=%s duration_s=%.1f results=%s",
                node.node_id,
                node.parent_id,
                node.raw_reward,
                node.is_success,
                node.eval_duration_s,
                json.dumps(_node_result_metrics(node), sort_keys=True),
            )
        return nodes

    def _select_ea_seeds(
        self,
        *,
        cycle: int,
        candidates: list[BasinNode],
        idea_agent: IdeaAgent,
    ) -> list[BasinNode]:
        eligible = [
            node for node in candidates
            if node.candidate.promotable and math.isfinite(node.raw_reward)
        ]
        if not eligible:
            return []
        prompt = "\n".join(
            [
                "## EA Seed Selection Task",
                f"Cycle {cycle}: select {min(self.config.ea_seed_count, len(eligible))} diverse nodes for small local EA.",
                "Prefer high raw_reward, but explicitly preserve diversity across components and alpha/beta/omega/mixing basins.",
                "Return only node IDs from the candidate table. Do not invent IDs.",
                "",
                "## Candidate Table",
                json.dumps([_node_prompt_brief(node) for node in sorted(eligible, key=lambda item: item.raw_reward, reverse=True)], indent=2, ensure_ascii=False),
                "",
                "## Required JSON",
                "{",
                '  "selected_node_ids": ["N00001", "N00007"],',
                '  "rationale": "why these seeds cover distinct basins"',
                "}",
            ]
        )
        logger.info("%s", log_block(f"BASIN_EA_SEED_SELECT_INPUT cycle={cycle}", prompt, color="blue"))
        try:
            payload = self._llm_json(
                idea_agent,
                prompt,
                f"basin EA seed selection cycle={cycle}",
                temperature=0.25,
            )
            ids = [str(item) for item in payload.get("selected_node_ids", [])]
        except Exception as exc:
            logger.warning("BASIN_EA_SEED_SELECT_FAILED cycle=%d: %s", cycle, exc)
            ids = []
        by_id = {node.node_id: node for node in eligible}
        selected: list[BasinNode] = []
        for node_id in ids:
            node = by_id.get(node_id)
            if node is not None and node not in selected:
                selected.append(node)
            if len(selected) >= self.config.ea_seed_count:
                break
        if len(selected) < min(self.config.ea_seed_count, len(eligible)):
            for node in self._diverse_top(eligible):
                if node not in selected:
                    selected.append(node)
                if len(selected) >= self.config.ea_seed_count:
                    break
        logger.info(
            "BASIN_EA_SEED_SELECTED cycle=%d ids=%s",
            cycle,
            [node.node_id for node in selected],
        )
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
                label=f"cycle{cycle}:seed{seed.node_id}",
            )
            try:
                result = local.search(
                    root_xc=seed.xc,
                    root_result=root_result,
                    validation_runner=validation_runner,
                    runner=runner,
                    baseline_validation_result=baseline_validation,
                    initial_population_vectors=[seed.candidate.parameters],
                    baseline_candidate=seed.candidate,
                )
            except Exception as exc:
                logger.warning("BASIN_LOCAL_EA_FAILED cycle=%d seed=%s: %s", cycle, seed.node_id, exc)
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

    def _run_task_local_ea(
        self,
        *,
        cycle: int,
        seeds: list[BasinNode],
    ) -> list[BasinNode]:
        if self._task is None:
            raise RuntimeError("No active SearchTask configured")
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
                include_global_hf=self.scorer.config.include_global_hf,
                include_role_scales=self.scorer.config.include_role_scales,
                role_scale_bounds=self.scorer.config.role_scale_bounds,
                include_component_mixing=self.scorer.config.include_component_mixing,
                component_mixing_bounds=self.scorer.config.component_mixing_bounds,
                include_component_ext_params=self.scorer.config.include_component_ext_params,
                component_pool=self.scorer.config.component_pool,
            )
            local = LocalEvolutionarySearch(
                config,
                seed_xc=seed.xc,
                seed_parameters=seed.candidate.parameters,
                basin_config=self.config,
                label=f"cycle{cycle}:seed{seed.node_id}",
            )
            try:
                result = self._task_local_ea_search(local, seed=seed, cycle=cycle)
            except Exception as exc:
                logger.warning("BASIN_LOCAL_EA_FAILED cycle=%d seed=%s: %s", cycle, seed.node_id, exc)
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
                    rationale=f"task local EA refinement from {seed.node_id}",
                    action_detail=dict(candidate.parameters),
                    task_evaluation=SearchTaskEvaluation.from_dict(
                        candidate.xc.metadata.get("task_evaluation")
                    ),
                )
                node.eval_duration_s = candidate.xc.metadata.get("basin_eval_duration_s")
                collected.append(node)
        return collected

    def _task_local_ea_search(
        self,
        local: LocalEvolutionarySearch,
        *,
        seed: BasinNode,
        cycle: int,
    ) -> EvolutionSearchResult:
        if self._task is None:
            raise RuntimeError("No active SearchTask configured")
        parameters = local._parameter_space(seed.xc)
        if not parameters:
            raise ValueError("Task local EA has no tunable parameters for this seed")
        baseline = copy.deepcopy(seed.candidate)
        evaluated: list[EvolutionCandidate] = [baseline]
        evaluated_by_sig: dict[str, EvolutionCandidate] = {
            baseline.xc.effective_signature(): baseline,
        }
        population_vectors = local._initial_population(
            baseline.parameters,
            parameters,
            seed_vectors=[seed.candidate.parameters],
        )
        best = baseline
        generation_summaries: list[dict[str, Any]] = []
        for generation in range(1, local.config.generations + 1):
            population: list[EvolutionCandidate] = []
            pending: list[tuple[str, EvolutionCandidate]] = []
            for vector in population_vectors:
                xc = local._build_candidate(seed.xc, vector, generation)
                try:
                    self._task.validate_functional(xc)
                except ValueError as exc:
                    logger.info(
                        "BASIN_LOCAL_EA_REJECT cycle=%d parent=%s generation=%d reason=%s",
                        cycle,
                        seed.node_id,
                        generation,
                        exc,
                    )
                    continue
                effective_sig = xc.effective_signature()
                cached = evaluated_by_sig.get(effective_sig)
                if cached is not None:
                    population.append(cached)
                    continue
                pending.append((
                    effective_sig,
                    EvolutionCandidate(
                        xc=xc,
                        parameters=dict(vector),
                        generation=generation,
                    ),
                ))
            if pending:
                starts = {sig: time.time() for sig, _candidate in pending}
                for sig, candidate in pending:
                    logger.info(
                        "BASIN_EVAL_START temp_node=%s stage=local_ea parent=%s name=%s",
                        f"{local.label}:{sig[:8]}",
                        seed.node_id,
                        candidate.xc.name,
                    )
                evaluations = self._task.evaluate_many(
                    [candidate.xc for _sig, candidate in pending],
                    labels=[f"{seed.node_id}_ea_g{generation}_{index:02d}" for index in range(len(pending))],
                    stage=f"cycle_{cycle:02d}_local_ea",
                )
                if len(evaluations) != len(pending):
                    raise RuntimeError(
                        f"Task returned {len(evaluations)} local-EA evaluations for {len(pending)} candidates"
                    )
                for (effective_sig, candidate), evaluation in zip(pending, evaluations):
                    if seed.task_evaluation is not None:
                        relative = self._task.relative_feedback(evaluation, seed.task_evaluation)
                        if relative:
                            evaluation.proposal_feedback["relative_to_parent"] = relative
                    self._task.apply_to_candidate(candidate, evaluation)
                    duration = time.time() - starts.get(effective_sig, time.time())
                    candidate.xc.metadata["basin_eval_duration_s"] = evaluation.walltime_s or duration
                    candidate.xc.metadata["task_evaluation"] = evaluation.to_dict()
                    evaluated.append(candidate)
                    evaluated_by_sig[effective_sig] = candidate
                    population.append(candidate)
                    logger.info(
                        "BASIN_EVAL_END temp_node=%s stage=local_ea parent=%s raw_reward=%.6f "
                        "promotable=%s duration_s=%.1f results=%s",
                        f"{local.label}:{effective_sig[:8]}",
                        seed.node_id,
                        candidate.fitness,
                        candidate.promotable,
                        candidate.xc.metadata["basin_eval_duration_s"],
                        json.dumps(
                            {
                                "task": evaluation.task_name,
                                "valid": evaluation.valid,
                                "objective": evaluation.objective,
                                "failure_reason": evaluation.failure_reason,
                            },
                            sort_keys=True,
                        ),
                    )
            ranked = sorted(population, key=EvolutionarySearch._rank_key, reverse=True)
            if ranked and EvolutionarySearch._rank_key(ranked[0]) > EvolutionarySearch._rank_key(best):
                best = ranked[0]
            diversity = local._population_diversity(ranked, parameters)
            generation_summaries.append(
                {
                    "generation": generation,
                    "unique_candidates": len({item.xc.effective_signature() for item in ranked}),
                    "promotable_candidates": sum(item.promotable for item in ranked),
                    "best_name": best.xc.name,
                    "best_fitness": best.fitness,
                    "best_parameters": dict(best.parameters),
                    **diversity,
                }
            )
            population_vectors = local._next_population(ranked, parameters)
        return EvolutionSearchResult(
            best=best,
            baseline=baseline,
            evaluated=evaluated,
            generation_summaries=generation_summaries,
            parameters=parameters,
        )

    def _update_digests(self, *, cycle: int, cycle_nodes: list[BasinNode], idea_agent: IdeaAgent) -> None:
        successes = [node for node in cycle_nodes if node.is_success]
        failures = [node for node in cycle_nodes if not node.is_success]
        component_aliases, _alias_to_component = _component_alias_maps(self.config.component_pool)
        anonymize_components = bool(self.config.anonymize_component_names)
        prompt = "\n".join(
            [
                "## Basin Reflection Task",
                "Summarize durable lessons for the next LLM proposal cycle.",
                "Do not propose new points. Use several concise causal bullets when useful; do not collapse distinct lessons into one line.",
                (
                    "Component names are blinded in this control. Refer only to anonymous component IDs such as X001/C001 and never infer or mention real Libxc names."
                    if anonymize_components
                    else ""
                ),
                "",
                "## Previous Success Digest",
                _mask_component_text(self.success_digest, component_aliases) if anonymize_components else self.success_digest,
                "",
                "## Previous Failure Digest",
                _mask_component_text(self.failure_digest, component_aliases) if anonymize_components else self.failure_digest,
                "",
                "## Current Cycle Successful Nodes",
                json.dumps(
                    [
                        _node_reflection_brief(
                            node,
                            component_aliases=component_aliases if anonymize_components else None,
                        )
                        for node in successes
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
                "## Current Cycle Failed Or Non-improving Nodes",
                json.dumps(
                    [
                        _node_reflection_brief(
                            node,
                            component_aliases=component_aliases if anonymize_components else None,
                        )
                        for node in failures
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
                "## Required JSON",
                "{",
                '  "success_digest": "updated success lessons, allowing several causal bullets",',
                '  "failure_digest": "updated failure lessons, allowing several causal bullets"',
                "}",
            ]
        )
        logger.info("%s", log_block(f"BASIN_REFLECTION_INPUT cycle={cycle}", prompt, color="blue"))
        try:
            payload = self._llm_json(idea_agent, prompt, f"basin reflection cycle={cycle}", temperature=0.2)
            self.success_digest = str(payload.get("success_digest") or self.success_digest)
            self.failure_digest = str(payload.get("failure_digest") or self.failure_digest)
        except Exception as exc:
            logger.warning("BASIN_REFLECTION_FAILED cycle=%d: %s", cycle, exc)
            self.success_digest = _deterministic_digest(
                "success",
                successes,
                previous=self.success_digest,
                max_lines=self.config.digest_max_lines,
            )
            self.failure_digest = _deterministic_digest(
                "failure",
                failures,
                previous=self.failure_digest,
                max_lines=self.config.digest_max_lines,
            )
            if anonymize_components:
                self.success_digest = _mask_component_text(self.success_digest, component_aliases)
                self.failure_digest = _mask_component_text(self.failure_digest, component_aliases)
        logger.info(
            "%s",
            log_block(
                f"BASIN_REFLECTION_OUTPUT cycle={cycle}",
                json.dumps(
                    {
                        "success_digest": self.success_digest,
                        "failure_digest": self.failure_digest,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                color="cyan",
            ),
        )

    def _load_resume_nodes(self, path: Path, *, through_cycle: int | None) -> int:
        if not path.exists():
            raise FileNotFoundError(f"Basin resume nodes file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records = payload.get("nodes") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"Basin resume nodes file must contain a list of nodes: {path}")
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
            candidate = EvolutionCandidate(
                xc=xc,
                parameters={key: float(value) for key, value in (record.get("parameters") or {}).items()},
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
            task_evaluation = SearchTaskEvaluation.from_dict(record.get("task_evaluation"))
            node_id = str(record.get("node_id") or f"N{max_node_number + 1:05d}")
            node = BasinNode(
                node_id=node_id,
                cycle=cycle,
                stage=str(record.get("stage") or "resume"),
                parent_id=str(record.get("parent_id") or ""),
                xc=xc,
                candidate=candidate,
                task_evaluation=task_evaluation,
                rationale=str(record.get("rationale") or ""),
                action_detail=dict(record.get("action_detail") or {}),
                eval_duration_s=record.get("eval_duration_s"),
                nearest_node_id=str(record.get("nearest_node_id") or ""),
                nearest_delta=record.get("nearest_delta"),
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
            previous="Loaded basin resume history.",
            max_lines=self.config.digest_max_lines,
        )
        self.failure_digest = _deterministic_digest(
            "failure",
            failures,
            previous="Loaded basin resume history.",
            max_lines=self.config.digest_max_lines,
        )
        return through_cycle if through_cycle is not None else max_cycle

    def _underexplored_directions(self) -> str:
        groups: dict[str, list[BasinNode]] = {}
        for node in self.nodes:
            if node.stage == "root":
                continue
            groups.setdefault(_scaffold_key(node.xc), []).append(node)
        candidates: list[tuple[float, str]] = []
        for scaffold, nodes in groups.items():
            finite_nodes = [node for node in nodes if math.isfinite(node.raw_reward)]
            if not finite_nodes:
                continue
            best = max(finite_nodes, key=lambda item: item.raw_reward)
            best_reward = best.raw_reward
            success_count = sum(node.is_success for node in nodes)
            local_ea_count = sum(node.stage == "local_ea" for node in nodes)
            proposal_count = sum(node.stage == "proposal" for node in nodes)
            reasons: list[str] = []
            promising = bool(best.candidate.promotable and (best.is_success or best.stage == "root"))
            if len(nodes) < 3 and promising:
                reasons.append("sparse scaffold evidence")
            if self.config.enable_local_ea and local_ea_count == 0 and promising:
                reasons.append("no local EA refinement yet")
            if promising and len(nodes) < 8:
                reasons.append("promising but shallow coverage")
            if promising and len(nodes) >= 3 and _parameter_span_count(nodes) < 3:
                reasons.append("high reward with narrow parameter coverage")
            if not reasons:
                continue
            priority = best_reward + 0.05 * success_count - 0.01 * len(nodes)
            candidates.append((priority, "\n".join([
                f"- scaffold: {scaffold}",
                f"  evidence: evaluated={len(nodes)}, proposals={proposal_count}, local_ea={local_ea_count}, successes={success_count}",
                f"  best: {best.node_id} raw_reward={_fmt(best_reward)} worst_guard={_fmt(best.candidate.worst_validation_ratio)}",
                f"  best_metrics: {_node_result_line(best)}",
                f"  tried_ranges: {_parameter_range_summary(nodes)}",
                f"  why_underexplored: {', '.join(reasons)}",
                "  instruction: propose a genuinely different basin point from an eligible parent; avoid only repeating the tried parameter range unless changing the component scaffold.",
            ])))
        if not candidates:
            return "Triggered because the previous batch produced fewer than two novel accepted points, but no promising underexplored scaffold was found by the deterministic coverage check. Prefer a new component scaffold or a clearly different alpha/beta/omega region from the recent nodes."
        candidates.sort(key=lambda item: item[0], reverse=True)
        header = "Triggered because the previous batch produced fewer than two novel accepted points. Use these coverage gaps to avoid repeating dense or already-rejected regions."
        return header + "\n" + "\n".join(item for _score, item in candidates[:8])

    def _proposal_context_nodes(self, *, cycle: int, cycle_evaluated: list[BasinNode]) -> list[BasinNode]:
        """Single compact history block: current proposals plus sampled previous-cycle EA groups."""
        selected: list[BasinNode] = []
        seen: set[str] = set()

        def add_many(items: list[BasinNode]) -> None:
            for node in items:
                if node.node_id in seen:
                    continue
                seen.add(node.node_id)
                selected.append(node)

        add_many([node for node in cycle_evaluated if node.stage == "proposal"])
        previous_cycle = int(cycle) - 1
        if previous_cycle >= 1:
            groups: dict[str, list[BasinNode]] = {}
            for node in self.nodes:
                if node.cycle == previous_cycle and node.stage == "local_ea" and node.parent_id:
                    groups.setdefault(node.parent_id, []).append(node)
            ranked_groups = sorted(
                groups.items(),
                key=lambda item: max((node.raw_reward for node in item[1]), default=float("-inf")),
                reverse=True,
            )
            for _seed_id, group_nodes in ranked_groups[:2]:
                add_many(_reward_spread_sample(group_nodes, self.config.prompt_history_per_ea_group))
        return selected

    def _proposal_context_breakdown(self, *, cycle: int, cycle_evaluated: list[BasinNode]) -> dict[str, Any]:
        previous_cycle = int(cycle) - 1
        groups: dict[str, list[BasinNode]] = {}
        if previous_cycle >= 1:
            for node in self.nodes:
                if node.cycle == previous_cycle and node.stage == "local_ea" and node.parent_id:
                    groups.setdefault(node.parent_id, []).append(node)
        sampled_groups = []
        for seed_id, group_nodes in sorted(
            groups.items(),
            key=lambda item: max((node.raw_reward for node in item[1]), default=float("-inf")),
            reverse=True,
        )[:2]:
            sampled = _reward_spread_sample(group_nodes, self.config.prompt_history_per_ea_group)
            sampled_groups.append(
                {
                    "ea_seed_parent_id": seed_id,
                    "sampled_nodes": [node.node_id for node in sampled],
                    "reward_range": [
                        min((node.raw_reward for node in group_nodes), default=None),
                        max((node.raw_reward for node in group_nodes), default=None),
                    ],
                }
            )
        return {
            "current_cycle_proposals": [node.node_id for node in cycle_evaluated if node.stage == "proposal"],
            "previous_cycle_local_ea_groups": sampled_groups,
        }

    def _success_parent_pool(self) -> list[BasinNode]:
        successes = [node for node in self.nodes if node.is_success]
        ranked = sorted(
            successes,
            key=lambda item: (item.raw_reward, -item.candidate.worst_validation_ratio),
            reverse=True,
        )
        selected: list[BasinNode] = []
        selected_by_group: dict[str, list[BasinNode]] = {}
        selected_directions_by_group: dict[str, set[str]] = {}
        skipped_by_group: dict[str, int] = {}
        for node in ranked:
            group_key = self._parent_pool_group_key(node)
            group_selected = selected_by_group.setdefault(group_key, [])
            if len(group_selected) >= self.config.parent_pool_max_per_seed:
                skipped_by_group[group_key] = skipped_by_group.get(group_key, 0) + 1
                continue
            direction_key = self._parent_pool_direction_key(node)
            group_directions = selected_directions_by_group.setdefault(group_key, set())
            if direction_key in group_directions:
                skipped_by_group[group_key] = skipped_by_group.get(group_key, 0) + 1
                continue
            selected.append(node)
            group_selected.append(node)
            group_directions.add(direction_key)
            if len(selected) >= self.config.parent_pool_size:
                break
        if selected:
            group_counts = {key: len(value) for key, value in selected_by_group.items() if value}
            logger.info(
                "BASIN_PARENT_POOL selected=%s group_counts=%s skipped_by_group=%s",
                [node.node_id for node in selected],
                group_counts,
                {key: value for key, value in skipped_by_group.items() if value},
            )
        return selected

    def _parent_pool_group_key(self, node: BasinNode) -> str:
        if node.stage == "local_ea" and node.parent_id:
            return node.parent_id
        return node.node_id

    def _parent_pool_direction_key(self, node: BasinNode) -> str:
        if node.stage != "local_ea" or not node.parent_id:
            return f"proposal:{node.node_id}"
        parent = self._nodes_by_id.get(node.parent_id)
        if parent is None:
            return f"local:{node.node_id}"
        deltas = _signed_parameter_directions(node.xc, parent.xc)
        if not deltas:
            return "local:no-material-direction"
        return "|".join(deltas[:4])

    def _diverse_top(self, nodes: list[BasinNode]) -> list[BasinNode]:
        ranked = sorted(nodes, key=lambda item: item.raw_reward, reverse=True)
        if not self.config.enforce_novelty_threshold:
            return ranked
        selected: list[BasinNode] = []
        for node in ranked:
            if not selected:
                selected.append(node)
                continue
            if all(_novelty_delta(node.xc, other.xc) >= self.config.novelty_threshold for other in selected):
                selected.append(node)
        for node in ranked:
            if node not in selected:
                selected.append(node)
        return selected

    def _nearest_existing(
        self,
        xc: XCFunctional,
        *,
        exclude_node_id: str = "",
    ) -> tuple[str, float | None]:
        if not self.nodes:
            return "", None
        best_id = ""
        best_delta = float("inf")
        for node in self.nodes:
            if node.node_id == exclude_node_id:
                continue
            delta = _novelty_delta(xc, node.xc)
            if delta < best_delta:
                best_delta = delta
                best_id = node.node_id
        if not best_id:
            return "", None
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
        prompt_label: str = "",
        register_signature: bool = True,
        task_evaluation: SearchTaskEvaluation | None = None,
    ) -> BasinNode:
        node_id = self._allocate_node_id()
        node = BasinNode(
            node_id=node_id,
            cycle=cycle,
            stage=stage,
            parent_id=parent_id,
            xc=xc,
            candidate=candidate,
            task_evaluation=task_evaluation,
            rationale=rationale,
            action_detail=action_detail,
            prompt_label=prompt_label,
        )
        self.nodes.append(node)
        self._nodes_by_id[node_id] = node
        if register_signature:
            self._effective_signatures.add(xc.effective_signature())
        return node

    def _discard_node(self, node: BasinNode) -> None:
        """Remove an un-evaluated rejected proposal from prompt-visible state."""
        self.nodes = [item for item in self.nodes if item.node_id != node.node_id]
        self._nodes_by_id.pop(node.node_id, None)

    def _allocate_node_id(self) -> str:
        node_id = f"N{self._next_id:05d}"
        self._next_id += 1
        return node_id

    def _peek_next_node_id(self) -> str:
        return f"N{self._next_id:05d}"

    def _read_parameters(self, xc: XCFunctional) -> dict[str, float]:
        parameters = self.scorer._parameter_space(xc)
        return self.scorer._read_parameters(xc, parameters) if parameters else {}

    def _write_artifacts(
        self,
        output_dir: Path,
        cycle_summaries: list[dict[str, Any]],
        *,
        knowledge: KnowledgeEngine | None,
        update_graph: bool = False,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        nodes_path = output_dir / "basin_agent_nodes.json"
        nodes_path.write_text(
            json.dumps([node.to_dict() for node in self.nodes], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "basin_agent_summary.json").write_text(
            json.dumps(
                {
                    "cycle_summaries": cycle_summaries,
                    "success_digest": self.success_digest,
                    "failure_digest": self.failure_digest,
                    "best": self.nodes and max(self.nodes, key=lambda item: EvolutionarySearch._rank_key(item.candidate)).to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._write_csv(output_dir / "basin_agent_nodes.csv")
        self._write_core_artifacts(output_dir)
        if not update_graph or knowledge is None or not self.config.update_functional_graph:
            return
        try:
            graph_report = knowledge.ingest_basin_agent_graph(
                nodes_path,
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Basin functional-graph update failed: %s", exc)
            raise
        else:
            if graph_report["nodes_added"] or graph_report["edges_added"]:
                logger.info(
                    "Basin functional graph updated: nodes_added=%d edges_added=%d "
                    "skipped_invalid_nodes=%d skipped_edges=%d",
                    graph_report["nodes_added"],
                    graph_report["edges_added"],
                    graph_report["skipped_invalid_nodes"],
                    graph_report["skipped_edges"],
                )

    def _write_core_artifacts(self, output_dir: Path) -> None:
        if not self.config.core_output_dir:
            return
        core_dir = Path(self.config.core_output_dir)
        core_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "basin_agent_nodes.json",
            "basin_agent_nodes.csv",
            "basin_agent_summary.json",
        ):
            src = output_dir / filename
            if src.exists():
                shutil.copy2(src, core_dir / filename)

    def _write_csv(self, path: Path) -> None:
        subset_names = sorted({
            name
            for node in self.nodes
            for benchmark in (node.candidate.search_result, node.candidate.validation_result)
            if benchmark is not None
            for name in _subset_names_from_result(benchmark)
        })
        parameter_names = sorted({key for node in self.nodes for key in node.candidate.parameters})
        fields = [
            "node_id",
            "cycle",
            "stage",
            "parent_id",
            "success_vs_parent",
            "parent_raw_reward",
            "raw_reward",
            "tree_reward",
            "promotable",
            "failure_reason",
            "guard_penalty",
            "worst_validation_ratio",
            "eval_duration_s",
            "nearest_node_id",
            "nearest_delta",
            "name",
            "effective_signature",
            *parameter_names,
        ]
        for subset in subset_names:
            fields.extend([f"{subset}_mae", f"{subset}_failures"])
        fields.extend(["action_detail_json", "xc_json"])
        fields.extend(["task_objective_json", "proposal_feedback_json"])
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for node in sorted(self.nodes, key=lambda item: item.node_id):
                row: dict[str, Any] = {
                    "node_id": node.node_id,
                    "cycle": node.cycle,
                    "stage": node.stage,
                    "parent_id": node.parent_id,
                    "success_vs_parent": node.is_success,
                    "parent_raw_reward": node.candidate.xc.metadata.get("basin_parent_reward"),
                    "raw_reward": node.raw_reward,
                    "tree_reward": node.candidate.tree_reward,
                    "promotable": node.candidate.promotable,
                    "failure_reason": node.candidate.failure_reason,
                    "guard_penalty": node.candidate.guard_penalty,
                    "worst_validation_ratio": node.candidate.worst_validation_ratio,
                    "eval_duration_s": node.eval_duration_s,
                    "nearest_node_id": node.nearest_node_id,
                    "nearest_delta": node.nearest_delta,
                    "name": node.xc.name,
                    "effective_signature": node.xc.effective_signature(),
                    "action_detail_json": json.dumps(node.action_detail, sort_keys=True, default=str),
                    "xc_json": json.dumps(node.xc.to_dict(), sort_keys=True, default=str),
                    "task_objective_json": json.dumps(
                        node.task_evaluation.objective if node.task_evaluation else {},
                        sort_keys=True,
                        default=str,
                    ),
                    "proposal_feedback_json": json.dumps(
                        node.task_evaluation.proposal_feedback if node.task_evaluation else {},
                        sort_keys=True,
                        default=str,
                    ),
                    **node.candidate.parameters,
                }
                for subset in subset_names:
                    row[f"{subset}_mae"] = (
                        _subset_metric_from_result(node.candidate.search_result, subset, "mae_kcal")
                        if _subset_metric_from_result(node.candidate.search_result, subset, "mae_kcal") is not None
                        else _subset_metric_from_result(node.candidate.validation_result, subset, "mae_kcal")
                    )
                    row[f"{subset}_failures"] = (
                        _subset_metric_from_result(node.candidate.search_result, subset, "n_failures")
                        if _subset_metric_from_result(node.candidate.search_result, subset, "n_failures") is not None
                        else _subset_metric_from_result(node.candidate.validation_result, subset, "n_failures")
                    )
                writer.writerow(row)

    def _llm_json(self, idea_agent: IdeaAgent, prompt: str, label: str, *, temperature: float) -> dict[str, Any]:
        import httpx
        import openai

        client = openai.OpenAI(
            api_key=idea_agent.api_key or "dummy",
            base_url=idea_agent.base_url,
            timeout=getattr(idea_agent, "llm_timeout_s", 120.0),
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        )
        started = time.time()
        response = idea_agent._with_llm_retries(
            lambda: client.chat.completions.create(
                model=idea_agent.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=False,
            ),
            openai_module=openai,
            operation=label,
        )
        content = response.choices[0].message.content or ""
        logger.info(
            "%s",
            log_block(
                f"LLM basin response ({label}, chars={len(content)}, wall={time.time() - started:.1f}s)",
                pretty_json_or_text(content),
                color="cyan",
            ),
        )
        malformed_content = _normalize_json_string_expressions(content)
        try:
            return _extract_json_object(malformed_content)
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error: Exception = exc

        max_repair_attempts = 3
        for repair_attempt in range(1, max_repair_attempts + 1):
            logger.warning(
                "LLM basin response returned invalid JSON during %s; requesting JSON-only repair %d/%d: %s",
                label,
                repair_attempt,
                max_repair_attempts,
                parse_error,
            )
            repair_started = time.time()
            repair_prompt = (
                f"JSON parser error: {parse_error}\n\n"
                "Malformed response follows:\n"
                f"{malformed_content}"
            )
            repair = idea_agent._with_llm_retries(
                lambda: client.chat.completions.create(
                    model=idea_agent.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Repair malformed JSON. Return only one valid JSON object with no markdown. "
                                "Use literal JSON values only: no comments, function calls, string methods such as "
                                ".replace(), or Python expressions. Do not change, add, remove, reorder, summarize, "
                                "or reinterpret any scientific content."
                            ),
                        },
                        {"role": "user", "content": repair_prompt},
                    ],
                    temperature=0.0,
                    stream=False,
                ),
                openai_module=openai,
                operation=f"{label} JSON repair {repair_attempt}/{max_repair_attempts}",
            )
            repaired_content = repair.choices[0].message.content or ""
            logger.info(
                "%s",
                log_block(
                    f"LLM basin JSON repair response ({label}, attempt={repair_attempt}/{max_repair_attempts}, "
                    f"chars={len(repaired_content)}, wall={time.time() - repair_started:.1f}s)",
                    pretty_json_or_text(repaired_content),
                    color="yellow",
                ),
            )
            malformed_content = _normalize_json_string_expressions(repaired_content)
            try:
                return _extract_json_object(malformed_content)
            except (json.JSONDecodeError, ValueError) as exc:
                parse_error = exc

        raise ValueError(
            f"LLM basin response remained invalid JSON after {max_repair_attempts} repair attempts during "
            f"{label}: {parse_error}"
        )



_JSON_STRING_REPLACE_RE = re.compile(
    r'("(?:\\.|[^"\\])*")\s*\.replace\(\s*("(?:\\.|[^"\\])*")\s*,\s*'
    r'("(?:\\.|[^"\\])*")\s*\)'
)


def _normalize_json_string_expressions(content: str) -> str:
    """Evaluate only literal JSON-string ``.replace`` calls emitted by an LLM."""

    def _replace(match: re.Match[str]) -> str:
        try:
            value = json.loads(match.group(1))
            old = json.loads(match.group(2))
            new = json.loads(match.group(3))
        except (json.JSONDecodeError, TypeError):
            return match.group(0)
        if not all(isinstance(item, str) for item in (value, old, new)):
            return match.group(0)
        return json.dumps(value.replace(old, new), ensure_ascii=False)

    return _JSON_STRING_REPLACE_RE.sub(_replace, str(content or ""))


def _normalize_basin_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize fragile LLM action schemas before the generic parser sees them."""
    if not isinstance(payload, dict):
        return payload
    normalized = copy.deepcopy(payload)
    action_type = str(normalized.get("action_type") or "").strip()
    changes = normalized.get("changes")
    if not isinstance(changes, dict):
        return normalized

    if action_type == "replace_component":
        flat: dict[str, Any] = {}
        nested = changes.get("replace") or changes.get("replace_component")
        source = nested if isinstance(nested, dict) else changes
        old_value = (
            source.get("old_name")
            or source.get("old_component")
            or source.get("from")
            or changes.get("old_name")
            or changes.get("old_component")
            or changes.get("from")
        )
        new_value = (
            source.get("new_name")
            or source.get("new_component")
            or source.get("to")
            or changes.get("new_name")
            or changes.get("new_component")
            or changes.get("to")
        )
        old_name = _component_name(old_value)
        new_name = _component_name(new_value)
        if old_name:
            flat["old_name"] = old_name
        if new_name:
            flat["new_name"] = new_name
        for key in ("new_xctype", "new_ext_params", "new_ext_param_bounds"):
            if key in source:
                flat[key] = source[key]
            elif key in changes:
                flat[key] = changes[key]
        # Keep numeric edits out of replace_component. Use a separate rewire action.
        normalized["changes"] = flat
        return normalized

    if action_type == "add_component":
        component = changes.get("component") or changes.get("new_component") or changes.get("add")
        if isinstance(component, dict):
            flat = dict(component)
            for key, value in changes.items():
                if key not in {"component", "new_component", "add"}:
                    flat.setdefault(key, value)
            normalized["changes"] = flat
        return normalized

    if action_type == "remove_component":
        if "name" not in changes:
            name = _component_name(changes.get("component") or changes.get("old_component") or changes.get("remove"))
            if name:
                normalized["changes"] = {**changes, "name": name}
        return normalized

    return normalized


def _component_name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("name", "component", "id"):
            if isinstance(value.get(key), str):
                return value[key]
        return ""
    return str(value) if value is not None else ""

def _result_metrics(result: BenchmarkResult | dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        subsets = result.get("subsets")
        if isinstance(subsets, dict):
            return {
                str(name): {
                    "mae_kcal": _safe_get_metric(subset, "mae_kcal"),
                    "n_failures": _safe_get_metric(subset, "n_failures"),
                    "n_reactions": _safe_get_metric(subset, "n_reactions"),
                }
                for name, subset in subsets.items()
                if isinstance(subset, dict)
            }
        materials = result.get("materials")
        if isinstance(materials, dict):
            metrics: dict[str, Any] = {}
            for name, payload in materials.items():
                if not isinstance(payload, dict):
                    continue
                band_gap = payload.get("band_gap") or {}
                metrics[str(name)] = {
                    "gap_ev": _safe_get_metric(band_gap, "gap_ev"),
                    "reference_gap_ev": _safe_get_metric(payload, "reference_gap_ev"),
                    "walltime_s": _safe_get_metric(payload, "walltime_s"),
                    "ok": payload.get("ok"),
                    "stage": payload.get("stage"),
                }
            return metrics
        return result
    return {
        name: {
            "mae_kcal": subset.mae_kcal,
            "n_failures": subset.n_failures,
            "n_reactions": subset.n_reactions,
        }
        for name, subset in result.subsets.items()
    }


def _subset_result(candidate: EvolutionCandidate, subset: str):
    if isinstance(candidate.search_result, BenchmarkResult) and subset in candidate.search_result.subsets:
        return candidate.search_result.subsets[subset]
    if isinstance(candidate.validation_result, BenchmarkResult) and subset in candidate.validation_result.subsets:
        return candidate.validation_result.subsets[subset]
    return None


def _subset_names_from_result(result: BenchmarkResult | dict[str, Any] | None) -> set[str]:
    if result is None:
        return set()
    if isinstance(result, BenchmarkResult):
        return set(result.subsets)
    if isinstance(result, dict):
        if isinstance(result.get("subsets"), dict):
            return {str(key) for key in result["subsets"]}
        if isinstance(result.get("materials"), dict):
            return {str(key) for key in result["materials"]}
    return set()


def _subset_metric_from_result(result: BenchmarkResult | dict[str, Any] | None, subset: str, key: str) -> Any:
    if result is None:
        return None
    if isinstance(result, BenchmarkResult):
        subset_result = result.subsets.get(subset)
        return getattr(subset_result, key, None) if subset_result is not None else None
    if isinstance(result, dict):
        subsets = result.get("subsets")
        if isinstance(subsets, dict) and isinstance(subsets.get(subset), dict):
            return subsets[subset].get(key)
        materials = result.get("materials")
        if isinstance(materials, dict) and isinstance(materials.get(subset), dict):
            payload = materials[subset]
            if key == "mae_kcal":
                return (payload.get("band_gap") or {}).get("gap_ev")
            if key == "n_failures":
                return 0 if payload.get("ok") else 1
    return None


def _safe_get_metric(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _runner_uses_docker_queue(runner: Any) -> bool:
    return str(getattr(runner, "execution_backend", "")) == "slurm_docker_queue" and hasattr(runner, "evaluate_many")


def _reward_spread_sample(nodes: list[BasinNode], limit: int) -> list[BasinNode]:
    """Pick worst, middle, and best points so prompt history is not top-only."""
    if not nodes:
        return []
    limit = max(1, int(limit))
    ranked = sorted(nodes, key=lambda item: (item.raw_reward, item.node_id))
    if len(ranked) <= limit:
        return ranked
    if limit == 1:
        return [ranked[-1]]
    indices = [round(idx * (len(ranked) - 1) / (limit - 1)) for idx in range(limit)]
    selected: list[BasinNode] = []
    seen: set[int] = set()
    for index in indices:
        index = max(0, min(len(ranked) - 1, int(index)))
        if index in seen:
            continue
        seen.add(index)
        selected.append(ranked[index])
    for node in ranked:
        if len(selected) >= limit:
            break
        if node not in selected:
            selected.append(node)
    return selected


def _component_alias_maps(component_pool: str) -> tuple[dict[str, str], dict[str, str]]:
    roles = component_roles_for_pool(component_pool)
    counters = {"exchange": 0, "correlation": 0}
    true_to_alias: dict[str, str] = {}
    alias_to_true: dict[str, str] = {}
    for name, role in sorted(roles.items(), key=lambda item: (item[1], item[0])):
        if role not in counters:
            continue
        counters[role] += 1
        prefix = "X" if role == "exchange" else "C"
        alias = f"{prefix}{counters[role]:03d}"
        true_to_alias[name] = alias
        alias_to_true[alias] = name
    return true_to_alias, alias_to_true


def _first_alias_for_role(component_aliases: dict[str, str], role: str, *, exclude: str = "") -> str:
    prefix = "X" if role == "exchange" else "C"
    aliases = sorted(alias for alias in component_aliases.values() if alias.startswith(prefix))
    if exclude:
        aliases = [alias for alias in aliases if alias != exclude]
    return aliases[0] if aliases else f"{prefix}001"


def _format_anonymized_component_catalogue(component_aliases: dict[str, str]) -> str:
    items = []
    for _name, alias in sorted(component_aliases.items(), key=lambda item: item[1]):
        role = "exchange" if alias.startswith("X") else "correlation"
        items.append({"id": alias, "role": role})
    return json.dumps(
        {
            "schema": "anonymous-component-catalogue-v1",
            "description": (
                "Component identities are blinded for this control. "
                "Use these anonymous IDs exactly; X*** are exchange components and C*** are correlation components."
            ),
            "components": items,
        },
        indent=2,
        ensure_ascii=False,
    )


def _component_prompt_name(name: str, component_aliases: dict[str, str] | None) -> str:
    if not component_aliases:
        return name
    return component_aliases.get(name, name)


def _mask_parameter_keys(payload: dict[str, Any], component_aliases: dict[str, str] | None) -> dict[str, Any]:
    if not component_aliases:
        return dict(payload)
    masked: dict[str, Any] = {}
    for key, value in dict(payload).items():
        masked[_mask_parameter_key(str(key), component_aliases)] = value
    return masked


def _mask_parameter_key(key: str, component_aliases: dict[str, str]) -> str:
    if key.startswith("mixing:"):
        name = key.split(":", 1)[1]
        return f"mixing:{component_aliases.get(name, name)}"
    for name, alias in component_aliases.items():
        if key.startswith(f"{name}."):
            return f"{alias}.{key.split('.', 1)[1]}"
    return key


def _mask_action_detail(payload: Any, component_aliases: dict[str, str] | None) -> Any:
    if not component_aliases:
        return payload
    if isinstance(payload, list):
        return [_mask_action_detail(item, component_aliases) for item in payload]
    if isinstance(payload, dict):
        masked: dict[str, Any] = {}
        for key, value in payload.items():
            masked_key = _mask_parameter_key(str(key), component_aliases)
            if key in {"name", "old_name", "new_name"} and isinstance(value, str):
                masked[masked_key] = component_aliases.get(value, value)
            elif key == "changes" and isinstance(value, dict):
                masked[masked_key] = _mask_action_detail(value, component_aliases)
            else:
                masked[masked_key] = _mask_action_detail(value, component_aliases)
        return masked
    if isinstance(payload, str):
        return _mask_component_text(payload, component_aliases)
    return payload


def _mask_component_text(text: str, component_aliases: dict[str, str] | None) -> str:
    if not component_aliases or not text:
        return str(text or "")
    masked = str(text)
    for name, alias in sorted(component_aliases.items(), key=lambda item: len(item[0]), reverse=True):
        masked = masked.replace(name, alias)
    return masked


def _translate_anonymized_action_payload(payload: dict[str, Any], alias_to_component: dict[str, str]) -> dict[str, Any]:
    translated = copy.deepcopy(payload)
    changes = translated.get("changes")
    if isinstance(changes, dict):
        translated["changes"] = _translate_anonymized_changes(changes, alias_to_component)
    return translated


def _translate_anonymized_changes(changes: dict[str, Any], alias_to_component: dict[str, str]) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    for key, value in changes.items():
        new_key = str(key)
        if new_key.startswith("mixing:"):
            alias = new_key.split(":", 1)[1]
            new_key = f"mixing:{alias_to_component.get(alias, alias)}"
        elif "." in new_key:
            alias, suffix = new_key.split(".", 1)
            if alias in alias_to_component:
                new_key = f"{alias_to_component[alias]}.{suffix}"
        if key in {"name", "old_name", "new_name"} and isinstance(value, str):
            translated[new_key] = alias_to_component.get(value, value)
        elif isinstance(value, dict):
            translated[new_key] = _translate_anonymized_changes(value, alias_to_component)
        else:
            translated[new_key] = value
    return translated


def _format_component_prior_text(priors: list[Any], *, limit: int) -> str:
    if not priors:
        return "No component benchmark priors available."
    lines = [
        "## Component Benchmark Priors (Associative, Not Causal)",
        "These components occur inside previously benchmarked whole functionals. Use them as weak hints, not proof that the component alone caused the result.",
    ]
    for prior in priors[: max(1, int(limit))]:
        guards = ", ".join(
            f"{subset}={float(mae):.2f}"
            for subset, mae in sorted(getattr(prior, "guard_metrics", {}).items())
        ) or "n/a"
        lines.append(
            f"- {prior.component_name} ({prior.role}): best_associated_functional={prior.best_functional}, "
            f"target_mae={prior.target_mae:.2f}, target_gain_vs_current={prior.target_gain:+.2f}, "
            f"association_score={prior.association_score:+.2f}, guards={guards}, "
            f"target_failures={prior.target_failures}, total_failures={prior.failures}, "
            f"observed_in={prior.occurrences} functionals"
        )
    return "\n".join(lines)


def _node_prompt_brief(node: BasinNode, *, component_aliases: dict[str, str] | None = None) -> dict[str, Any]:
    payload = {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "cycle": node.cycle,
        "stage": node.stage,
        "success_vs_parent": node.is_success,
        "parent_raw_reward": node.candidate.xc.metadata.get("basin_parent_reward"),
        "raw_reward": node.raw_reward,
        "tree_reward": node.candidate.tree_reward,
        "failure_reason": _mask_component_text(node.candidate.failure_reason, component_aliases),
        "worst_guard": node.candidate.worst_validation_ratio,
        "parameters": _mask_parameter_keys(node.candidate.parameters, component_aliases),
        "components": [
            (
                {
                    "name": _component_prompt_name(component.name, component_aliases),
                    "role": component.role,
                    "mixing": component.mixing_factor,
                }
                if component_aliases
                else {
                    "name": component.name,
                    "role": component.role,
                    "mixing": component.mixing_factor,
                    "ext_params": component.ext_params,
                }
            )
            for component in node.xc.components
        ],
        "hf_exchange": node.xc.hf_exchange.to_dict() if hasattr(node.xc.hf_exchange, "to_dict") else {
            "alpha": node.xc.hf_exchange.alpha,
            "beta": node.xc.hf_exchange.beta,
            "omega": node.xc.hf_exchange.omega,
        },
        "search": _result_metrics(node.candidate.search_result),
        "guard": _result_metrics(node.candidate.validation_result),
        "rationale": _mask_component_text(node.rationale, component_aliases),
    }
    if node.task_evaluation is not None:
        payload["task_evaluation"] = _mask_action_detail(
            node.task_evaluation.to_dict(),
            component_aliases,
        )
    return payload


def _node_reflection_brief(node: BasinNode, *, component_aliases: dict[str, str] | None = None) -> dict[str, Any]:
    """Keep every cycle node visible while avoiding duplicated XC serialization."""
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "success_vs_parent": node.is_success,
        "raw_reward": node.raw_reward,
        "rationale": _mask_component_text(node.rationale, component_aliases),
        "actions": _mask_action_detail(node.action_detail.get("actions", node.action_detail), component_aliases),
        "components": [
            {
                "name": _component_prompt_name(component.name, component_aliases),
                "role": component.role,
                "mixing": component.mixing_factor,
            }
            for component in node.xc.components
        ],
        "task_evaluation": (
            _mask_action_detail(node.task_evaluation.to_dict(), component_aliases)
            if node.task_evaluation
            else None
        ),
    }


def _subset_mae(candidate: EvolutionCandidate, subset: str) -> str:
    sr = _subset_result(candidate, subset)
    return f"{sr.mae_kcal:.3f}" if sr else "n/a"



def _node_result_metrics(node: BasinNode) -> dict[str, Any]:
    if node.task_evaluation is not None:
        return {
            "task": node.task_evaluation.task_name,
            "valid": node.task_evaluation.valid,
            "reward": node.task_evaluation.reward,
            "objective": node.task_evaluation.objective,
            "proposal_feedback": node.task_evaluation.proposal_feedback,
            "failure_reason": node.task_evaluation.failure_reason,
            "walltime_s": node.task_evaluation.walltime_s,
        }
    metrics: dict[str, Any] = {}
    for result in (node.candidate.search_result, node.candidate.validation_result):
        if result is None:
            continue
        for name, subset in _result_metrics(result).items():
            if not isinstance(subset, dict):
                metrics[name] = subset
                continue
            mae = subset.get("mae_kcal")
            gap = subset.get("gap_ev")
            metrics[name] = {
                "mae": None if mae is None else round(float(mae), 4),
                "gap_ev": None if gap is None else round(float(gap), 4),
                "failures": subset.get("n_failures"),
                "ok": subset.get("ok"),
                "stage": subset.get("stage"),
            }
    metrics["worst_guard"] = None if not math.isfinite(node.candidate.worst_validation_ratio) else round(float(node.candidate.worst_validation_ratio), 4)
    metrics["failure_reason"] = node.candidate.failure_reason
    return metrics

def _node_result_line(node: BasinNode | None) -> str:
    if node is None:
        return "unknown"
    task_text = ""
    if node.task_evaluation is not None:
        task_text = f" task={json.dumps(node.task_evaluation.to_dict(), ensure_ascii=False, sort_keys=True)}"
    return (
        f"{node.node_id} raw_reward={_fmt(node.raw_reward)} "
        f"success={node.is_success} search={_result_metrics(node.candidate.search_result)} "
        f"guard={_result_metrics(node.candidate.validation_result)}{task_text}"
    )


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


def _component_set(xc: XCFunctional) -> set[tuple[str, str]]:
    return {(component.role, component.name) for component in xc.components}


def _benchmark_result_from_node_metrics(xc_name: str, payload: dict[str, Any]) -> BenchmarkResult | None:
    if not payload:
        return None

    def _float_metric(metrics: dict[str, Any], primary: str, fallback: str) -> float:
        value = metrics.get(primary)
        if value is None:
            value = metrics.get(fallback)
        return 0.0 if value is None else float(value)

    def _int_metric(metrics: dict[str, Any], primary: str, fallback: str) -> int:
        value = metrics.get(primary)
        if value is None:
            value = metrics.get(fallback)
        return 0 if value is None else int(value)

    result = BenchmarkResult(xc_name=xc_name)
    for subset_name, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        subset = str(metrics.get("subset") or subset_name)
        result.subsets[subset] = SubsetResult(
            subset=subset,
            mae_kcal=_float_metric(metrics, "mae_kcal", "mae"),
            rms_kcal=_float_metric(metrics, "rms_kcal", "rms"),
            max_error_kcal=_float_metric(metrics, "max_error_kcal", "max"),
            n_reactions=_int_metric(metrics, "n_reactions", "n"),
            n_failures=_int_metric(metrics, "n_failures", "failures"),
            walltime_s=_float_metric(metrics, "walltime_s", "wall"),
            failure_details=[str(item) for item in (metrics.get("failure_details") or [])],
        )
    if not result.subsets:
        return None
    result.compute_overall(aggregate_mode="subset_mean")
    return result


def _scaffold_key(xc: XCFunctional) -> str:
    parts = []
    for component in sorted(xc.components, key=lambda item: (item.role, item.name)):
        parts.append(f"{component.role}:{component.name}")
    return "; ".join(parts) or "empty-scaffold"


def _parameter_range_summary(nodes: list[BasinNode]) -> str:
    keys = [
        "alpha",
        "beta",
        "omega",
        "scale:exchange",
        "scale:correlation",
        "mixing:GGA_C_PBE",
        "mixing:GGA_C_PBE_SOL",
        "mixing:GGA_C_P86",
        "mixing:GGA_C_APBE",
        "mixing:MGGA_C_TPSS",
        "mixing:MGGA_C_SCAN",
    ]
    maps = [_parameter_map(node.xc) for node in nodes]
    parts: list[str] = []
    for key in keys:
        values = [float(item[key]) for item in maps if key in item]
        if not values:
            continue
        lo = min(values)
        hi = max(values)
        parts.append(f"{key}={lo:.3f}" if abs(hi - lo) < 1e-9 else f"{key}=[{lo:.3f},{hi:.3f}]")
    return "; ".join(parts) or "no numeric range summary"


def _parameter_span_count(nodes: list[BasinNode]) -> int:
    maps = [_parameter_map(node.xc) for node in nodes]
    keys = set().union(*(item.keys() for item in maps)) if maps else set()
    count = 0
    for key in keys:
        values = [float(item[key]) for item in maps if key in item]
        if len(values) >= 2 and max(values) - min(values) >= _direction_threshold(key):
            count += 1
    return count


def _signed_parameter_directions(left: XCFunctional, right: XCFunctional) -> list[str]:
    left_values = _parameter_map(left)
    right_values = _parameter_map(right)
    changes: list[tuple[float, str]] = []
    for key in sorted(set(left_values) | set(right_values)):
        delta = float(left_values.get(key, 0.0)) - float(right_values.get(key, 0.0))
        threshold = _direction_threshold(key)
        if abs(delta) < threshold:
            continue
        sign = "+" if delta > 0.0 else "-"
        changes.append((abs(delta), f"{key}:{sign}"))
    changes.sort(reverse=True)
    return [item for _magnitude, item in changes]


def _direction_threshold(key: str) -> float:
    if key in {"alpha", "beta", "omega"}:
        return 0.01
    if key.startswith("scale:"):
        return 0.025
    if key.startswith("mixing:"):
        return 0.025
    if key.endswith("._d") or key.endswith(".d"):
        return 0.05
    return 0.005


def _node_ref(node: BasinNode) -> str:
    source = "EA" if node.stage == "local_ea" else node.stage
    parent = node.parent_id or "none"
    return f"{node.node_id}(parent:{parent},source:{source},reward:{_fmt(node.raw_reward)},success:{node.is_success})"


def _format_node_refs(nodes: list[BasinNode]) -> str:
    if not nodes:
        return "None."
    return "[" + ", ".join(_node_ref(node) for node in nodes) + "]"


def _novelty_delta(left: XCFunctional, right: XCFunctional) -> float:
    if _component_set(left) != _component_set(right):
        return 1.0
    left_values = _parameter_map(left)
    right_values = _parameter_map(right)
    keys = set(left_values) | set(right_values)
    if not keys:
        return 0.0
    return max(abs(left_values.get(key, 0.0) - right_values.get(key, 0.0)) for key in keys)

def _deterministic_digest(kind: str, nodes: list[BasinNode], *, previous: str, max_lines: int = 48) -> str:
    ranked = sorted(nodes, key=lambda item: item.raw_reward, reverse=True)[:12]
    lines = [previous, f"Recent {kind} nodes:"]
    for node in ranked:
        lines.append(
            f"- {node.node_id}: reward={_fmt(node.raw_reward)} parent={node.parent_id} "
            f"params={json.dumps(node.candidate.parameters, sort_keys=True)}"
        )
    return "\n".join(lines[-max(8, int(max_lines)):])
