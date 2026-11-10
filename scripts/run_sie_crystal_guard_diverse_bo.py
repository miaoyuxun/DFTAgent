#!/usr/bin/env python3
"""Run a diverse-BO baseline on the SIE4x4 + ZnS/MgO guard task."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xc.benchmark.runner import BenchmarkRunner  # noqa: E402
from xc.benchmark.vc_manifest_runner import (  # noqa: E402
    VCManifestBenchmarkRunner,
    VCManifestRunnerConfig,
)
from xc.idea.basin_agent import (  # noqa: E402
    BasinAgentResult,
    BasinNode,
    LocalEvolutionarySearch,
    _deterministic_digest,
)
from xc.idea.diverse_bo_basin import DiverseBOBasinConfig, DiverseBOBasinSearch  # noqa: E402
from xc.idea.evolutionary import (  # noqa: E402
    EvolutionCandidate,
    EvolutionConfig,
    EvolutionarySearch,
    EvolutionSearchResult,
)
from xc.search_tasks import (  # noqa: E402
    CrystalGuardSlurmDockerEvaluator,
    CrystalGuardSlurmEvaluatorConfig,
    CrystalGuardVCEvaluator,
    CrystalGuardVCEvaluatorConfig,
    CrystalGuardPersistentVCEvaluator,
    CrystalGuardPersistentVCEvaluatorConfig,
    SIECrystalGuardSearchTask,
    SIECrystalGuardTaskConfig,
    make_mtask_pw92_functional,
)
from xc.search_tasks.base import SearchTaskEvaluation  # noqa: E402
from xc.xc_functional import XCFunctional  # noqa: E402
from xc.zno_component_registry import validate_component_pool_lock  # noqa: E402


logger = logging.getLogger(__name__)


def _completed_bo_batch_count(nodes: list[BasinNode], *, batch_size: int) -> int:
    """Recover the last completed batch number for an interrupted cycle."""
    batch_numbers: list[int] = []
    for node in nodes:
        label = str((node.action_detail or {}).get("label") or "")
        marker = "_batch"
        if marker not in label:
            continue
        suffix = label.rsplit(marker, 1)[1].lstrip("_")
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            batch_numbers.append(int(digits))
    if batch_numbers:
        return max(batch_numbers)
    return math.ceil(len(nodes) / max(1, int(batch_size)))


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _resolve_image(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("$"):
        return os.environ.get(value[1:].strip("{}"), "")
    return os.path.expandvars(value) or os.environ.get("DFTAGENT_IMAGE", "")


def _resolve_env_value(value: str, *, fallback_env: str) -> str:
    value = str(value or "").strip()
    if value.startswith("$"):
        return os.environ.get(value[1:].strip("{}"), "")
    return os.path.expandvars(value) or os.environ.get(fallback_env, "")


def _apply_evaluation_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    evaluation = dict(config.get("evaluation") or {})
    if args.slurm_nodelist:
        evaluation["nodelist"] = str(args.slurm_nodelist).strip()
    if args.slurm_nodes is not None:
        evaluation["nodes"] = int(args.slurm_nodes)
    if args.slurm_ntasks_per_node is not None:
        evaluation["ntasks_per_node"] = int(args.slurm_ntasks_per_node)
    if args.slurm_cpus_per_node is not None:
        evaluation["cpus_per_node"] = int(args.slurm_cpus_per_node)
    if args.slurm_total_cpus is not None:
        evaluation["total_cpus"] = int(args.slurm_total_cpus)
    if args.max_concurrent is not None:
        evaluation["max_concurrent"] = int(args.max_concurrent)
    if args.slurm_extra_sbatch:
        existing = list(evaluation.get("slurm_extra_args") or [])
        existing.extend(str(item).strip() for item in args.slurm_extra_sbatch if str(item).strip())
        evaluation["slurm_extra_args"] = existing
    config["evaluation"] = evaluation


def _slurm_placement_args(common_eval: dict) -> tuple[str, list[str], list[str]]:
    raw_extra = list(common_eval.get("slurm_extra_args") or [])
    nodelist = str(common_eval.get("nodelist") or common_eval.get("slurm_nodelist") or "").strip()
    benchmark_extra: list[str] = []
    crystal_extra_sbatch: list[str] = []
    for item in raw_extra:
        text = str(item).strip()
        if not text:
            continue
        if text.startswith("--nodelist="):
            nodelist = text.split("=", 1)[1].strip()
            continue
        if text in {"--nodelist", "-w"}:
            logger.warning("Ignoring incomplete nodelist extra arg %r; use --slurm-nodelist n09,n10", text)
            continue
        benchmark_extra.append(text)
        crystal_extra_sbatch.append(text)
    if nodelist:
        benchmark_extra.insert(0, f"--nodelist={nodelist}")
    return nodelist, benchmark_extra, crystal_extra_sbatch


def _short_reason(text: str, *, max_len: int = 120) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return "unknown"
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _fmt_log_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4g}"
        return str(value)
    return str(value)


def _guard_log_text(evaluation: SearchTaskEvaluation) -> str:
    guards = (evaluation.proposal_feedback or {}).get("crystal_guards") or {}
    parts: list[str] = []
    for material in ("ZnS", "MgO"):
        guard = guards.get(material) or {}
        if not guard:
            continue
        status = "ok" if guard.get("valid") else "fail"
        gap = _fmt_log_value(guard.get("gap_ev"))
        reason = _short_reason(str(guard.get("failure_reason") or "-"), max_len=60)
        parts.append(f"{material}:{status},gap={gap},reason={reason}")
    return ";".join(parts) if parts else "-"


class TaskDiverseBOBasinSearch(DiverseBOBasinSearch):
    """Task-aware wrapper around the existing diverse-BO proposal policy."""

    def _pool_candidate_allowed(self, xc: XCFunctional) -> bool:
        task = getattr(self, "_task_filter", None)
        if task is None:
            return True
        try:
            task.validate_functional(xc)
        except Exception:
            return False
        return True

    def search_task(
        self,
        *,
        root_xc,
        task: SIECrystalGuardSearchTask,
        root_evaluation: SearchTaskEvaluation,
        output_dir: Path,
    ) -> BasinAgentResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._task_filter = task

        if self.config.resume_nodes_path:
            self._load_resume_nodes(
                Path(self.config.resume_nodes_path),
                through_cycle=self.config.resume_through_cycle,
            )
            if not any(node.stage == "root" for node in self.nodes):
                raise ValueError("Task-aware BO resume history contains no root node")
            logger.info(
                "TASK_DIVERSE_BO_RESUME loaded_nodes=%d evaluated=%d source=%s",
                len(self.nodes),
                sum(node.stage != "root" for node in self.nodes),
                self.config.resume_nodes_path,
            )
        else:
            root_candidate = EvolutionCandidate(
                xc=root_xc,
                parameters=self._read_parameters(root_xc),
                generation=0,
            )
            task.apply_to_candidate(root_candidate, root_evaluation)
            root_candidate.xc.metadata["task_evaluation"] = root_evaluation.to_dict()
            root_node = self._register_node(
                cycle=0,
                stage="root",
                parent_id="",
                xc=root_candidate.xc,
                candidate=root_candidate,
                rationale="bootstrap root for task-aware diverse BO baseline",
                action_detail={"action": "root"},
            )
            root_node.task_evaluation = root_evaluation
            root_node.candidate.xc.metadata["basin_parent_reward"] = root_node.raw_reward

        cycle_summaries: list[dict[str, Any]] = []
        for cycle in range(1, self.config.cycles + 1):
            evaluated = [
                node
                for node in self.nodes
                if node.cycle == cycle and node.stage == "proposal"
            ]
            batch_index = _completed_bo_batch_count(
                evaluated,
                batch_size=self.config.proposal_batch_size,
            )
            if evaluated:
                logger.info(
                    "TASK_DIVERSE_BO_CYCLE_RESUME cycle=%d evaluated=%d/%d next_batch=%d",
                    cycle,
                    len(evaluated),
                    self.config.proposals_per_cycle,
                    batch_index + 1,
                )
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
                    "TASK_DIVERSE_BO_BATCH_START cycle=%d batch=%d evaluated_total=%d/%d count=%d",
                    cycle,
                    batch_index,
                    len(evaluated),
                    self.config.proposals_per_cycle,
                    count,
                )
                proposed = self._propose_batch(cycle=cycle, batch_index=batch_index, count=count)
                batch_evaluated = self._evaluate_task_nodes(
                    proposed,
                    task=task,
                    stage=f"cycle_{cycle:02d}_bo_batch_{batch_index:02d}",
                )
                evaluated.extend(batch_evaluated)
                self._write_artifacts(output_dir, cycle_summaries)
                logger.info(
                    "TASK_DIVERSE_BO_BATCH_DONE cycle=%d batch=%d evaluated_batch=%d evaluated_total=%d/%d",
                    cycle,
                    batch_index,
                    len(batch_evaluated),
                    len(evaluated),
                    self.config.proposals_per_cycle,
                )

            selected: list[BasinNode] = []
            ea_nodes: list[BasinNode] = []
            if self.config.enable_local_ea:
                selected = self._select_ea_seeds(evaluated)
                ea_nodes = self._run_task_local_ea(cycle=cycle, seeds=selected, task=task)
                self._write_artifacts(output_dir, cycle_summaries)

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
            self._write_artifacts(output_dir, cycle_summaries)

        result = BasinAgentResult(
            nodes=self.nodes,
            cycle_summaries=cycle_summaries,
            success_digest=self.success_digest,
            failure_digest=self.failure_digest,
        )
        self._write_artifacts(output_dir, cycle_summaries)
        return result

    def _run_task_local_ea(
        self,
        *,
        cycle: int,
        seeds: list[BasinNode],
        task: SIECrystalGuardSearchTask,
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
                include_global_hf=self.scorer.config.include_global_hf,
                include_role_scales=self.scorer.config.include_role_scales,
                role_scale_bounds=self.scorer.config.role_scale_bounds,
                include_component_mixing=self.scorer.config.include_component_mixing,
                include_component_ext_params=self.scorer.config.include_component_ext_params,
                component_pool=self.scorer.config.component_pool,
            )
            local = LocalEvolutionarySearch(
                config,
                seed_xc=seed.xc,
                seed_parameters=seed.candidate.parameters,
                basin_config=self.config,
                label=f"dbo_cycle{cycle}:seed{seed.node_id}",
            )
            try:
                result = self._task_local_ea_search(local, seed=seed, cycle=cycle, task=task)
            except Exception as exc:
                logger.warning("TASK_DIVERSE_BO_LOCAL_EA_FAILED cycle=%d seed=%s: %s", cycle, seed.node_id, exc)
                continue
            for candidate in result.evaluated[1:]:
                candidate.xc.metadata["basin_parent_id"] = seed.node_id
                candidate.xc.metadata["basin_parent_reward"] = seed.raw_reward
                evaluation_payload = candidate.xc.metadata.get("task_evaluation") or {}
                node = self._register_node(
                    cycle=cycle,
                    stage="local_ea",
                    parent_id=seed.node_id,
                    xc=candidate.xc,
                    candidate=candidate,
                    rationale=f"task local EA refinement from {seed.node_id}",
                    action_detail=dict(candidate.parameters),
                )
                node.task_evaluation = SearchTaskEvaluation.from_dict(evaluation_payload)
                node.eval_duration_s = candidate.xc.metadata.get("basin_eval_duration_s")
                collected.append(node)
        return collected

    def _task_local_ea_search(
        self,
        local: LocalEvolutionarySearch,
        *,
        seed: BasinNode,
        cycle: int,
        task: SIECrystalGuardSearchTask,
    ) -> EvolutionSearchResult:
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
                        "TASK_DIVERSE_BO_EVAL_START temp_node=%s stage=local_ea parent=%s name=%s",
                        f"{local.label}:{sig[:8]}",
                        seed.node_id,
                        candidate.xc.name,
                    )
                evaluations = task.evaluate_many(
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
                        relative = task.relative_feedback(evaluation, seed.task_evaluation)
                        if relative:
                            evaluation.proposal_feedback["relative_to_parent"] = relative
                    task.apply_to_candidate(candidate, evaluation)
                    duration = time.time() - starts.get(effective_sig, time.time())
                    candidate.xc.metadata["basin_eval_duration_s"] = evaluation.walltime_s or duration
                    candidate.xc.metadata["task_evaluation"] = evaluation.to_dict()
                    evaluated.append(candidate)
                    evaluated_by_sig[effective_sig] = candidate
                    population.append(candidate)
                    logger.info(
                        "TASK_DIVERSE_BO_EVAL_END temp_node=%s stage=local_ea parent=%s raw_reward=%.6f "
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

    def _candidate_pool(self, *, cycle: int, batch_index: int):
        pool = super()._candidate_pool(cycle=cycle, batch_index=batch_index)
        task = getattr(self, "_task_filter", None)
        if task is None:
            return pool

        filtered = []
        rejected: dict[str, int] = {}
        for item in pool:
            try:
                task.validate_functional(item.xc)
            except Exception as exc:
                reason = _short_reason(str(exc))
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            filtered.append(item)
        if rejected:
            top_reasons = sorted(rejected.items(), key=lambda kv: kv[1], reverse=True)[:5]
            logger.info(
                "TASK_DIVERSE_BO_CANDIDATE_POOL_FILTERED cycle=%d batch=%d raw=%d kept=%d rejected=%d reasons=%s",
                cycle,
                batch_index,
                len(pool),
                len(filtered),
                len(pool) - len(filtered),
                top_reasons,
            )
        return filtered

    def _evaluate_task_nodes(
        self,
        nodes: list[BasinNode],
        *,
        task: SIECrystalGuardSearchTask,
        stage: str,
    ) -> list[BasinNode]:
        if not nodes:
            return []
        started = time.time()
        valid_nodes: list[BasinNode] = []
        invalid: dict[str, SearchTaskEvaluation] = {}
        for node in nodes:
            node.eval_started_at = started
            try:
                task.validate_functional(node.xc)
                valid_nodes.append(node)
            except Exception as exc:
                invalid[node.node_id] = SearchTaskEvaluation(
                    task_name=task.name,
                    valid=False,
                    failure_reason=f"task_validation_failed:{exc}",
                    proposal_feedback={"validation_error": str(exc)},
                    walltime_s=0.0,
                )

        evaluated_by_id: dict[str, SearchTaskEvaluation] = dict(invalid)
        if valid_nodes:
            evaluations = task.evaluate_many(
                [node.xc for node in valid_nodes],
                labels=[node.node_id for node in valid_nodes],
                stage=stage,
            )
            if len(evaluations) != len(valid_nodes):
                raise RuntimeError(f"Task returned {len(evaluations)} evaluations for {len(valid_nodes)} BO nodes")
            evaluated_by_id.update({node.node_id: evaluation for node, evaluation in zip(valid_nodes, evaluations)})

        for node in nodes:
            evaluation = evaluated_by_id[node.node_id]
            parent = self._nodes_by_id.get(node.parent_id)
            parent_evaluation = parent.task_evaluation if parent is not None else None
            relative = task.relative_feedback(evaluation, parent_evaluation)
            if relative:
                evaluation.proposal_feedback["relative_to_parent"] = relative
            task.apply_to_candidate(node.candidate, evaluation)
            node.candidate.xc.metadata["task_evaluation"] = evaluation.to_dict()
            node.task_evaluation = evaluation
            node.eval_finished_at = time.time()
            node.eval_duration_s = evaluation.walltime_s
            if node.eval_duration_s is None:
                node.eval_duration_s = node.eval_finished_at - (node.eval_started_at or node.eval_finished_at)
            node.stage = "proposal"
            logger.info(
                "TASK_DIVERSE_BO_EVAL_END node=%s parent=%s raw_reward=%.6f valid=%s reason=%s "
                "sie_mae=%s sie_failures=%s guards=%s duration_s=%.1f",
                node.node_id,
                node.parent_id,
                node.raw_reward,
                evaluation.valid,
                evaluation.failure_reason or "-",
                _fmt_log_value(evaluation.objective.get("sie4x4_mae_kcal")),
                _fmt_log_value(evaluation.objective.get("sie4x4_failures")),
                _guard_log_text(evaluation),
                node.eval_duration_s,
            )
        return nodes

    def _write_csv(self, path: Path) -> None:
        fields = [
            "node_id", "cycle", "stage", "parent_id", "name", "raw_reward",
            "tree_reward", "promotable", "success_vs_parent", "failure_reason",
            "guard_penalty", "worst_validation_ratio", "eval_duration_s",
            "nearest_node_id", "nearest_delta", "sie4x4_mae_kcal",
            "sie4x4_failures", "sie_failure_penalty", "crystal_guard_penalty",
            "ZnS_gap_ev", "ZnS_guard_unit", "MgO_gap_ev", "MgO_guard_unit",
            "action_detail_json", "parameters_json", "task_evaluation_json", "xc_json",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for node in self.nodes:
                objective = node.task_evaluation.objective if node.task_evaluation else {}
                feedback = node.task_evaluation.proposal_feedback if node.task_evaluation else {}
                guards = feedback.get("crystal_guards") or {}
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
                    "sie4x4_mae_kcal": objective.get("sie4x4_mae_kcal"),
                    "sie4x4_failures": objective.get("sie4x4_failures"),
                    "sie_failure_penalty": objective.get("sie_failure_penalty"),
                    "crystal_guard_penalty": objective.get("guard_penalty"),
                    "ZnS_gap_ev": (guards.get("ZnS") or {}).get("gap_ev"),
                    "ZnS_guard_unit": (guards.get("ZnS") or {}).get("guard_unit"),
                    "MgO_gap_ev": (guards.get("MgO") or {}).get("gap_ev"),
                    "MgO_guard_unit": (guards.get("MgO") or {}).get("guard_unit"),
                    "action_detail_json": json.dumps(node.action_detail, sort_keys=True, default=str),
                    "parameters_json": json.dumps(node.candidate.parameters, sort_keys=True, default=str),
                    "task_evaluation_json": json.dumps(
                        node.task_evaluation.to_dict() if node.task_evaluation else {},
                        sort_keys=True,
                        default=str,
                    ),
                    "xc_json": json.dumps(node.xc.to_dict(), sort_keys=True, default=str),
                })


def _root_functional(task_values: dict[str, Any]) -> XCFunctional:
    name = str(task_values.get("root_functional", "mTASK-PW92"))
    if name.upper().replace("_", "-") == "MTASK-PW92":
        return make_mtask_pw92_functional()
    return XCFunctional.from_known(name)


def _build_task(config: dict, output_dir: Path):
    common_eval = dict(config.get("evaluation") or {})
    evaluation_backend = str(common_eval.get("backend", "slurm_docker")).strip().lower()
    if evaluation_backend not in {"slurm_docker", "vc", "persistent_vc"}:
        raise ValueError("evaluation.backend must be slurm_docker, vc, or persistent_vc")
    image = _resolve_image(common_eval.get("image", "$DFTAGENT_IMAGE"))
    vc_prefix = _resolve_env_value(
        common_eval.get("vc_prefix", "$DFTAGENT_VC_PREFIX"),
        fallback_env="DFTAGENT_VC_PREFIX",
    )
    if evaluation_backend == "slurm_docker" and not image:
        raise ValueError("evaluation.image or DFTAGENT_IMAGE is required")
    if evaluation_backend == "vc" and not vc_prefix:
        raise ValueError("evaluation.vc_prefix or DFTAGENT_VC_PREFIX is required")
    persistent_queue = _resolve_env_value(
        common_eval.get("persistent_queue_dir", "$DFTAGENT_PERSISTENT_VC_QUEUE"),
        fallback_env="DFTAGENT_PERSISTENT_VC_QUEUE",
    )
    if evaluation_backend == "persistent_vc" and not persistent_queue:
        raise ValueError(
            "evaluation.persistent_queue_dir or DFTAGENT_PERSISTENT_VC_QUEUE is required"
        )
    slurm_nodelist, slurm_extra_args, crystal_extra_sbatch = _slurm_placement_args(common_eval)

    task_values = dict(config.get("task") or {})
    root_xc = _root_functional(task_values)
    protected_components = tuple(str(name) for name in task_values.get("protected_components", ()))
    minimum_component_mixing = {
        str(name): float(value)
        for name, value in dict(task_values.get("minimum_component_mixing") or {}).items()
    }
    root_components = {component.name: component for component in root_xc.components}
    missing_protected = sorted(set(protected_components) - set(root_components))
    if missing_protected:
        raise ValueError(
            f"Protected components are absent from root {root_xc.name!r}: {missing_protected}"
        )
    fixed_component_ext_params = {
        name: dict(root_components[name].ext_params)
        for name in protected_components
    }
    root_crystal_paths = {}
    for material, value in dict(task_values.get("root_crystal_result_paths") or {}).items():
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        root_crystal_paths[str(material)] = str(path)
    task_config = SIECrystalGuardTaskConfig(
        component_pool=str(task_values.get("component_pool", config.get("search", {}).get("component_pool", "zno_gap"))),
        baseline_sie_mae_kcal=float(task_values.get("baseline_sie_mae_kcal", 10.0)),
        baseline_sie_failures=int(task_values.get("baseline_sie_failures", 0)),
        baseline_sie_reactions=int(task_values.get("baseline_sie_reactions", 16)),
        zns_reference_gap_ev=float(task_values.get("zns_reference_gap_ev", 3.83)),
        mgo_reference_gap_ev=float(task_values.get("mgo_reference_gap_ev", 7.77)),
        zns_free_tolerance_ev=float(task_values.get("zns_free_tolerance_ev", 0.50)),
        mgo_free_tolerance_ev=float(task_values.get("mgo_free_tolerance_ev", 0.20)),
        guard_scale_ev=float(task_values.get("guard_scale_ev", 0.30)),
        guard_mean_weight=float(task_values.get("guard_mean_weight", 0.25)),
        guard_max_weight=float(task_values.get("guard_max_weight", 0.10)),
        sie_fail_penalties=tuple(float(x) for x in task_values.get("sie_fail_penalties", [0.15, 0.15, 0.25, 0.25])),
        extra_sie_fail_penalty=float(task_values.get("extra_sie_fail_penalty", 0.50)),
        forbid_hf_exchange=bool(task_values.get("forbid_hf_exchange", True)),
        allow_ext_param_rewire=bool(task_values.get("allow_ext_param_rewire", False)),
        protected_components=protected_components,
        minimum_component_mixing=minimum_component_mixing,
        fixed_component_ext_params=fixed_component_ext_params,
        root_label=str(task_values.get("root_label", root_xc.name)),
        crystal_guard_materials=tuple(task_values.get("crystal_guard_materials", ["ZnS", "MgO"])),
        root_crystal_result_paths=root_crystal_paths,
    )

    benchmark_values = dict(config.get("benchmark_evaluation") or {})
    crystal_values = dict(config.get("crystal_evaluation") or {})
    if evaluation_backend in {"vc", "persistent_vc"}:
        container_repo_root = str(common_eval.get("container_repo_root", "/code/remote/DFTAgent"))
        sie_runner = VCManifestBenchmarkRunner(
            VCManifestRunnerConfig(
                output_dir=output_dir / "sie_evaluations",
                vc_prefix=vc_prefix if evaluation_backend == "vc" else "",
                persistent_queue_dir=Path(persistent_queue) if evaluation_backend == "persistent_vc" else None,
                container_repo_root=container_repo_root,
                python_executable=str(common_eval.get("python_executable", "python3")),
                conda_sh=str(common_eval.get("conda_sh", "")),
                conda_env=str(common_eval.get("conda_env", "")),
                pre_cmd=str(common_eval.get("pre_cmd", "")),
                gmtkn55_dir=Path(benchmark_values.get("gmtkn55_dir", ROOT / "benchmark" / "GMTKN55")),
                subsets=("SIE4x4",),
                basis=str(benchmark_values.get("basis", "def2-TZVP")),
                scf_max_cycle=int(benchmark_values.get("scf_max_cycle", 200)),
                scf_conv_tol=float(benchmark_values.get("scf_conv_tol", 1e-8)),
                n_workers=int(benchmark_values.get("n_workers", 8)),
                n_threads=int(benchmark_values.get("n_threads", 1)),
                molecules_per_task=int(benchmark_values.get("molecules_per_task", 8)),
                aggregate_mode=str(benchmark_values.get("aggregate_mode", "reaction_mean")),
                concurrent_subset_jobs=int(common_eval.get("max_concurrent", 16)),
                vc_tasks_per_submit=int(common_eval.get("vc_tasks_per_submit", 0)),
                poll_seconds=float(common_eval.get("queue_monitor_poll_s", 10)),
                wait_timeout_seconds=float(common_eval.get("wait_timeout_seconds", 0)),
            ),
            repo_root=ROOT,
        )
        if evaluation_backend == "persistent_vc":
            crystal_evaluator = CrystalGuardPersistentVCEvaluator(
                CrystalGuardPersistentVCEvaluatorConfig(
                    output_dir=output_dir / "crystal_guard_evaluations",
                    queue_dir=Path(persistent_queue),
                    cpus_per_candidate=int(common_eval.get("cpus_per_candidate", 8)),
                    hard_timeout_s=int(common_eval.get("hard_timeout_s", 7200)),
                    poll_seconds=float(common_eval.get("queue_monitor_poll_s", 2)),
                    wait_timeout_seconds=float(common_eval.get("wait_timeout_seconds", 0)),
                    band_mesh=tuple(int(x) for x in crystal_values.get("band_mesh", [2, 2, 2])),
                    materials=tuple(task_config.crystal_guard_materials),
                ),
                repo_root=ROOT,
            )
        else:
            crystal_evaluator = CrystalGuardVCEvaluator(
                CrystalGuardVCEvaluatorConfig(
                    output_dir=output_dir / "crystal_guard_evaluations",
                    vc_prefix=vc_prefix,
                    container_repo_root=container_repo_root,
                    python_executable=str(common_eval.get("python_executable", "python3")),
                    conda_sh=str(common_eval.get("conda_sh", "")),
                    conda_env=str(common_eval.get("conda_env", "")),
                    pre_cmd=str(common_eval.get("pre_cmd", "")),
                    cpus_per_candidate=int(common_eval.get("cpus_per_candidate", 8)),
                    max_concurrent_submissions=int(common_eval.get("max_concurrent", 16)),
                    hard_timeout_s=int(common_eval.get("hard_timeout_s", 7200)),
                    band_mesh=tuple(int(x) for x in crystal_values.get("band_mesh", [2, 2, 2])),
                    materials=tuple(task_config.crystal_guard_materials),
                ),
                repo_root=ROOT,
            )
    else:
        sie_runner = BenchmarkRunner(
            gmtkn55_dir=Path(benchmark_values.get("gmtkn55_dir", ROOT / "benchmark" / "GMTKN55")),
            basis=str(benchmark_values.get("basis", "def2-TZVP")),
            subsets=["SIE4x4"],
            n_workers=int(benchmark_values.get("n_workers", 16)),
            n_threads=int(benchmark_values.get("n_threads", 1)),
            execution_backend="slurm_docker_queue",
            slurm_partition=str(common_eval.get("partition", "compute")),
            slurm_mem=str(common_eval.get("memory", "120G")),
            slurm_docker_image=image,
            slurm_docker_container_repo_root=str(common_eval.get("container_repo_root", "/workspace/DFTAgent")),
            slurm_docker_time=str(common_eval.get("slurm_time", "02:10:00")),
            slurm_docker_total_cpus=int(common_eval.get("total_cpus", 256)),
            slurm_docker_cpus_per_worker=int(common_eval.get("cpus_per_candidate", 16)),
            slurm_docker_worker_count=int(common_eval.get("max_concurrent", 16)),
            slurm_docker_nodes=int(common_eval.get("nodes", 0)),
            slurm_docker_ntasks_per_node=int(common_eval.get("ntasks_per_node", 0)),
            slurm_docker_cpus_per_node=int(common_eval.get("cpus_per_node", 128)),
            slurm_docker_submit_mode=str(common_eval.get("slurm_submit_mode", "allocation")),
            slurm_docker_array_max_running=int(common_eval.get("array_max_running", 0)),
            slurm_docker_queue_dispatch_mode=str(common_eval.get("queue_dispatch_mode", "candidate_dedicated")),
            slurm_docker_thread_policy=str(benchmark_values.get("thread_policy", "atom_adaptive")),
            slurm_docker_atom_thread_bins=str(benchmark_values.get("atom_thread_bins", "2:1,6:2,10:4,999:16")),
            slurm_docker_max_threads_per_sample=int(benchmark_values.get("max_threads_per_sample", 16)),
            slurm_docker_molecules_per_task=int(benchmark_values.get("molecules_per_task", 1)),
            slurm_docker_output_dir=output_dir / "sie_evaluations",
            slurm_extra_args=slurm_extra_args,
            aggregate_mode=str(benchmark_values.get("aggregate_mode", "reaction_mean")),
            scf_max_cycle=int(benchmark_values.get("scf_max_cycle", 200)),
            scf_conv_tol=float(benchmark_values.get("scf_conv_tol", 1e-8)),
        )

        crystal_cfg = CrystalGuardSlurmEvaluatorConfig(
            output_dir=output_dir / "crystal_guard_evaluations",
            image=image,
            container_repo_root=str(common_eval.get("container_repo_root", "/workspace/DFTAgent")),
            partition=str(common_eval.get("partition", "compute")),
            cpus_per_candidate=int(common_eval.get("cpus_per_candidate", 16)),
            max_concurrent=int(common_eval.get("max_concurrent", 16)),
            memory=str(common_eval.get("memory", "120G")),
            hard_timeout_s=int(common_eval.get("hard_timeout_s", 7200)),
            slurm_time=str(common_eval.get("slurm_time", "02:10:00")),
            slurm_nodelist=slurm_nodelist,
            extra_sbatch=crystal_extra_sbatch,
            band_mesh=tuple(int(x) for x in crystal_values.get("band_mesh", [2, 2, 2])),
            materials=tuple(task_config.crystal_guard_materials),
            adaptive_straggler_cutoff=bool(common_eval.get("adaptive_straggler_cutoff", True)),
            straggler_cutoff_start_fraction=float(common_eval.get("straggler_cutoff_start_fraction", 0.75)),
            straggler_cutoff_min_multiplier=float(common_eval.get("straggler_cutoff_min_multiplier", 1.5)),
            straggler_cutoff_mad_multiplier=float(common_eval.get("straggler_cutoff_mad_multiplier", 3.0)),
            straggler_cutoff_max_fraction=float(common_eval.get("straggler_cutoff_max_fraction", 0.05)),
            poll_s=float(common_eval.get("queue_monitor_poll_s", 30)),
        )
        crystal_evaluator = CrystalGuardSlurmDockerEvaluator(crystal_cfg, repo_root=ROOT)
    task = SIECrystalGuardSearchTask(
        sie_runner=sie_runner,
        crystal_evaluator=crystal_evaluator,
        config=task_config,
    )
    return task, image, root_xc, minimum_component_mixing, evaluation_backend, vc_prefix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "sie_crystal_guard_diverse_bo.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume-nodes",
        type=Path,
        default=None,
        help="Resume the same fixed budget from an interrupted diverse-BO nodes JSON",
    )
    parser.add_argument("--slurm-nodelist", default="", help="Restrict Slurm jobs to nodes such as n09,n10")
    parser.add_argument("--slurm-nodes", type=int, default=None, help="Override evaluation.nodes for fixed-node allocation")
    parser.add_argument("--slurm-ntasks-per-node", type=int, default=None, help="Override evaluation.ntasks_per_node")
    parser.add_argument("--slurm-cpus-per-node", type=int, default=None, help="Override evaluation.cpus_per_node")
    parser.add_argument("--slurm-total-cpus", type=int, default=None, help="Override evaluation.total_cpus")
    parser.add_argument("--max-concurrent", type=int, default=None, help="Override evaluation.max_concurrent")
    parser.add_argument(
        "--slurm-extra-sbatch",
        action="append",
        default=[],
        help="Extra #SBATCH option, e.g. --exclude=n01 or --reservation=foo",
    )
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    _apply_evaluation_cli_overrides(config, args)
    output_dir = args.output_dir or Path(config.get("output_dir", "results/run_sie_crystal_guard_diverse_bo"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, str(config.get("log_level", "INFO")).upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    search_values = dict(config.get("search") or {})
    if args.seed is not None:
        search_values["seed"] = int(args.seed)
    if args.resume_nodes is not None:
        search_values["resume_nodes_path"] = str(args.resume_nodes)
    core_dir = None if args.output_dir is not None else (
        search_values.get("core_output_dir") or config.get("core_output_dir")
    )
    if not core_dir:
        core_dir = str(output_dir) + "_core"
    elif not Path(core_dir).is_absolute():
        core_dir = str(ROOT / core_dir)
    search_values["core_output_dir"] = core_dir
    search_config = DiverseBOBasinConfig(**search_values)

    task_values = dict(config.get("task") or {})
    task_component_pool = str(task_values.get("component_pool", search_config.component_pool))
    if task_component_pool != search_config.component_pool:
        raise ValueError(
            "search.component_pool and task.component_pool must match: "
            f"{search_config.component_pool!r} != {task_component_pool!r}"
        )
    pool_signature = validate_component_pool_lock(
        search_config.component_pool,
        config.get("component_pool_lock"),
    )
    logger.info("COMPONENT_POOL_LOCK_OK %s", pool_signature)

    task, image, root_xc, minimum_component_mixing, evaluation_backend, vc_prefix = _build_task(config, output_dir)
    effective_config = {
        "output_dir": str(output_dir),
        "core_output_dir": str(core_dir),
        "search": search_values,
        "task": config.get("task") or {},
        "component_pool_lock": config.get("component_pool_lock") or {},
        "component_pool_signature": pool_signature,
        "evaluation": {
            **dict(config.get("evaluation") or {}),
            "backend": evaluation_backend,
            "image": image,
            "vc_prefix": vc_prefix,
        },
        "benchmark_evaluation": config.get("benchmark_evaluation") or {},
        "crystal_evaluation": config.get("crystal_evaluation") or {},
    }
    (output_dir / "effective_sie_crystal_guard_diverse_bo_config.json").write_text(
        json.dumps(effective_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(core_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        output_dir / "effective_sie_crystal_guard_diverse_bo_config.json",
        Path(core_dir) / "effective_sie_crystal_guard_diverse_bo_config.json",
    )
    if args.validate_config:
        print("SIE+crystal diverse-BO configuration is valid; no PySCF call was made.", flush=True)
        return 0

    root_evaluation = task.root_evaluation_from_existing()
    (output_dir / "root_evaluation_source.json").write_text(
        json.dumps(root_evaluation.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(output_dir / "root_evaluation_source.json", Path(core_dir) / "root_evaluation_source.json")

    scorer_config = EvolutionConfig(
        seed=search_config.seed,
        component_pool=search_config.component_pool,
        include_global_hf=False,
        include_role_scales=True,
        role_scale_bounds=(0.8, 1.2),
        include_component_mixing=True,
        component_mixing_bounds={
            name: (minimum, 1.0)
            for name, minimum in minimum_component_mixing.items()
        },
        include_component_ext_params=False,
        exact_mixing_sum=True,
        evaluation_concurrency=search_config.proposal_evaluation_concurrency,
    )
    search = TaskDiverseBOBasinSearch(config=search_config, scorer_config=scorer_config)
    result = search.search_task(
        root_xc=root_xc,
        task=task,
        root_evaluation=root_evaluation,
        output_dir=output_dir,
    )
    for name in ("diverse_bo_basin_nodes.csv", "diverse_bo_basin_nodes.json", "diverse_bo_basin_summary.json"):
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, Path(core_dir) / name)
    print(json.dumps(result.summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
