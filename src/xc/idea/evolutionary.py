"""Black-box evolutionary optimization for continuous XC parameters."""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import random
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ..benchmark.result import BenchmarkResult
from ..xc_functional import XCFunctional
from ..zno_component_registry import SUPPORTED_COMPONENT_POOLS, component_roles_for_pool
from .actions import Action, ActionType, apply_action
from .reward import (
    DEFAULT_GUARD_TOLERANCES,
    guard_regression_penalty,
    target_guard_raw_reward,
    tree_reward_from_raw,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvolutionParameter:
    """One continuous parameter exposed to the evolutionary optimizer."""

    key: str
    lower: float
    upper: float
    kind: str = "ext_param"
    role: str = ""

    def clip(self, value: float) -> float:
        return min(self.upper, max(self.lower, float(value)))


@dataclass
class EvolutionConfig:
    population_size: int = 12
    generations: int = 8
    elite_size: int = 4
    tournament_size: int = 3
    crossover_rate: float = 0.75
    mutation_rate: float = 0.35
    mutation_scale: float = 0.12
    validation_regression_limit: float = float("inf")
    target_gain_weight: float = 2.0
    guard_regression_penalty: float = 1.0
    guard_mean_weight: float = 0.25
    guard_max_weight: float = 0.10
    guard_tolerances: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GUARD_TOLERANCES)
    )
    runtime_failure_penalty: float = 0.05
    seed: int = 42
    alpha_bounds: tuple[float, float] = (0.0, 0.50)
    beta_bounds: tuple[float, float] = (0.0, 1.0)
    omega_bounds: tuple[float, float] = (0.0, 0.50)
    component_names: tuple[str, ...] = ()
    component_pool: str = "expanded"
    include_global_hf: bool = True
    include_role_scales: bool = False
    role_scale_bounds: tuple[float, float] = (0.8, 1.2)
    include_component_mixing: bool = True
    component_mixing_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    include_component_ext_params: bool = True
    exact_mixing_sum: bool = False
    diversity_injection_fraction: float = 0.0
    diversity_injection_patience: int = 0
    diversity_injection_threshold: float = 0.0
    diversity_injection_max_events: int = 0
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.01
    early_stop_diversity_threshold: float = 0.0
    evaluation_concurrency: int = 1

    def __post_init__(self) -> None:
        self.population_size = max(2, int(self.population_size))
        self.generations = max(1, int(self.generations))
        self.elite_size = max(1, min(int(self.elite_size), self.population_size))
        self.tournament_size = max(2, int(self.tournament_size))
        self.crossover_rate = min(1.0, max(0.0, float(self.crossover_rate)))
        self.mutation_rate = min(1.0, max(0.0, float(self.mutation_rate)))
        self.mutation_scale = max(0.0, float(self.mutation_scale))
        self.exact_mixing_sum = bool(self.exact_mixing_sum)
        normalized_mixing_bounds: dict[str, tuple[float, float]] = {}
        for name, bounds in dict(self.component_mixing_bounds or {}).items():
            if len(bounds) != 2:
                raise ValueError(f"component_mixing_bounds[{name!r}] must contain two values")
            lower, upper = (float(bounds[0]), float(bounds[1]))
            if upper < lower:
                lower, upper = upper, lower
            normalized_mixing_bounds[str(name)] = (max(0.0, lower), min(1.0, upper))
        self.component_mixing_bounds = normalized_mixing_bounds
        self.diversity_injection_fraction = min(
            1.0, max(0.0, float(self.diversity_injection_fraction))
        )
        self.diversity_injection_patience = max(0, int(self.diversity_injection_patience))
        self.diversity_injection_threshold = max(0.0, float(self.diversity_injection_threshold))
        self.diversity_injection_max_events = max(0, int(self.diversity_injection_max_events))
        self.early_stop_patience = max(0, int(self.early_stop_patience))
        self.early_stop_min_delta = max(0.0, float(self.early_stop_min_delta))
        self.early_stop_diversity_threshold = max(0.0, float(self.early_stop_diversity_threshold))
        self.evaluation_concurrency = max(1, int(self.evaluation_concurrency))
        if self.component_pool not in SUPPORTED_COMPONENT_POOLS:
            raise ValueError(f"Unsupported component_pool: {self.component_pool}")
        lo, hi = self.role_scale_bounds
        lo = float(lo)
        hi = float(hi)
        if hi < lo:
            lo, hi = hi, lo
        self.role_scale_bounds = (lo, hi)


@dataclass
class MixedEvolutionConfig(EvolutionConfig):
    """Configuration for EA over both component scaffolds and parameters."""

    structural_mutation_rate: float = 0.25
    max_components_per_role: int = 2
    initial_structure_edits: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        self.structural_mutation_rate = min(1.0, max(0.0, float(self.structural_mutation_rate)))
        self.max_components_per_role = max(1, int(self.max_components_per_role))
        self.initial_structure_edits = max(1, int(self.initial_structure_edits))


@dataclass
class EvolutionCandidate:
    xc: XCFunctional
    parameters: dict[str, float]
    generation: int
    search_result: BenchmarkResult | None = None
    validation_result: BenchmarkResult | None = None
    fitness: float = float("-inf")
    promotable: bool = False
    failure_reason: str = ""
    validation_ratio: float | None = None
    validation_subset_ratios: dict[str, float] = field(default_factory=dict)
    worst_validation_ratio: float = float("inf")
    guard_penalty: float = 0.0
    guard_regression_units: dict[str, float] = field(default_factory=dict)
    tree_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        def _metric(obj: Any, key: str) -> Any:
            if obj is None:
                return None
            if isinstance(obj, dict):
                if key == "overall_mae":
                    return obj.get("overall_mae")
                if key == "total_failures":
                    return obj.get("total_failures")
                return obj.get(key)
            return getattr(obj, key, None)

        return {
            "name": self.xc.name,
            "signature": self.xc.structural_signature(),
            "effective_signature": self.xc.effective_signature(),
            "generation": self.generation,
            "parameters": dict(self.parameters),
            "fitness": self.fitness,
            "promotable": self.promotable,
            "failure_reason": self.failure_reason,
            "validation_ratio": self.validation_ratio,
            "validation_subset_ratios": dict(self.validation_subset_ratios),
            "worst_validation_ratio": self.worst_validation_ratio,
            "guard_penalty": self.guard_penalty,
            "guard_regression_units": dict(self.guard_regression_units),
            "tree_reward": self.tree_reward,
            "search_mae": _metric(self.search_result, "overall_mae"),
            "validation_mae": _metric(self.validation_result, "overall_mae"),
            "search_failures": _metric(self.search_result, "total_failures"),
            "validation_failures": _metric(self.validation_result, "total_failures"),
            "xc_dict": self.xc.to_dict(),
        }


@dataclass
class EvolutionSearchResult:
    best: EvolutionCandidate
    baseline: EvolutionCandidate
    evaluated: list[EvolutionCandidate]
    generation_summaries: list[dict[str, Any]]
    parameters: list[EvolutionParameter]

    @property
    def summary(self) -> dict[str, Any]:
        incumbent = self.baseline
        promotions = 0
        for candidate in self.evaluated[1:]:
            if EvolutionarySearch._rank_key(candidate) > EvolutionarySearch._rank_key(incumbent):
                incumbent = candidate
                promotions += 1
        return {
            "generations": len(self.generation_summaries),
            "evaluated_nodes": len(self.evaluated),
            "promotable_nodes": sum(candidate.promotable for candidate in self.evaluated),
            "promotions_after_root": promotions,
            "best": self.best.to_dict(),
            "parameter_space": [
                {
                    "key": item.key,
                    "lower": item.lower,
                    "upper": item.upper,
                    "kind": item.kind,
                    "role": item.role,
                }
                for item in self.parameters
            ],
        }


class EvolutionarySearch:
    """Optimize continuous parameters using full black-box benchmarks."""

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()
        self.random = random.Random(self.config.seed)
        self.last_result: EvolutionSearchResult | None = None

    def search(
        self,
        *,
        root_xc: XCFunctional,
        root_result: BenchmarkResult,
        validation_runner,
        runner,
        baseline_validation_result: BenchmarkResult | None = None,
        initial_population_vectors: list[dict[str, float]] | None = None,
        on_evaluation: Callable[[EvolutionCandidate], None] | None = None,
        baseline_candidate: EvolutionCandidate | None = None,
    ) -> EvolutionSearchResult:
        parameters = self._parameter_space(root_xc)
        if not parameters:
            raise ValueError("Evolutionary search has no tunable parameters for this functional")

        baseline_validation = baseline_validation_result or validation_runner.evaluate(root_xc)
        baseline = (
            copy.deepcopy(baseline_candidate)
            if baseline_candidate is not None
            else EvolutionCandidate(
                xc=copy.deepcopy(root_xc),
                parameters=self._read_parameters(root_xc, parameters),
                generation=0,
                search_result=root_result,
                validation_result=baseline_validation,
            )
        )
        if baseline_candidate is None:
            self._score(baseline, root_result, baseline_validation)

        logger.info(
            "Evolutionary search start: population=%d generations=%d parameters=%s baseline_fitness=%.6f",
            self.config.population_size,
            self.config.generations,
            [item.key for item in parameters],
            baseline.fitness,
        )

        evaluated: list[EvolutionCandidate] = [baseline]
        evaluated_by_sig: dict[str, EvolutionCandidate] = {
            baseline.xc.effective_signature(): baseline,
        }
        population_vectors = self._initial_population(
            baseline.parameters,
            parameters,
            seed_vectors=initial_population_vectors,
        )
        best = baseline
        generation_summaries: list[dict[str, Any]] = []
        best_fitness_reference = baseline.fitness
        stagnant_generations = 0
        diversity_injection_events = 0

        for generation in range(1, self.config.generations + 1):
            population: list[EvolutionCandidate] = []
            pending: list[tuple[str, EvolutionCandidate]] = []
            for vector in population_vectors:
                xc = self._build_candidate(root_xc, vector, generation)
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
                logger.info(
                    "Evolution generation %d evaluating %d new candidates with concurrency=%d",
                    generation,
                    len(pending),
                    self.config.evaluation_concurrency,
                )
            completed_candidates = self._evaluate_pending_candidates(
                pending,
                runner=runner,
                validation_runner=validation_runner,
                root_result=root_result,
                baseline_validation=baseline_validation,
            )
            for effective_sig, candidate in completed_candidates:
                evaluated.append(candidate)
                evaluated_by_sig[effective_sig] = candidate
                population.append(candidate)
                if on_evaluation is not None:
                    on_evaluation(candidate)
                logger.info(
                    "Evolution candidate generation=%d name=%s fitness=%.6f tree_reward=%.6f "
                    "promotable=%s search_mae=%.4f validation_mae=%.4f worst_guard=%.4f "
                    "guard_penalty=%.4f guard_units=%s params=%s",
                    generation,
                    candidate.xc.name,
                    candidate.fitness,
                    candidate.tree_reward,
                    candidate.promotable,
                    candidate.search_result.overall_mae,
                    candidate.validation_result.overall_mae,
                    candidate.worst_validation_ratio,
                    candidate.guard_penalty,
                    json.dumps(candidate.guard_regression_units, sort_keys=True),
                    json.dumps(candidate.parameters, sort_keys=True),
                )

            ranked = sorted(population, key=self._rank_key, reverse=True)
            if ranked and self._rank_key(ranked[0]) > self._rank_key(best):
                best = ranked[0]
                logger.info(
                    "Evolution new best: generation=%d name=%s fitness=%.6f search_mae=%.4f "
                    "worst_guard=%.4f guard_penalty=%.4f params=%s",
                    generation,
                    best.xc.name,
                    best.fitness,
                    best.search_result.overall_mae,
                    best.worst_validation_ratio,
                    best.guard_penalty,
                    json.dumps(best.parameters, sort_keys=True),
                )

            summary = self._generation_summary(generation, ranked, best, parameters)
            if best.fitness > best_fitness_reference + self.config.early_stop_min_delta:
                best_fitness_reference = best.fitness
                stagnant_generations = 0
            else:
                stagnant_generations += 1
            summary["stagnant_generations"] = stagnant_generations
            summary["diversity_injection_event"] = False
            generation_summaries.append(summary)
            logger.info(
                "Evolution generation %d complete: unique=%d promotable=%d best_fitness=%.6f "
                "best_search_mae=%.4f best_worst_guard=%.4f diversity_pairwise=%.4f "
                "diversity_std=%.4f stagnant=%d",
                generation,
                summary["unique_candidates"],
                summary["promotable_candidates"],
                summary["best_fitness"],
                summary["best_search_mae"],
                summary["best_worst_guard_ratio"],
                summary["mean_pairwise_parameter_distance"],
                summary["mean_parameter_std"],
                stagnant_generations,
            )

            inject_count = 0
            can_inject = (
                self.config.diversity_injection_fraction > 0.0
                and self.config.diversity_injection_patience > 0
                and stagnant_generations >= self.config.diversity_injection_patience
                and summary["mean_pairwise_parameter_distance"] <= self.config.diversity_injection_threshold
                and diversity_injection_events < self.config.diversity_injection_max_events
            )
            if can_inject:
                inject_count = max(1, round(self.config.population_size * self.config.diversity_injection_fraction))
                diversity_injection_events += 1
                stagnant_generations = 0
                summary["diversity_injection_event"] = True
                summary["diversity_injection_count"] = inject_count
                logger.info(
                    "Evolution diversity injection: generation=%d inject_count=%d event=%d/%d "
                    "pairwise=%.4f threshold=%.4f",
                    generation,
                    inject_count,
                    diversity_injection_events,
                    self.config.diversity_injection_max_events,
                    summary["mean_pairwise_parameter_distance"],
                    self.config.diversity_injection_threshold,
                )
            elif (
                self.config.early_stop_patience > 0
                and stagnant_generations >= self.config.early_stop_patience
                and summary["mean_pairwise_parameter_distance"] <= self.config.early_stop_diversity_threshold
            ):
                logger.info(
                    "Evolution early stop: generation=%d stagnant=%d pairwise=%.4f threshold=%.4f "
                    "best_fitness=%.6f",
                    generation,
                    stagnant_generations,
                    summary["mean_pairwise_parameter_distance"],
                    self.config.early_stop_diversity_threshold,
                    best.fitness,
                )
                break
            population_vectors = self._next_population(ranked, parameters, inject_count=inject_count)

        result = EvolutionSearchResult(
            best=best,
            baseline=baseline,
            evaluated=evaluated,
            generation_summaries=generation_summaries,
            parameters=parameters,
        )
        self.last_result = result
        return result

    def _evaluate_pending_candidates(
        self,
        pending: list[tuple[str, EvolutionCandidate]],
        *,
        runner,
        validation_runner,
        root_result: BenchmarkResult,
        baseline_validation: BenchmarkResult,
    ) -> list[tuple[str, EvolutionCandidate]]:
        if not pending:
            return []

        def _run(item: tuple[str, EvolutionCandidate]) -> tuple[str, EvolutionCandidate]:
            effective_sig, candidate = item
            candidate.search_result = runner.evaluate(candidate.xc)
            candidate.validation_result = validation_runner.evaluate(candidate.xc)
            self._score(candidate, root_result, baseline_validation)
            return effective_sig, candidate

        if self.config.evaluation_concurrency <= 1 or len(pending) <= 1:
            return [_run(item) for item in pending]

        completed: list[tuple[str, EvolutionCandidate]] = []
        with ThreadPoolExecutor(
            max_workers=min(self.config.evaluation_concurrency, len(pending))
        ) as executor:
            futures = [executor.submit(_run, item) for item in pending]
            for future in as_completed(futures):
                completed.append(future.result())
        return completed

    def write_artifacts(self, result: EvolutionSearchResult, results_dir: Path | str) -> None:
        output_dir = Path(results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evolutionary_history.json").write_text(
            json.dumps([candidate.to_dict() for candidate in result.evaluated], indent=2)
        )
        (output_dir / "evolutionary_summary.json").write_text(
            json.dumps(
                {
                    **result.summary,
                    "generation_summaries": result.generation_summaries,
                },
                indent=2,
            )
        )
        if result.generation_summaries:
            generation_fields = sorted({
                key
                for summary in result.generation_summaries
                for key in summary
                if key != "best_parameters"
            })
            with (output_dir / "evolutionary_generation_summaries.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=generation_fields)
                writer.writeheader()
                for summary in result.generation_summaries:
                    writer.writerow({
                        key: json.dumps(value, sort_keys=True, default=str)
                        if isinstance(value, (dict, list)) else value
                        for key, value in summary.items()
                        if key in generation_fields
                    })
        subset_names = sorted({
            name
            for candidate in result.evaluated
            for benchmark in (candidate.search_result, candidate.validation_result)
            if benchmark is not None
            for name in benchmark.subsets
        })
        parameter_names = [parameter.key for parameter in result.parameters]
        # Mixed-structure search can expose different parameters per scaffold.
        # Keep the root order stable, then append every additional observed key.
        for candidate in result.evaluated:
            for key in sorted(candidate.parameters):
                if key not in parameter_names:
                    parameter_names.append(key)
        fields = [
            "evaluation_index",
            "generation",
            "name",
            "signature",
            "effective_signature",
            "fitness",
            "promotable",
            "failure_reason",
            "search_overall_mae",
            "validation_overall_mae",
            "validation_ratio",
            "worst_validation_ratio",
            "guard_penalty",
            "tree_reward",
            "action_detail_json",
            "structure_operations_json",
            "xc_json",
            *parameter_names,
        ]
        for subset in subset_names:
            fields.extend([
                f"{subset}_mae",
                f"{subset}_ratio",
                f"{subset}_guard_regression_unit",
                f"{subset}_failures",
            ])
        with (output_dir / "evolutionary_history.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, candidate in enumerate(result.evaluated, start=1):
                row: dict[str, Any] = {
                    "evaluation_index": index,
                    "generation": candidate.generation,
                    "name": candidate.xc.name,
                    "signature": candidate.xc.structural_signature(),
                    "effective_signature": candidate.xc.effective_signature(),
                    "fitness": candidate.fitness,
                    "promotable": candidate.promotable,
                    "failure_reason": candidate.failure_reason,
                    "search_overall_mae": (
                        candidate.search_result.overall_mae if candidate.search_result else None
                    ),
                    "validation_overall_mae": (
                        candidate.validation_result.overall_mae if candidate.validation_result else None
                    ),
                    "validation_ratio": candidate.validation_ratio,
                    "worst_validation_ratio": candidate.worst_validation_ratio,
                    "guard_penalty": candidate.guard_penalty,
                    "tree_reward": candidate.tree_reward,
                    "action_detail_json": json.dumps(
                        candidate.xc.metadata.get("action_detail", {}),
                        sort_keys=True,
                        default=str,
                    ),
                    "structure_operations_json": json.dumps(
                        candidate.xc.metadata.get("structure_operations", []),
                        sort_keys=True,
                        default=str,
                    ),
                    "xc_json": json.dumps(candidate.xc.to_dict(), sort_keys=True, default=str),
                    **candidate.parameters,
                }
                for subset in subset_names:
                    benchmark = None
                    if candidate.search_result and subset in candidate.search_result.subsets:
                        benchmark = candidate.search_result
                    elif candidate.validation_result and subset in candidate.validation_result.subsets:
                        benchmark = candidate.validation_result
                    subset_result = benchmark.subsets.get(subset) if benchmark else None
                    row[f"{subset}_mae"] = subset_result.mae_kcal if subset_result else None
                    row[f"{subset}_ratio"] = candidate.validation_subset_ratios.get(subset)
                    row[f"{subset}_guard_regression_unit"] = candidate.guard_regression_units.get(subset)
                    row[f"{subset}_failures"] = subset_result.n_failures if subset_result else None
                writer.writerow(row)

    def _parameter_space(self, xc: XCFunctional) -> list[EvolutionParameter]:
        parameters: list[EvolutionParameter] = []
        if self.config.include_global_hf:
            parameters.extend([
                EvolutionParameter("alpha", *self.config.alpha_bounds, kind="global_hf"),
                EvolutionParameter("beta", *self.config.beta_bounds, kind="global_hf"),
                EvolutionParameter("omega", *self.config.omega_bounds, kind="global_hf"),
            ])
        if self.config.include_role_scales:
            parameters.extend([
                EvolutionParameter(
                    "scale:exchange",
                    *self.config.role_scale_bounds,
                    kind="role_scale",
                    role="exchange",
                ),
                EvolutionParameter(
                    "scale:correlation",
                    *self.config.role_scale_bounds,
                    kind="role_scale",
                    role="correlation",
                ),
            ])
        allowed_components = set(self.config.component_names)
        for component in sorted(xc.components, key=lambda item: item.name):
            if allowed_components and component.name not in allowed_components:
                continue
            if self.config.include_component_mixing and component.role in {"exchange", "correlation"}:
                lower, upper = self.config.component_mixing_bounds.get(component.name, (0.0, 1.0))
                parameters.append(
                    EvolutionParameter(
                        f"mixing:{component.name}",
                        lower,
                        upper,
                        kind="mixing",
                        role=component.role,
                    )
                )
            if self.config.include_component_ext_params:
                for name, bounds in sorted(component.ext_param_bounds.items()):
                    parameters.append(
                        EvolutionParameter(
                            f"{component.name}.{name}",
                            float(bounds[0]),
                            float(bounds[1]),
                            kind="ext_param",
                            role=component.role,
                        )
                    )
        return parameters

    @staticmethod
    def _read_parameters(
        xc: XCFunctional,
        parameters: list[EvolutionParameter],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        components = {component.name: component for component in xc.components}
        for parameter in parameters:
            if parameter.key in {"alpha", "beta", "omega"}:
                values[parameter.key] = float(getattr(xc.hf_exchange, parameter.key))
                continue
            if parameter.key.startswith("mixing:"):
                component_name = parameter.key[len("mixing:"):]
                values[parameter.key] = float(components[component_name].mixing_factor)
                continue
            if parameter.key.startswith("scale:"):
                role = parameter.key[len("scale:"):]
                values[parameter.key] = float(xc.role_scales.get(role, 1.0))
                continue
            component_name, parameter_name = parameter.key.split(".", 1)
            values[parameter.key] = float(components[component_name].ext_params[parameter_name])
        return values

    def _initial_population(
        self,
        baseline: dict[str, float],
        parameters: list[EvolutionParameter],
        *,
        seed_vectors: list[dict[str, float]] | None = None,
    ) -> list[dict[str, float]]:
        population: list[dict[str, float]] = []
        seen: set[tuple[float, ...]] = set()

        def _key(vector: dict[str, float]) -> tuple[float, ...]:
            return tuple(round(float(vector[item.key]), 10) for item in parameters)

        def _append(vector: dict[str, float]) -> None:
            if len(population) >= self.config.population_size:
                return
            repaired = self._repair_vector(dict(vector), parameters)
            key = _key(repaired)
            if key in seen:
                return
            seen.add(key)
            population.append(repaired)

        _append(baseline)
        for vector in seed_vectors or []:
            _append(vector)

        remaining = self.config.population_size - len(population)
        if remaining <= 0:
            return population

        for vector in self._latin_vectors(remaining, parameters):
            _append(vector)

        while len(population) < self.config.population_size:
            _append(self._random_vector(parameters))
        return population

    def _latin_vectors(
        self,
        count: int,
        parameters: list[EvolutionParameter],
    ) -> list[dict[str, float]]:
        if count <= 0:
            return []
        permutations = []
        for _parameter in parameters:
            slots = list(range(count))
            self.random.shuffle(slots)
            permutations.append(slots)
        vectors = []
        for index in range(count):
            vector = {}
            for parameter_index, parameter in enumerate(parameters):
                fraction = (permutations[parameter_index][index] + self.random.random()) / max(count, 1)
                vector[parameter.key] = parameter.lower + fraction * (parameter.upper - parameter.lower)
            vectors.append(self._repair_vector(vector, parameters))
        return vectors

    def _random_vector(
        self,
        parameters: list[EvolutionParameter],
    ) -> dict[str, float]:
        return self._repair_vector(
            {
                parameter.key: self.random.uniform(parameter.lower, parameter.upper)
                for parameter in parameters
            },
            parameters,
        )

    def _next_population(
        self,
        ranked: list[EvolutionCandidate],
        parameters: list[EvolutionParameter],
        *,
        inject_count: int = 0,
    ) -> list[dict[str, float]]:
        if not ranked:
            raise RuntimeError("Evolutionary search produced an empty population")
        elites = [
            self._repair_vector(dict(candidate.parameters), parameters)
            for candidate in ranked[: self.config.elite_size]
        ]
        population = list(elites)
        seen = {self._vector_key(vector, parameters) for vector in population}

        def _append(vector: dict[str, float]) -> bool:
            repaired = self._repair_vector(vector, parameters)
            key = self._vector_key(repaired, parameters)
            if key in seen:
                return False
            seen.add(key)
            population.append(repaired)
            return True

        inject_count = max(0, min(int(inject_count), self.config.population_size - len(population)))
        crossover_target = self.config.population_size - inject_count
        attempts = 0
        while len(population) < crossover_target and attempts < self.config.population_size * 50:
            attempts += 1
            left = self._tournament(ranked)
            if self.random.random() < self.config.crossover_rate:
                right = self._tournament(ranked)
                child = self._crossover(left.parameters, right.parameters, parameters)
            else:
                child = dict(left.parameters)
            _append(self._mutate(child, parameters))

        for vector in self._latin_vectors(inject_count, parameters):
            if len(population) >= self.config.population_size:
                break
            _append(vector)

        while len(population) < self.config.population_size:
            _append(self._random_vector(parameters))
        return population

    @staticmethod
    def _vector_key(
        vector: dict[str, float],
        parameters: list[EvolutionParameter],
    ) -> tuple[float, ...]:
        return tuple(round(float(vector[item.key]), 10) for item in parameters)

    def _tournament(self, population: list[EvolutionCandidate]) -> EvolutionCandidate:
        size = min(self.config.tournament_size, len(population))
        competitors = self.random.sample(population, size)
        return max(competitors, key=self._rank_key)

    def _crossover(
        self,
        left: dict[str, float],
        right: dict[str, float],
        parameters: list[EvolutionParameter],
    ) -> dict[str, float]:
        child = {}
        for parameter in parameters:
            weight = self.random.random()
            child[parameter.key] = parameter.clip(
                weight * left[parameter.key] + (1.0 - weight) * right[parameter.key]
            )
        return child

    def _mutate(
        self,
        vector: dict[str, float],
        parameters: list[EvolutionParameter],
    ) -> dict[str, float]:
        mutated = dict(vector)
        changed = False
        for parameter in parameters:
            if self.random.random() >= self.config.mutation_rate:
                continue
            sigma = self.config.mutation_scale * (parameter.upper - parameter.lower)
            mutated[parameter.key] = parameter.clip(
                mutated[parameter.key] + self.random.gauss(0.0, sigma)
            )
            changed = True
        if not changed:
            parameter = self.random.choice(parameters)
            sigma = self.config.mutation_scale * (parameter.upper - parameter.lower)
            mutated[parameter.key] = parameter.clip(
                mutated[parameter.key] + self.random.gauss(0.0, sigma)
            )
        return self._repair_vector(mutated, parameters)

    def _repair_vector(
        self,
        vector: dict[str, float],
        parameters: list[EvolutionParameter],
    ) -> dict[str, float]:
        """Project random EA vectors back into the hard XC sum-rule domain."""
        repaired = {
            parameter.key: parameter.clip(vector.get(parameter.key, parameter.lower))
            for parameter in parameters
        }
        alpha = repaired.get("alpha", 0.0)
        if "beta" in repaired and alpha + repaired["beta"] > 1.0:
            repaired["beta"] = max(0.0, 1.0 - alpha)

        def _scale(keys: list[str], limit: float) -> None:
            total = sum(repaired[key] for key in keys)
            if total <= limit or total <= 0:
                return
            scale = max(0.0, limit) / total
            for key in keys:
                repaired[key] *= scale

        def _project_exact(keys: list[str], target: float) -> None:
            if not keys:
                return
            by_key = {parameter.key: parameter for parameter in parameters}
            lower_total = sum(by_key[key].lower for key in keys)
            upper_total = sum(by_key[key].upper for key in keys)
            target = min(upper_total, max(lower_total, float(target)))

            # Project onto a bounded simplex without violating protected
            # component floors such as the HLE16 backbone mixing >= 0.5.
            for _ in range(len(keys) * 2 + 2):
                delta = target - sum(repaired[key] for key in keys)
                if abs(delta) <= 1e-12:
                    break
                if delta > 0:
                    adjustable = [key for key in keys if repaired[key] < by_key[key].upper - 1e-12]
                    capacities = {key: by_key[key].upper - repaired[key] for key in adjustable}
                else:
                    adjustable = [key for key in keys if repaired[key] > by_key[key].lower + 1e-12]
                    capacities = {key: repaired[key] - by_key[key].lower for key in adjustable}
                capacity = sum(capacities.values())
                if not adjustable or capacity <= 1e-12:
                    break
                amount = min(abs(delta), capacity)
                direction = 1.0 if delta > 0 else -1.0
                for key in adjustable:
                    repaired[key] += direction * amount * capacities[key] / capacity
                    repaired[key] = by_key[key].clip(repaired[key])

        exchange_mix = [
            parameter.key
            for parameter in parameters
            if parameter.kind == "mixing" and parameter.role == "exchange"
        ]
        correlation_mix = [
            parameter.key
            for parameter in parameters
            if parameter.kind == "mixing" and parameter.role == "correlation"
        ]
        exchange_target = max(0.0, 1.0 - alpha)
        if self.config.exact_mixing_sum:
            _project_exact(exchange_mix, exchange_target)
            _project_exact(correlation_mix, 1.0)
        else:
            _scale(exchange_mix, exchange_target)
            _scale(correlation_mix, 1.0)
        return repaired

    def _build_candidate(
        self,
        root_xc: XCFunctional,
        parameters: dict[str, float],
        generation: int,
    ) -> XCFunctional:
        search_parameters = self._parameter_space(root_xc)
        parameters = self._repair_vector(parameters, search_parameters)
        candidate = copy.deepcopy(root_xc)
        components = {component.name: component for component in candidate.components}
        for key, value in parameters.items():
            if key in {"alpha", "beta", "omega"}:
                setattr(candidate.hf_exchange, key, float(value))
                continue
            if key.startswith("mixing:"):
                component_name = key[len("mixing:"):]
                components[component_name].mixing_factor = float(value)
                continue
            if key.startswith("scale:"):
                role = key[len("scale:"):]
                candidate.role_scales[role] = float(value)
                continue
            component_name, parameter_name = key.split(".", 1)
            components[component_name].ext_params[parameter_name] = float(value)

        parameter_label = "_".join(
            f"{key.replace(':', '_').split('.')[-1].lstrip('_')}={value:.5g}"
            for key, value in sorted(parameters.items())
        )
        candidate.name = f"{root_xc.name}__evo_g{generation}_{parameter_label}"
        if len(candidate.name) > 150:
            candidate.name = f"{root_xc.name}__evo_g{generation}_{candidate.structural_signature()}"
        candidate.metadata.update(
            {
                "action": "evolutionary_rewire",
                "action_detail": dict(parameters),
                "search_algorithm": "evolutionary",
                "generation": generation,
            }
        )
        return candidate

    def _score(
        self,
        candidate: EvolutionCandidate,
        baseline_search: BenchmarkResult,
        baseline_validation: BenchmarkResult,
    ) -> None:
        search_result = candidate.search_result
        validation_result = candidate.validation_result
        if search_result is None or validation_result is None:
            candidate.failure_reason = "evaluation_failure"
            return
        search_ok = sum(item.n_reactions for item in search_result.subsets.values()) > 0
        validation_ok = sum(item.n_reactions for item in validation_result.subsets.values()) > 0
        hard_violations = [
            violation for violation in candidate.xc.validate()
            if violation.startswith("HARD:")
        ]
        if not search_ok or not validation_ok:
            candidate.failure_reason = "evaluation_failure"
            candidate.fitness = float("-inf")
            candidate.tree_reward = 0.0
            return
        if hard_violations:
            candidate.failure_reason = "hard_constraint_violation"
            candidate.fitness = float("-inf")
            candidate.tree_reward = 0.0
            return
        if search_result.total_failures > 0:
            candidate.failure_reason = "search_runtime_failure"
            candidate.fitness = float("-inf")
            candidate.tree_reward = 0.0
            return

        candidate.validation_ratio = (
            validation_result.overall_mae / max(baseline_validation.overall_mae, 1e-9)
        )
        for name, subset in validation_result.subsets.items():
            baseline_subset = baseline_validation.subsets.get(name)
            if baseline_subset and baseline_subset.n_reactions > 0:
                candidate.validation_subset_ratios[name] = (
                    subset.mae_kcal / max(baseline_subset.mae_kcal, 1e-9)
                )
        candidate.worst_validation_ratio = max(
            [candidate.validation_ratio, *candidate.validation_subset_ratios.values()]
        )
        candidate.guard_penalty, candidate.guard_regression_units = guard_regression_penalty(
            validation_result,
            baseline_validation,
            tolerances=self.config.guard_tolerances,
            mean_weight=self.config.guard_mean_weight,
            max_weight=self.config.guard_max_weight,
        )
        candidate.promotable = True
        candidate.failure_reason = ""

        search_ratio = search_result.overall_mae / max(baseline_search.overall_mae, 1e-9)
        candidate.fitness = target_guard_raw_reward(
            search_ratio=search_ratio,
            guard_penalty=self.config.guard_regression_penalty * candidate.guard_penalty,
            target_gain_weight=self.config.target_gain_weight,
            runtime_failure_penalty=self.config.runtime_failure_penalty,
            validation_failures=validation_result.total_failures,
        )
        candidate.tree_reward = tree_reward_from_raw(candidate.fitness)

    @staticmethod
    def _rank_key(candidate: EvolutionCandidate) -> tuple[float, float, float]:
        return (
            1.0 if candidate.promotable else 0.0,
            candidate.fitness,
            -candidate.worst_validation_ratio if math.isfinite(candidate.worst_validation_ratio) else -1e9,
        )

    def _generation_summary(
        self,
        generation: int,
        ranked: list[EvolutionCandidate],
        best: EvolutionCandidate,
        parameters: list[EvolutionParameter] | None = None,
    ) -> dict[str, Any]:
        diversity = self._population_diversity(ranked, parameters or [])
        return {
            "generation": generation,
            "unique_candidates": len({item.xc.effective_signature() for item in ranked}),
            "promotable_candidates": sum(item.promotable for item in ranked),
            "best_name": best.xc.name,
            "best_fitness": best.fitness,
            "best_search_mae": best.search_result.overall_mae,
            "best_validation_mae": best.validation_result.overall_mae,
            "best_worst_guard_ratio": best.worst_validation_ratio,
            "best_guard_penalty": best.guard_penalty,
            "best_tree_reward": best.tree_reward,
            "best_parameters": dict(best.parameters),
            **diversity,
        }

    @staticmethod
    def _population_diversity(
        ranked: list[EvolutionCandidate],
        parameters: list[EvolutionParameter],
    ) -> dict[str, float]:
        if len(ranked) < 2 or not parameters:
            return {
                "mean_pairwise_parameter_distance": 0.0,
                "mean_parameter_std": 0.0,
            }
        keys = [parameter.key for parameter in parameters]
        bounds = {
            parameter.key: (
                float(parameter.lower),
                float(parameter.upper) if parameter.upper > parameter.lower else float(parameter.lower) + 1.0,
            )
            for parameter in parameters
        }

        vectors: list[list[float]] = []
        for candidate in ranked:
            vector = []
            for key in keys:
                lo, hi = bounds[key]
                value = float(candidate.parameters.get(key, lo))
                vector.append((value - lo) / (hi - lo))
            vectors.append(vector)

        pairwise_total = 0.0
        pairwise_count = 0
        for left_index in range(len(vectors)):
            for right_index in range(left_index + 1, len(vectors)):
                pairwise_total += math.sqrt(
                    sum(
                        (left - right) ** 2
                        for left, right in zip(vectors[left_index], vectors[right_index])
                    )
                )
                pairwise_count += 1
        mean_pairwise = pairwise_total / pairwise_count if pairwise_count else 0.0

        std_total = 0.0
        for column in zip(*vectors):
            mean = sum(column) / len(column)
            variance = sum((value - mean) ** 2 for value in column) / len(column)
            std_total += math.sqrt(variance)
        mean_std = std_total / len(keys) if keys else 0.0
        return {
            "mean_pairwise_parameter_distance": mean_pairwise,
            "mean_parameter_std": mean_std,
        }


class MixedEvolutionarySearch(EvolutionarySearch):
    """EA that evolves a sparse XC component scaffold and its parameters together.

    Each individual owns an ``XCFunctional`` rather than sharing one fixed
    parameter space.  This makes add/replace/remove legitimate genetic moves
    while retaining the established continuous EA scorer and guard handling.
    """

    def __init__(self, config: MixedEvolutionConfig | None = None):
        config = config or MixedEvolutionConfig()
        super().__init__(config)
        self.config: MixedEvolutionConfig = config

    def search(
        self,
        *,
        root_xc: XCFunctional,
        root_result: BenchmarkResult,
        validation_runner,
        runner,
        baseline_validation_result: BenchmarkResult | None = None,
        initial_scaffolds: list[tuple[str, XCFunctional]] | None = None,
        initialization_mode: str = "random_structure",
        known_library_signatures: set[str] | None = None,
        on_evaluation: Callable[[EvolutionCandidate], None] | None = None,
    ) -> EvolutionSearchResult:
        """Run mixed EA from random or supplied component scaffolds.

        ``initial_scaffolds`` are evaluated unchanged first.  Remaining first
        generation slots use parameter-randomized copies of those scaffolds;
        absent supplied scaffolds, they use random legal component edits of
        ``root_xc``.  All later generations use the same structural and
        continuous mutation operators.
        """
        baseline_validation = baseline_validation_result or validation_runner.evaluate(root_xc)
        root_parameters = self._parameter_space(root_xc)
        baseline = EvolutionCandidate(
            xc=copy.deepcopy(root_xc),
            parameters=self._read_parameters(root_xc, root_parameters),
            generation=0,
            search_result=root_result,
            validation_result=baseline_validation,
        )
        self._score(baseline, root_result, baseline_validation)

        if initialization_mode not in {"random_structure", "latin_root"}:
            raise ValueError(f"Unsupported mixed EA initialization mode: {initialization_mode}")
        blocked = set(known_library_signatures or set())
        logger.info(
            "Mixed evolutionary search start: population=%d generations=%d structural_mutation=%.2f "
            "max_components_per_role=%d initialization=%s supplied_scaffolds=%d baseline_fitness=%.6f",
            self.config.population_size,
            self.config.generations,
            self.config.structural_mutation_rate,
            self.config.max_components_per_role,
            initialization_mode,
            len(initial_scaffolds or []),
            baseline.fitness,
        )

        evaluated: list[EvolutionCandidate] = [baseline]
        evaluated_by_sig: dict[str, EvolutionCandidate] = {
            baseline.xc.effective_signature(): baseline,
        }
        population_xcs = self._initial_scaffolds(
            root_xc,
            initial_scaffolds=initial_scaffolds,
            initialization_mode=initialization_mode,
            blocked_signatures=blocked,
        )
        best = baseline
        generation_summaries: list[dict[str, Any]] = []

        for generation in range(1, self.config.generations + 1):
            population: list[EvolutionCandidate] = []
            pending: list[tuple[str, EvolutionCandidate]] = []
            for xc in population_xcs:
                effective_sig = xc.effective_signature()
                cached = evaluated_by_sig.get(effective_sig)
                if cached is not None:
                    population.append(cached)
                    continue

                pending.append((
                    effective_sig,
                    EvolutionCandidate(
                        xc=xc,
                        parameters=self._read_parameters(xc, self._parameter_space(xc)),
                        generation=generation,
                    ),
                ))

            if pending:
                logger.info(
                    "Mixed evolution generation %d evaluating %d new candidates with concurrency=%d",
                    generation,
                    len(pending),
                    self.config.evaluation_concurrency,
                )
            completed_candidates = self._evaluate_pending_candidates(
                pending,
                runner=runner,
                validation_runner=validation_runner,
                root_result=root_result,
                baseline_validation=baseline_validation,
            )
            for effective_sig, candidate in completed_candidates:
                evaluated.append(candidate)
                evaluated_by_sig[effective_sig] = candidate
                population.append(candidate)
                if on_evaluation is not None:
                    on_evaluation(candidate)
                logger.info(
                    "Mixed evolution candidate generation=%d name=%s fitness=%.6f tree_reward=%.6f "
                    "promotable=%s search_mae=%.4f validation_mae=%.4f worst_guard=%.4f "
                    "guard_penalty=%.4f structure=%s params=%s",
                    generation,
                    candidate.xc.name,
                    candidate.fitness,
                    candidate.tree_reward,
                    candidate.promotable,
                    candidate.search_result.overall_mae,
                    candidate.validation_result.overall_mae,
                    candidate.worst_validation_ratio,
                    candidate.guard_penalty,
                    json.dumps(candidate.xc.metadata.get("structure_operations", []), sort_keys=True),
                    json.dumps(candidate.parameters, sort_keys=True),
                )

            ranked = sorted(population, key=self._rank_key, reverse=True)
            if ranked and self._rank_key(ranked[0]) > self._rank_key(best):
                best = ranked[0]
                logger.info(
                    "Mixed evolution new best: generation=%d name=%s fitness=%.6f search_mae=%.4f "
                    "worst_guard=%.4f structure=%s params=%s",
                    generation,
                    best.xc.name,
                    best.fitness,
                    best.search_result.overall_mae,
                    best.worst_validation_ratio,
                    json.dumps(best.xc.metadata.get("structure_operations", []), sort_keys=True),
                    json.dumps(best.parameters, sort_keys=True),
                )

            summary = self._generation_summary(generation, ranked, best)
            summary["mixed_structure_search"] = True
            generation_summaries.append(summary)
            logger.info(
                "Mixed evolution generation %d complete: unique=%d promotable=%d best_fitness=%.6f "
                "best_search_mae=%.4f best_worst_guard=%.4f",
                generation,
                summary["unique_candidates"],
                summary["promotable_candidates"],
                summary["best_fitness"],
                summary["best_search_mae"],
                summary["best_worst_guard_ratio"],
            )
            population_xcs = self._next_mixed_population(
                ranked,
                root_xc=root_xc,
                generation=generation + 1,
                blocked_signatures=blocked,
            )

        result = EvolutionSearchResult(
            best=best,
            baseline=baseline,
            evaluated=evaluated,
            generation_summaries=generation_summaries,
            parameters=root_parameters,
        )
        self.last_result = result
        return result

    def _initial_scaffolds(
        self,
        root_xc: XCFunctional,
        *,
        initial_scaffolds: list[tuple[str, XCFunctional]] | None,
        initialization_mode: str,
        blocked_signatures: set[str],
    ) -> list[XCFunctional]:
        if initialization_mode == "latin_root":
            parameters = self._parameter_space(root_xc)
            baseline_vector = self._read_parameters(root_xc, parameters)
            vectors = self._initial_population(baseline_vector, parameters)
            population: list[XCFunctional] = []
            for index, vector in enumerate(vectors):
                candidate = (
                    copy.deepcopy(root_xc)
                    if index == 0
                    else self._build_candidate(root_xc, vector, generation=1)
                )
                candidate.metadata.update({
                    "action": "mixed_evolution_latin_initialization",
                    "action_detail": {"parameters": dict(vector)},
                    "search_algorithm": "mixed_evolutionary",
                    "structure_source": "llm_component_union_latin",
                    "structure_operations": copy.deepcopy(
                        root_xc.metadata.get("llm_component_union", [])
                    ),
                })
                if index:
                    self._name_mixed(candidate, generation=1, root_name=root_xc.name)
                population.append(candidate)
            logger.info(
                "Mixed EA Latin initialization: root=%s population=%d parameters=%s",
                root_xc.name,
                len(population),
                [parameter.key for parameter in parameters],
            )
            return population

        population: list[XCFunctional] = [copy.deepcopy(root_xc)]
        supplied = [(label, copy.deepcopy(xc)) for label, xc in initial_scaffolds or []]
        seen = {root_xc.effective_signature()}

        # Preserve supplied LLM structures exactly before creating any numerical variants.
        for label, scaffold in supplied:
            if len(population) >= self.config.population_size:
                break
            if scaffold.effective_signature() in seen:
                continue
            scaffold.metadata.update({
                "action": "mixed_evolution_initial_structure",
                "action_detail": {"source": "llm_structure_initial", "label": label},
                "search_algorithm": "mixed_evolutionary",
                "structure_source": "llm_initial",
                "structure_operations": copy.deepcopy(scaffold.metadata.get("action_detail", {}).get("actions", [])),
            })
            self._name_mixed(scaffold, generation=1)
            population.append(scaffold)
            seen.add(scaffold.effective_signature())

        source_scaffolds = supplied or [("baseline", root_xc)]
        attempts = 0
        while len(population) < self.config.population_size and attempts < self.config.population_size * 30:
            attempts += 1
            if supplied:
                label, base = source_scaffolds[(len(population) - 1) % len(source_scaffolds)]
                candidate = self._randomize_parameters(copy.deepcopy(base), generation=1)
                candidate.metadata.update({
                    "structure_source": "llm_initial_parameter_lhs",
                    "initial_scaffold_label": label,
                })
            else:
                candidate, operations = self._random_structure_variant(root_xc)
                candidate = self._randomize_parameters(candidate, generation=1)
                candidate.metadata.update({
                    "structure_source": "algorithmic_random",
                    "structure_operations": operations,
                })
            self._name_mixed(candidate, generation=1)
            signature = candidate.effective_signature()
            if signature in seen or self._is_blocked(candidate, blocked_signatures):
                continue
            population.append(candidate)
            seen.add(signature)

        if len(population) < self.config.population_size:
            logger.warning(
                "Mixed EA initialized only %d/%d unique candidates after %d attempts",
                len(population), self.config.population_size, attempts,
            )
        return population

    def _next_mixed_population(
        self,
        ranked: list[EvolutionCandidate],
        *,
        root_xc: XCFunctional,
        generation: int,
        blocked_signatures: set[str],
    ) -> list[XCFunctional]:
        if not ranked:
            raise RuntimeError("Mixed evolutionary search produced an empty population")
        population = [copy.deepcopy(candidate.xc) for candidate in ranked[: self.config.elite_size]]
        seen = {candidate.effective_signature() for candidate in population}
        attempts = 0
        while len(population) < self.config.population_size and attempts < self.config.population_size * 40:
            attempts += 1
            left = self._tournament(ranked)
            right = self._tournament(ranked) if self.random.random() < self.config.crossover_rate else None
            child = self._mixed_child(left, right, root_xc=root_xc, generation=generation)
            signature = child.effective_signature()
            if signature in seen or self._is_blocked(child, blocked_signatures):
                continue
            population.append(child)
            seen.add(signature)
        if len(population) < self.config.population_size:
            logger.warning(
                "Mixed EA generation %d filled only %d/%d unique candidates",
                generation, len(population), self.config.population_size,
            )
        return population

    def _mixed_child(
        self,
        left: EvolutionCandidate,
        right: EvolutionCandidate | None,
        *,
        root_xc: XCFunctional,
        generation: int,
    ) -> XCFunctional:
        base = copy.deepcopy(left.xc)
        parameters = self._parameter_space(base)
        vector = self._read_parameters(base, parameters)
        if right is not None:
            right_values = self._read_parameters(right.xc, self._parameter_space(right.xc))
            for parameter in parameters:
                if parameter.key in right_values:
                    weight = self.random.random()
                    vector[parameter.key] = parameter.clip(
                        weight * vector[parameter.key] + (1.0 - weight) * right_values[parameter.key]
                    )
        vector = self._mutate(vector, parameters)
        child = self._build_candidate(base, vector, generation)
        operations: list[dict[str, Any]] = []
        if self.random.random() < self.config.structural_mutation_rate:
            child, operation = self._mutate_structure(child)
            if operation is not None:
                operations.append(operation)

        child.metadata.update({
            "action": "evolutionary_mixed",
            "action_detail": {
                "continuous": self._read_parameters(child, self._parameter_space(child)),
                "structure": operations,
            },
            "search_algorithm": "mixed_evolutionary",
            "generation": generation,
            "parent_names": [left.xc.name, right.xc.name if right is not None else ""],
            "structure_operations": operations,
        })
        self._name_mixed(child, generation=generation, root_name=root_xc.name)
        return child

    def _random_structure_variant(self, root_xc: XCFunctional) -> tuple[XCFunctional, list[dict[str, Any]]]:
        candidate = copy.deepcopy(root_xc)
        operations: list[dict[str, Any]] = []
        edits = self.random.randint(1, self.config.initial_structure_edits)
        for _ in range(edits):
            candidate, operation = self._mutate_structure(candidate)
            if operation is None:
                break
            operations.append(operation)
        return candidate, operations

    def _mutate_structure(self, xc: XCFunctional) -> tuple[XCFunctional, dict[str, Any] | None]:
        options: list[Action] = []
        for role in ("exchange", "correlation"):
            components = [component for component in xc.components if component.role == role]
            names = {component.name for component in components}
            component_pool = component_roles_for_pool(self.config.component_pool)
            # HJS-PBE is the LC-wPBE exchange backbone.  It is omitted from
            # PURE_COMPONENT_ROLES for proposal prompting, but a genetic
            # replacement must be able to mutate back to this valid component.
            if self.config.component_pool != "zno_gap":
                component_pool["GGA_X_HJS_PBE"] = "exchange"
            candidates = sorted(
                name for name, component_role in component_pool.items()
                if component_role == role and name not in names
            )
            for component in components:
                for new_name in candidates:
                    options.append(Action(
                        ActionType.REPLACE_COMPONENT,
                        changes={"old_name": component.name, "new_name": new_name},
                    ))
            if len(components) < self.config.max_components_per_role:
                for new_name in candidates:
                    options.append(Action(
                        ActionType.ADD_COMPONENT,
                        changes={"name": new_name, "role": role},
                    ))
            if len(components) > 1:
                for component in components:
                    options.append(Action(
                        ActionType.REMOVE_COMPONENT,
                        changes={"name": component.name},
                    ))

        self.random.shuffle(options)
        for action in options:
            try:
                before = next(
                    (component for component in xc.components if component.name == action.changes.get("name")),
                    None,
                )
                removed_weight = before.mixing_factor if before is not None else 0.0
                candidate = apply_action(xc, action)
                if action.action_type == ActionType.REMOVE_COMPONENT and removed_weight > 0:
                    role = before.role if before is not None else ""
                    recipients = [component for component in candidate.components if component.role == role]
                    if recipients:
                        max(recipients, key=lambda component: component.mixing_factor).mixing_factor += removed_weight
                if any(item.startswith("HARD:") for item in candidate.validate()):
                    continue
                return candidate, {
                    "op": action.action_type.value,
                    "changes": copy.deepcopy(action.changes),
                }
            except (ValueError, KeyError, TypeError):
                continue
        return copy.deepcopy(xc), None

    def _randomize_parameters(self, xc: XCFunctional, *, generation: int) -> XCFunctional:
        parameters = self._parameter_space(xc)
        vector = {
            parameter.key: self.random.uniform(parameter.lower, parameter.upper)
            for parameter in parameters
        }
        return self._build_candidate(xc, self._repair_vector(vector, parameters), generation)

    @staticmethod
    def _is_blocked(xc: XCFunctional, blocked_signatures: set[str]) -> bool:
        return bool({xc.structural_signature(), xc.effective_signature()} & blocked_signatures)

    @staticmethod
    def _name_mixed(xc: XCFunctional, *, generation: int, root_name: str | None = None) -> None:
        prefix = (root_name or xc.name).split("__", 1)[0]
        xc.name = f"{prefix}__mixed_g{generation}_{xc.effective_signature()}"
