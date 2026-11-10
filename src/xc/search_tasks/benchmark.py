"""Adapter for the original molecular benchmark and guard objective."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from ..benchmark.result import BenchmarkResult
from ..xc_functional import XCFunctional
from .base import SearchTaskEvaluation


def _result_metrics(result: BenchmarkResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        "overall_mae": result.overall_mae,
        "total_failures": result.total_failures,
        "subsets": {
            name: {
                "mae_kcal": subset.mae_kcal,
                "n_failures": subset.n_failures,
            }
            for name, subset in result.subsets.items()
        },
    }


class BenchmarkSearchTask:
    """Preserve the historical SIE+guard scoring behind the task interface."""

    name = "molecular_benchmark"
    use_benchmark_component_priors = True

    def __init__(
        self,
        *,
        scorer: Any,
        root_result: BenchmarkResult,
        baseline_validation: BenchmarkResult,
        runner: Any,
        validation_runner: Any,
        evaluation_concurrency: int = 1,
    ) -> None:
        self.scorer = scorer
        self.root_result = root_result
        self.baseline_validation = baseline_validation
        self.runner = runner
        self.validation_runner = validation_runner
        self.evaluation_concurrency = max(1, int(evaluation_concurrency))

    def validate_functional(self, xc: XCFunctional) -> None:
        hard = [item for item in xc.validate() if item.startswith("HARD:")]
        if hard:
            raise ValueError("; ".join(hard))

    def from_existing(
        self,
        search_result: BenchmarkResult,
        validation_result: BenchmarkResult,
        candidate: Any,
    ) -> SearchTaskEvaluation:
        candidate.search_result = search_result
        candidate.validation_result = validation_result
        self.scorer._score(candidate, self.root_result, self.baseline_validation)
        return self._evaluation(candidate)

    def evaluate_many(
        self,
        xcs: Sequence[XCFunctional],
        *,
        labels: Sequence[str],
        stage: str,
    ) -> list[SearchTaskEvaluation]:
        del labels
        candidates = [self._empty_candidate(xc) for xc in xcs]
        if self._has_batch_api(self.runner) and self._has_batch_api(self.validation_runner):
            search_results = self.runner.evaluate_many(list(xcs), label_prefix=f"{stage}_search")
            guard_results = self.validation_runner.evaluate_many(list(xcs), label_prefix=f"{stage}_guard")
            return [
                self._score_results(candidate, search, guard)
                for candidate, search, guard in zip(candidates, search_results, guard_results)
            ]

        def run(index: int) -> tuple[int, SearchTaskEvaluation]:
            candidate = candidates[index]
            search = self.runner.evaluate(candidate.xc)
            guard = self.validation_runner.evaluate(candidate.xc)
            return index, self._score_results(candidate, search, guard)

        if self.evaluation_concurrency <= 1 or len(candidates) <= 1:
            return [run(index)[1] for index in range(len(candidates))]
        ordered: list[SearchTaskEvaluation | None] = [None] * len(candidates)
        with ThreadPoolExecutor(max_workers=min(self.evaluation_concurrency, len(candidates))) as executor:
            futures = [executor.submit(run, index) for index in range(len(candidates))]
            for future in as_completed(futures):
                index, evaluation = future.result()
                ordered[index] = evaluation
        return [item for item in ordered if item is not None]

    def apply_to_candidate(self, candidate: Any, evaluation: SearchTaskEvaluation) -> None:
        raw = evaluation.raw_result or {}
        candidate.search_result = raw.get("search_result")
        candidate.validation_result = raw.get("validation_result")
        candidate.fitness = evaluation.reward
        candidate.promotable = evaluation.valid
        candidate.failure_reason = evaluation.failure_reason
        candidate.tree_reward = float(raw.get("tree_reward", 0.0))
        candidate.guard_penalty = float(raw.get("guard_penalty", 0.0))
        candidate.worst_validation_ratio = float(raw.get("worst_validation_ratio", float("inf")))
        candidate.validation_ratio = raw.get("validation_ratio")
        candidate.validation_subset_ratios = dict(raw.get("validation_subset_ratios") or {})
        candidate.guard_regression_units = dict(raw.get("guard_regression_units") or {})

    def prompt_instructions(self) -> str:
        return (
            "The task reward is the existing molecular target/guard trade-off. "
            "Search and guard MAEs are objective evidence used by this task."
        )

    def relative_feedback(
        self,
        child: SearchTaskEvaluation,
        parent: SearchTaskEvaluation | None,
    ) -> dict[str, Any]:
        if parent is None:
            return {}
        return {
            "reward": {"parent": parent.reward, "child": child.reward, "delta": child.reward - parent.reward},
        }

    def _score_results(
        self,
        candidate: Any,
        search_result: BenchmarkResult,
        validation_result: BenchmarkResult,
    ) -> SearchTaskEvaluation:
        candidate.search_result = search_result
        candidate.validation_result = validation_result
        self.scorer._score(candidate, self.root_result, self.baseline_validation)
        return self._evaluation(candidate)

    def _evaluation(self, candidate: Any) -> SearchTaskEvaluation:
        return SearchTaskEvaluation(
            task_name=self.name,
            valid=bool(candidate.promotable and math.isfinite(candidate.fitness)),
            reward=candidate.fitness,
            objective={
                "search": _result_metrics(candidate.search_result),
                "guard": _result_metrics(candidate.validation_result),
                "guard_penalty": candidate.guard_penalty,
                "worst_validation_ratio": candidate.worst_validation_ratio,
            },
            proposal_feedback={},
            failure_reason=candidate.failure_reason,
            walltime_s=sum(
                result.total_walltime_s
                for result in (candidate.search_result, candidate.validation_result)
                if result is not None
            ),
            raw_result={
                "search_result": candidate.search_result,
                "validation_result": candidate.validation_result,
                "tree_reward": candidate.tree_reward,
                "guard_penalty": candidate.guard_penalty,
                "worst_validation_ratio": candidate.worst_validation_ratio,
                "validation_ratio": candidate.validation_ratio,
                "validation_subset_ratios": candidate.validation_subset_ratios,
                "guard_regression_units": candidate.guard_regression_units,
            },
        )

    @staticmethod
    def _has_batch_api(runner: Any) -> bool:
        return getattr(runner, "execution_backend", "") == "slurm_docker_queue" and callable(
            getattr(runner, "evaluate_many", None)
        )

    @staticmethod
    def _empty_candidate(xc: XCFunctional) -> Any:
        from ..idea.evolutionary import EvolutionCandidate

        return EvolutionCandidate(xc=xc, parameters={}, generation=0)
