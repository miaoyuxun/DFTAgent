"""Task adapter for maximizing the negative mean MAE of benchmark subsets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from ..benchmark.result import BenchmarkResult, SubsetResult
from ..idea.reward import tree_reward_from_raw
from ..xc_functional import LIBXC_PARAMS, NAME_TO_ID, XCFunctional
from ..zno_component_registry import validate_functional_for_pool
from .base import SearchTaskEvaluation


@dataclass(frozen=True)
class MeanMAETaskConfig:
    """Configuration for an equal-subset-weight molecular objective."""

    subsets: tuple[str, ...]
    component_pool: str = "zno_gap"
    max_failures_per_subset: int = 0
    allow_hf_exchange: bool = True
    allow_ext_param_rewire: bool = False

    def __post_init__(self) -> None:
        normalized = tuple(str(name).strip() for name in self.subsets if str(name).strip())
        if not normalized:
            raise ValueError("MeanMAETaskConfig.subsets cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"MeanMAETaskConfig.subsets contains duplicates: {normalized}")
        object.__setattr__(self, "subsets", normalized)
        if self.max_failures_per_subset < 0:
            raise ValueError("max_failures_per_subset must be non-negative")


class MeanMAEBenchmarkTask:
    """Score candidates by the negative arithmetic mean of subset MAEs.

    Every configured subset receives equal weight, independent of its number of
    reactions. Incomplete benchmark results are rejected instead of allowing a
    candidate to improve by failing difficult reactions.
    """

    name = "benchmark_mean_mae"
    use_benchmark_component_priors = False

    def __init__(self, *, runner: Any, config: MeanMAETaskConfig) -> None:
        self.runner = runner
        self.config = config

    def validate_functional(self, xc: XCFunctional) -> None:
        validate_functional_for_pool(xc, component_pool=self.config.component_pool)
        hf = xc.hf_exchange
        beta = float(hf.beta)
        omega = float(hf.omega)
        if beta >= 1e-6 and omega < 1e-4:
            raise ValueError("Range-separated HF requires beta > 0 and omega >= 1e-4 together")
        if beta < 1e-6 and omega > 1e-12 and not any(
            component.name == "GGA_X_HJS_PBE" or component.role == "exchange_correlation"
            for component in xc.components
        ):
            raise ValueError("omega has no effect when beta = 0 for ordinary full-range exchange")
        if not self.config.allow_hf_exchange:
            if any(abs(float(value)) > 1e-12 for value in (hf.alpha, hf.beta, hf.omega)):
                raise ValueError("This benchmark task forbids HF exchange")
        if not self.config.allow_ext_param_rewire:
            for component in xc.components:
                actual = dict(getattr(component, "ext_params", None) or {})
                component_id = NAME_TO_ID.get(component.name)
                expected = None
                if component_id is not None:
                    expected = {
                        key: float(default)
                        for key, default, _bounds in LIBXC_PARAMS.get(component_id, [])
                    }
                if expected is not None:
                    if set(actual) != set(expected) or any(
                        abs(float(actual[key]) - float(value)) > 1e-12
                        for key, value in expected.items()
                    ):
                        raise ValueError(
                            f"Component {component.name!r} internal/ext parameters are fixed"
                        )
                elif actual:
                    raise ValueError("Component internal/ext parameters are fixed for this task")
        hard = [item for item in xc.validate() if item.startswith("HARD:")]
        if hard:
            raise ValueError("; ".join(hard))

    def evaluate_many(
        self,
        xcs: Sequence[XCFunctional],
        *,
        labels: Sequence[str],
        stage: str,
    ) -> list[SearchTaskEvaluation]:
        del labels
        results = self.runner.evaluate_many(list(xcs), label_prefix=stage)
        if len(results) != len(xcs):
            raise RuntimeError(
                f"Benchmark runner returned {len(results)} results for {len(xcs)} candidates"
            )
        return [self.from_result(result) for result in results]

    def from_result(self, result: BenchmarkResult) -> SearchTaskEvaluation:
        subset_payload: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        maes: list[float] = []
        for name in self.config.subsets:
            subset = result.subsets.get(name)
            payload = _subset_payload(subset)
            subset_payload[name] = payload
            mae = _finite_or_none(getattr(subset, "mae_kcal", None))
            if subset is None or subset.n_reactions <= 0 or mae is None:
                failures.append(f"{name}:missing_or_invalid")
                continue
            if int(subset.n_failures) > self.config.max_failures_per_subset:
                failures.append(
                    f"{name}:failures={int(subset.n_failures)}>"
                    f"{self.config.max_failures_per_subset}"
                )
            maes.append(mae)

        valid = not failures and len(maes) == len(self.config.subsets)
        mean_mae = sum(maes) / len(maes) if len(maes) == len(self.config.subsets) else None
        reward = -float(mean_mae) if valid and mean_mae is not None else float("-inf")
        return SearchTaskEvaluation(
            task_name=self.name,
            valid=valid,
            reward=reward,
            objective={
                "subsets": subset_payload,
                "mean_mae_kcal": mean_mae,
                "reward_definition": "negative arithmetic mean of configured subset MAEs",
                "equal_subset_weighting": True,
            },
            proposal_feedback={
                "subset_mae_kcal": {
                    name: payload.get("mae_kcal") for name, payload in subset_payload.items()
                },
                "lower_mae_is_better": True,
            },
            failure_reason=",".join(failures),
            walltime_s=float(result.total_walltime_s or 0.0),
            raw_result=result,
        )

    def apply_to_candidate(self, candidate: Any, evaluation: SearchTaskEvaluation) -> None:
        result = evaluation.raw_result if isinstance(evaluation.raw_result, BenchmarkResult) else None
        candidate.search_result = result
        candidate.validation_result = None
        candidate.fitness = evaluation.reward
        candidate.promotable = evaluation.valid
        candidate.failure_reason = evaluation.failure_reason
        candidate.guard_penalty = 0.0
        candidate.worst_validation_ratio = 1.0 if evaluation.valid else float("inf")
        candidate.tree_reward = tree_reward_from_raw(evaluation.reward) if evaluation.valid else 0.0

    def prompt_instructions(self) -> str:
        subset_list = ", ".join(self.config.subsets)
        hf_text = (
            "HF exchange is editable through alpha, beta, and omega."
            if self.config.allow_hf_exchange
            else "HF exchange is forbidden."
        )
        ext_text = (
            "Component internal/ext parameters may be edited."
            if self.config.allow_ext_param_rewire
            else "Component internal/ext parameters are fixed."
        )
        return "\n".join(
            [
                f"Objective subsets: {subset_list}.",
                "For each valid candidate, compute the arithmetic mean of the three subset MAEs in kcal/mol.",
                "raw_reward = -mean_MAE, so larger raw_reward is better and lower MAE is better.",
                "The subsets have equal weight; this is not a reaction-count-weighted aggregate.",
                f"Any missing subset or more than {self.config.max_failures_per_subset} failed reactions in a subset makes the candidate invalid.",
                hf_text,
                ext_text,
                "Do not optimize walltime or any unreported scientific property; only the stated mean MAE defines ranking.",
            ]
        )

    def relative_feedback(
        self,
        child: SearchTaskEvaluation,
        parent: SearchTaskEvaluation | None,
    ) -> dict[str, Any]:
        if parent is None:
            return {}
        parent_subsets = dict(parent.objective.get("subsets") or {})
        child_subsets = dict(child.objective.get("subsets") or {})
        return {
            "reward": _before_after(parent.reward, child.reward),
            "mean_mae_kcal": _before_after(
                parent.objective.get("mean_mae_kcal"),
                child.objective.get("mean_mae_kcal"),
            ),
            "subset_mae_kcal": {
                name: _before_after(
                    (parent_subsets.get(name) or {}).get("mae_kcal"),
                    (child_subsets.get(name) or {}).get("mae_kcal"),
                )
                for name in self.config.subsets
            },
        }


def _subset_payload(subset: SubsetResult | None) -> dict[str, Any]:
    if subset is None:
        return {}
    return subset.to_dict(include_reaction_errors=False)


def _before_after(before: Any, after: Any) -> dict[str, Any]:
    payload = {"parent": before, "child": after}
    try:
        before_number = float(before)
        after_number = float(after)
    except (TypeError, ValueError):
        return payload
    if math.isfinite(before_number) and math.isfinite(after_number):
        payload["delta"] = after_number - before_number
    return payload


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
