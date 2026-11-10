"""Template for connecting a custom evaluator and reward to both search methods."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from xc.idea.reward import tree_reward_from_raw
from xc.search_tasks.base import SearchTaskEvaluation
from xc.xc_functional import XCFunctional


BatchEvaluator = Callable[
    [Sequence[XCFunctional], Sequence[str], str],
    Sequence[Mapping[str, Any]],
]
RewardFunction = Callable[[Mapping[str, Any]], float]


class CallbackRewardTask:
    """Small adapter around a user-supplied batch evaluator and reward function."""

    name = "callback_reward"
    use_benchmark_component_priors = False

    def __init__(
        self,
        *,
        evaluator: BatchEvaluator,
        reward_function: RewardFunction,
        instructions: str,
    ) -> None:
        self.evaluator = evaluator
        self.reward_function = reward_function
        self.instructions = instructions

    def validate_functional(self, xc: XCFunctional) -> None:
        hard_errors = [item for item in xc.validate() if item.startswith("HARD:")]
        if hard_errors:
            raise ValueError("; ".join(hard_errors))

    def evaluate_many(
        self,
        xcs: Sequence[XCFunctional],
        *,
        labels: Sequence[str],
        stage: str,
    ) -> list[SearchTaskEvaluation]:
        measurements = list(self.evaluator(xcs, labels, stage))
        if len(measurements) != len(xcs):
            raise RuntimeError(
                f"Evaluator returned {len(measurements)} results for {len(xcs)} candidates"
            )

        evaluations = []
        for measurement in measurements:
            valid = bool(measurement.get("valid", True))
            try:
                reward = float(self.reward_function(measurement))
            except Exception as exc:
                valid = False
                reward = float("-inf")
                failure_reason = f"reward_failed:{exc}"
            else:
                failure_reason = str(measurement.get("failure_reason") or "")
            if not valid or not math.isfinite(reward):
                valid = False
                reward = float("-inf")

            evaluations.append(
                SearchTaskEvaluation(
                    task_name=self.name,
                    valid=valid,
                    reward=reward,
                    objective=dict(measurement.get("objective") or {}),
                    proposal_feedback=dict(measurement.get("diagnostics") or {}),
                    failure_reason=failure_reason,
                    walltime_s=measurement.get("walltime_s"),
                    raw_result=dict(measurement),
                )
            )
        return evaluations

    def apply_to_candidate(self, candidate: Any, evaluation: SearchTaskEvaluation) -> None:
        candidate.search_result = evaluation.raw_result
        candidate.validation_result = None
        candidate.fitness = evaluation.reward
        candidate.promotable = evaluation.valid
        candidate.failure_reason = evaluation.failure_reason
        candidate.guard_penalty = float(evaluation.objective.get("penalty") or 0.0)
        candidate.worst_validation_ratio = 1.0 if evaluation.valid else float("inf")
        candidate.tree_reward = (
            tree_reward_from_raw(evaluation.reward) if evaluation.valid else 0.0
        )

    def prompt_instructions(self) -> str:
        return self.instructions

    def relative_feedback(
        self,
        child: SearchTaskEvaluation,
        parent: SearchTaskEvaluation | None,
    ) -> dict[str, Any]:
        if parent is None:
            return {}
        return {
            "reward": {
                "parent": parent.reward,
                "child": child.reward,
                "delta": child.reward - parent.reward,
            },
            "objective": {
                key: _before_after(parent.objective.get(key), child.objective.get(key))
                for key in sorted(set(parent.objective) | set(child.objective))
            },
        }


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


def example_reward(measurement: Mapping[str, Any]) -> float:
    """Example: improve a target error while continuously penalizing a guard."""

    objective = measurement["objective"]
    target_error = float(objective["target_error"])
    guard_error = float(objective["guard_error"])
    free_guard_tolerance = 0.10
    guard_penalty = max(0.0, guard_error - free_guard_tolerance)
    return 1.0 - target_error - 0.5 * guard_penalty
