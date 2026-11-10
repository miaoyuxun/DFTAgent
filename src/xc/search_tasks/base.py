"""Task-neutral interfaces for scientific search evaluations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from ..xc_functional import XCFunctional


@dataclass
class SearchTaskEvaluation:
    """Serializable result consumed by a search algorithm.

    ``objective`` contains only quantities that define the score. Additional
    observations belong in ``proposal_feedback`` and must not silently alter
    ranking or parent eligibility.
    """

    task_name: str
    valid: bool
    reward: float = float("-inf")
    objective: dict[str, Any] = field(default_factory=dict)
    proposal_feedback: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    walltime_s: float | None = None
    raw_result: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.valid = bool(self.valid)
        self.reward = float(self.reward)
        if not self.valid or not math.isfinite(self.reward):
            self.valid = False
            self.reward = float("-inf")
        if self.walltime_s is not None:
            self.walltime_s = float(self.walltime_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "valid": self.valid,
            "reward": self.reward,
            "objective": dict(self.objective),
            "proposal_feedback": dict(self.proposal_feedback),
            "failure_reason": self.failure_reason,
            "walltime_s": self.walltime_s,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SearchTaskEvaluation | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            task_name=str(payload.get("task_name") or "unknown"),
            valid=bool(payload.get("valid", False)),
            reward=float(payload.get("reward", float("-inf"))),
            objective=dict(payload.get("objective") or {}),
            proposal_feedback=dict(payload.get("proposal_feedback") or {}),
            failure_reason=str(payload.get("failure_reason") or ""),
            walltime_s=payload.get("walltime_s"),
        )


@runtime_checkable
class SearchTask(Protocol):
    """Minimal contract between a scientific task and a search algorithm."""

    name: str
    use_benchmark_component_priors: bool

    def validate_functional(self, xc: XCFunctional) -> None:
        """Raise ``ValueError`` when a candidate is outside the task domain."""

    def evaluate_many(
        self,
        xcs: Sequence[XCFunctional],
        *,
        labels: Sequence[str],
        stage: str,
    ) -> list[SearchTaskEvaluation]:
        """Evaluate candidates and preserve input ordering."""

    def apply_to_candidate(self, candidate: Any, evaluation: SearchTaskEvaluation) -> None:
        """Populate the generic ranking fields on a search candidate."""

    def prompt_instructions(self) -> str:
        """Describe the objective, validity rules, and diagnostic semantics."""

    def relative_feedback(
        self,
        child: SearchTaskEvaluation,
        parent: SearchTaskEvaluation | None,
    ) -> dict[str, Any]:
        """Return task-aware child-versus-parent observations for the LLM."""
