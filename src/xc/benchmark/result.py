"""Benchmark result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubsetResult:
    subset: str
    mae_kcal: float
    rms_kcal: float
    max_error_kcal: float
    n_reactions: int
    n_failures: int
    walltime_s: float
    per_reaction_errors: list[float] = field(default_factory=list)
    failure_details: list[str] = field(default_factory=list)

    def to_dict(self, *, include_reaction_errors: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subset": self.subset,
            "mae_kcal": self.mae_kcal,
            "rms_kcal": self.rms_kcal,
            "max_error_kcal": self.max_error_kcal,
            "n_reactions": self.n_reactions,
            "n_failures": self.n_failures,
            "walltime_s": self.walltime_s,
            "failure_details": list(self.failure_details),
        }
        if include_reaction_errors:
            payload["per_reaction_errors"] = list(self.per_reaction_errors)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SubsetResult":
        return cls(
            subset=str(payload.get("subset", "")),
            mae_kcal=float(payload.get("mae_kcal", 0.0)),
            rms_kcal=float(payload.get("rms_kcal", 0.0)),
            max_error_kcal=float(payload.get("max_error_kcal", 0.0)),
            n_reactions=int(payload.get("n_reactions", 0)),
            n_failures=int(payload.get("n_failures", 0)),
            walltime_s=float(payload.get("walltime_s", 0.0)),
            per_reaction_errors=[float(value) for value in payload.get("per_reaction_errors", [])],
            failure_details=[str(value) for value in payload.get("failure_details", [])],
        )


@dataclass
class BenchmarkResult:
    xc_name: str
    subsets: dict[str, SubsetResult] = field(default_factory=dict)
    overall_mae: float = 0.0
    overall_rms: float = 0.0
    total_walltime_s: float = 0.0
    total_failures: int = 0
    timestamp: str = ""

    def compute_overall(self, *, aggregate_mode: str = "reaction_mean") -> None:
        import numpy as np

        if aggregate_mode not in {"reaction_mean", "subset_mean"}:
            raise ValueError(f"Unsupported aggregate_mode: {aggregate_mode}")

        if aggregate_mode == "subset_mean":
            maes = [sr.mae_kcal for sr in self.subsets.values() if sr.n_reactions > 0]
            if maes:
                self.overall_mae = float(np.mean(maes))
                rmses = [sr.rms_kcal for sr in self.subsets.values() if sr.n_reactions > 0]
                self.overall_rms = float(np.mean(rmses)) if rmses else 0.0
        else:
            all_errors = []
            for sr in self.subsets.values():
                all_errors.extend(sr.per_reaction_errors)
            if all_errors:
                self.overall_mae = float(np.mean(all_errors))
                self.overall_rms = float(np.sqrt(np.mean(np.array(all_errors) ** 2)))
            else:
                # Fallback when per_reaction_errors is empty (e.g. pre-cached results).
                # Use equal subset weighting so cached and uncached summary files remain usable.
                maes = [sr.mae_kcal for sr in self.subsets.values() if sr.n_reactions > 0]
                if maes:
                    self.overall_mae = float(np.mean(maes))
                    rmses = [sr.rms_kcal for sr in self.subsets.values() if sr.n_reactions > 0]
                    self.overall_rms = float(np.mean(rmses)) if rmses else 0.0
        self.total_walltime_s = sum(sr.walltime_s for sr in self.subsets.values())
        self.total_failures = sum(sr.n_failures for sr in self.subsets.values())
