"""Shared target/guard reward helpers for MCTS and evolutionary search."""

from __future__ import annotations

import math
from typing import Mapping

from ..benchmark.result import BenchmarkResult


DEFAULT_GUARD_TOLERANCES: dict[str, float] = {
    "BH76": 3.0,
    "G21EA": 1.0,
    "G21IP": 0.5,
}


def guard_regression_penalty(
    candidate: BenchmarkResult,
    baseline: BenchmarkResult,
    *,
    tolerances: Mapping[str, float] | None = None,
    mean_weight: float = 0.25,
    max_weight: float = 0.10,
) -> tuple[float, dict[str, float]]:
    """Return a softly aggregated absolute-MAE guard regression penalty."""
    tolerances = tolerances or DEFAULT_GUARD_TOLERANCES
    units: dict[str, float] = {}
    for subset_name, subset in candidate.subsets.items():
        baseline_subset = baseline.subsets.get(subset_name)
        if baseline_subset is None or baseline_subset.n_reactions <= 0:
            continue
        tolerance = max(float(tolerances.get(subset_name, 1.0)), 1e-9)
        units[subset_name] = max(
            0.0,
            float(subset.mae_kcal) - float(baseline_subset.mae_kcal),
        ) / tolerance
    if not units:
        return 0.0, {}
    values = list(units.values())
    penalty = mean_weight * (sum(values) / len(values)) + max_weight * max(values)
    return penalty, units


def target_guard_raw_reward(
    *,
    search_ratio: float,
    guard_penalty: float,
    target_gain_weight: float = 2.0,
    runtime_failure_penalty: float = 0.05,
    validation_failures: int = 0,
) -> float:
    """Return the uncapped score used to compare valid candidates."""
    return (
        0.5
        + target_gain_weight * (1.0 - search_ratio)
        - guard_penalty
        - runtime_failure_penalty * max(0, validation_failures)
    )


def tree_reward_from_raw(raw_reward: float) -> float:
    """Map uncapped valid-candidate reward to [0, 1], keeping invalid at zero."""
    if not math.isfinite(raw_reward):
        return 0.0
    value = 2.0 * (raw_reward - 0.5)
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
