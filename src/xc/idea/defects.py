"""Canonical node-level defect tags emitted during MCTS evaluation."""

from __future__ import annotations

from ..benchmark.result import BenchmarkResult


def classify_node_defects(
    *,
    search_result: BenchmarkResult,
    validation_result: BenchmarkResult | None,
    validation_ratio: float | None,
    validation_subset_ratios: dict[str, float] | None,
    baseline_mae: float,
    raw_scores: dict[str, int],
    hard_violations: list[str],
    soft_violations: list[str],
) -> list[str]:
    tags: list[str] = []
    if hard_violations:
        tags.append("hard_constraint_violation")
    elif soft_violations:
        tags.append("soft_constraint_pressure")
    worst_validation_ratio = max([validation_ratio or 1.0, *(validation_subset_ratios or {}).values()])
    if validation_ratio is not None and worst_validation_ratio > 1.5:
        tags.append("validation_regression")
    elif validation_ratio is not None and worst_validation_ratio > 1.05:
        tags.append("weak_generalization")
    if search_result.total_failures > 0:
        tags.append("search_runtime_failure")
    if validation_result is not None and validation_result.total_failures > 0:
        tags.append("validation_runtime_failure")
    if sum(sr.n_reactions for sr in search_result.subsets.values()) == 0:
        tags.append("search_evaluation_failure")
    if validation_result is not None and sum(sr.n_reactions for sr in validation_result.subsets.values()) == 0:
        tags.append("validation_evaluation_failure")
    if search_result.overall_mae > baseline_mae * 1.05:
        tags.append("search_regression")
    if raw_scores.get("improvement", 0) <= 2:
        tags.append("stagnant_improvement")
    if raw_scores.get("novelty", 0) <= 1:
        tags.append("low_novelty")
    if raw_scores.get("parsimony", 0) <= 2:
        tags.append("complexity_pressure")
    if raw_scores.get("evidence_consistency", 0) <= 1:
        tags.append("weak_evidence")
    return list(dict.fromkeys(tags))
