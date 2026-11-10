"""Tests for the black-box evolutionary XC parameter optimizer."""

from __future__ import annotations

import json
from pathlib import Path

from xc.benchmark.result import BenchmarkResult, SubsetResult
from xc.idea.evolutionary import EvolutionConfig, EvolutionarySearch
from xc.xc_functional import XCFunctional


def _result(name: str, values: dict[str, float]) -> BenchmarkResult:
    result = BenchmarkResult(xc_name=name)
    for subset, mae in values.items():
        result.subsets[subset] = SubsetResult(
            subset=subset,
            mae_kcal=mae,
            rms_kcal=mae,
            max_error_kcal=mae,
            n_reactions=4,
            n_failures=0,
            walltime_s=0.0,
            per_reaction_errors=[mae] * 4,
        )
    result.compute_overall()
    return result


class SearchRunner:
    def evaluate(self, xc):
        alpha = xc.hf_exchange.alpha
        beta = next(c for c in xc.components if c.name == "GGA_C_PBE").ext_params["_beta"]
        # Higher alpha helps SIE; beta near 0.06 provides a smaller second gain.
        mae = 14.0 - 28.0 * (alpha - 0.25) + 25.0 * abs(beta - 0.06)
        return _result(xc.name, {"SIE4x4": mae})


class GuardRunner:
    def evaluate(self, xc):
        alpha = xc.hf_exchange.alpha
        beta = next(c for c in xc.components if c.name == "GGA_C_PBE").ext_params["_beta"]
        # Lower beta creates enough headroom to safely raise alpha toward 0.36.
        g21ip = 4.0 * (1.0 + 0.75 * max(0.0, alpha - 0.30) - 1.5 * max(0.0, 0.0667 - beta))
        return _result(
            xc.name,
            {
                "BH76": 5.0 - 5.0 * (alpha - 0.25),
                "G21EA": 8.0,
                "G21IP": g21ip,
            },
        )


def test_evolutionary_search_improves_continuous_target_guard_reward():
    root = XCFunctional.from_known("PBE0")
    root_result = SearchRunner().evaluate(root)
    root_validation = GuardRunner().evaluate(root)
    search = EvolutionarySearch(EvolutionConfig(
        population_size=14,
        generations=6,
        elite_size=4,
        seed=7,
    ))

    result = search.search(
        root_xc=root,
        root_result=root_result,
        runner=SearchRunner(),
        validation_runner=GuardRunner(),
        baseline_validation_result=root_validation,
    )

    assert result.best.promotable
    assert result.best.search_result.overall_mae < root_result.overall_mae
    assert result.best.fitness > result.baseline.fitness
    assert result.best.parameters["alpha"] > root.hf_exchange.alpha
    assert abs(result.best.parameters["GGA_C_PBE._beta"] - 0.06) < 0.03


def test_evolutionary_candidate_preserves_exchange_sum_rule():
    root = XCFunctional.from_known("PBE0")
    search = EvolutionarySearch(EvolutionConfig())
    parameter_space = search._parameter_space(root)
    keys = {parameter.key for parameter in parameter_space}
    assert {"alpha", "beta", "omega", "mixing:GGA_X_PBE", "GGA_X_PBE._mu"} <= keys
    parameters = search._read_parameters(root, parameter_space)
    parameters["alpha"] = 0.36

    candidate = search._build_candidate(root, parameters, generation=1)
    exchange_sum = sum(
        component.mixing_factor
        for component in candidate.components
        if component.role == "exchange"
    )

    assert abs(exchange_sum + candidate.hf_exchange.alpha - 1.0) < 1e-9
    assert not [item for item in candidate.validate() if item.startswith("HARD:")]


def test_evolutionary_mutation_always_changes_at_least_one_parameter():
    root = XCFunctional.from_known("PBE0")
    search = EvolutionarySearch(EvolutionConfig(seed=17, mutation_rate=1.0))
    parameter_space = search._parameter_space(root)
    vector = {
        parameter.key: parameter.lower + 0.25 * (parameter.upper - parameter.lower)
        for parameter in parameter_space
    }
    mutated = search._mutate(vector, parameter_space)
    changed = [
        key for key, value in vector.items()
        if abs(mutated[key] - value) > 1e-12
    ]
    assert len(changed) >= 1


def test_evolutionary_search_writes_reproducible_artifacts(tmp_path: Path):
    root = XCFunctional.from_known("PBE0")
    search = EvolutionarySearch(EvolutionConfig(
        population_size=4,
        generations=1,
        elite_size=2,
        seed=11,
    ))
    result = search.search(
        root_xc=root,
        root_result=SearchRunner().evaluate(root),
        runner=SearchRunner(),
        validation_runner=GuardRunner(),
        baseline_validation_result=GuardRunner().evaluate(root),
    )

    search.write_artifacts(result, tmp_path)
    summary = json.loads((tmp_path / "evolutionary_summary.json").read_text())
    history = json.loads((tmp_path / "evolutionary_history.json").read_text())

    assert summary["evaluated_nodes"] == len(result.evaluated)
    assert summary["parameter_space"][0]["key"] == "alpha"
    assert history[0]["name"] == "PBE0"
    assert (tmp_path / "evolutionary_history.csv").exists()
