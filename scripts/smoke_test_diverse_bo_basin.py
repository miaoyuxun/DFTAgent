"""Smoke test for diverse_bo_basin without running PySCF.

The fake runner returns deterministic benchmark results from XC parameters, so
this checks proposal generation, scoring, local EA, and artifact writing without
touching GMTKN55 or quantum chemistry backends.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xc.benchmark.result import BenchmarkResult, SubsetResult
from xc.idea.diverse_bo_basin import DiverseBOBasinConfig, DiverseBOBasinSearch
from xc.idea.evolutionary import EvolutionConfig
from xc.xc_functional import LEGACY_PURE_COMPONENT_ROLES
from xc.xc_functional import XCFunctional


class FakeRunner:
    """Deterministic, cheap benchmark runner for smoke tests."""

    def __init__(self, subsets: list[str], *, kind: str):
        self.subsets = subsets
        self.kind = kind

    def evaluate(self, xc: XCFunctional) -> BenchmarkResult:
        values = _feature_values(xc)
        if self.kind == "search":
            base = 9.0
            # Broad basin near the values that have often been useful in real runs.
            mae = (
                base
                - 3.5 * values.get("omega", 0.0)
                + 1.8 * abs(values.get("alpha", 0.0) - 0.30)
                + 0.9 * abs(values.get("beta", 0.0) - 0.65)
                - 0.5 * values.get("has_apbe_c", 0.0)
                - 0.35 * values.get("has_pbesol_c", 0.0)
            )
        else:
            base = 4.0
            mae = (
                base
                + 0.7 * abs(values.get("alpha", 0.0) - 0.20)
                + 0.8 * max(0.0, values.get("omega", 0.0) - 0.35)
                + 0.4 * values.get("has_mgga_c", 0.0)
            )
        mae = max(0.1, float(mae))

        result = BenchmarkResult(xc_name=xc.name)
        for index, subset in enumerate(self.subsets):
            subset_mae = mae * (1.0 + 0.03 * index)
            result.subsets[subset] = SubsetResult(
                subset=subset,
                mae_kcal=subset_mae,
                rms_kcal=subset_mae * 1.15,
                max_error_kcal=subset_mae * 2.0,
                n_reactions=8,
                n_failures=0,
                walltime_s=0.01,
                per_reaction_errors=[subset_mae] * 8,
            )
        result.compute_overall()
        return result


def _feature_values(xc: XCFunctional) -> dict[str, float]:
    names = {component.name for component in xc.components}
    values = {
        "alpha": float(xc.hf_exchange.alpha),
        "beta": float(xc.hf_exchange.beta),
        "omega": float(xc.hf_exchange.omega),
        "has_apbe_c": 1.0 if "GGA_C_APBE" in names else 0.0,
        "has_pbesol_c": 1.0 if "GGA_C_PBE_SOL" in names else 0.0,
        "has_mgga_c": 1.0 if any(component.role == "correlation" and component.xctype == "MGGA" for component in xc.components) else 0.0,
    }
    for component in xc.components:
        values[f"mixing:{component.name}"] = float(component.mixing_factor)
    return values


def main() -> None:
    output_dir = ROOT / "results" / "smoke_diverse_bo_basin"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    parameter_output_dir = ROOT / "results" / "smoke_diverse_bo_basin_parameter"
    if parameter_output_dir.exists():
        shutil.rmtree(parameter_output_dir)

    root = XCFunctional.from_known("HYB_GGA_XC_LC_WPBE")
    search_runner = FakeRunner(["SIE4x4"], kind="search")
    guard_runner = FakeRunner(["BH76", "G21EA", "G21IP"], kind="guard")

    root_result = search_runner.evaluate(root)
    guard_result = guard_runner.evaluate(root)
    config = DiverseBOBasinConfig(
        cycles=1,
        proposals_per_cycle=4,
        proposal_batch_size=2,
        max_proposal_batches=3,
        candidate_pool_size=64,
        proposal_evaluation_concurrency=1,
        ea_seed_count=2,
        ea_population_size=4,
        ea_generations=1,
        ea_elite_size=1,
        novelty_threshold=0.03,
        single_step_proposals=True,
        component_pool="legacy",
        seed=7,
    )
    scorer_config = EvolutionConfig(
        population_size=4,
        generations=1,
        elite_size=1,
        exact_mixing_sum=True,
        seed=7,
    )
    searcher = DiverseBOBasinSearch(config=config, scorer_config=scorer_config)
    result = searcher.search(
        root_xc=root,
        root_result=root_result,
        baseline_validation_result=guard_result,
        runner=search_runner,
        validation_runner=guard_runner,
        output_dir=output_dir,
    )

    assert len(result.nodes) >= 5, f"expected root + proposals, got {len(result.nodes)}"
    assert (output_dir / "diverse_bo_basin_nodes.csv").exists()
    assert (output_dir / "diverse_bo_basin_nodes.json").exists()
    assert (output_dir / "diverse_bo_basin_summary.json").exists()
    assert result.best.search_result is not None
    assert result.best.validation_result is not None
    proposal_nodes = [node for node in result.nodes if node.stage == "proposal"]
    assert proposal_nodes, "expected at least one evaluated BO proposal"
    assert all(
        len(node.action_detail.get("structure", [])) <= 1
        for node in proposal_nodes
    ), "single-step BO proposals must contain at most one operation"
    legacy_names = set(LEGACY_PURE_COMPONENT_ROLES) | {"GGA_X_HJS_PBE"}
    for node in proposal_nodes:
        for component in node.xc.components:
            if component.role in {"exchange", "correlation"}:
                assert component.name in legacy_names, (
                    f"legacy BO emitted expanded component {component.name}"
                )

    parameter_config = DiverseBOBasinConfig(
        cycles=1,
        proposals_per_cycle=2,
        proposal_batch_size=2,
        max_proposal_batches=2,
        candidate_pool_size=32,
        proposal_evaluation_concurrency=1,
        ea_seed_count=0,
        ea_population_size=2,
        ea_generations=0,
        ea_elite_size=1,
        novelty_threshold=0.01,
        structural_mutation_probability=0.0,
        single_step_proposals=True,
        seed=11,
    )
    parameter_searcher = DiverseBOBasinSearch(config=parameter_config, scorer_config=scorer_config)
    parameter_result = parameter_searcher.search(
        root_xc=root,
        root_result=root_result,
        baseline_validation_result=guard_result,
        runner=search_runner,
        validation_runner=guard_runner,
        output_dir=parameter_output_dir,
    )
    parameter_nodes = [node for node in parameter_result.nodes if node.stage == "proposal"]
    parameter_local_ea_nodes = [node for node in parameter_result.nodes if node.stage == "local_ea"]
    assert parameter_nodes, "expected parameter-only BO proposals"
    assert not parameter_local_ea_nodes, "ea_seed_count=0 should disable local EA in diverse BO"
    assert all(
        len(node.action_detail.get("structure", [])) == 1
        and node.action_detail["structure"][0].get("step_type") in {"parameter", "parameter_chain"}
        for node in parameter_nodes
    ), "forced parameter single-step BO should only emit one parameter move"

    print(
        "diverse_bo_basin smoke ok: "
        f"nodes={len(result.nodes)} parameter_nodes={len(parameter_result.nodes)} best={result.best.xc.name} "
        f"reward={result.best.fitness:.4f} "
        f"SIE={result.best.search_result.overall_mae:.4f}"
    )


if __name__ == "__main__":
    main()
