"""Search task: SIE4x4 primary reward with configurable crystal-gap guards."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..benchmark.result import BenchmarkResult, SubsetResult
from ..idea.reward import tree_reward_from_raw
from ..xc_functional import LIBXC_PARAMS, NAME_TO_ID, XCFunctional
from ..zno_component_registry import validate_functional_for_pool
from .base import SearchTaskEvaluation


@dataclass(frozen=True)
class SIECrystalGuardTaskConfig:
    component_pool: str = "zno_gap"
    baseline_sie_mae_kcal: float = 10.0
    baseline_sie_failures: int = 0
    baseline_sie_reactions: int = 16
    zns_reference_gap_ev: float = 3.83
    mgo_reference_gap_ev: float = 7.77
    zns_free_tolerance_ev: float = 0.50
    mgo_free_tolerance_ev: float = 0.20
    guard_scale_ev: float = 0.30
    guard_mean_weight: float = 0.25
    guard_max_weight: float = 0.10
    sie_fail_penalties: tuple[float, ...] = (0.15, 0.15, 0.25, 0.25)
    extra_sie_fail_penalty: float = 0.50
    forbid_hf_exchange: bool = True
    allow_ext_param_rewire: bool = False
    protected_components: tuple[str, ...] = ()
    minimum_component_mixing: dict[str, float] = field(default_factory=dict)
    fixed_component_ext_params: dict[str, dict[str, float]] = field(default_factory=dict)
    root_label: str = "mTASK-PW92-root"
    crystal_guard_materials: tuple[str, ...] = ("ZnS", "MgO")
    root_crystal_result_paths: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.baseline_sie_mae_kcal <= 0:
            raise ValueError("baseline_sie_mae_kcal must be positive")
        if self.guard_scale_ev <= 0:
            raise ValueError("guard_scale_ev must be positive")
        for name, minimum in self.minimum_component_mixing.items():
            if not 0.0 <= float(minimum) <= 1.0:
                raise ValueError(f"minimum mixing for {name} must be within [0, 1]")


class SIECrystalGuardSearchTask:
    """Combine a molecular SIE objective with hard crystal-gap guards."""

    name = "sie_crystal_guard"
    use_benchmark_component_priors = False

    def __init__(
        self,
        *,
        sie_runner: Any,
        crystal_evaluator: Callable[..., Sequence[dict[str, Any]]],
        config: SIECrystalGuardTaskConfig | None = None,
    ) -> None:
        self.sie_runner = sie_runner
        self.crystal_evaluator = crystal_evaluator
        self.config = config or SIECrystalGuardTaskConfig()

    def validate_functional(self, xc: XCFunctional) -> None:
        validate_functional_for_pool(xc, component_pool=self.config.component_pool)
        components = {component.name: component for component in xc.components}
        for name in self.config.protected_components:
            component = components.get(name)
            if component is None:
                raise ValueError(f"Protected backbone component {name!r} cannot be removed or replaced")
            minimum = float(self.config.minimum_component_mixing.get(name, 0.0))
            if float(component.mixing_factor) < minimum - 1e-12:
                raise ValueError(
                    f"Protected backbone component {name!r} mixing must be >= {minimum:.6g}"
                )
        if self.config.forbid_hf_exchange:
            hf = xc.hf_exchange
            if any(abs(float(value)) > 1e-12 for value in (hf.alpha, hf.beta, hf.omega)):
                raise ValueError("This SIE+crystal task forbids alpha, beta, omega, and all HF exchange")
        if not self.config.allow_ext_param_rewire:
            for component in xc.components:
                actual = dict(getattr(component, "ext_params", None) or {})
                expected = self.config.fixed_component_ext_params.get(component.name)
                if expected is None and component.name in NAME_TO_ID:
                    expected = {
                        key: float(default)
                        for key, default, _bounds in LIBXC_PARAMS.get(NAME_TO_ID[component.name], [])
                    }
                if expected is not None:
                    if set(actual) != set(expected) or any(
                        abs(float(actual[key]) - float(value)) > 1e-12
                        for key, value in expected.items()
                    ):
                        raise ValueError(
                            f"Protected component {component.name!r} internal/ext parameters are fixed"
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
        sie_results = self.sie_runner.evaluate_many(list(xcs), label_prefix=f"{stage}_sie")
        crystal_payloads = list(self.crystal_evaluator(list(xcs), labels=list(labels), stage=f"{stage}_crystal"))
        if len(sie_results) != len(xcs):
            raise RuntimeError(f"SIE runner returned {len(sie_results)} results for {len(xcs)} candidates")
        if len(crystal_payloads) != len(xcs):
            raise RuntimeError(
                f"Crystal evaluator returned {len(crystal_payloads)} results for {len(xcs)} candidates"
            )
        return [
            self.from_results(sie_result=sie, crystal_payload=crystal)
            for sie, crystal in zip(sie_results, crystal_payloads)
        ]

    def root_evaluation_from_existing(self) -> SearchTaskEvaluation:
        crystal_payload = {
            "ok": True,
            "materials": {
                material: _load_json(Path(path))
                for material, path in self.config.root_crystal_result_paths.items()
                if str(path).strip()
            },
        }
        sie = BenchmarkResult(xc_name=self.config.root_label)
        sie.subsets["SIE4x4"] = SubsetResult(
            subset="SIE4x4",
            mae_kcal=float(self.config.baseline_sie_mae_kcal),
            rms_kcal=float(self.config.baseline_sie_mae_kcal),
            max_error_kcal=0.0,
            n_reactions=int(self.config.baseline_sie_reactions),
            n_failures=int(self.config.baseline_sie_failures),
            walltime_s=0.0,
        )
        sie.compute_overall()
        evaluation = self.from_results(sie_result=sie, crystal_payload=crystal_payload)
        evaluation.proposal_feedback["reused_existing_root"] = True
        return evaluation

    def from_results(
        self,
        *,
        sie_result: BenchmarkResult,
        crystal_payload: Mapping[str, Any],
    ) -> SearchTaskEvaluation:
        sie_subset = sie_result.subsets.get("SIE4x4")
        sie_mae = _finite_or_none(getattr(sie_subset, "mae_kcal", None))
        sie_failures = int(getattr(sie_subset, "n_failures", 0) or 0)
        sie_reactions = int(getattr(sie_subset, "n_reactions", 0) or 0)
        sie_missing = sie_subset is None or sie_mae is None or sie_reactions <= 0
        crystal = self._crystal_guard_summary(crystal_payload)
        failures: list[str] = []
        if sie_missing:
            failures.append("missing_sie4x4_mae")
        for material, summary in crystal.items():
            if not summary.get("valid"):
                failures.append(f"{material}_guard_failed:{summary.get('failure_reason') or 'invalid'}")
        guard_hard_failed = any(item.endswith("_guard_failed") or "_guard_failed:" in item for item in failures)

        sie_ratio = (sie_mae / self.config.baseline_sie_mae_kcal) if sie_mae is not None else None
        sie_fail_penalty = self._sie_failure_penalty(sie_failures)
        guard_units = {
            material: float(summary.get("guard_unit") or 0.0)
            for material, summary in crystal.items()
        }
        if guard_units:
            guard_mean = sum(guard_units.values()) / len(guard_units)
            guard_max = max(guard_units.values())
        else:
            guard_mean = 0.0
            guard_max = 0.0
        guard_penalty = self.config.guard_mean_weight * guard_mean + self.config.guard_max_weight * guard_max
        valid = (not sie_missing) and (not guard_hard_failed)
        reward = (
            0.5 + 2.0 * (1.0 - float(sie_ratio)) - guard_penalty - sie_fail_penalty
            if valid and sie_ratio is not None
            else float("-inf")
        )
        return SearchTaskEvaluation(
            task_name=self.name,
            valid=valid,
            reward=reward,
            objective={
                "sie4x4_mae_kcal": sie_mae,
                "baseline_sie4x4_mae_kcal": self.config.baseline_sie_mae_kcal,
                "sie4x4_ratio_to_root": sie_ratio,
                "sie4x4_failures": sie_failures,
                "sie_failure_penalty": sie_fail_penalty,
                "guard_penalty": guard_penalty,
                "guard_units": guard_units,
                "reward_definition": (
                    "0.5 + 2*(1 - SIE4x4_MAE/root_SIE4x4_MAE) "
                    "- crystal_guard_penalty - segmented_SIE_failure_penalty"
                ),
            },
            proposal_feedback={
                "crystal_guards": crystal,
                "sie4x4": _benchmark_subset_payload(sie_subset),
                "timeout_policy": "adaptive slow-tail cutoff plus hard walltime; runtime is not a continuous reward term",
            },
            failure_reason=",".join(failures),
            walltime_s=float(sie_result.total_walltime_s or 0.0)
            + sum(_finite_or_none(item.get("walltime_s")) or 0.0 for item in crystal.values()),
            raw_result={
                "sie_result": _benchmark_result_payload(sie_result),
                "crystal_payload": dict(crystal_payload),
            },
        )

    def apply_to_candidate(self, candidate: Any, evaluation: SearchTaskEvaluation) -> None:
        raw = evaluation.raw_result or {}
        candidate.search_result = raw.get("sie_result")
        candidate.validation_result = raw.get("crystal_payload")
        candidate.fitness = evaluation.reward
        candidate.promotable = evaluation.valid
        candidate.failure_reason = evaluation.failure_reason
        candidate.guard_penalty = float(evaluation.objective.get("guard_penalty") or 0.0)
        guard_units = evaluation.objective.get("guard_units") or {}
        candidate.worst_validation_ratio = max([1.0, *(float(value) for value in guard_units.values())])
        candidate.tree_reward = tree_reward_from_raw(evaluation.reward) if evaluation.valid else 0.0

    def prompt_instructions(self) -> str:
        guard_lines = []
        for material in self.config.crystal_guard_materials:
            guard_lines.append(
                f"{material} reference gap: {self._reference_for(material):.3f} eV; "
                f"free tolerance {self._free_tolerance_for(material):.2f} eV."
            )
        protected_lines = []
        if self.config.protected_components:
            protected_lines.extend([
                "The following root-backbone components are mandatory and cannot be removed or replaced: "
                + ", ".join(self.config.protected_components) + ".",
                "Their minimum mixing factors are: "
                + ", ".join(
                    f"{name} >= {float(self.config.minimum_component_mixing.get(name, 0.0)):.2f}"
                    for name in self.config.protected_components
                )
                + ".",
            ])
        return "\n".join(
            [
                "Primary objective: reduce SIE4x4 MAE. Larger raw_reward is better.",
                f"Root SIE4x4 MAE baseline: {self.config.baseline_sie_mae_kcal:.3f} kcal/mol.",
                "raw_reward = 0.5 + 2*(1 - SIE4x4_MAE/root_SIE4x4_MAE) - guard_penalty - SIE_failure_penalty.",
                "SIE4x4 failures are not hard failures; the first two failures cost 0.15 each, the third and fourth cost 0.25 each, and later failures cost 0.50 each.",
                f"{', '.join(self.config.crystal_guard_materials)} are hard guards: timeout, SCF failure, invalid gap, metallic/fractional state, or missing result makes the candidate invalid.",
                *guard_lines,
                "Crystal guard penalty uses excess absolute gap error beyond the free tolerance, scaled by guard_scale_ev.",
                "Runtime is diagnostic-only unless adaptive cutoff or hard walltime marks the crystal guard failed.",
                "HF exchange is forbidden. Do not propose alpha, beta, omega, or any hybrid/RSH behavior.",
                "Component internal/ext parameters are fixed. You may add/remove/replace components, change component mixing factors, and tune role-level scales `scale:exchange` and `scale:correlation`.",
                *protected_lines,
                "`scale:exchange` and `scale:correlation` are bounded in [0.8, 1.2]. They scale the total semilocal exchange or correlation contribution after same-role component mixing is normalized.",
                "Molecular component priors are disabled; do not infer crystal suitability from molecular mass-benchmark priors.",
            ]
        )

    def relative_feedback(
        self,
        child: SearchTaskEvaluation,
        parent: SearchTaskEvaluation | None,
    ) -> dict[str, Any]:
        if parent is None:
            return {}
        keys = (
            "sie4x4_mae_kcal",
            "sie4x4_ratio_to_root",
            "sie4x4_failures",
            "guard_penalty",
            "sie_failure_penalty",
        )
        return {
            "reward": _before_after(parent.reward, child.reward),
            "objective": {
                key: _before_after(parent.objective.get(key), child.objective.get(key))
                for key in keys
            },
            "crystal_guards": {
                material: _material_delta(
                    (parent.proposal_feedback.get("crystal_guards") or {}).get(material) or {},
                    (child.proposal_feedback.get("crystal_guards") or {}).get(material) or {},
                )
                for material in self.config.crystal_guard_materials
            },
        }

    def _crystal_guard_summary(self, payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        materials = dict(payload.get("materials") or {})
        summary: dict[str, dict[str, Any]] = {}
        for material in self.config.crystal_guard_materials:
            item = dict(materials.get(material) or {})
            band_gap = item.get("band_gap") or {}
            gap = _finite_or_none(band_gap.get("gap_ev"))
            reference = self._reference_for(material)
            tolerance = self._free_tolerance_for(material)
            error = (gap - reference) if gap is not None else None
            abs_error = abs(error) if error is not None else None
            guard_unit = (
                max(0.0, abs_error - tolerance) / self.config.guard_scale_ev
                if abs_error is not None
                else None
            )
            ok = bool(item.get("ok", False))
            band_overlap = bool(band_gap.get("band_overlap", False))
            valid = ok and gap is not None and not band_overlap and not item.get("timed_out") and not item.get("runtime_cutoff")
            failure_reason = ""
            if not valid:
                failure_reason = str(item.get("error") or item.get("failed_stage") or item.get("stage") or "invalid_gap")
            summary[material] = {
                "valid": valid,
                "gap_ev": gap,
                "reference_gap_ev": reference,
                "gap_error_ev": error,
                "absolute_gap_error_ev": abs_error,
                "free_tolerance_ev": tolerance,
                "guard_unit": guard_unit,
                "walltime_s": _finite_or_none(item.get("walltime_s")),
                "stage": item.get("stage"),
                "scf_cycles": (item.get("strict_scf") or {}).get("cycles"),
                "scf_gap_ev": _finite_or_none((item.get("scf_gap") or {}).get("gap_ev")),
                "gamma_gap_ev": _finite_or_none(band_gap.get("gamma_gap_ev")),
                "minimum_direct_gap_ev": _finite_or_none(band_gap.get("minimum_direct_gap_ev")),
                "band_overlap": band_overlap,
                "timed_out": bool(item.get("timed_out") or item.get("runtime_cutoff")),
                "failure_reason": failure_reason,
            }
        return summary

    def _reference_for(self, material: str) -> float:
        if material == "ZnS":
            return self.config.zns_reference_gap_ev
        if material == "MgO":
            return self.config.mgo_reference_gap_ev
        raise KeyError(material)

    def _free_tolerance_for(self, material: str) -> float:
        if material == "ZnS":
            return self.config.zns_free_tolerance_ev
        if material == "MgO":
            return self.config.mgo_free_tolerance_ev
        raise KeyError(material)

    def _sie_failure_penalty(self, failures: int) -> float:
        count = max(0, int(failures))
        penalty = 0.0
        for index in range(min(count, len(self.config.sie_fail_penalties))):
            penalty += float(self.config.sie_fail_penalties[index])
        if count > len(self.config.sie_fail_penalties):
            penalty += (count - len(self.config.sie_fail_penalties)) * float(self.config.extra_sie_fail_penalty)
        return penalty


def _benchmark_subset_payload(subset: SubsetResult | None) -> dict[str, Any]:
    if subset is None:
        return {}
    return subset.to_dict(include_reaction_errors=False)


def _benchmark_result_payload(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "xc_name": result.xc_name,
        "overall_mae": result.overall_mae,
        "overall_rms": result.overall_rms,
        "total_walltime_s": result.total_walltime_s,
        "total_failures": result.total_failures,
        "subsets": {
            name: subset.to_dict(include_reaction_errors=False)
            for name, subset in result.subsets.items()
        },
    }


def _material_delta(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gap_ev": _before_after(parent.get("gap_ev"), child.get("gap_ev")),
        "absolute_gap_error_ev": _before_after(parent.get("absolute_gap_error_ev"), child.get("absolute_gap_error_ev")),
        "guard_unit": _before_after(parent.get("guard_unit"), child.get("guard_unit")),
        "valid": {"parent": parent.get("valid"), "child": child.get("valid")},
        "walltime_s": _before_after(parent.get("walltime_s"), child.get("walltime_s")),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _before_after(before: Any, after: Any) -> dict[str, Any]:
    payload = {"parent": before, "child": after}
    before_number = _finite_or_none(before)
    after_number = _finite_or_none(after)
    if before_number is not None and after_number is not None:
        payload["delta"] = after_number - before_number
    return payload
