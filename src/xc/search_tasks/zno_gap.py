"""Task definition for self-consistent bulk-ZnO gap discovery."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..idea.reward import tree_reward_from_raw
from ..xc_functional import ComponentSpec, XCFunctional
from ..zno_component_registry import validate_functional_for_pool
from .base import SearchTaskEvaluation


@dataclass(frozen=True)
class ZnOGapTaskConfig:
    reference_gap_ev: float = 3.44
    reward_mode: str = "negative_absolute_error"
    component_pool: str = "zno_gap"

    def __post_init__(self) -> None:
        if self.reference_gap_ev <= 0:
            raise ValueError("reference_gap_ev must be positive")
        if self.reward_mode != "negative_absolute_error":
            raise ValueError(f"Unsupported ZnO reward_mode: {self.reward_mode}")


class ZnOGapSearchTask:
    """Score ZnO using only sampled band-gap absolute error.

    The evaluator callable receives ``(xcs, labels, stage)`` and returns raw
    per-functional JSON payloads. It can be backed by Slurm, a local smoke
    evaluator, or fixtures in unit tests without changing basin-search logic.
    """

    name = "zno_gap"
    use_benchmark_component_priors = False

    def __init__(
        self,
        evaluator: Callable[..., Sequence[dict[str, Any]]],
        *,
        config: ZnOGapTaskConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or ZnOGapTaskConfig()

    def validate_functional(self, xc: XCFunctional) -> None:
        validate_functional_for_pool(xc, component_pool=self.config.component_pool)
        hf = xc.hf_exchange
        if any(abs(float(value)) > 1e-12 for value in (hf.alpha, hf.beta, hf.omega)):
            raise ValueError("ZnO semilocal search forbids alpha, beta, omega, and all HF exchange")
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
        raw_results = list(self.evaluator(list(xcs), labels=list(labels), stage=stage))
        if len(raw_results) != len(xcs):
            raise RuntimeError(
                f"ZnO evaluator returned {len(raw_results)} results for {len(xcs)} candidates"
            )
        return [self.from_payload(payload) for payload in raw_results]

    def from_payload(self, payload: dict[str, Any]) -> SearchTaskEvaluation:
        strict = payload.get("strict_scf") or {}
        gap = strict.get("sampled_kpoint_gap") or payload.get("sampled_kpoint_gap") or {}
        diagnostics = payload.get("gap_diagnostics") or strict.get("gap_diagnostics") or gap
        sampled_gap = _finite_or_none(gap.get("gap_ev", diagnostics.get("sampled_gap_ev")))
        timed_out = bool(payload.get("timed_out", False) or payload.get("stage") == "timeout")
        strict_converged = bool(strict.get("converged", False))
        gap_valid = bool(gap.get("valid", sampled_gap is not None)) and sampled_gap is not None
        metallic = bool(gap.get("is_metallic", diagnostics.get("is_metallic", False)))
        fractional = bool(
            gap.get("has_fractional_occupations", diagnostics.get("has_fractional_occupations", False))
        )

        failures = []
        if timed_out:
            failures.append("timeout")
        if not strict_converged:
            failures.append("strict_scf_not_converged")
        if not gap_valid:
            failures.append("invalid_gap")
        if metallic:
            failures.append("metallic")
        if fractional:
            failures.append("fractional_occupations")
        valid = not failures
        absolute_error = (
            abs(float(sampled_gap) - self.config.reference_gap_ev)
            if sampled_gap is not None
            else None
        )
        reward = -absolute_error if valid and absolute_error is not None else float("-inf")

        feedback = {
            "gamma_gap_ev": _finite_or_none(diagnostics.get("gamma_gap_ev")),
            "minimum_direct_gap_ev": _finite_or_none(diagnostics.get("minimum_direct_gap_ev")),
            "directness_delta_ev": _finite_or_none(diagnostics.get("directness_delta_ev")),
            "vbm_kpoint_index": diagnostics.get("vbm_kpoint_index"),
            "cbm_kpoint_index": diagnostics.get("cbm_kpoint_index"),
            "is_direct_gap": diagnostics.get("is_direct_gap"),
            "is_gamma_direct_gap": diagnostics.get("is_gamma_direct_gap"),
            "is_metallic": metallic,
            "has_fractional_occupations": fractional,
            "strict_scf_converged": strict_converged,
            "strict_scf_cycles": strict.get("scf_cycles"),
            "preconditioning_converged": (payload.get("preconditioning") or {}).get("converged"),
            "preconditioning_cycles": (payload.get("preconditioning") or {}).get("scf_cycles"),
            "timed_out": timed_out,
            "last_stage": payload.get("stage_before_timeout") or payload.get("stage"),
            "last_completed_cycle": payload.get("last_completed_cycle"),
            "recent_cycle_walltimes_s": list(payload.get("recent_cycle_walltimes_s") or []),
            "numerical_diagnostics_only": {
                "total_energy_hartree": strict.get("total_energy_hartree"),
                "vbm_hartree": gap.get("vbm_hartree"),
                "cbm_hartree": gap.get("cbm_hartree"),
            },
        }
        return SearchTaskEvaluation(
            task_name=self.name,
            valid=valid,
            reward=reward,
            objective={
                "sampled_gap_ev": sampled_gap,
                "reference_gap_ev": self.config.reference_gap_ev,
                "absolute_error_ev": absolute_error,
                "reward_definition": "-abs(sampled_gap_ev - reference_gap_ev)",
            },
            proposal_feedback=feedback,
            failure_reason=",".join(failures) or str(payload.get("error") or ""),
            walltime_s=_finite_or_none(payload.get("walltime_total_s")),
            raw_result=payload,
        )

    def apply_to_candidate(self, candidate: Any, evaluation: SearchTaskEvaluation) -> None:
        candidate.search_result = None
        candidate.validation_result = None
        candidate.fitness = evaluation.reward
        candidate.promotable = evaluation.valid
        candidate.failure_reason = evaluation.failure_reason
        candidate.guard_penalty = 0.0
        candidate.worst_validation_ratio = 1.0 if evaluation.valid else float("inf")
        candidate.tree_reward = tree_reward_from_raw(evaluation.reward) if evaluation.valid else 0.0

    def prompt_instructions(self) -> str:
        return "\n".join(
            [
                "The only optimization objective is the self-consistent bulk-ZnO sampled gap.",
                f"Reference gap: {self.config.reference_gap_ev:.3f} eV.",
                "raw_reward = -abs(sampled_gap_ev - reference_gap_ev); larger is better.",
                "Gamma/direct-gap, metallicity, SCF-cycle, and walltime fields are diagnostic-only and never improve ranking.",
                "Timeout is a hard execution cutoff, not a continuous penalty: any completed 1 h and 1.5 h candidates are equally acceptable if their gap reward is equal.",
                "Timeout, failed strict SCF, invalid gap, metallic state, or fractional occupations make a point invalid and ineligible as a parent.",
                "Total energy and unaligned absolute VBM/CBM are diagnostic-only; do not optimize their magnitude.",
                "HF exchange is forbidden. Do not propose alpha, beta, or omega changes.",
                "Component internal/ext parameters are fixed. Only add/remove/replace components and change component mixing factors.",
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
            "sampled_gap_ev",
            "absolute_error_ev",
        )
        objective_delta = {
            key: _before_after(parent.objective.get(key), child.objective.get(key))
            for key in keys
        }
        diagnostic_keys = (
            "gamma_gap_ev",
            "minimum_direct_gap_ev",
            "directness_delta_ev",
            "is_gamma_direct_gap",
            "strict_scf_cycles",
            "preconditioning_cycles",
        )
        diagnostics = {
            key: _before_after(parent.proposal_feedback.get(key), child.proposal_feedback.get(key))
            for key in diagnostic_keys
        }
        return {
            "reward": _before_after(parent.reward, child.reward),
            "objective": objective_delta,
            "diagnostics_only": diagnostics,
        }


def make_mtask_pw92_functional() -> XCFunctional:
    """Return the semilocal mTASK-PW92 starting scaffold used by the ZnO task."""
    return XCFunctional(
        name="mTASK-PW92",
        components=[
            ComponentSpec(name="MGGA_X_MTASK", xctype="MGGA", role="exchange"),
            ComponentSpec(name="LDA_C_PW", xctype="LDA", role="correlation"),
        ],
    )


def load_existing_zno_root_evaluation(
    source: Path | str,
    *,
    task: ZnOGapSearchTask,
    expected_kmesh: Sequence[int] = (2, 2, 2),
    expected_basis: str = "gth-dzvp-molopt-sr",
    expected_pseudo: str = "gth-pbe",
    assumed_cpus: int | None = None,
) -> tuple[SearchTaskEvaluation, dict[str, Any], Path]:
    """Load and validate a completed mTASK-PW92 result for the root node.

    A missing or incompatible result is a hard error. In particular, this
    helper never falls back to evaluating the root again.
    """
    result_path = _resolve_existing_result_path(Path(source))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    calculation = payload.get("calculation") or {}
    strict = payload.get("strict_scf") or {}
    effective_spec = calculation.get("xc_effective_spec") or {}

    mismatches: list[str] = []
    if not bool(payload.get("ok", False)):
        mismatches.append("result is not marked ok")
    if str(payload.get("stage") or "") != "complete":
        mismatches.append(f"stage={payload.get('stage')!r}, expected 'complete'")
    if not bool(strict.get("converged", False)):
        mismatches.append("strict SCF is not converged")
    if tuple(int(value) for value in calculation.get("kmesh") or ()) != tuple(expected_kmesh):
        mismatches.append(
            f"kmesh={calculation.get('kmesh')!r}, expected {list(expected_kmesh)!r}"
        )
    if str(calculation.get("basis") or "").lower() != expected_basis.lower():
        mismatches.append(
            f"basis={calculation.get('basis')!r}, expected {expected_basis!r}"
        )
    if str(calculation.get("pseudopotential") or "").lower() != expected_pseudo.lower():
        mismatches.append(
            f"pseudopotential={calculation.get('pseudopotential')!r}, expected {expected_pseudo!r}"
        )
    components = {
        item.strip().upper()
        for item in str(effective_spec.get("xc_code") or "").split(",")
        if item.strip()
    }
    if components != {"MGGA_X_MTASK", "LDA_C_PW"}:
        mismatches.append(
            f"xc_code={effective_spec.get('xc_code')!r}, expected mTASK-PW92"
        )
    hyb = tuple(float(value) for value in effective_spec.get("hyb") or ())
    if len(hyb) != 3 or any(abs(value) > 1e-12 for value in hyb):
        mismatches.append(f"hyb={effective_spec.get('hyb')!r}, expected [0, 0, 0]")
    if str((payload.get("structure") or {}).get("reduced_formula") or "") != "ZnO":
        mismatches.append("structure is not reduced_formula='ZnO'")
    if mismatches:
        raise ValueError(
            f"Existing ZnO root result {result_path} is incompatible: " + "; ".join(mismatches)
        )

    evaluation = task.from_payload(payload)
    if not evaluation.valid:
        raise ValueError(
            f"Existing ZnO root result {result_path} is invalid: {evaluation.failure_reason}"
        )
    evaluation.proposal_feedback.update(
        {
            "reused_existing_result": True,
            "existing_result_path": str(result_path),
            "evaluation_resource_cpus_assumed": assumed_cpus,
            "evaluation_resource_note": (
                "Historical approximately-15-CPU root result treated as the "
                "16-CPU search baseline; recorded walltime is retained unchanged."
                if assumed_cpus == 16
                else "Historical root result; recorded walltime is retained unchanged."
            ),
        }
    )
    return evaluation, payload, result_path


def _resolve_existing_result_path(source: Path) -> Path:
    if source.is_file():
        return source
    if not source.exists():
        raise FileNotFoundError(
            f"Existing ZnO root result was not found: {source}. "
            "The root is never re-evaluated automatically."
        )
    preferred = source / "mTASK-PW92_ZnO_k222.json"
    if preferred.is_file():
        return preferred
    candidates = sorted(source.glob("*ZnO*k222*.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not identify one ZnO root result JSON under {source}; "
        f"candidates={[path.name for path in candidates]}"
    )


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
