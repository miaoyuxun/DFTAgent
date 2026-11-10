"""Component ablation — evaluate XC functional with components removed one at a time.

Produces AblationResult entries for each component, showing the causal
contribution of that component to each benchmark subset. This is the
key input for IdeaAgent: it tells the agent which components are
helpful or harmful on which subsets.

Pattern borrowed from Xcientist's ablation_results:
  {component, op, result, metric, value, analysis, method_context, confidence}

Normalization mirrors Xcientist's _normalize_ablation_component_entry():
  - result labels normalized to: positive | negative | inconclusive
  - confidence clamped to [0, 1]
  - method_context added for traceability
  - dual storage: raw results + normalized entries
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ..xc_functional import XCFunctional, ComponentSpec, HFExchange
from ..benchmark.result import BenchmarkResult, SubsetResult
from .component_taxonomy import classify_component, ComponentClassification

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..benchmark.runner import BenchmarkRunner

# --- Normalized result labels (Xcientist convention) ---
RESULT_POSITIVE = "positive"       # component is helpful (removing it hurt)
RESULT_NEGATIVE = "negative"       # component is harmful (removing it helped)
RESULT_INCONCLUSIVE = "inconclusive"  # removing had no significant effect

VALID_RESULT_LABELS = {RESULT_POSITIVE, RESULT_NEGATIVE, RESULT_INCONCLUSIVE}

# Delta thresholds in kcal/mol for classification
_DELTA_HELPFUL = 1.0      # delta > this → positive (component helpful)
_DELTA_HARMFUL = -1.0     # delta < this → negative (component harmful)


@dataclass
class AblationResult:
    """One component-level ablation result.

    Mirrors Xcientist's ablation_results schema:
    - component: which component was ablated
    - op: what operation was performed (remove, zero_mixing, zero_alpha)
    - result: positive/negative/inconclusive — did removing this component hurt or help?
    - metric: which benchmark metric (e.g., SIE4x4_mae)
    - value: the MAE value when this component is removed
    - baseline_value: the MAE value with the full functional
    - delta: value - baseline_value (positive = removing hurt, component is helpful)
    - analysis: human-readable explanation
    - method_context: what functional/setup this ablation was run on
    - confidence: how confident we are in this causal claim
    """
    component: str
    component_family: str
    op: str  # "remove", "zero_mixing", "zero_alpha", "zero_omega"
    result: str  # "positive" | "negative" | "inconclusive"
    metric: str
    value: float
    baseline_value: float
    delta: float  # value - baseline_value
    analysis: str
    method_context: str = ""
    confidence: float = 0.5


@dataclass
class AblationReport:
    """Full ablation report for a functional."""
    functional_name: str
    baseline_mae: float
    ablation_results: list[AblationResult] = field(default_factory=list)
    # Raw results before normalization (Xcientist dual storage)
    raw_results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalization (mirrors Xcientist's _normalize_ablation_component_entry)
# ---------------------------------------------------------------------------

def normalize_ablation_component_entry(raw: dict) -> dict:
    """Normalize a raw ablation entry to canonical form.

    Mirrors Xcientist's _normalize_ablation_component_entry():
    - result label normalized to positive/negative/inconclusive
    - confidence clamped to [0, 1]
    - method_context added if missing
    - required fields validated
    """
    required = {"component", "op", "result", "metric", "value"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"Missing required ablation fields: {missing}")

    normalized = dict(raw)

    # Normalize result label
    result = str(normalized["result"]).lower().strip()
    if result in VALID_RESULT_LABELS:
        normalized["result"] = result
    elif result in ("helpful", "good", "beneficial"):
        normalized["result"] = RESULT_POSITIVE
    elif result in ("harmful", "bad", "detrimental"):
        normalized["result"] = RESULT_NEGATIVE
    else:
        normalized["result"] = RESULT_INCONCLUSIVE

    # Clamp confidence
    conf = normalized.get("confidence", 0.5)
    normalized["confidence"] = max(0.0, min(1.0, float(conf)))

    # Ensure method_context
    if "method_context" not in normalized or not normalized["method_context"]:
        normalized["method_context"] = normalized.get("functional_name", "unknown")

    # Ensure component_family
    if "component_family" not in normalized or not normalized["component_family"]:
        cls = classify_component(normalized["component"])
        normalized["component_family"] = cls.family

    # Ensure baseline_value and delta
    if "baseline_value" not in normalized:
        normalized["baseline_value"] = None
    if "delta" not in normalized and "baseline_value" in normalized and normalized["baseline_value"] is not None:
        normalized["delta"] = normalized["value"] - normalized["baseline_value"]

    return normalized


def normalize_ablation_results_payload(payload: list[dict]) -> list[dict]:
    """Normalize a list of raw ablation entries.

    Mirrors Xcientist's normalize_ablation_results_payload():
    - Each entry normalized via normalize_ablation_component_entry()
    - Invalid entries logged and skipped
    """
    normalized = []
    for i, raw in enumerate(payload):
        try:
            normalized.append(normalize_ablation_component_entry(raw))
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping invalid ablation entry {i}: {e}")
    return normalized


# ---------------------------------------------------------------------------
# Ingestion (mirrors Xcientist's ingest_ablation_results)
# ---------------------------------------------------------------------------

def ingest_ablation_results(
    report: AblationReport,
    knowledge_engine: Any = None,
) -> list[dict]:
    """Ingest an AblationReport into normalized form, optionally storing in knowledge engine.

    Mirrors Xcientist's ingest_ablation_results():
    1. Convert AblationResult entries to raw dicts
    2. Normalize each entry
    3. Store in knowledge engine if provided
    4. Return normalized entries

    Args:
        report: AblationReport from run_component_ablation()
        knowledge_engine: Optional KnowledgeEngine for persistent storage

    Returns:
        List of normalized ablation entries
    """
    # Step 1: Convert to raw dicts
    raw_entries = []
    for ar in report.ablation_results:
        raw_entries.append({
            "component": ar.component,
            "component_family": ar.component_family,
            "op": ar.op,
            "result": ar.result,
            "metric": ar.metric,
            "value": ar.value,
            "baseline_value": ar.baseline_value,
            "delta": ar.delta,
            "analysis": ar.analysis,
            "method_context": ar.method_context or report.functional_name,
            "confidence": ar.confidence,
            "functional_name": report.functional_name,
        })

    # Step 2: Normalize
    normalized = normalize_ablation_results_payload(raw_entries)

    # Step 3: Store in knowledge engine if provided
    if knowledge_engine is not None:
        for entry in normalized:
            try:
                from .knowledge_engine import AblationEntry
                ae = AblationEntry(
                    component=entry["component"],
                    component_family=entry.get("component_family", "unknown"),
                    functional_name=entry.get("functional_name", report.functional_name),
                    result=entry["result"],
                    metric=entry["metric"],
                    value=entry["value"],
                    analysis=entry.get("analysis", ""),
                    confidence=entry.get("confidence", 0.5),
                )
                knowledge_engine.add_ablation(ae)
            except Exception as e:
                logger.warning(f"Failed to store ablation entry for {entry['component']}: {e}")

    logger.info(
        f"Ingested {len(normalized)} ablation results for {report.functional_name} "
        f"(from {len(raw_entries)} raw entries)"
    )
    return normalized


# ---------------------------------------------------------------------------
# Core ablation runner
# ---------------------------------------------------------------------------

def run_component_ablation(
    xc: XCFunctional,
    runner: BenchmarkRunner,
    baseline_result: BenchmarkResult | None = None,
) -> AblationReport:
    """Run component-level ablation on an XCFunctional.

    For each component, remove it (or zero its mixing factor) and
    re-evaluate on the same subsets. Compare against baseline to
    determine the causal contribution of each component.

    Args:
        xc: The XCFunctional to ablate.
        runner: BenchmarkRunner to use for evaluation.
        baseline_result: Pre-computed baseline result (avoids re-running).

    Returns:
        AblationReport with per-component causal evidence.
    """
    # Get baseline
    if baseline_result is None:
        baseline_result = runner.evaluate(xc)
    baseline_mae = baseline_result.overall_mae

    report = AblationReport(
        functional_name=xc.name,
        baseline_mae=baseline_mae,
    )

    # Ablate each component
    for i, comp in enumerate(xc.components):
        # Skip dispersion components (they're optional, ablation is obvious)
        if comp.role == "dispersion":
            continue

        # Skip compound exchange_correlation components (can't remove them individually)
        if comp.role == "exchange_correlation":
            continue

        # Strategy: zero the mixing factor (keeps component registered but removes its contribution)
        ablated_xc = _ablate_by_zero_mixing(xc, i)

        # Validate the ablated functional
        violations = ablated_xc.validate()
        if violations:
            logger.info(f"Skipping ablation of {comp.name}: violations={violations}")
            continue

        # Evaluate ablated functional
        try:
            ablated_result = runner.evaluate(ablated_xc)
        except Exception as e:
            logger.warning(f"Ablation evaluation failed for {comp.name}: {e}")
            continue

        # Compute per-subset deltas
        cls = classify_component(comp.name, comp.role)
        for subset_name, sr in ablated_result.subsets.items():
            baseline_sr = baseline_result.subsets.get(subset_name)
            if baseline_sr is None or sr.n_reactions == 0 or baseline_sr.n_reactions == 0:
                continue

            delta = sr.mae_kcal - baseline_sr.mae_kcal
            result_label = _classify_delta(delta)

            ar = AblationResult(
                component=comp.name,
                component_family=cls.family,
                op="zero_mixing",
                result=result_label,
                metric=f"{subset_name}_mae",
                value=sr.mae_kcal,
                baseline_value=baseline_sr.mae_kcal,
                delta=delta,
                analysis=_make_analysis(comp.name, "zero_mixing", subset_name,
                                        baseline_sr.mae_kcal, sr.mae_kcal, delta),
                method_context=xc.name,
                confidence=0.7 if abs(delta) > 2.0 else 0.5,
            )
            report.ablation_results.append(ar)
            report.raw_results.append({
                "component": comp.name, "op": "zero_mixing",
                "result": result_label, "metric": f"{subset_name}_mae",
                "value": sr.mae_kcal, "baseline_value": baseline_sr.mae_kcal,
                "delta": delta, "functional_name": xc.name,
            })

    # Also ablate HF exchange if present
    if xc.hf_exchange.alpha > 0:
        _ablate_and_record(
            xc, runner, baseline_result, report,
            ablate_fn=_ablate_hf_exchange,
            component_name="HF_EXCHANGE",
            component_family="hf_exchange.global",
            op="zero_alpha",
        )

    # Also ablate RSH omega if present
    if xc.hf_exchange.omega > 0:
        _ablate_and_record(
            xc, runner, baseline_result, report,
            ablate_fn=_ablate_rsh,
            component_name="RSH_OMEGA",
            component_family="hf_exchange.range_separated",
            op="zero_omega",
        )

    return report


def _classify_delta(delta: float) -> str:
    """Classify a delta into positive/negative/inconclusive.

    delta > 0: removing the component HURT → component is helpful → "positive"
    delta < 0: removing the component HELPED → component is harmful → "negative"
    delta ≈ 0: removing had no effect → "inconclusive"
    """
    if delta > _DELTA_HELPFUL:
        return RESULT_POSITIVE
    elif delta < _DELTA_HARMFUL:
        return RESULT_NEGATIVE
    else:
        return RESULT_INCONCLUSIVE


def _make_analysis(component: str, op: str, subset: str,
                   baseline: float, ablated: float, delta: float) -> str:
    """Generate human-readable analysis for an ablation result."""
    if delta > _DELTA_HELPFUL:
        verdict = "component is helpful"
    elif delta < _DELTA_HARMFUL:
        verdict = "component is harmful"
    else:
        verdict = "component has negligible effect"
    return (f"Removing {component} ({op}) on {subset}: "
            f"MAE changed from {baseline:.2f} to {ablated:.2f} "
            f"(delta={delta:+.2f}, {verdict})")


def _ablate_and_record(
    xc: XCFunctional,
    runner: BenchmarkRunner,
    baseline_result: BenchmarkResult,
    report: AblationReport,
    ablate_fn,
    component_name: str,
    component_family: str,
    op: str,
) -> None:
    """Run a single ablation and record results into the report."""
    ablated_xc = ablate_fn(xc)
    violations = ablated_xc.validate()
    if violations:
        logger.info(f"Skipping ablation of {component_name}: violations={violations}")
        return

    try:
        ablated_result = runner.evaluate(ablated_xc)
    except Exception as e:
        logger.warning(f"Ablation evaluation failed for {component_name}: {e}")
        return

    for subset_name, sr in ablated_result.subsets.items():
        baseline_sr = baseline_result.subsets.get(subset_name)
        if baseline_sr is None or sr.n_reactions == 0 or baseline_sr.n_reactions == 0:
            continue

        delta = sr.mae_kcal - baseline_sr.mae_kcal
        result_label = _classify_delta(delta)

        ar = AblationResult(
            component=component_name,
            component_family=component_family,
            op=op,
            result=result_label,
            metric=f"{subset_name}_mae",
            value=sr.mae_kcal,
            baseline_value=baseline_sr.mae_kcal,
            delta=delta,
            analysis=_make_analysis(component_name, op, subset_name,
                                    baseline_sr.mae_kcal, sr.mae_kcal, delta),
            method_context=xc.name,
            confidence=0.7 if abs(delta) > 2.0 else 0.5,
        )
        report.ablation_results.append(ar)
        report.raw_results.append({
            "component": component_name, "op": op,
            "result": result_label, "metric": f"{subset_name}_mae",
            "value": sr.mae_kcal, "baseline_value": baseline_sr.mae_kcal,
            "delta": delta, "functional_name": xc.name,
        })


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def _ablate_by_zero_mixing(xc: XCFunctional, component_index: int) -> XCFunctional:
    """Create a copy of xc with one component's mixing_factor set to 0."""
    new = copy.deepcopy(xc)
    new.name = f"{xc.name}_ablated_{new.components[component_index].name}"
    new.components[component_index].mixing_factor = 0.0
    new.metadata["ablation"] = f"zero_mixing:{new.components[component_index].name}"
    return new


def _ablate_hf_exchange(xc: XCFunctional) -> XCFunctional:
    """Create a copy of xc with HF exchange alpha set to 0."""
    new = copy.deepcopy(xc)
    new.name = f"{xc.name}_ablated_HF"
    new.hf_exchange.alpha = 0.0
    new.hf_exchange.beta = 0.0
    # Redistribute: all exchange comes from DFT components
    exch_sum = sum(c.mixing_factor for c in new.components if c.role == "exchange")
    if exch_sum > 0:
        target = 1.0
        scale = target / exch_sum
        for c in new.components:
            if c.role == "exchange":
                c.mixing_factor *= scale
    new.metadata["ablation"] = "zero_alpha"
    return new


def _ablate_rsh(xc: XCFunctional) -> XCFunctional:
    """Create a copy of xc with RSH omega set to 0 (becomes a global hybrid)."""
    new = copy.deepcopy(xc)
    new.name = f"{xc.name}_ablated_RSH"
    new.hf_exchange.omega = 0.0
    new.metadata["ablation"] = "zero_omega"
    return new


# ---------------------------------------------------------------------------
# Formatting (for prompt injection)
# ---------------------------------------------------------------------------

def format_ablation_report(report: AblationReport) -> str:
    """Format an AblationReport for prompt injection.

    Follows Xcientist's pattern: component-level causal evidence
    presented as structured text, with interpretation guide.
    """
    lines = []
    lines.append(f"## Component Ablation Results for {report.functional_name}")
    lines.append(f"Baseline overall MAE: {report.baseline_mae:.2f} kcal/mol\n")

    # Interpretation guide (mirrors Xcientist's re_analysis_replan semantics)
    lines.append("### Interpretation Guide")
    lines.append("- **positive** result = removing this component HURT performance → component is **helpful**, keep it")
    lines.append("- **negative** result = removing this component HELPED performance → component is **harmful**, consider removing or replacing it")
    lines.append("- **inconclusive** result = removing had no significant effect → component is negligible on this metric\n")

    # Group by component
    by_component: dict[str, list[AblationResult]] = {}
    for ar in report.ablation_results:
        if ar.component not in by_component:
            by_component[ar.component] = []
        by_component[ar.component].append(ar)

    for comp_name, results in by_component.items():
        lines.append(f"### {comp_name}")
        for ar in results:
            sign = "+" if ar.delta > 0 else ""
            lines.append(
                f"- {ar.metric}: baseline={ar.baseline_value:.2f}, "
                f"ablated={ar.value:.2f}, delta={sign}{ar.delta:.2f} "
                f"→ {ar.result} (confidence={ar.confidence:.1f})"
            )
        # Summary: is this component helpful or harmful overall?
        deltas = [ar.delta for ar in results]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        if avg_delta > _DELTA_HELPFUL:
            lines.append(f"  **Overall: this component is helpful** (avg delta={avg_delta:+.2f})")
        elif avg_delta < _DELTA_HARMFUL:
            lines.append(f"  **Overall: this component is harmful** (avg delta={avg_delta:+.2f})")
        else:
            lines.append(f"  **Overall: inconclusive** (avg delta={avg_delta:+.2f})")
        lines.append("")

    return "\n".join(lines)
