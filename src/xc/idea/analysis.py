"""Structured benchmark and ablation interpretation for the XC loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..benchmark.result import BenchmarkResult
from .ablation import AblationReport, RESULT_INCONCLUSIVE, RESULT_NEGATIVE, RESULT_POSITIVE


@dataclass
class Observation:
    observation_id: str
    kind: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "kind": self.kind,
            "payload": self.payload,
        }


@dataclass
class Finding:
    finding_id: str
    kind: str
    summary: str
    evidence_refs: list[str]
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "kind": self.kind,
            "summary": self.summary,
            "evidence_refs": self.evidence_refs,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkAnalysisArtifact:
    round_index: int
    observations: list[Observation]
    ablation_findings: list[Finding]
    research_notes: list[str]
    open_questions: list[str]
    next_tests: list[str]

    @property
    def findings(self) -> list[Finding]:
        return self.ablation_findings

    @property
    def evidence_refs(self) -> list[str]:
        refs: list[str] = []
        for finding in self.ablation_findings:
            refs.extend(finding.evidence_refs)
        return list(dict.fromkeys(refs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "observations": [item.to_dict() for item in self.observations],
            "ablation_findings": [item.to_dict() for item in self.ablation_findings],
            "research_notes": self.research_notes,
            "open_questions": self.open_questions,
            "next_tests": self.next_tests,
        }


class BenchmarkAnalyzer:
    """Extract hard evidence before any idea-generation stage runs."""

    def analyze(
        self,
        *,
        round_index: int,
        result: BenchmarkResult,
        ablation_report: AblationReport | None,
    ) -> BenchmarkAnalysisArtifact:
        observations = self._metric_observations(round_index, result)
        findings = self._ablation_findings(round_index, ablation_report)
        notes = [finding.summary for finding in findings]
        open_questions: list[str] = []
        next_tests: list[str] = []

        if not findings:
            open_questions.append("No component-level causal evidence is available yet.")
            next_tests.append("Run component ablation on the current best functional.")
        else:
            inconclusive = [f for f in findings if f.kind == RESULT_INCONCLUSIVE]
            if inconclusive:
                open_questions.append(
                    "Some components remain causally unresolved after ablation."
                )
                next_tests.append(
                    "Repeat or refine ablation for inconclusive components before overcommitting."
                )

        return BenchmarkAnalysisArtifact(
            round_index=round_index,
            observations=observations,
            ablation_findings=findings,
            research_notes=notes,
            open_questions=open_questions,
            next_tests=next_tests,
        )

    def _metric_observations(
        self, round_index: int, result: BenchmarkResult
    ) -> list[Observation]:
        rows: list[Observation] = [
            Observation(
                observation_id=f"obs:r{round_index}:overall",
                kind="benchmark_overall",
                payload={"xc_name": result.xc_name, "overall_mae": result.overall_mae},
            )
        ]
        for subset, subset_result in sorted(result.subsets.items()):
            rows.append(
                Observation(
                    observation_id=f"obs:r{round_index}:subset:{subset}",
                    kind="benchmark_subset",
                    payload={
                        "xc_name": result.xc_name,
                        "subset": subset,
                        "mae_kcal": subset_result.mae_kcal,
                        "n_reactions": subset_result.n_reactions,
                        "n_failures": subset_result.n_failures,
                    },
                )
            )
        return rows

    def _ablation_findings(
        self, round_index: int, report: AblationReport | None
    ) -> list[Finding]:
        if report is None:
            return []
        findings: list[Finding] = []
        for idx, entry in enumerate(report.ablation_results, start=1):
            ref = f"ablation:r{round_index}:{entry.component}:{entry.metric}"
            if entry.result == RESULT_POSITIVE:
                summary = (
                    f"{entry.component} is supported by ablation: removing it worsened "
                    f"{entry.metric} by {entry.delta:+.3f}."
                )
            elif entry.result == RESULT_NEGATIVE:
                summary = (
                    f"{entry.component} appears harmful or redundant: removing it improved "
                    f"{entry.metric} by {abs(entry.delta):.3f}."
                )
            else:
                summary = (
                    f"{entry.component} remains inconclusive on {entry.metric}; "
                    f"ablation delta was {entry.delta:+.3f}."
                )
            findings.append(
                Finding(
                    finding_id=f"finding:r{round_index}:{idx:03d}",
                    kind=entry.result,
                    summary=summary,
                    evidence_refs=[ref],
                    confidence=entry.confidence,
                    metadata={
                        "component": entry.component,
                        "component_family": entry.component_family,
                        "metric": entry.metric,
                        "delta": entry.delta,
                    },
                )
            )
        return findings
