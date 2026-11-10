"""Certification handoff payloads for downstream PySCF validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .analysis import BenchmarkAnalysisArtifact
from .search_state import ProposalState


@dataclass
class IdeaHandoff:
    candidate_name: str
    xc_code: str
    originating_skill: str
    search_path: dict[str, Any]
    evidence_refs: list[str]
    research_notes: list[str]
    expected_mechanism: str
    validation_plan: dict[str, Any]
    ablation_coverage: list[str]
    pyscf_payload: dict[str, Any]

    @classmethod
    def from_state(
        cls,
        state: ProposalState,
        analysis: BenchmarkAnalysisArtifact | None,
    ) -> "IdeaHandoff":
        xc = state.xc
        ablation_coverage = []
        if analysis:
            ablation_coverage = [
                item.metadata.get("component", "")
                for item in analysis.ablation_findings
                if item.metadata.get("component")
            ]
        return cls(
            candidate_name=xc.name,
            xc_code=xc.to_xc_code(),
            originating_skill=state.active_skill,
            search_path={
                "parent_sig": state.parent_sig,
                "structural_sig": state.structural_sig,
                "hypothesis_id": state.hypothesis_id,
            },
            evidence_refs=list(dict.fromkeys(state.evidence_refs + state.memory_refs)),
            research_notes=list(analysis.research_notes) if analysis else [],
            expected_mechanism=state.core_contribution,
            validation_plan=state.edit_plan or {},
            ablation_coverage=ablation_coverage,
            pyscf_payload={
                "name": xc.name,
                "xc_code": xc.to_xc_code(),
                "components": [
                    {
                        "name": item.name,
                        "xctype": item.xctype,
                        "role": item.role,
                        "mixing_factor": item.mixing_factor,
                        "ext_params": item.ext_params,
                    }
                    for item in xc.components
                ],
                "hf_exchange": {
                    "alpha": xc.hf_exchange.alpha,
                    "beta": xc.hf_exchange.beta,
                    "omega": xc.hf_exchange.omega,
                },
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "xc_code": self.xc_code,
            "originating_skill": self.originating_skill,
            "search_path": self.search_path,
            "evidence_refs": self.evidence_refs,
            "research_notes": self.research_notes,
            "expected_mechanism": self.expected_mechanism,
            "validation_plan": self.validation_plan,
            "ablation_coverage": self.ablation_coverage,
            "pyscf_payload": self.pyscf_payload,
        }
