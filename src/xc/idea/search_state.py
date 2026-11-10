"""ProposalState — rich semantic state for MCTS nodes.

Analogous to Xcientist's IdeaState: carries not just the functional
but also semantic context (core_contribution, rationale, component_explanations,
memory_refs, edit_plan, skill_metrics) and search statistics (visits, reward).

Every state carries reasoning alongside parameters, mirroring Xcientist's
pattern where each IdeaState has 17 fields including title, abstract,
core_contribution, method, risks, rationale, memory_refs, components,
component_explanations, root_domains, paper_graph_context, edit_plan,
skill_metrics, and signature.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..xc_functional import XCFunctional
from ..benchmark.result import BenchmarkResult
from .actions import ActionType

# Limits for text/list fields (mirrors Xcientist's UNIFORM_CLIP_TEXT_LIMIT)
_MAX_TEXT_LEN = 10000
_MAX_LIST_LEN = 16
_MAX_COMPONENTS = 32


@dataclass
class ProposalState:
    """Rich semantic state for one MCTS node.

    Fields organized in 5 layers mirroring Xcientist's IdeaState:
    - Physical layer: xc, result, structural_sig
    - Semantic layer: core_contribution, rationale, memory_refs,
      component_explanations, paper_graph_context
    - Search layer: parent_sig, depth, visit_count, total_reward,
      active_skill, action_taken, action_detail, edit_plan, skill_metrics
    - Evaluation layer: evaluation (multi-dimensional scores + qualitative)
    - Provenance layer: tags, root_domains
    """

    # --- Physical layer ---
    xc: XCFunctional
    result: BenchmarkResult | None = None
    validation_result: BenchmarkResult | None = None
    structural_sig: str = ""

    # --- Semantic layer (mirrors Xcientist's IdeaState) ---
    core_contribution: str = ""
    rationale: str = ""
    memory_refs: list[str] = field(default_factory=list)
    component_explanations: dict[str, str] = field(default_factory=dict)
    paper_graph_context: str = ""
    hypothesis_id: str = ""
    hypothesis_status: str = "active"
    evidence_refs: list[str] = field(default_factory=list)
    analysis_refs: list[str] = field(default_factory=list)
    parent_hypothesis_id: str | None = None

    # --- Search layer ---
    finding_tags: list[str] = field(default_factory=list)
    defect_tags: list[str] = field(default_factory=list)
    active_skill: str = ""
    action_taken: ActionType | None = None
    action_detail: dict = field(default_factory=dict)
    proposal_id: str = ""
    proposal_index: int = 0
    edit_plan: dict[str, Any] | None = None
    skill_metrics: dict[str, Any] = field(default_factory=dict)
    parent_sig: str | None = None
    depth: int = 0
    visit_count: int = 0
    total_reward: float = 0.0

    # --- Evaluation layer ---
    evaluation: dict[str, float] = field(default_factory=dict)
    raw_scores: dict[str, int] = field(default_factory=dict)
    reward: float = 0.0
    raw_reward: float = 0.0
    validation_ratio: float | None = None
    validation_subset_ratios: dict[str, float] = field(default_factory=dict)
    promotable: bool = False
    promotion_status: str = "pending"
    failure_reason: str = ""

    # --- Provenance layer ---
    tags: list[str] = field(default_factory=list)
    root_domains: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.structural_sig and self.xc:
            self.structural_sig = self.xc.structural_signature()
        # Clip text fields
        self.core_contribution = self.core_contribution[:_MAX_TEXT_LEN]
        self.rationale = self.rationale[:_MAX_TEXT_LEN]
        self.paper_graph_context = self.paper_graph_context[:_MAX_TEXT_LEN]
        # Clip lists
        if len(self.memory_refs) > _MAX_LIST_LEN:
            self.memory_refs = self.memory_refs[:_MAX_LIST_LEN]
        if len(self.evidence_refs) > _MAX_LIST_LEN:
            self.evidence_refs = self.evidence_refs[:_MAX_LIST_LEN]
        if len(self.analysis_refs) > _MAX_LIST_LEN:
            self.analysis_refs = self.analysis_refs[:_MAX_LIST_LEN]
        if len(self.tags) > _MAX_LIST_LEN:
            self.tags = self.tags[:_MAX_LIST_LEN]
        if len(self.root_domains) > 2:
            self.root_domains = self.root_domains[:2]
        # Normalize component_explanations to match xc.components
        if self.xc and self.component_explanations:
            valid_names = {c.name for c in self.xc.components}
            self.component_explanations = {
                k: v[:_MAX_TEXT_LEN]
                for k, v in self.component_explanations.items()
                if k in valid_names
            }

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visit_count if self.visit_count > 0 else 0.0

    def compute_signature(self) -> str:
        """SHA-256 hash of canonical content (mirrors Xcientist IdeaState.signature).

        Hashes core_contribution + rationale + sorted(tags) + sorted(components)
        + component_explanations to capture semantic identity, not just physical.
        """
        payload = json.dumps({
            "core_contribution": self.core_contribution.lower().strip(),
            "rationale": self.rationale.lower().strip()[:500],
            "tags": sorted(set(self.tags)),
            "components": sorted(
                c.name for c in (self.xc.components if self.xc else [])
            ),
            "component_explanations": {
                k: v.lower().strip()[:200]
                for k, v in sorted(self.component_explanations.items())
            },
            "root_domains": sorted(self.root_domains),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_prompt_view(self, heading: str = "Current Proposal") -> str:
        """Render this state as a rich prompt section (mirrors Xcientist's to_prompt_view)."""
        lines = [f"## {heading}"]
        if self.core_contribution:
            lines.append(f"**Core contribution**: {self.core_contribution}")
        if self.rationale:
            lines.append(f"**Rationale**: {self.rationale}")
        if self.tags:
            lines.append(f"**Tags**: {', '.join(self.tags)}")
        if self.finding_tags:
            lines.append(f"**Finding tags**: {', '.join(self.finding_tags)}")
        if self.active_skill:
            lines.append(f"**Active skill**: {self.active_skill}")
        if self.component_explanations:
            lines.append("**Component roles**:")
            for name, explanation in sorted(self.component_explanations.items()):
                lines.append(f"  - {name}: {explanation}")
        if self.paper_graph_context:
            lines.append(f"**Literature context**: {self.paper_graph_context}")
        if self.evaluation:
            lines.append("**Evaluation scores**:")
            for metric, score in sorted(self.evaluation.items()):
                lines.append(f"  - {metric}: {score:.3f}")
        return "\n".join(lines)

    def to_payload(self) -> dict[str, Any]:
        """Full structured payload (mirrors Xcientist's to_payload)."""
        payload = {
            "structural_sig": self.structural_sig,
            "core_contribution": self.core_contribution,
            "rationale": self.rationale,
            "memory_refs": self.memory_refs,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_status": self.hypothesis_status,
            "evidence_refs": self.evidence_refs,
            "analysis_refs": self.analysis_refs,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "component_explanations": self.component_explanations,
            "components_with_explanations": [
                {"component": name, "explanation": explanation}
                for name, explanation in sorted(self.component_explanations.items())
            ],
            "paper_graph_context": self.paper_graph_context,
            "finding_tags": self.finding_tags,
            "defect_tags": self.defect_tags,
            "active_skill": self.active_skill,
            "action_taken": self.action_taken.value if self.action_taken else None,
            "action_detail": self.action_detail,
            "proposal_id": self.proposal_id,
            "proposal_index": self.proposal_index,
            "edit_plan": self.edit_plan,
            "skill_metrics": self.skill_metrics,
            "tags": self.tags,
            "root_domains": self.root_domains,
            "depth": self.depth,
            "visit_count": self.visit_count,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "evaluation": self.evaluation,
            "raw_scores": self.raw_scores,
            "reward": self.reward,
            "raw_reward": self.raw_reward,
            "validation_ratio": self.validation_ratio,
            "validation_subset_ratios": self.validation_subset_ratios,
            "promotable": self.promotable,
            "promotion_status": self.promotion_status,
            "failure_reason": self.failure_reason,
            "defect_tags": self.defect_tags,
            "parent_sig": self.parent_sig,
        }
        if self.xc:
            payload["xc_name"] = self.xc.name
            payload["xc_components"] = [
                {"name": c.name, "xctype": c.xctype, "role": c.role,
                 "mixing_factor": c.mixing_factor, "ext_params": c.ext_params}
                for c in self.xc.components
            ]
            payload["hf_exchange"] = {
                "alpha": self.xc.hf_exchange.alpha,
                "beta": self.xc.hf_exchange.beta,
                "omega": self.xc.hf_exchange.omega,
            }
        if self.result:
            payload["overall_mae"] = self.result.overall_mae
            payload["subsets"] = {
                name: {"mae_kcal": sr.mae_kcal, "n_reactions": sr.n_reactions}
                for name, sr in self.result.subsets.items()
            }
        if self.validation_result:
            payload["validation_mae"] = self.validation_result.overall_mae
            payload["validation_subsets"] = {
                name: {"mae_kcal": sr.mae_kcal, "n_reactions": sr.n_reactions}
                for name, sr in self.validation_result.subsets.items()
            }
        return payload

    def to_summary(self) -> dict[str, Any]:
        """Compact node view for traces and run-level diagnostics."""
        return {
            "xc_name": self.xc.name if self.xc else "",
            "structural_sig": self.structural_sig,
            "parent_sig": self.parent_sig,
            "depth": self.depth,
            "active_skill": self.active_skill,
            "action_taken": self.action_taken.value if self.action_taken else None,
            "proposal_id": self.proposal_id,
            "proposal_index": self.proposal_index,
            "search_mae": self.result.overall_mae if self.result else None,
            "validation_mae": self.validation_result.overall_mae if self.validation_result else None,
            "validation_ratio": self.validation_ratio,
            "validation_subset_ratios": self.validation_subset_ratios,
            "reward": self.reward,
            "raw_reward": self.raw_reward,
            "promotable": self.promotable,
            "promotion_status": self.promotion_status,
            "failure_reason": self.failure_reason,
            "visit_count": self.visit_count,
            "avg_reward": self.avg_reward,
            "analysis_refs": self.analysis_refs,
            "evidence_refs": self.evidence_refs,
        }
