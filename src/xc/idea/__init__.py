"""XC idea agent package."""

from .agent import IdeaAgent
from .actions import Action, ActionType, apply_action
from .component_taxonomy import classify_component, extract_component_families, family_summary
from .knowledge_engine import KnowledgeEngine, KnowledgeBundle, format_knowledge_bundle
from .ablation import (
    AblationResult, AblationReport,
    run_component_ablation, ingest_ablation_results,
    format_ablation_report,
    normalize_ablation_component_entry, normalize_ablation_results_payload,
    RESULT_POSITIVE, RESULT_NEGATIVE, RESULT_INCONCLUSIVE,
)

__all__ = [
    "IdeaAgent", "Action", "ActionType", "apply_action",
    "classify_component", "extract_component_families", "family_summary",
    "KnowledgeEngine", "KnowledgeBundle", "format_knowledge_bundle",
    "AblationResult", "AblationReport",
    "run_component_ablation", "ingest_ablation_results",
    "format_ablation_report",
    "normalize_ablation_component_entry", "normalize_ablation_results_payload",
    "RESULT_POSITIVE", "RESULT_NEGATIVE", "RESULT_INCONCLUSIVE",
]
