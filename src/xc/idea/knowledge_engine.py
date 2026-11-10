"""Knowledge engine — SQLite-backed structured memory for XC optimization.

Three tables:
- knowledge: domain knowledge, empirical rules, anti-patterns
- ablation_evidence: component-level causal evidence from past evaluations
- search_trace: full record of every candidate evaluated

No FAISS needed — XC component space is finite and structured (679 libxc functionals),
so SQLite with structured queries + FTS5 full-text search is sufficient.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from ..xc_functional import (
    XCFunctional,
    ComponentSpec,
    HFExchange,
    NAME_TO_ID,
    PURE_COMPONENT_ROLES,
)
from ..benchmark.result import BenchmarkResult, SubsetResult
from ..json_utils import json_safe_copy, safe_json_dumps
from .analysis import BenchmarkAnalysisArtifact, Observation, Finding
from .component_taxonomy import classify_component, extract_component_families, ComponentClassification
from .ablation import AblationReport, AblationResult, ingest_ablation_results, run_component_ablation
from .functional_graph import (
    EXPECTED_SUBSETS,
    GUARD_SUBSETS,
    TARGET_SUBSETS,
    FunctionalEdge,
    FunctionalGraph,
    FunctionalNode,
    append_basin_agent_artifact,
    load_functional_graph,
)

if TYPE_CHECKING:
    from ..benchmark.runner import BenchmarkRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for knowledge retrieval
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeEntry:
    category: str  # 'domain' | 'empirical' | 'constraint' | 'anti_pattern'
    key: str
    content: str
    confidence: float = 1.0
    source: str = "observation"


@dataclass
class AblationEntry:
    component: str
    component_family: str
    functional_name: str
    result: str  # 'positive' | 'negative' | 'inconclusive'
    metric: str
    value: float
    analysis: str
    confidence: float = 0.5


@dataclass
class TraceEntry:
    name: str
    parent_name: str
    action_type: str
    action_detail: str
    signature: str
    search_mae: float
    validation_mae: float
    status: str  # 'accepted' | 'rejected' | 'failed'
    finding_tags: str
    core_contribution: str = ""
    rationale: str = ""
    component_explanations: str = ""  # JSON dict
    edit_plan: str = ""  # JSON dict
    skill_metrics: str = ""  # JSON dict
    active_skill: str = ""
    root_domains: str = ""  # JSON list


@dataclass
class HypothesisEntry:
    hypothesis_id: str
    statement: str
    target_defects: str
    status: str
    confidence: float
    support_refs: str = ""
    counter_refs: str = ""


@dataclass
class MechanismCardEntry:
    mechanism_key: str
    statement: str
    target_subset: str = ""
    target_delta: float = 0.0
    validation_penalty: float = 0.0
    risk_score: float = 0.0
    support_count: int = 0
    target_defects: list[str] = field(default_factory=list)
    component_families: list[str] = field(default_factory=list)
    expected_effects: dict[str, Any] = field(default_factory=dict)
    risk_subsets: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "provisional"


@dataclass
class FailurePatternEntry:
    pattern_key: str
    symptom: str
    root_cause: str = ""
    repair_hint: str = ""
    action_type: str = ""
    component_family: str = ""
    hard_block: bool = False
    occurrences: int = 1
    confidence: float = 0.5


@dataclass
class RiskPatternEntry:
    pattern_key: str
    target_subset: str
    risk_subset: str
    symptom: str
    action_type: str = ""
    active_skill: str = ""
    component_family: str = ""
    occurrences: int = 1
    avg_ratio: float = 1.0
    confidence: float = 0.5


@dataclass
class LibxcAnchorEntry:
    functional_name: str
    source: str
    structural_signature: str
    similarity: float
    anchor_type: str = "nearest_neighbor"
    target_score: float = 0.0
    validation_penalty: float = 0.0
    subset_metrics: dict[str, Any] = field(default_factory=dict)
    component_families: list[str] = field(default_factory=list)
    defect_tags: list[str] = field(default_factory=list)


@dataclass
class ComponentBenchmarkPriorEntry:
    component_name: str
    role: str
    occurrences: int
    best_functional: str
    association_score: float
    target_mae: float
    target_gain: float
    guard_metrics: dict[str, float] = field(default_factory=dict)
    failures: int = 0
    target_failures: int = 0


@dataclass
class CandidatePriorEntry:
    key: str
    skill: str
    action_type: str
    prior_score: float
    risk_score: float
    evidence_refs: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class ActionOutcomeEntry:
    pattern_key: str
    verdict: str  # 'promising' | 'risky'
    action_type: str
    active_skill: str = ""
    component_family: str = ""
    target_delta: float = 0.0
    risk_score: float = 0.0
    worst_regression_ratio: float = 1.0
    occurrences: int = 1
    supports: dict[str, float] = field(default_factory=dict)
    regressions: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    confidence: float = 0.5


@dataclass
class KnowledgeBundle:
    """Assembled context for IdeaAgent proposal prompt."""
    ablation_evidence: list[AblationEntry] = field(default_factory=list)
    relevant_traces: list[TraceEntry] = field(default_factory=list)
    mechanism_cards: list[MechanismCardEntry] = field(default_factory=list)
    failure_patterns: list[FailurePatternEntry] = field(default_factory=list)
    risk_patterns: list[RiskPatternEntry] = field(default_factory=list)
    libxc_anchors: list[LibxcAnchorEntry] = field(default_factory=list)
    component_benchmark_priors: list[ComponentBenchmarkPriorEntry] = field(default_factory=list)
    candidate_priors: list[CandidatePriorEntry] = field(default_factory=list)
    action_outcomes: list[ActionOutcomeEntry] = field(default_factory=list)
    active_hypotheses: list[HypothesisEntry] = field(default_factory=list)
    # Latest ablation report (direct from run_component_ablation)
    ablation_report: AblationReport | None = None


# ---------------------------------------------------------------------------
# KnowledgeEngine
# ---------------------------------------------------------------------------

class KnowledgeEngine:
    """SQLite-backed knowledge engine for XC functional optimization."""

    def __init__(
        self,
        db_path: Path | str = "results/knowledge.db",
        *,
        functional_graph_path: Path | str | None = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_tables()
        self.functional_graph_path = (
            Path(functional_graph_path) if functional_graph_path is not None else None
        )
        self.functional_graph: FunctionalGraph | None = None
        if self.functional_graph_path is not None:
            self._load_functional_graph()

    def _load_functional_graph(self) -> None:
        """Load permanent graph memory without making search depend on it."""
        path = self.functional_graph_path
        if path is None:
            return
        try:
            self.functional_graph = load_functional_graph(path)
        except (OSError, TypeError, ValueError) as exc:
            self.functional_graph = None
            logger.warning("Functional graph unavailable at %s: %s", path, exc)
            return
        logger.info(
            "Loaded functional graph from %s: nodes=%d edges=%d",
            path,
            len(self.functional_graph.nodes),
            len(self.functional_graph.edges),
        )

    def ingest_basin_agent_graph(
        self,
        artifact_path: Path | str,
    ) -> dict[str, int]:
        """Refresh selected Basin nodes in the standalone functional graph."""
        if self.functional_graph_path is None:
            return {
                "nodes_added": 0,
                "edges_added": 0,
                "skipped_invalid_nodes": 0,
                "skipped_edges": 0,
            }
        report = append_basin_agent_artifact(
            self.functional_graph_path,
            artifact_path,
        )
        self._load_functional_graph()
        return report

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'observation',
                created_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_key ON knowledge(key);

            CREATE TABLE IF NOT EXISTS ablation_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                component_family TEXT NOT NULL,
                functional_name TEXT NOT NULL,
                result TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                analysis TEXT,
                confidence REAL DEFAULT 0.5,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ablation_component ON ablation_evidence(component);
            CREATE INDEX IF NOT EXISTS idx_ablation_family ON ablation_evidence(component_family);

            CREATE TABLE IF NOT EXISTS search_trace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                parent_name TEXT,
                action_type TEXT NOT NULL,
                action_detail TEXT,
                signature TEXT NOT NULL UNIQUE,
                search_mae REAL,
                validation_mae REAL,
                status TEXT NOT NULL,
                defect_tags TEXT,
                core_contribution TEXT DEFAULT '',
                rationale TEXT DEFAULT '',
                component_explanations TEXT DEFAULT '',
                edit_plan TEXT DEFAULT '',
                skill_metrics TEXT DEFAULT '',
                active_skill TEXT DEFAULT '',
                root_domains TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trace_status ON search_trace(status);
            CREATE INDEX IF NOT EXISTS idx_trace_action ON search_trace(action_type);

            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                target_defects TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                support_refs TEXT DEFAULT '',
                counter_refs TEXT DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_hypothesis_status ON hypotheses(status);

            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS findings (
                finding_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS functional_evaluations (
                eval_id TEXT PRIMARY KEY,
                structural_signature TEXT NOT NULL,
                functional_name TEXT NOT NULL,
                source TEXT NOT NULL,
                xc_json TEXT NOT NULL,
                xc_code TEXT DEFAULT '',
                component_families_json TEXT NOT NULL,
                hf_exchange_json TEXT NOT NULL,
                subset_metrics_json TEXT NOT NULL,
                benchmark_protocol_hash TEXT NOT NULL,
                benchmark_context_json TEXT DEFAULT '{}',
                status TEXT NOT NULL,
                failure_reason TEXT DEFAULT '',
                defect_tags_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_functional_eval_unique
                ON functional_evaluations(structural_signature, source, benchmark_protocol_hash);
            CREATE INDEX IF NOT EXISTS idx_functional_eval_sig
                ON functional_evaluations(structural_signature);
            CREATE INDEX IF NOT EXISTS idx_functional_eval_source
                ON functional_evaluations(source);

            CREATE TABLE IF NOT EXISTS edit_effects (
                effect_id TEXT PRIMARY KEY,
                parent_eval_id TEXT NOT NULL,
                child_eval_id TEXT NOT NULL,
                parent_signature TEXT NOT NULL,
                child_signature TEXT NOT NULL,
                action_type TEXT NOT NULL,
                active_skill TEXT DEFAULT '',
                action_detail_json TEXT NOT NULL,
                component_families_json TEXT NOT NULL,
                subset_effects_json TEXT NOT NULL,
                promotable INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                generalization_risk REAL DEFAULT 0.0,
                support_refs_json TEXT NOT NULL,
                counter_refs_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_edit_effect_unique
                ON edit_effects(parent_eval_id, child_eval_id, action_type, active_skill);
            CREATE INDEX IF NOT EXISTS idx_edit_effect_action
                ON edit_effects(action_type);
            CREATE INDEX IF NOT EXISTS idx_edit_effect_skill
                ON edit_effects(active_skill);

            CREATE TABLE IF NOT EXISTS edit_subset_effects (
                effect_id TEXT NOT NULL,
                subset TEXT NOT NULL,
                panel TEXT NOT NULL,
                before_mae REAL NOT NULL,
                after_mae REAL NOT NULL,
                improvement_delta REAL NOT NULL,
                ratio REAL NOT NULL,
                is_target INTEGER DEFAULT 0,
                is_regression INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY(effect_id, subset)
            );
            CREATE INDEX IF NOT EXISTS idx_edit_subset_target
                ON edit_subset_effects(subset, is_target);
            CREATE INDEX IF NOT EXISTS idx_edit_subset_regression
                ON edit_subset_effects(is_regression);

            CREATE TABLE IF NOT EXISTS mechanism_cards (
                mechanism_key TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                target_subset TEXT DEFAULT '',
                target_delta REAL DEFAULT 0.0,
                validation_penalty REAL DEFAULT 0.0,
                risk_score REAL DEFAULT 0.0,
                support_count INTEGER DEFAULT 0,
                target_defects_json TEXT NOT NULL,
                component_families_json TEXT NOT NULL,
                expected_effects_json TEXT NOT NULL,
                risk_subsets_json TEXT NOT NULL,
                guardrails_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mechanism_confidence
                ON mechanism_cards(confidence);
            CREATE INDEX IF NOT EXISTS idx_mechanism_target
                ON mechanism_cards(target_subset);

            CREATE TABLE IF NOT EXISTS failure_patterns (
                pattern_key TEXT PRIMARY KEY,
                symptom TEXT NOT NULL,
                root_cause TEXT DEFAULT '',
                repair_hint TEXT DEFAULT '',
                action_type TEXT DEFAULT '',
                component_family TEXT DEFAULT '',
                hard_block INTEGER DEFAULT 0,
                occurrences INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_failure_action
                ON failure_patterns(action_type);

            CREATE TABLE IF NOT EXISTS risk_patterns (
                pattern_key TEXT PRIMARY KEY,
                target_subset TEXT NOT NULL,
                risk_subset TEXT NOT NULL,
                symptom TEXT NOT NULL,
                action_type TEXT DEFAULT '',
                active_skill TEXT DEFAULT '',
                component_family TEXT DEFAULT '',
                occurrences INTEGER DEFAULT 1,
                avg_ratio REAL DEFAULT 1.0,
                confidence REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_risk_target
                ON risk_patterns(target_subset);
            CREATE INDEX IF NOT EXISTS idx_risk_skill
                ON risk_patterns(active_skill);
        """)
        for table, columns in {
            "functional_evaluations": [
                ("benchmark_context_json", "TEXT DEFAULT '{}'"),
            ],
            "mechanism_cards": [
                ("target_subset", "TEXT DEFAULT ''"),
                ("target_delta", "REAL DEFAULT 0.0"),
                ("validation_penalty", "REAL DEFAULT 0.0"),
                ("risk_score", "REAL DEFAULT 0.0"),
                ("support_count", "INTEGER DEFAULT 0"),
            ],
        }.items():
            for col, typ in columns:
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass
        # Add columns to existing tables that may lack the new fields
        for col, typ in [
            ("core_contribution", "TEXT DEFAULT ''"),
            ("rationale", "TEXT DEFAULT ''"),
            ("component_explanations", "TEXT DEFAULT ''"),
            ("edit_plan", "TEXT DEFAULT ''"),
            ("skill_metrics", "TEXT DEFAULT ''"),
            ("active_skill", "TEXT DEFAULT ''"),
            ("root_domains", "TEXT DEFAULT ''"),
        ]:
            try:
                self.conn.execute(f"ALTER TABLE search_trace ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass  # column already exists
        self.conn.commit()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_knowledge(self, entry: KnowledgeEntry) -> int:
        """Add or update a knowledge entry."""
        now = datetime.now().isoformat()
        existing = self.conn.execute(
            "SELECT id FROM knowledge WHERE key = ?", (entry.key,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE knowledge SET content=?, confidence=?, source=? WHERE key=?",
                (entry.content, entry.confidence, entry.source, entry.key),
            )
            self.conn.commit()
            return existing[0]

        cur = self.conn.execute(
            "INSERT INTO knowledge (category, key, content, confidence, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entry.category, entry.key, entry.content, entry.confidence, entry.source, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_ablation(self, entry: AblationEntry) -> int:
        cur = self.conn.execute(
            "INSERT INTO ablation_evidence "
            "(component, component_family, functional_name, result, metric, value, "
            "analysis, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.component, entry.component_family, entry.functional_name,
                entry.result, entry.metric, entry.value, entry.analysis,
                entry.confidence, datetime.now().isoformat(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_trace(self, entry: TraceEntry) -> int:
        existing = self.conn.execute(
            "SELECT id FROM search_trace WHERE signature = ?", (entry.signature,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE search_trace SET status=?, search_mae=?, validation_mae=?, "
                "defect_tags=?, core_contribution=?, rationale=?, "
                "component_explanations=?, edit_plan=?, skill_metrics=?, "
                "active_skill=?, root_domains=? WHERE signature=?",
                (entry.status, entry.search_mae, entry.validation_mae,
                 entry.finding_tags, entry.core_contribution, entry.rationale,
                 entry.component_explanations, entry.edit_plan,
                 entry.skill_metrics, entry.active_skill,
                 entry.root_domains, entry.signature),
            )
            self.conn.commit()
            return existing[0]

        cur = self.conn.execute(
            "INSERT INTO search_trace "
            "(name, parent_name, action_type, action_detail, signature, "
            "search_mae, validation_mae, status, defect_tags, "
            "core_contribution, rationale, component_explanations, "
            "edit_plan, skill_metrics, active_skill, root_domains, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.name, entry.parent_name, entry.action_type, entry.action_detail,
                entry.signature, entry.search_mae, entry.validation_mae,
                entry.status, entry.finding_tags,
                entry.core_contribution, entry.rationale,
                entry.component_explanations, entry.edit_plan,
                entry.skill_metrics, entry.active_skill,
                entry.root_domains, datetime.now().isoformat(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def record_analysis(self, artifact: BenchmarkAnalysisArtifact) -> None:
        now = datetime.now().isoformat()
        for observation in artifact.observations:
            self.conn.execute(
                "INSERT OR REPLACE INTO observations (observation_id, kind, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    observation.observation_id,
                    observation.kind,
                    json.dumps(observation.payload),
                    now,
                ),
            )
        for finding in artifact.ablation_findings:
            self.conn.execute(
                "INSERT OR REPLACE INTO findings "
                "(finding_id, kind, summary, evidence_refs, confidence, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    finding.finding_id,
                    finding.kind,
                    finding.summary,
                    json.dumps(finding.evidence_refs),
                    finding.confidence,
                    json.dumps(finding.metadata),
                    now,
                ),
            )
        self.conn.commit()

    def load_recent_findings(self, limit: int = 20) -> list[Finding]:
        rows = self.conn.execute(
            "SELECT finding_id, kind, summary, evidence_refs, confidence, metadata "
            "FROM findings ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            Finding(
                finding_id=row[0],
                kind=row[1],
                summary=row[2],
                evidence_refs=json.loads(row[3] or "[]"),
                confidence=row[4],
                metadata=json.loads(row[5] or "{}"),
            )
            for row in rows
        ]

    def upsert_hypothesis(self, entry: HypothesisEntry) -> None:
        self.conn.execute(
            "INSERT INTO hypotheses "
            "(hypothesis_id, statement, target_defects, status, confidence, "
            "support_refs, counter_refs, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(hypothesis_id) DO UPDATE SET "
            "statement=excluded.statement, target_defects=excluded.target_defects, "
            "status=excluded.status, confidence=excluded.confidence, "
            "support_refs=excluded.support_refs, counter_refs=excluded.counter_refs, "
            "updated_at=excluded.updated_at",
            (
                entry.hypothesis_id, entry.statement, entry.target_defects,
                entry.status, entry.confidence, entry.support_refs,
                entry.counter_refs, datetime.now().isoformat(),
            ),
        )
        self.conn.commit()

    def load_hypotheses(self) -> list[HypothesisEntry]:
        rows = self.conn.execute(
            "SELECT hypothesis_id, statement, target_defects, status, confidence, "
            "support_refs, counter_refs FROM hypotheses ORDER BY updated_at DESC"
        ).fetchall()
        return [HypothesisEntry(*r) for r in rows]

    # ------------------------------------------------------------------
    # Unified benchmark memory
    # ------------------------------------------------------------------

    def record_functional_evaluation(
        self,
        xc: XCFunctional,
        result: BenchmarkResult,
        *,
        source: str,
        validation_result: BenchmarkResult | None = None,
        status: str = "complete",
        failure_reason: str = "",
        defect_tags: list[str] | None = None,
        benchmark_context: dict[str, Any] | None = None,
    ) -> str:
        """Record one benchmarked functional, regardless of where it came from."""
        protocol_hash = self._benchmark_protocol_hash(result, validation_result, benchmark_context)
        sig = xc.structural_signature()
        eval_id = self._stable_id("eval", sig, source, protocol_hash)
        now = datetime.now().isoformat()
        subset_metrics = {
            "search": self._result_to_metrics(result),
            "validation": self._result_to_metrics(validation_result) if validation_result else {},
            "search_overall_mae": self._valid_overall_mae(result),
            "validation_overall_mae": self._valid_overall_mae(validation_result) if validation_result else None,
        }
        try:
            xc_code = xc.to_xc_code()
        except Exception:
            xc_code = ""
        component_families = [
            {
                "name": item.name,
                "macro_role": item.macro_role,
                "family": item.family,
            }
            for item in extract_component_families(xc)
        ]
        hf_exchange = {
            "alpha": xc.hf_exchange.alpha,
            "beta": xc.hf_exchange.beta,
            "omega": xc.hf_exchange.omega,
        }
        self.conn.execute(
            "INSERT INTO functional_evaluations "
            "(eval_id, structural_signature, functional_name, source, xc_json, xc_code, "
            "component_families_json, hf_exchange_json, subset_metrics_json, "
            "benchmark_protocol_hash, benchmark_context_json, status, failure_reason, "
            "defect_tags_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(structural_signature, source, benchmark_protocol_hash) DO UPDATE SET "
            "functional_name=excluded.functional_name, xc_json=excluded.xc_json, "
            "xc_code=excluded.xc_code, component_families_json=excluded.component_families_json, "
            "hf_exchange_json=excluded.hf_exchange_json, subset_metrics_json=excluded.subset_metrics_json, "
            "benchmark_context_json=excluded.benchmark_context_json, "
            "status=excluded.status, failure_reason=excluded.failure_reason, "
            "defect_tags_json=excluded.defect_tags_json, updated_at=excluded.updated_at",
            (
                eval_id,
                sig,
                xc.name,
                source,
                safe_json_dumps(xc.to_dict(), sort_keys=True),
                xc_code,
                json.dumps(component_families, sort_keys=True),
                json.dumps(hf_exchange, sort_keys=True),
                json.dumps(subset_metrics, sort_keys=True),
                protocol_hash,
                json.dumps(benchmark_context or {}, sort_keys=True),
                status,
                failure_reason,
                json.dumps(defect_tags or []),
                now,
                now,
            ),
        )
        self.conn.commit()
        return eval_id

    def latest_evaluation_id(
        self,
        structural_signature: str,
        *,
        source: str | None = None,
    ) -> str | None:
        if source:
            row = self.conn.execute(
                "SELECT eval_id FROM functional_evaluations "
                "WHERE structural_signature=? AND source=? ORDER BY updated_at DESC LIMIT 1",
                (structural_signature, source),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT eval_id FROM functional_evaluations "
                "WHERE structural_signature=? ORDER BY updated_at DESC LIMIT 1",
                (structural_signature,),
            ).fetchone()
        return row[0] if row else None

    def record_edit_effect(
        self,
        *,
        parent_eval_id: str,
        child_eval_id: str,
        action_type: str,
        action_detail: dict,
        active_skill: str = "",
        promotable: bool = False,
        status: str = "evaluated",
        evidence_refs: list[str] | None = None,
        target_subsets: list[str] | None = None,
    ) -> str | None:
        """Record parent->child benchmark deltas as reusable causal evidence."""
        parent = self._load_functional_eval(parent_eval_id)
        child = self._load_functional_eval(child_eval_id)
        if not parent or not child:
            return None
        subset_effects, risk, support_refs, counter_refs = self._compute_subset_effects(parent, child)
        if not subset_effects:
            logger.info(
                "Skipping edit effect with no valid parent measurements: parent=%s child=%s action=%s",
                parent_eval_id,
                child_eval_id,
                action_type,
            )
            return None
        support_refs.extend(evidence_refs or [])
        effect_id = self._stable_id("effect", parent_eval_id, child_eval_id, action_type, active_skill)
        child_families = [
            item.get("family", "")
            for item in json.loads(child["component_families_json"] or "[]")
            if item.get("family")
        ]
        self.conn.execute(
            "INSERT INTO edit_effects "
            "(effect_id, parent_eval_id, child_eval_id, parent_signature, child_signature, "
            "action_type, active_skill, action_detail_json, component_families_json, "
            "subset_effects_json, promotable, status, generalization_risk, "
            "support_refs_json, counter_refs_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(parent_eval_id, child_eval_id, action_type, active_skill) DO UPDATE SET "
            "action_detail_json=excluded.action_detail_json, component_families_json=excluded.component_families_json, "
            "subset_effects_json=excluded.subset_effects_json, promotable=excluded.promotable, "
            "status=excluded.status, generalization_risk=excluded.generalization_risk, "
            "support_refs_json=excluded.support_refs_json, counter_refs_json=excluded.counter_refs_json",
            (
                effect_id,
                parent_eval_id,
                child_eval_id,
                parent["structural_signature"],
                child["structural_signature"],
                action_type,
                active_skill,
                safe_json_dumps(action_detail, sort_keys=True),
                json.dumps(sorted(set(child_families))),
                json.dumps(subset_effects, sort_keys=True),
                int(promotable),
                status,
                risk,
                json.dumps(list(dict.fromkeys(support_refs))[:12]),
                json.dumps(list(dict.fromkeys(counter_refs))[:12]),
                datetime.now().isoformat(),
            ),
        )
        self.conn.commit()
        self._record_edit_subset_effects(effect_id, subset_effects, target_subsets or [])
        self._record_risk_patterns_for_effect(
            effect_id=effect_id,
            subset_effects=subset_effects,
            action_type=action_type,
            active_skill=active_skill,
            component_families=sorted(set(child_families)),
            target_subsets=target_subsets or [],
        )
        self.refresh_mechanism_cards([effect_id])
        outcome = self._summarize_action_outcome(
            pattern_key=self._action_pattern_key(
                parent=parent,
                child=child,
                action_type=action_type,
                action_detail=action_detail,
                component_families=sorted(set(child_families)),
            ),
            subset_effects=subset_effects,
            action_type=action_type,
            active_skill=active_skill,
            component_family=sorted(set(child_families))[0] if child_families else "",
            target_subsets=target_subsets or [],
        )
        logger.info(
            "Action outcome recorded: pattern=%s verdict=%s skill=%s target_delta=%+.3f "
            "risk=%.3f worst_regression=%.2f supports=%s regressions=%s",
            outcome.pattern_key,
            outcome.verdict,
            active_skill or "n/a",
            outcome.target_delta,
            outcome.risk_score,
            outcome.worst_regression_ratio,
            outcome.supports,
            outcome.regressions,
        )
        return effect_id

    def record_failure_pattern(
        self,
        *,
        symptom: str,
        action_type: str = "",
        component_family: str = "",
        root_cause: str = "",
        repair_hint: str = "",
        hard_block: bool = False,
    ) -> str:
        normalized = self._normalize_key("failure", action_type, component_family, symptom)
        now = datetime.now().isoformat()
        existing = self.conn.execute(
            "SELECT occurrences, confidence, root_cause, repair_hint FROM failure_patterns WHERE pattern_key=?",
            (normalized,),
        ).fetchone()
        if existing:
            occurrences = int(existing[0]) + 1
            confidence = min(0.95, float(existing[1]) + 0.05)
            self.conn.execute(
                "UPDATE failure_patterns SET occurrences=?, confidence=?, root_cause=?, "
                "repair_hint=?, hard_block=?, updated_at=? WHERE pattern_key=?",
                (
                    occurrences,
                    confidence,
                    root_cause or existing[2],
                    repair_hint or existing[3] or self._default_repair_hint(symptom),
                    int(hard_block),
                    now,
                    normalized,
                ),
            )
        else:
            self.conn.execute(
                "INSERT INTO failure_patterns "
                "(pattern_key, symptom, root_cause, repair_hint, action_type, component_family, "
                "hard_block, occurrences, confidence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    normalized,
                    symptom,
                    root_cause,
                    repair_hint or self._default_repair_hint(symptom),
                    action_type,
                    component_family,
                    int(hard_block),
                    1,
                    0.65 if hard_block else 0.5,
                    now,
                    now,
                ),
            )
        self.conn.commit()
        return normalized

    def record_risk_pattern(
        self,
        *,
        target_subset: str,
        risk_subset: str,
        symptom: str,
        action_type: str = "",
        active_skill: str = "",
        component_family: str = "",
        ratio: float = 1.0,
    ) -> str:
        """Record scientific/validation regression separately from hard failures."""
        key = self._normalize_key("risk", target_subset, risk_subset, action_type, active_skill, component_family, symptom)
        now = datetime.now().isoformat()
        existing = self.conn.execute(
            "SELECT occurrences, avg_ratio, confidence FROM risk_patterns WHERE pattern_key=?",
            (key,),
        ).fetchone()
        if existing:
            occurrences = int(existing[0]) + 1
            avg_ratio = ((float(existing[1]) * int(existing[0])) + ratio) / occurrences
            confidence = min(0.95, float(existing[2]) + 0.05)
            self.conn.execute(
                "UPDATE risk_patterns SET occurrences=?, avg_ratio=?, confidence=?, updated_at=? "
                "WHERE pattern_key=?",
                (occurrences, avg_ratio, confidence, now, key),
            )
        else:
            self.conn.execute(
                "INSERT INTO risk_patterns "
                "(pattern_key, target_subset, risk_subset, symptom, action_type, active_skill, "
                "component_family, occurrences, avg_ratio, confidence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    target_subset,
                    risk_subset,
                    symptom,
                    action_type,
                    active_skill,
                    component_family,
                    1,
                    ratio,
                    0.55,
                    now,
                    now,
                ),
            )
        self.conn.commit()
        return key

    def refresh_mechanism_cards(self, effect_ids: list[str] | None = None) -> int:
        """Aggregate edit effects into target-scoped mechanism cards."""
        del effect_ids  # current implementation refreshes all lightweight aggregates
        rows = self.conn.execute(
            "SELECT ee.effect_id, ee.action_type, ee.active_skill, ee.component_families_json, "
            "ese.subset, ese.panel, ese.improvement_delta, ese.ratio, ese.is_target, "
            "ese.before_mae, ese.after_mae, ee.support_refs_json, ee.counter_refs_json "
            "FROM edit_effects ee JOIN edit_subset_effects ese ON ee.effect_id = ese.effect_id "
            "ORDER BY ee.created_at DESC"
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row | tuple]] = {}
        grouped_targets: dict[str, str] = {}
        rows_by_effect: dict[str, list[sqlite3.Row | tuple]] = {}
        for row in rows:
            rows_by_effect.setdefault(str(row[0]), []).append(row)
        for effect_rows in rows_by_effect.values():
            targets = [str(row[4]) for row in effect_rows if int(row[8] or 0)]
            if not targets:
                targets = [str(row[4]) for row in effect_rows if str(row[5] or "") == "search"]
            if not targets and effect_rows:
                targets = [str(effect_rows[0][4])]
            families = json.loads(effect_rows[0][3] or "[]")
            family = families[0] if families else "unknown"
            for target in dict.fromkeys(targets):
                key = self._normalize_key("mechanism", target, effect_rows[0][1], effect_rows[0][2] or "unspecified", family)
                grouped.setdefault(key, []).extend(effect_rows)
                grouped_targets[key] = target

        count = 0
        now = datetime.now().isoformat()
        for key, items in grouped.items():
            subset_totals: dict[str, list[float]] = {}
            risk_subsets: list[str] = []
            support_refs: list[str] = []
            counter_refs: list[str] = []
            families: set[str] = set()
            action_type = str(items[0][1])
            skill = str(items[0][2] or "")
            target_subset = grouped_targets.get(key, str(items[0][4]))
            for item in items:
                families.update(json.loads(item[3] or "[]"))
                subset = str(item[4])
                panel = str(item[5] or "")
                delta = float(item[6] or 0.0)
                ratio = float(item[7] or 1.0)
                before_mae = float(item[9] or 0.0)
                after_mae = float(item[10] or 0.0)
                if after_mae <= 0.0 and before_mae > 0.0:
                    delta = before_mae - 999.0
                    ratio = 999.0 / max(before_mae, 1e-6)
                subset_totals.setdefault(subset, []).append(delta)
                if subset != target_subset and (delta < -0.25 or ratio > 1.05):
                    risk_subsets.append(subset)
                support_refs.extend(json.loads(item[11] or "[]"))
                counter_refs.extend(json.loads(item[12] or "[]"))
            expected = {
                subset: round(sum(values) / len(values), 4)
                for subset, values in sorted(subset_totals.items())
                if values
            }
            target_delta = float(expected.get(target_subset, 0.0))
            validation_penalty = round(sum(abs(v) for subset, v in expected.items() if subset != target_subset and v < 0), 4)
            risk_score = max(0.0, min(1.0, validation_penalty / 5.0))
            target_defects = self._defects_from_target(target_subset, target_delta, risk_score)
            statement = (
                f"{action_type}/{skill or 'unspecified'} on {', '.join(sorted(families)) or 'unknown family'} "
                f"targets {target_subset}: avg_delta={target_delta:+.3f} kcal/mol "
                f"over {len(items)} observation(s); validation_penalty={validation_penalty:.3f}."
            )
            guardrails = [
                f"Monitor {subset} regression"
                for subset in sorted(set(risk_subsets))
            ][:6]
            confidence = min(0.9, 0.42 + 0.12 * len(items) - 0.10 * risk_score)
            status = "supported" if len(items) >= 3 and target_delta > 0 and risk_score < 0.5 else "provisional"
            self.conn.execute(
                "INSERT INTO mechanism_cards "
                "(mechanism_key, statement, target_subset, target_delta, validation_penalty, "
                "risk_score, support_count, target_defects_json, component_families_json, "
                "expected_effects_json, risk_subsets_json, guardrails_json, evidence_refs_json, "
                "confidence, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(mechanism_key) DO UPDATE SET "
                "statement=excluded.statement, target_subset=excluded.target_subset, "
                "target_delta=excluded.target_delta, validation_penalty=excluded.validation_penalty, "
                "risk_score=excluded.risk_score, support_count=excluded.support_count, "
                "target_defects_json=excluded.target_defects_json, "
                "component_families_json=excluded.component_families_json, "
                "expected_effects_json=excluded.expected_effects_json, risk_subsets_json=excluded.risk_subsets_json, "
                "guardrails_json=excluded.guardrails_json, evidence_refs_json=excluded.evidence_refs_json, "
                "confidence=excluded.confidence, status=excluded.status, updated_at=excluded.updated_at",
                (
                    key,
                    statement,
                    target_subset,
                    target_delta,
                    validation_penalty,
                    risk_score,
                    len(items),
                    json.dumps(target_defects),
                    json.dumps(sorted(families)),
                    json.dumps(expected, sort_keys=True),
                    json.dumps(sorted(set(risk_subsets))),
                    json.dumps(guardrails),
                    json.dumps(list(dict.fromkeys(support_refs + counter_refs))[:16]),
                    confidence,
                    status,
                    now,
                    now,
                ),
            )
            count += 1
        self.conn.commit()
        return count

    # ------------------------------------------------------------------
    # Retrieval for IdeaAgent
    # ------------------------------------------------------------------

    def retrieve_for_proposal(
        self,
        current_xc: XCFunctional,
        current_result: BenchmarkResult,
    ) -> KnowledgeBundle:
        """Assemble recent evidence context for IdeaAgent proposal."""
        return self.retrieve_for_node(current_xc, current_result)

    def retrieve_for_node(
        self,
        current_xc: XCFunctional,
        current_result: BenchmarkResult,
        *,
        defect_tags: list[str] | None = None,
        target_subsets: list[str] | None = None,
        active_skill: str = "",
        retrieval_mode: str = "proposal",
        limit: int = 5,
    ) -> KnowledgeBundle:
        """Retrieve compact, node-relevant memory for expansion and prompting."""
        bundle = KnowledgeBundle()
        targets = target_subsets or self._target_subsets_from_defects(defect_tags or []) or list(current_result.subsets)
        bundle.ablation_evidence = self._query_relevant_ablation(current_xc, defect_tags or [], limit=12)
        bundle.relevant_traces = self._query_relevant_traces(active_skill, defect_tags or [], limit=5)
        bundle.mechanism_cards = self._query_mechanism_cards(
            current_xc=current_xc,
            defect_tags=defect_tags or [],
            target_subsets=targets,
            active_skill=active_skill,
            limit=limit,
        )
        bundle.failure_patterns = self._query_failure_patterns(
            current_xc=current_xc,
            defect_tags=defect_tags or [],
            active_skill=active_skill,
            hard_only=(retrieval_mode == "repair"),
            limit=limit,
        )
        bundle.risk_patterns = self._query_risk_patterns(
            target_subsets=targets,
            active_skill=active_skill,
            limit=limit,
        )
        bundle.action_outcomes = self._query_action_outcomes(
            target_subsets=targets,
            active_skill=active_skill,
            limit=limit,
        )
        bundle.libxc_anchors = self._query_libxc_anchors(current_xc, current_result, defect_tags or [], targets, limit=limit)
        bundle.component_benchmark_priors = self._query_component_benchmark_priors(
            current_xc,
            current_result,
            targets,
            limit=max(8, limit * 2),
        )
        bundle.candidate_priors = self._compute_candidate_priors(
            active_skill=active_skill,
            target_subsets=targets,
            limit=limit,
        )
        bundle.active_hypotheses = [
            item for item in self.load_hypotheses()
            if item.status in {"active", "supported", "weakened"}
        ][:limit]
        if bundle.action_outcomes:
            risky = sum(1 for item in bundle.action_outcomes if item.verdict == "risky")
            promising = sum(1 for item in bundle.action_outcomes if item.verdict == "promising")
            logger.info(
                "Retrieved action outcome patterns for skill=%s targets=%s: risky=%d promising=%d top=%s",
                active_skill or "any",
                ",".join(targets) or "any",
                risky,
                promising,
                ", ".join(item.pattern_key for item in bundle.action_outcomes[:3]),
            )
        if bundle.component_benchmark_priors:
            logger.info(
                "Retrieved component benchmark priors for targets=%s: %s",
                ",".join(targets) or "any",
                "; ".join(
                    f"{item.component_name}:best={item.best_functional}:"
                    f"target_gain={item.target_gain:+.2f}:score={item.association_score:+.2f}:"
                    f"target_failures={item.target_failures}:failures={item.failures}"
                    for item in bundle.component_benchmark_priors[:8]
                ),
            )
        return bundle

    def retrieve_parent_pool_graph(
        self,
        parents: list[tuple[str, XCFunctional]],
        *,
        node_budget: int = 24,
        edge_budget: int = 32,
    ) -> dict[str, Any] | None:
        """Build one bounded, deduplicated graph around a Basin parent pool."""
        if self.functional_graph is None or not parents:
            return None

        unique_parents: list[tuple[str, XCFunctional]] = []
        seen_parent_ids: set[str] = set()
        for parent_id, xc in parents:
            parent_id = str(parent_id)
            if not parent_id or parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)
            unique_parents.append((parent_id, xc))

        try:
            subgraph = self._build_parent_pool_evolution_subgraph(
                unique_parents,
                node_budget=node_budget,
                edge_budget=edge_budget,
            )
        except Exception as exc:
            logger.warning("Functional graph parent-pool query failed: %s", exc)
            return None

        if subgraph is not None:
            logger.info(
                "Retrieved parent-pool functional graph: parents=%d anchors=%d nodes=%d edges=%d",
                len(unique_parents),
                len(subgraph.get("parent_anchors", [])),
                len(subgraph.get("nodes", [])),
                len(subgraph.get("edges", [])),
            )
        return subgraph

    def _build_parent_pool_evolution_subgraph(
        self,
        parents: list[tuple[str, XCFunctional]],
        *,
        node_budget: int,
        edge_budget: int,
    ) -> dict[str, Any] | None:
        graph = self.functional_graph
        if graph is None:
            return None

        parent_anchors: list[dict[str, Any]] = []
        anchor_ids: list[str] = []
        exact_anchor_ids: set[str] = set()
        for parent_id, xc in parents:
            structural_nodes = graph.nodes_by_structural_signature.get(
                xc.structural_signature(), []
            )
            anchor = next(
                (
                    node
                    for node in structural_nodes
                    if node.name == xc.name
                    and node.conversion.get("method") == "basin_agent_xc_dict"
                    and any(
                        source_file.endswith(f"#{parent_id}")
                        for source_file in node.source_files
                    )
                ),
                None,
            )
            if anchor is None:
                anchor = next(
                    (
                        node
                        for node in structural_nodes
                        if node.state.get("operability") == "full"
                    ),
                    None,
                )
            exact_match = anchor is not None
            if exact_match:
                match_score = 1.0
            else:
                nearest = next(
                    (
                        item
                        for item in graph.query_neighbors(
                            xc,
                            top_k=len(graph.nodes),
                            operability="full",
                        )
                        if not self._is_basin_graph_node(
                            graph.node_by_id[str(item["node_id"])]
                        )
                        and self._positive_number(item.get("match_score"))
                    ),
                    None,
                )
                if nearest is None:
                    continue
                anchor = graph.node_by_id[str(nearest["node_id"])]
                match_score = float(nearest["match_score"])

            parent_anchors.append(
                {
                    "parent_id": parent_id,
                    "parent_name": xc.name,
                    "anchor_node_id": anchor.id,
                    "match_score": round(match_score, 6),
                }
            )
            if anchor.id not in anchor_ids:
                anchor_ids.append(anchor.id)
            if exact_match:
                exact_anchor_ids.add(anchor.id)

        if not anchor_ids:
            return None

        edge_limit = max(0, int(edge_budget))
        node_limit = max(
            len(anchor_ids),
            min(
                max(1, int(node_budget)),
                max(24, len(anchor_ids) + edge_limit),
            ),
        )
        selected_nodes = set(anchor_ids)
        selected_edges: dict[str, tuple[FunctionalEdge, int]] = {}
        exact_reachable_nodes = set(exact_anchor_ids)

        edge_queues: dict[str, Iterator[tuple[FunctionalEdge, int]]] = {}
        for anchor_id in anchor_ids:
            candidates = self._incident_graph_edges(
                [anchor_id],
                path_depth=1,
                include_basin_nodes=anchor_id in exact_anchor_ids,
            )
            edge_queues[anchor_id] = iter(
                sorted(
                    candidates.values(),
                    key=lambda item: self._parent_pool_graph_edge_rank(
                        item[0],
                        path_depth=item[1],
                    ),
                )
            )

        def add_next_edge(anchor_id: str) -> bool:
            for edge, depth in edge_queues[anchor_id]:
                if edge.id in selected_edges:
                    continue
                if not self._graph_edge_is_target_actionable(edge):
                    continue
                new_nodes = {edge.source, edge.target}.difference(selected_nodes)
                if len(selected_nodes) + len(new_nodes) > node_limit:
                    continue
                selected_edges[edge.id] = (edge, depth)
                selected_nodes.update(new_nodes)
                if anchor_id in exact_anchor_ids:
                    exact_reachable_nodes.update((edge.source, edge.target))
                return True
            return False

        for _round in range(2):
            for anchor_id in anchor_ids:
                if len(selected_edges) >= edge_limit:
                    break
                add_next_edge(anchor_id)

        first_hop_nodes = sorted(selected_nodes.difference(anchor_ids))
        if first_hop_nodes and len(selected_edges) < edge_limit:
            exact_first_hop = sorted(set(first_hop_nodes) & exact_reachable_nodes)
            nonexact_first_hop = sorted(set(first_hop_nodes) - exact_reachable_nodes)
            second_candidates = self._incident_graph_edges(
                nonexact_first_hop,
                path_depth=2,
                include_basin_nodes=False,
            )
            second_candidates.update(
                self._incident_graph_edges(
                    exact_first_hop,
                    path_depth=2,
                    include_basin_nodes=True,
                )
            )
            self._select_parent_pool_graph_edges(
                second_candidates,
                selected_nodes=selected_nodes,
                selected_edges=selected_edges,
                max_add=edge_limit - len(selected_edges),
                node_budget=node_limit,
            )

        ordered_node_ids = list(anchor_ids)
        for edge, _depth in selected_edges.values():
            for node_id in (edge.source, edge.target):
                if node_id not in ordered_node_ids:
                    ordered_node_ids.append(node_id)

        return {
            "parent_anchors": parent_anchors,
            "nodes": [
                self._compact_parent_pool_graph_node(graph.node_by_id[node_id])
                for node_id in ordered_node_ids
            ],
            "edges": [
                self._compact_parent_pool_graph_edge(
                    edge,
                    path_depth=path_depth,
                )
                for edge, path_depth in selected_edges.values()
            ],
            "limits": {
                "max_depth": 2,
                "max_nodes": node_limit,
                "max_edges": edge_limit,
            },
        }

    def _incident_graph_edges(
        self,
        node_ids: list[str],
        *,
        path_depth: int,
        include_basin_nodes: bool,
    ) -> dict[str, tuple[FunctionalEdge, int]]:
        graph = self.functional_graph
        if graph is None:
            return {}
        candidates: dict[str, tuple[FunctionalEdge, int]] = {}
        for node_id in node_ids:
            for edge in graph.outgoing_edges.get(node_id, []):
                if not include_basin_nodes and self._edge_has_basin_node(edge):
                    continue
                if not self._graph_edge_is_target_actionable(edge):
                    continue
                candidates.setdefault(edge.id, (edge, path_depth))
        return candidates

    @staticmethod
    def _is_basin_graph_node(node: FunctionalNode) -> bool:
        return node.conversion.get("method") == "basin_agent_xc_dict"

    def _edge_has_basin_node(self, edge: FunctionalEdge) -> bool:
        graph = self.functional_graph
        if graph is None:
            return False
        return any(
            self._is_basin_graph_node(graph.node_by_id[node_id])
            for node_id in (edge.source, edge.target)
        )

    def _select_parent_pool_graph_edges(
        self,
        candidates: dict[str, tuple[FunctionalEdge, int]],
        *,
        selected_nodes: set[str],
        selected_edges: dict[str, tuple[FunctionalEdge, int]],
        max_add: int,
        node_budget: int,
    ) -> None:
        ranked = sorted(
            candidates.values(),
            key=lambda item: self._parent_pool_graph_edge_rank(
                item[0],
                path_depth=item[1],
            ),
        )
        ordered: list[tuple[FunctionalEdge, int]] = []
        for relevance in ("direct", "coupled", "context"):
            match = next(
                (
                    item for item in ranked
                    if self._parent_pool_graph_edge_relevance(item[0]) == relevance
                ),
                None,
            )
            if match is not None:
                ordered.append(match)
        risky = next((item for item in ranked if self._graph_edge_is_risky(item[0])), None)
        if risky is not None:
            ordered.append(risky)
        ordered.extend(ranked)

        added = 0
        seen: set[str] = set()
        for edge, depth in ordered:
            if added >= max_add or edge.id in seen or edge.id in selected_edges:
                continue
            seen.add(edge.id)
            if not self._graph_edge_is_target_actionable(edge):
                continue
            new_nodes = {edge.source, edge.target}.difference(selected_nodes)
            if len(selected_nodes) + len(new_nodes) > node_budget:
                continue
            selected_edges[edge.id] = (edge, depth)
            selected_nodes.update(new_nodes)
            added += 1

    def _parent_pool_graph_edge_rank(
        self,
        edge: FunctionalEdge,
        *,
        path_depth: int,
    ) -> tuple[Any, ...]:
        relevance_rank = {"direct": 0, "coupled": 1, "context": 2}
        relevance = self._parent_pool_graph_edge_relevance(edge)
        target_subset = TARGET_SUBSETS[0]
        target_delta = self._metric_delta(edge, target_subset)
        target_mae = self._graph_node_subset_mae(edge.target, target_subset)
        guard_regressions = [
            delta
            for subset in GUARD_SUBSETS
            if (delta := self._metric_delta(edge, subset)) is not None and delta > 0
        ]
        completeness = sum(
            self._metric_delta(edge, subset) is not None
            for subset in EXPECTED_SUBSETS
        ) / len(EXPECTED_SUBSETS)
        return (
            relevance_rank[relevance],
            path_depth,
            target_delta is None,
            target_delta if target_delta is not None else float("inf"),
            target_mae is None,
            target_mae if target_mae is not None else float("inf"),
            len(guard_regressions),
            sum(guard_regressions),
            -completeness,
            edge.id,
        )

    @staticmethod
    def _parent_pool_graph_edge_relevance(edge: FunctionalEdge) -> str:
        action_type = str(edge.operation.get("action_type", ""))
        if action_type == "composite":
            steps = edge.operation.get("steps", [])
            step_count = len(steps) if isinstance(steps, list) else 0
            return "coupled" if step_count > 1 else "direct"
        if action_type in {"add", "remove", "replace", "rewire"}:
            return "coupled" if edge.operation.get("composite") else "direct"
        return "context"

    def _compact_parent_pool_graph_node(self, node: FunctionalNode) -> dict[str, Any]:
        xc_dict = node.state.get("xc_dict", {})
        if not isinstance(xc_dict, dict):
            xc_dict = {}
        components = []
        for component in xc_dict.get("components", []) or []:
            if not isinstance(component, dict):
                continue
            compact = {
                key: json_safe_copy(component[key])
                for key in ("name", "role", "xctype", "mixing_factor")
                if key in component
            }
            components.append(compact)
        metrics = {
            subset: {
                key: payload.get(key)
                for key in ("mae_kcal", "n_failures")
            }
            for subset in EXPECTED_SUBSETS
            if isinstance((payload := node.metrics.get(subset)), dict)
        }
        return {
            "node_id": node.id,
            "name": node.name,
            "rung": node.semantic.get("rung"),
            "role": node.semantic.get("role"),
            "components": components,
            "hybrid": json_safe_copy(xc_dict.get("hybrid", {})),
            "metrics": metrics,
            "data_status": {
                "operability": node.state.get("operability"),
                "conversion_status": node.conversion.get("status"),
                "conversion_method": node.conversion.get("method"),
                "conversion_warning_count": len(node.conversion.get("warnings", [])),
                "metric_completeness": round(len(metrics) / len(EXPECTED_SUBSETS), 6),
            },
        }

    def _compact_parent_pool_graph_edge(
        self,
        edge: FunctionalEdge,
        *,
        path_depth: int,
    ) -> dict[str, Any]:
        graph = self.functional_graph
        if graph is None:
            raise RuntimeError("functional graph is not loaded")
        source = graph.node_by_id[edge.source]
        rewire_changes = edge.operation.get("rewire_changes", {})
        if not isinstance(rewire_changes, dict):
            rewire_changes = {}
        parameter_changes = [
            {
                "parameter": str(parameter),
                "source_value": self._graph_node_parameter_value(source, str(parameter)),
                "target_value": json_safe_copy(target_value),
            }
            for parameter, target_value in sorted(rewire_changes.items())
        ]
        metric_delta = {
            subset: {
                key: payload.get(key)
                for key in (
                    "delta_mae",
                    "failure_delta",
                )
            }
            for subset in EXPECTED_SUBSETS
            if isinstance((payload := edge.metric_delta.get(subset)), dict)
        }
        llm_analysis = edge.semantic.get("llm_analysis", {})
        return {
            "edge_id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "path_depth": path_depth,
            "relevance": self._parent_pool_graph_edge_relevance(edge),
            "risk": self._graph_edge_is_risky(edge),
            "operation": {
                "action_type": edge.operation.get("action_type"),
                "composite": bool(edge.operation.get("composite")),
                "structural_change": json_safe_copy(edge.operation.get("structural_change", {})),
                "parameter_changes": parameter_changes,
                "primitive_action_count": edge.operation.get("primitive_action_count"),
                "steps": json_safe_copy(edge.operation.get("steps", [])),
            },
            "metric_delta": metric_delta,
            "data_status": {
                "metric_completeness": round(len(metric_delta) / len(EXPECTED_SUBSETS), 6),
                "llm_analysis_status": (
                    llm_analysis.get("status") if isinstance(llm_analysis, dict) else "unknown"
                ),
            },
        }

    def _graph_edge_is_risky(self, edge: FunctionalEdge) -> bool:
        if any(
            (delta := self._metric_delta(edge, subset)) is not None and delta > 0
            for subset in GUARD_SUBSETS
        ):
            return True
        return any(
            self._positive_number(payload.get("failure_delta"))
            for payload in edge.metric_delta.values()
            if isinstance(payload, dict)
        )

    def _graph_edge_is_target_actionable(self, edge: FunctionalEdge) -> bool:
        """Keep only evaluated, target-clean, forward-improving graph moves."""

        target_subset = TARGET_SUBSETS[0]
        if not self._graph_node_has_no_subset_failures(edge.source, target_subset):
            return False
        if not self._graph_node_has_no_subset_failures(edge.target, target_subset):
            return False
        delta = self._metric_delta(edge, target_subset)
        if delta is None:
            return False
        return delta < 0

    def _graph_node_has_no_subset_failures(self, node_id: str, subset: str) -> bool:
        graph = self.functional_graph
        if graph is None or node_id not in graph.node_by_id:
            return False
        payload = graph.node_by_id[node_id].metrics.get(subset)
        if not isinstance(payload, dict):
            return False
        failures = payload.get("n_failures")
        try:
            return int(failures or 0) == 0
        except (TypeError, ValueError):
            return False

    def _graph_node_subset_mae(self, node_id: str, subset: str) -> float | None:
        graph = self.functional_graph
        if graph is None or node_id not in graph.node_by_id:
            return None
        payload = graph.node_by_id[node_id].metrics.get(subset)
        if not isinstance(payload, dict):
            return None
        try:
            value = float(payload.get("mae_kcal"))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _metric_delta(edge: FunctionalEdge, subset: str) -> float | None:
        payload = edge.metric_delta.get(subset, {})
        try:
            value = payload.get("delta_mae") if isinstance(payload, dict) else None
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _graph_node_parameter_value(node: FunctionalNode, parameter: str) -> Any:
        xc_dict = node.state.get("xc_dict", {})
        if not isinstance(xc_dict, dict):
            return None
        if parameter.startswith("hybrid.hf_exchange."):
            hybrid = xc_dict.get("hybrid", {})
            hf_exchange = hybrid.get("hf_exchange", {}) if isinstance(hybrid, dict) else {}
            return hf_exchange.get(parameter.rsplit(".", 1)[-1])
        components = xc_dict.get("components", [])
        if parameter.startswith("mixing:"):
            component_name = parameter.removeprefix("mixing:")
            for component in components:
                if isinstance(component, dict) and component.get("name") == component_name:
                    return component.get("mixing_factor")
            return None
        if "." not in parameter:
            return None
        component_name, ext_param = parameter.split(".", 1)
        for component in components:
            if not isinstance(component, dict) or component.get("name") != component_name:
                continue
            ext_params = component.get("ext_params", {})
            return ext_params.get(ext_param) if isinstance(ext_params, dict) else None
        return None

    @staticmethod
    def _positive_number(value: Any) -> bool:
        try:
            return value is not None and float(value) > 0.0
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Recording observations
    # ------------------------------------------------------------------

    def record_observation(
        self,
        xc: XCFunctional,
        result: BenchmarkResult,
        action_type: str,
        action_detail: dict,
        status: str,
        finding_tags: list[str] | None = None,
        ablation_report: AblationReport | None = None,
        core_contribution: str = "",
        rationale: str = "",
        component_explanations: dict | None = None,
        edit_plan: dict | None = None,
        skill_metrics: dict | None = None,
        active_skill: str = "",
        root_domains: list[str] | None = None,
        validation_mae: float = 0.0,
        validation_result: BenchmarkResult | None = None,
        parent_signature: str | None = None,
        promotable: bool = False,
        failure_reason: str = "",
        benchmark_context: dict[str, Any] | None = None,
        target_subsets: list[str] | None = None,
    ) -> None:
        """Record a search observation and extract knowledge from it.

        Args:
            xc: The XCFunctional that was evaluated.
            result: The BenchmarkResult from evaluation.
            action_type: What action was taken (e.g., 'rewire', 'add', 'replace').
            action_detail: Details of the action (e.g., {'alpha': 0.30}).
            status: 'accepted', 'rejected', or 'failed'.
            finding_tags: Labels attached to the structured analysis.
            ablation_report: Pre-computed AblationReport from run_component_ablation().
            core_contribution: One-sentence physical statement from ProposalState.
            rationale: Detailed rationale from ProposalState.
            component_explanations: Dict mapping component names to role descriptions.
            edit_plan: Structured edit plan from ProposalState.
            skill_metrics: Skill metrics from ProposalState.
            active_skill: Name of the skill used.
            root_domains: Root domain tags from ProposalState.
        """
        action_detail = json_safe_copy(action_detail)
        component_explanations = json_safe_copy(component_explanations or {})
        edit_plan = json_safe_copy(edit_plan or {})
        skill_metrics = json_safe_copy(skill_metrics or {})
        root_domains = json_safe_copy(root_domains or [])
        # 1. Write search trace with rich semantic content
        finding_str = ",".join(finding_tags or [])
        trace = TraceEntry(
            name=xc.name,
            parent_name=xc.metadata.get("parent", ""),
            action_type=action_type,
            action_detail=safe_json_dumps(action_detail),
            signature=xc.structural_signature(),
            search_mae=result.overall_mae,
            validation_mae=validation_mae,
            status=status,
            finding_tags=finding_str,
            core_contribution=core_contribution,
            rationale=rationale,
            component_explanations=safe_json_dumps(component_explanations),
            edit_plan=safe_json_dumps(edit_plan),
            skill_metrics=safe_json_dumps(skill_metrics),
            active_skill=active_skill,
            root_domains=safe_json_dumps(root_domains),
        )
        self.add_trace(trace)

        # 1b. Write the unified evaluation/effect memory used by MCTS priors.
        source = "mcts_candidate" if action_type != "pre-cached" else "manual_baseline"
        eval_id = self.record_functional_evaluation(
            xc,
            result,
            source=source,
            validation_result=validation_result,
            status=status,
            failure_reason=failure_reason,
            defect_tags=finding_tags or [],
            benchmark_context=benchmark_context,
        )
        if parent_signature:
            parent_eval_id = self.latest_evaluation_id(parent_signature, source=source)
            if parent_eval_id is None:
                parent_eval_id = self.latest_evaluation_id(parent_signature)
            if parent_eval_id and parent_eval_id != eval_id:
                self.record_edit_effect(
                    parent_eval_id=parent_eval_id,
                    child_eval_id=eval_id,
                    action_type=action_type,
                    action_detail=action_detail,
                    active_skill=active_skill,
                    promotable=promotable,
                    status=status,
                    evidence_refs=[],
                    target_subsets=target_subsets or list(result.subsets),
                )
        if failure_reason:
            if failure_reason == "validation_regression":
                for subset in (target_subsets or list(result.subsets)):
                    self.record_risk_pattern(
                        target_subset=subset,
                        risk_subset="validation",
                        symptom=failure_reason,
                        action_type=action_type,
                        active_skill=active_skill,
                    )
            else:
                self.record_failure_pattern(
                    symptom=failure_reason,
                    action_type=action_type,
                    hard_block=failure_reason in {"hard_constraint_violation", "evaluation_failure"},
                )

        # 2. Ingest ablation evidence
        # If a real AblationReport is provided, use it (Xcientist pattern).
        # Otherwise, create a lightweight action-level record.
        if ablation_report is not None:
            self.ingest_ablation_report(ablation_report)
        elif status in ("accepted", "rejected") and action_detail:
            self._record_action_evidence(xc, result, action_type, action_detail, status)

        # 3. Extract empirical rules from patterns
        self._extract_empirical_rules(action_type, status, result)

        # 4. Extract parameter curve points from REWIRE actions
        if action_type == "rewire" and status in ("accepted", "rejected"):
            for key, value in action_detail.items():
                if isinstance(value, (int, float)):
                    for subset_name, sr in result.subsets.items():
                        if sr.n_reactions > 0:
                            self.add_knowledge(KnowledgeEntry(
                                category="domain",
                                key=f"curve.live.{key}.{value:.4f}.{subset_name}",
                                content=f"{key}={value:.4f}: {subset_name} MAE={sr.mae_kcal:.3f} "
                                        f"kcal/mol (status={status})",
                                confidence=0.7 if status == "accepted" else 0.4,
                                source="live_search",
                            ))

    def ingest_ablation_report(self, report: AblationReport) -> list[dict]:
        """Ingest an AblationReport into the knowledge engine.

        Mirrors Xcientist's ingest_ablation_results():
        1. Convert AblationResult entries to normalized dicts
        2. Store each entry in ablation_evidence table
        3. Return normalized entries

        This is the primary path for real component-level causal evidence.
        """
        return ingest_ablation_results(report, knowledge_engine=self)

    def _record_action_evidence(
        self,
        xc: XCFunctional,
        result: BenchmarkResult,
        action_type: str,
        action_detail: dict,
        status: str,
    ) -> None:
        """Record lightweight action-level evidence when no AblationReport is available.

        Uses the atomic ActionType model: rewire, add, add_descriptor, remove, replace, enforce.
        """
        result_str = "positive" if status == "accepted" else "negative"

        # REWIRE: parameter tuning (alpha, omega, mixing, ext_params)
        if action_type == "rewire":
            for key, value in action_detail.items():
                # Special keys: alpha → HF_EXCHANGE, omega → RSH
                if key == "alpha":
                    comp = "HF_EXCHANGE"
                    family = "hf_exchange.global"
                    param = "alpha"
                elif key == "omega":
                    comp = "HF_EXCHANGE"
                    family = "hf_exchange.range_separated"
                    param = "omega"
                elif key.startswith("mixing:"):
                    comp = key.split(":", 1)[1]
                    family = classify_component(comp).family
                    param = "mixing_factor"
                else:
                    comp, param = key.split(".", 1) if "." in key else (key, key)
                    family = classify_component(comp).family

                for subset_name, sr in result.subsets.items():
                    if sr.n_reactions > 0:
                        self.add_ablation(AblationEntry(
                            component=comp, component_family=family,
                            functional_name=xc.name, result=result_str,
                            metric=f"{subset_name}_mae", value=sr.mae_kcal,
                            analysis=f"REWIRE {param}={value} → {status}",
                            confidence=0.4,
                        ))

        # ADD_COMPONENT / ADD_DESCRIPTOR: new component-like terms.
        elif action_type in {"add", "add_descriptor"}:
            if action_type == "add_descriptor":
                descriptor = action_detail.get("descriptor", action_detail)
                name = descriptor.get("name") if isinstance(descriptor, dict) else ""
                template = descriptor.get("template") if isinstance(descriptor, dict) else ""
                comp_name = name or f"DESC_{template or 'descriptor'}"
                for subset_name, sr in result.subsets.items():
                    if sr.n_reactions > 0:
                        self.add_ablation(AblationEntry(
                            component=comp_name,
                            component_family="descriptor.gga",
                            functional_name=xc.name,
                            result=result_str,
                            metric=f"{subset_name}_mae",
                            value=sr.mae_kcal,
                            analysis=f"ADD_DESCRIPTOR {template or comp_name} -> {status}",
                            confidence=0.3,
                        ))
                return
            # Check if it's a dispersion addition
            if "type" in action_detail:
                disp_type = action_detail.get("type", "D3BJ")
                for subset_name, sr in result.subsets.items():
                    if sr.n_reactions > 0:
                        self.add_ablation(AblationEntry(
                            component=f"DISP_{disp_type}",
                            component_family="correlation.dispersion",
                            functional_name=xc.name, result=result_str,
                            metric=f"{subset_name}_mae", value=sr.mae_kcal,
                            analysis=f"ADD dispersion {disp_type} → {status}",
                            confidence=0.3,
                        ))
            else:
                comp_name = action_detail.get("name", "")
                if comp_name:
                    cls = classify_component(comp_name)
                    for subset_name, sr in result.subsets.items():
                        if sr.n_reactions > 0:
                            self.add_ablation(AblationEntry(
                                component=comp_name, component_family=cls.family,
                                functional_name=xc.name, result=result_str,
                                metric=f"{subset_name}_mae", value=sr.mae_kcal,
                                analysis=f"ADD {comp_name} → {status}",
                                confidence=0.3,
                            ))

        # REMOVE_COMPONENT
        elif action_type == "remove":
            comp_name = action_detail.get("name", "")
            if comp_name:
                cls = classify_component(comp_name)
                for subset_name, sr in result.subsets.items():
                    if sr.n_reactions > 0:
                        self.add_ablation(AblationEntry(
                            component=comp_name, component_family=cls.family,
                            functional_name=xc.name, result=result_str,
                            metric=f"{subset_name}_mae", value=sr.mae_kcal,
                            analysis=f"REMOVE {comp_name} → {status}",
                            confidence=0.3,
                        ))

        # REPLACE_COMPONENT
        elif action_type == "replace":
            old_name = action_detail.get("old_name", "")
            new_name = action_detail.get("new_name", "")
            if new_name:
                cls = classify_component(new_name)
                for subset_name, sr in result.subsets.items():
                    if sr.n_reactions > 0:
                        self.add_ablation(AblationEntry(
                            component=new_name, component_family=cls.family,
                            functional_name=xc.name, result=result_str,
                            metric=f"{subset_name}_mae", value=sr.mae_kcal,
                            analysis=f"REPLACE {old_name}→{new_name} → {status}",
                            confidence=0.3,
                        ))

        # ENFORCE_CONSTRAINT
        elif action_type == "enforce":
            constraint = action_detail.get("constraint", "")
            for subset_name, sr in result.subsets.items():
                if sr.n_reactions > 0:
                    self.add_ablation(AblationEntry(
                        component=f"CONSTRAINT_{constraint}",
                        component_family="constraint",
                        functional_name=xc.name, result=result_str,
                        metric=f"{subset_name}_mae", value=sr.mae_kcal,
                        analysis=f"ENFORCE {constraint} → {status}",
                        confidence=0.3,
                    ))

    def _extract_empirical_rules(
        self,
        action_type: str,
        status: str,
        result: BenchmarkResult,
    ) -> None:
        """Extract simple empirical rules from repeated patterns."""
        # Check if this action type has been consistently positive or negative
        traces = self._query_traces_by_action(action_type, limit=10)
        if len(traces) >= 3:
            accepted = sum(1 for t in traces if t.status == "accepted")
            ratio = accepted / len(traces)
            if ratio >= 0.7:
                self.add_knowledge(KnowledgeEntry(
                    category="empirical",
                    key=f"action.{action_type}.effective",
                    content=f"Action {action_type} has {ratio:.0%} acceptance rate ({accepted}/{len(traces)})",
                    confidence=min(0.9, 0.5 + 0.1 * accepted),
                    source="derived",
                ))
            elif ratio <= 0.2:
                self.add_knowledge(KnowledgeEntry(
                    category="anti_pattern",
                    key=f"action.{action_type}.ineffective",
                    content=f"Action {action_type} has only {ratio:.0%} acceptance rate — consider alternatives",
                    confidence=min(0.9, 0.5 + 0.1 * (len(traces) - accepted)),
                    source="derived",
                ))

    # ------------------------------------------------------------------
    # Seeding domain knowledge
    # ------------------------------------------------------------------

    def seed_domain_knowledge(self) -> None:
        """Pre-populate DFT domain knowledge from established physics."""
        entries = [
            # SIE knowledge
            KnowledgeEntry(
                category="domain", key="sie.fix.increase_alpha",
                content="Self-interaction error (SIE) arises because approximate DFT exchange does not exactly "
                        "cancel the Hartree energy of an electron interacting with itself. The residual spurious "
                        "self-repulsion causes delocalization error: electrons spread too evenly across fragments, "
                        "underestimating energy barriers and overstabilizing fractional-charge states. Increasing "
                        "the exact (HF) exchange fraction alpha directly reduces SIE because HF exchange is "
                        "self-interaction-free by construction (E_x^HF[ρ_1e] = -E_H[ρ_1e]). PBE0 (alpha=0.25) "
                        "reduces SIE4x4 MAE from ~14 to ~8 kcal/mol vs PBE. B3LYP (alpha=0.20) is intermediate. "
                        "Further increasing alpha (0.30-0.50) continues to reduce SIE but at a cost: thermochemistry "
                        "(BH76, ISO34) follows a U-shaped curve, with optimal alpha around 0.25 for global hybrids.",
                confidence=0.95, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="sie.fix.range_separation",
                content="Range-separated hybrids (RSH) address SIE more efficiently than global hybrids by "
                        "splitting the Coulomb operator: 1/r = erf(ωr)/r + erfc(ωr)/r, where the short-range "
                        "part uses DFT exchange and the long-range part uses exact HF exchange. This is physically "
                        "motivated because SIE manifests most strongly at long range (fractional charges on "
                        "separated fragments). CAM-B3LYP (ω=0.33, alpha=0.19, beta=0.46) achieves 65% HF exchange "
                        "at long range vs 19% at short range. ωB97X (ω=0.30, alpha=0.16, beta=0.84) achieves "
                        "100% long-range HF exchange, effectively eliminating long-range SIE. The key advantage "
                        "over global hybrids: RSH can have high long-range HF fraction (fixing SIE and CT excitations) "
                        "while keeping moderate short-range HF fraction (preserving thermochemistry). The parameter "
                        "ω controls the crossover: small ω (0.1) → mostly DFT (like HSE06 for solids), large ω (0.4) "
                        "→ mostly HF (like LC-ωPBE for charge-transfer).",
                confidence=0.95, source="literature",
            ),
            # Barrier knowledge
            KnowledgeEntry(
                category="domain", key="barrier.fix.increase_alpha",
                content="Reaction barrier heights are systematically underestimated by pure GGA functionals because "
                        "the exchange enhancement factor F_x(s) is too large at intermediate reduced gradients s, "
                        "causing over-delocalization of electron density in the transition state. The transition "
                        "state has more diffuse density (higher s) than reactants, so excessive F_x(s) stabilizes "
                        "it too much relative to reactants, lowering the barrier. Exact (HF) exchange does not "
                        "suffer from this delocalization error because it is non-local and correctly localizes "
                        "electrons. Increasing alpha from 0 (pure GGA) to 0.20 (B3LYP) raises BH76 MAE from "
                        "~8 to ~5 kcal/mol. PBE0 (alpha=0.25) achieves ~4 kcal/mol. Further increases help "
                        "barriers but hurt thermochemistry due to the accuracy-accuracy tradeoff: the optimal "
                        "alpha for barriers (~0.50) is much higher than for thermochemistry (~0.25).",
                confidence=0.90, source="literature",
            ),
            # Noncovalent knowledge
            KnowledgeEntry(
                category="domain", key="noncovalent.fix.dispersion",
                content="Pure GGA and hybrid functionals cannot describe long-range dispersion (van der Waals) "
                        "interactions because their exchange-correlation potentials decay exponentially rather than "
                        "as the correct -C₆/r⁶ power law. This causes dramatic underbinding of noncovalent complexes: "
                        "S22 MAE >5 kcal/mol for pure GGAs. Grimme's D3/D3BJ correction adds the missing -s₆·C₆/r⁶ "
                        "- s₈·C₈/r⁸ terms with functional-specific damping. D3BJ (Becke-Johnson damping) is generally "
                        "preferred over D3 (zero-damping) because it avoids the double-counting problem at short range "
                        "and gives better thermochemistry. PBE-D3BJ: s₆=1.0, s₈=0.7224, a₁=0.4289, a₂=4.4407. "
                        "B3LYP-D3BJ: s₆=1.0, s₈=0.3950, a₁=0.2786, a₂=4.654. Using wrong parameters gives poor results.",
                confidence=0.95, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="noncovalent.fix.rsh",
                content="Range-separated hybrids with dispersion (e.g., ωB97X-D, ωB97M-V) provide excellent "
                        "noncovalent energies by combining two physical mechanisms: (1) the RSH handles long-range "
                        "exchange correctly (eliminating charge-transfer errors and recovering the correct asymptotic "
                        "exchange potential), and (2) the dispersion correction handles long-range correlation (the "
                        "attractive -C₆/r⁶ tail that DFT exchange alone cannot produce). ωB97X-D achieves S22 MAE "
                        "~0.2 kcal/mol, among the best for any semilocal functional. The VV10 nonlocal correlation "
                        "functional (used in ωB97M-V, SCAN-rVV10) is a more sophisticated alternative to D3/D3BJ "
                        "that computes C₆ coefficients self-consistently from the density rather than from reference data.",
                confidence=0.85, source="literature",
            ),
            # PBE-specific
            KnowledgeEntry(
                category="domain", key="PBE.strength.thermochemistry",
                content="PBE is good for thermochemistry and solid-state but poor for SIE, barriers, "
                        "and noncovalent interactions. Its parameters (kappa=0.804, mu=0.2195) are "
                        "derived from physical constraints, not fitted.",
                confidence=0.90, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="PBE_SOL.strength.solids",
                content="PBEsol (kappa=0.123, mu=0.10) is optimized for solids and surfaces. "
                        "It overbinds molecules — not suitable for thermochemistry benchmarks.",
                confidence=0.85, source="literature",
            ),
            # Constraint knowledge
            KnowledgeEntry(
                category="constraint", key="lieb_oxford",
                content="Lieb-Oxford bound: E_x ≤ -1.679 * ∫ρ^(4/3) dr. "
                        "Violating this means the exchange energy is too negative (unphysical). "
                        "PBE's kappa=0.804 was chosen to satisfy this bound.",
                confidence=0.95, source="literature",
            ),
            KnowledgeEntry(
                category="constraint", key="one_electron_sie",
                content="One-electron self-interaction: for a one-electron density, E_x must exactly "
                        "cancel E_H (E_x + E_H = 0). Pure LDA/GGA violate this; HF exchange satisfies it exactly.",
                confidence=0.95, source="literature",
            ),
            # Anti-patterns
            KnowledgeEntry(
                category="anti_pattern", key="overfitting.single_subset",
                content="Optimizing for a single subset often overfits — the functional becomes specialized "
                        "and regresses on other subsets. Always validate on held-out subsets.",
                confidence=0.90, source="observation",
            ),
            KnowledgeEntry(
                category="anti_pattern", key="alpha_too_high",
                content="alpha > 0.50 typically hurts thermochemistry significantly while only "
                        "marginally improving SIE. The sweet spot for global hybrids is 0.20-0.30.",
                confidence=0.85, source="literature",
            ),
            KnowledgeEntry(
                category="anti_pattern", key="too_many_components",
                content="Adding too many components (>6) makes the functional hard to interpret and "
                        "prone to overfitting. Prefer replacing components over adding many.",
                confidence=0.80, source="observation",
            ),
            # --- Additional entries from xc_functional_parameter_space.md ---
            # Alpha (HF exchange) parameter space
            KnowledgeEntry(
                category="domain", key="alpha.sweet_spot",
                content="The sweet spot for global hybrid alpha is 0.20-0.30. "
                        "alpha=0.20 (B3LYP) and alpha=0.25 (PBE0) are the most successful values. "
                        "alpha=0.25 comes from the adiabatic connection formula: E_xc = ∫₀¹ E_xc^λ dλ, "
                        "where the λ=0 limit gives exact exchange and λ=1 gives the Kohn-Sham solution. "
                        "The second-order term in the perturbation expansion yields alpha=1/4. "
                        "This theoretical justification makes PBE0's alpha=0.25 more principled than "
                        "B3LYP's alpha=0.20 (which was empirically fitted to atomization energies).",
                confidence=0.90, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="alpha.sie_monotonic",
                content="SIE4x4 MAE decreases monotonically with increasing alpha because exact HF exchange "
                        "is self-interaction-free: E_x^HF[ρ_1e] = -E_H[ρ_1e] exactly. Each 0.05 increase "
                        "in alpha reduces SIE4x4 MAE by ~1.4 kcal/mol (from ~14 at alpha=0 to ~4 at alpha=0.50). "
                        "However, thermochemistry (BH76, ISO34) follows a U-shaped curve: BH76 MAE decreases "
                        "monotonically (barriers benefit from exact exchange), but ISO34 MAE has a minimum "
                        "around alpha=0.20-0.25 and increases sharply above 0.30. The optimal alpha for "
                        "balanced performance is 0.20-0.25. Going above 0.30 requires careful justification "
                        "because ISO34 regression typically outweighs SIE4x4 improvement.",
                confidence=0.85, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="alpha.sum_rule",
                content="For hybrid functionals, alpha + DFT_exchange_mixing = 1.0 (sum rule). "
                        "When alpha increases, DFT exchange mixing must decrease proportionally. "
                        "B3LYP: alpha=0.20, LDA_X_mix=0.08, B88_mix=0.72.",
                confidence=0.95, source="literature",
            ),
            # Omega (range separation) parameter space
            KnowledgeEntry(
                category="domain", key="omega.physical_meaning",
                content="omega controls the range-separation cutoff in RSH functionals. "
                        "1/r = erf(omega*r)/r + erfc(omega*r)/r splits Coulomb into SR+LR. "
                        "Small omega → more DFT exchange (SR dominates). Large omega → more HF exchange (LR dominates).",
                confidence=0.90, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="omega.typical_values",
                content="Common omega values: HSE06=0.11 (short-range hybrid for solids), "
                        "CAM-B3LYP=0.33 (general-purpose RSH), wB97X=0.30 (long-range corrected), "
                        "LC-wPBE=0.40 (pure long-range correction). omega<0.1 ≈ global hybrid.",
                confidence=0.85, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="omega.beta_rsh",
                content="RSH has 3 parameters: alpha (SR-HF), beta (LR-HF extra), omega. "
                        "Total SR-HF = alpha, total LR-HF = alpha+beta. "
                        "CAM-B3LYP: alpha=0.19, beta=0.46 → SR=19%, LR=65%. "
                        "wB97X: alpha=0.157706, beta=0.842294 → LR=100%.",
                confidence=0.90, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="omega.degradation",
                content="Setting beta=0 and omega→inf degrades an RSH to a global hybrid. "
                        "Setting alpha=0 and beta=1.0 gives pure long-range correction (LC scheme). "
                        "RSH parameters can be continuously deformed between these extremes.",
                confidence=0.80, source="literature",
            ),
            # PBE parameter space
            KnowledgeEntry(
                category="domain", key="PBE.kappa_mu_effect",
                content="PBE exchange enhancement factor F(s) = 1 + kappa - kappa/(1 + mu*s^2/kappa). "
                        "kappa controls the upper bound of F(s) (Lieb-Oxford bound). "
                        "mu controls the gradient expansion coefficient for small s. "
                        "PBE: kappa=0.804, mu=0.2195. PBEsol: kappa=0.123, mu=0.10.",
                confidence=0.95, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="PBE.revpBE_direction",
                content="revPBE increases kappa from 0.804 to 1.245, relaxing the Lieb-Oxford bound. "
                        "This improves weak interactions but may violate the bound. "
                        "REWIRE kappa→1.245 moves PBE toward revPBE.",
                confidence=0.85, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="PBE.pbesol_direction",
                content="PBEsol reduces kappa (0.804→0.123) and mu (0.2195→0.10) simultaneously. "
                        "This compresses the enhancement factor, improving solid-state properties "
                        "but overbinding molecules. REWIRE both params moves PBE→PBEsol.",
                confidence=0.90, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="PBE.correlation_params",
                content="PBE correlation has beta=0.0667 and gamma=0.0311. "
                        "beta is the gradient expansion coefficient, gamma is a buffer parameter. "
                        "These are less commonly tuned than exchange params.",
                confidence=0.80, source="literature",
            ),
            # B88/B3LYP parameter space
            KnowledgeEntry(
                category="domain", key="B88.beta_param",
                content="B88 gradient correction has beta=0.0042 and gamma=6.0. "
                        "beta controls gradient correction strength. "
                        "B3LYP uses B88 with 72% mixing factor.",
                confidence=0.85, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="B3LYP.mixing_structure",
                content="B3LYP exchange: 20% HF + 8% LDA_X + 72% B88. "
                        "B3LYP correlation: 81% LYP + 19% VWN. "
                        "Exchange sum rule: alpha + LDA_mix + B88_mix = 1. "
                        "Correlation sum rule: LYP_mix + VWN_mix = 1.",
                confidence=0.90, source="literature",
            ),
            # SCAN/MGGA parameter space
            KnowledgeEntry(
                category="domain", key="SCAN.constraint_satisfaction",
                content="SCAN is the first meta-GGA satisfying all 17 known exact constraints. "
                        "It uses kinetic density tau via alpha=(tau-tau_unif)/tau_unif indicator "
                        "to distinguish covalent bonds from weak interactions.",
                confidence=0.90, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="SCAN.ext_params",
                content="SCAN exchange has 4 ext_params: c1=0.667, c2=0.800, d=1.240, k1=0.065. "
                        "These control the interpolation between single-orbital and uniform-gas limits. "
                        "SCAN correlation has no ext_params.",
                confidence=0.85, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="SCAN.r2scan_variant",
                content="r2SCAN fixes numerical issues in SCAN while maintaining accuracy. "
                        "r2SCAN has no ext_params (all hardcoded). "
                        "REPLACE_COMPONENT SCAN→r2SCAN improves numerical stability.",
                confidence=0.80, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="SCAN.scan0_hybrid",
                content="SCAN0 = 25% HF + 75% SCAN exchange + SCAN correlation. "
                        "Like PBE0 but with SCAN exchange. Better for SIE than PBE0. "
                        "ADD_COMPONENT HF (alpha=0.25) to SCAN → SCAN0.",
                confidence=0.85, source="literature",
            ),
            # Dimensionality of parameter space
            KnowledgeEntry(
                category="domain", key="param_space.dimensionality",
                content="Most common functionals have ≤7 tunable dimensions. "
                        "LDA: 0 dim. PBE: 4 dim (kappa, mu, beta, gamma). "
                        "B3LYP: 4 dim. PBE0: 5 dim. CAM-B3LYP: 6 dim. "
                        "Adding HF exchange adds 1 dim (alpha). Adding RSH adds 2 dim (beta, omega).",
                confidence=0.90, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="param_space.ext_params_prevalence",
                content="465 out of 679 libxc functionals (68.5%) have tunable ext_params. "
                        "Physics-driven functionals (PBE, SCAN) have 2-4 params. "
                        "Empirical functionals (Minnesota series) have 12-49 params. "
                        "REWIRE is applicable to most functionals.",
                confidence=0.85, source="literature",
            ),
            # LDA limitations
            KnowledgeEntry(
                category="anti_pattern", key="lda_no_params",
                content="LDA has 0 tunable dimensions — it is a single point in parameter space. "
                        "Cannot be improved by REWIRE. Must ADD_COMPONENT (gradient correction) "
                        "or ADD_COMPONENT (HF exchange) to move away from LDA.",
                confidence=0.95, source="literature",
            ),
            # MGGA vs GGA
            KnowledgeEntry(
                category="domain", key="mgga_vs_gga.exchange",
                content="Meta-GGA exchange (SCAN, TPSS) uses kinetic density tau as extra input. "
                        "This allows distinguishing covalent vs weak bonds. "
                        "MGGA typically better for SIE and barriers than GGA, but harder to hybridize.",
                confidence=0.85, source="literature",
            ),
            # Dispersion correction details
            KnowledgeEntry(
                category="domain", key="dispersion.d3bj_mechanism",
                content="D3BJ adds -s6*C6/r^6 - s8*C8/r^8 damping terms. "
                        "s6 is the global scaling (default 1.0 for most functionals). "
                        "s8 controls higher-order dispersion. BJ damping avoids the double-counting "
                        "problem of zero-damping (D3).",
                confidence=0.90, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="dispersion.functional_specific_s6",
                content="D3/D3BJ parameters (s6, s8, a1, a2) are functional-specific. "
                        "PBE-D3BJ: s6=1.0, s8=0.7224, a1=0.4289, a2=4.4407. "
                        "B3LYP-D3BJ: s6=1.0, s8=0.3950, a1=0.2786, a2=4.654. "
                        "Using wrong parameters gives poor results.",
                confidence=0.85, source="literature",
            ),
            # RSH continuous deformation
            KnowledgeEntry(
                category="domain", key="rsh.deformation_paths",
                content="RSH parameters can be continuously deformed: "
                        "B3LYP (alpha=0.20, beta=0, omega=0) → CAM-B3LYP (alpha=0.19, beta=0.46, omega=0.33) "
                        "→ LC-wBLYP (alpha=0, beta=1.0, omega=0.47). "
                        "This is a smooth path from global hybrid to long-range correction.",
                confidence=0.80, source="literature",
            ),
            # Mixing factor physics
            KnowledgeEntry(
                category="domain", key="mixing_factor.sum_rule",
                content="Exchange components must satisfy: alpha + sum(DFT_exchange_mixing) = 1. "
                        "Correlation components must satisfy: sum(correlation_mixing) = 1. "
                        "Violating sum rules gives unphysical results.",
                confidence=0.95, source="literature",
            ),
            KnowledgeEntry(
                category="domain", key="mixing_factor.adjustment",
                content="Adjusting mixing_factor changes the weight of a component. "
                        "REWIRE mixing:GGA_X_PBE from 0.75 to 0.70 reduces PBE exchange weight. "
                        "This is equivalent to increasing alpha by 0.05 (due to sum rule).",
                confidence=0.85, source="observation",
            ),
            # Functional family landscape
            KnowledgeEntry(
                category="domain", key="landscape.hybrid_gga_dominant",
                content="HYB_GGA_XC is the largest compound functional category (107 in libxc). "
                        "Contains B3LYP, PBE0, CAM-B3LYP, and most widely-used functionals. "
                        "GGA exchange-only (108) is the largest single-role category.",
                confidence=0.80, source="literature",
            ),
        ]
        for entry in entries:
            self.add_knowledge(entry)
        logger.info(f"Seeded {len(entries)} domain knowledge entries")

        # Load mass benchmark knowledge if available
        default_json = Path("knowledge/domain_knowledge.json")
        if default_json.exists():
            self.load_from_json(default_json)

    def load_from_json(self, json_path: Path | str) -> int:
        """Load domain knowledge from build_domain_knowledge.py output.

        Inserts parameter curves, ablation evidence, anti-patterns, and
        domain knowledge entries from the mass benchmark analysis.

        Returns total number of entries inserted.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            logger.warning(f"Knowledge JSON not found: {json_path}")
            return 0

        data = json.loads(json_path.read_text())
        total = 0

        # 1. Parameter curves → domain knowledge entries
        for curve in data.get("parameter_curves", []):
            key = curve.get("key", "")
            if curve.get("category") == "parameter_curve":
                content = (
                    f"{curve['parameter']}={curve['parameter_value']}: "
                    f"avg MAE on {curve['subset']} = {curve['avg_mae']:.3f} kcal/mol "
                    f"(n={curve['n_functionals']}, range [{curve['min_mae']:.3f}, {curve['max_mae']:.3f}])"
                )
                self.add_knowledge(KnowledgeEntry(
                    category="domain", key=f"curve.{key}",
                    content=content,
                    confidence=0.85, source="mass_benchmark",
                ))
                total += 1
            elif curve.get("category") == "family_performance":
                content = (
                    f"{curve['family']}: avg MAE on {curve['subset']} = {curve['avg_mae']:.3f} kcal/mol "
                    f"(n={curve['n_functionals']}, range [{curve['min_mae']:.3f}, {curve['max_mae']:.3f}])"
                )
                self.add_knowledge(KnowledgeEntry(
                    category="domain", key=f"family.{key}",
                    content=content,
                    confidence=0.85, source="mass_benchmark",
                ))
                total += 1

        # 2. Natural ablation evidence → ablation_evidence table
        for abl in data.get("ablation_evidence", []):
            pair = abl.get("functional_pair", ("", ""))
            param = abl.get("parameter_changed", "")
            values = abl.get("parameter_values", (0, 0))
            self.add_ablation(AblationEntry(
                component=abl.get("component", ""),
                component_family=abl.get("component_family", ""),
                functional_name=f"{pair[0]} vs {pair[1]}",
                result=abl.get("result", "inconclusive"),
                metric=f"{abl.get('subset', '')}_mae_delta",
                value=abl.get("delta", 0.0),
                analysis=f"{param} change {values[0]}→{values[1]}: "
                         f"MAE {abl.get('mae_before', 0):.3f}→{abl.get('mae_after', 0):.3f}",
                confidence=0.7,
            ))
            total += 1

        # 3. Anti-patterns → anti_pattern knowledge entries
        for pattern in data.get("anti_patterns", []):
            self.add_knowledge(KnowledgeEntry(
                category="anti_pattern",
                key=f"mass.{pattern.get('key', '')}",
                content=pattern.get("content", ""),
                confidence=pattern.get("confidence", 0.8),
                source="mass_benchmark",
            ))
            total += 1

        # 4. Domain knowledge entries
        for dk in data.get("domain_knowledge", []):
            self.add_knowledge(KnowledgeEntry(
                category="domain",
                key=f"mass.{dk.get('key', '')}",
                content=dk.get("content", ""),
                confidence=dk.get("confidence", 0.85),
                source="mass_benchmark",
            ))
            total += 1

        logger.info(f"Loaded {total} entries from {json_path}")
        return total

    def ingest_libxc_prerun(
        self,
        cache_dir: Path | str,
        *,
        source: str = "libxc_prerun",
        baseline_only: bool = False,
    ) -> int:
        """Import pre-run libxc benchmark JSON files into unified evaluation memory."""
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            logger.warning("libxc prerun cache not found: %s", cache_path)
            return 0
        if baseline_only:
            self.conn.execute("DELETE FROM functional_evaluations WHERE source=?", (source,))
            self.conn.commit()
        decomposition_report: dict[str, Any] = {}
        decomposition_path = cache_path / "libxc_components.json"
        if decomposition_path.exists():
            try:
                decomposition_raw = json.loads(decomposition_path.read_text())
                decomposition_report = decomposition_raw.get("entries", decomposition_raw)
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read libxc component decomposition report: %s", decomposition_path)
        total = 0
        for json_file in sorted(cache_path.glob("*.json")):
            if json_file.name == "libxc_components.json":
                continue
            try:
                raw = json.loads(json_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            meta = raw.get("__meta__") if isinstance(raw.get("__meta__"), dict) else {}
            name = str(raw.get("name") or raw.get("functional_name") or meta.get("xc_name") or json_file.stem)
            if baseline_only and not self._is_baseline_prerun_cache(json_file, raw, name):
                continue
            xc = self._xc_from_prerun_payload(
                name,
                raw,
                decomposition=decomposition_report.get(json_file.name),
            )
            if xc is None:
                continue
            xc.name = name
            result = BenchmarkResult(xc_name=xc.name)
            subset_payload = raw.get("subsets") if isinstance(raw.get("subsets"), dict) else raw
            for subset, payload in subset_payload.items():
                if not isinstance(payload, dict) or payload.get("mae_kcal") is None:
                    continue
                result.subsets[str(subset)] = self._subset_from_payload(str(subset), payload)
            if not result.subsets:
                continue
            result.compute_overall()
            defect_tags = self._defect_tags_from_result(result)
            self.record_functional_evaluation(
                xc,
                result,
                source=source,
                status="complete",
                failure_reason="",
                defect_tags=defect_tags,
                benchmark_context={"source": "legacy_cache", "cache_file": json_file.name},
            )
            total += 1
        logger.info(
            "Imported %d benchmark-library functionals into knowledge memory%s",
            total,
            f" using {decomposition_path.name}" if decomposition_report else "",
        )
        return total

    @staticmethod
    def _is_baseline_prerun_cache(json_file: Path, raw: dict[str, Any], name: str) -> bool:
        """Return True for the fixed benchmark-library baseline files.

        Generated search caches commonly carry names such as ``__editable__``,
        ``__evo_...``, ``__tpe_...`` or explicit search metadata.  Component
        benchmark priors should be based on the reference panel only, not on
        earlier optimizer trajectories.
        """
        meta = raw.get("__meta__") if isinstance(raw.get("__meta__"), dict) else {}
        xc_dict = meta.get("xc_dict") if isinstance(meta.get("xc_dict"), dict) else raw.get("xc_dict")
        metadata = xc_dict.get("metadata", {}) if isinstance(xc_dict, dict) else {}
        text = " ".join(
            str(item)
            for item in (
                json_file.stem,
                name,
                meta.get("xc_name", ""),
                metadata.get("action", ""),
                metadata.get("search_algorithm", ""),
            )
        ).lower()
        generated_markers = (
            "__",
            "_editable",
            "evo_",
            "tpe_",
            "rewire_",
            "refine_",
            "replace_",
            "remove_",
            "add_",
            "descriptor",
            "evolutionary",
            "optuna",
            "mcts",
        )
        return not any(marker in text for marker in generated_markers)

    # ------------------------------------------------------------------
    # Internal query helpers
    # ------------------------------------------------------------------

    def _query_knowledge_by_key_pattern(self, pattern: str) -> list[KnowledgeEntry]:
        rows = self.conn.execute(
            "SELECT category, key, content, confidence, source FROM knowledge "
            "WHERE key LIKE ? ORDER BY confidence DESC",
            (f"%{pattern}%",),
        ).fetchall()
        return [KnowledgeEntry(*r) for r in rows]

    def _query_knowledge_by_category(self, category: str) -> list[KnowledgeEntry]:
        rows = self.conn.execute(
            "SELECT category, key, content, confidence, source FROM knowledge "
            "WHERE category = ? ORDER BY confidence DESC",
            (category,),
        ).fetchall()
        return [KnowledgeEntry(*r) for r in rows]

    def _query_ablation_by_component(self, component: str, limit: int = 10) -> list[AblationEntry]:
        rows = self.conn.execute(
            "SELECT component, component_family, functional_name, result, metric, value, "
            "analysis, confidence FROM ablation_evidence "
            "WHERE component = ? ORDER BY confidence DESC, created_at DESC LIMIT ?",
            (component, limit),
        ).fetchall()
        return [AblationEntry(*r) for r in rows]

    def _query_ablation_by_family(self, family: str, limit: int = 10) -> list[AblationEntry]:
        rows = self.conn.execute(
            "SELECT component, component_family, functional_name, result, metric, value, "
            "analysis, confidence FROM ablation_evidence "
            "WHERE component_family = ? AND result != 'inconclusive' "
            "ORDER BY confidence DESC, created_at DESC LIMIT ?",
            (family, limit),
        ).fetchall()
        return [AblationEntry(*r) for r in rows]

    def _query_recent_ablation(self, limit: int = 10) -> list[AblationEntry]:
        rows = self.conn.execute(
            "SELECT component, component_family, functional_name, result, metric, value, "
            "analysis, confidence FROM ablation_evidence "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [AblationEntry(*r) for r in rows]

    def _query_traces_by_action(self, action_type: str, limit: int = 10) -> list[TraceEntry]:
        rows = self.conn.execute(
            "SELECT name, parent_name, action_type, action_detail, signature, "
            "search_mae, validation_mae, status, defect_tags, core_contribution, "
            "rationale, component_explanations, edit_plan, skill_metrics, "
            "active_skill, root_domains FROM search_trace "
            "WHERE action_type = ? ORDER BY created_at DESC LIMIT ?",
            (action_type, limit),
        ).fetchall()
        return [TraceEntry(*r) for r in rows]

    def _query_traces_by_skill(self, skill_name: str, limit: int = 10) -> list[TraceEntry]:
        rows = self.conn.execute(
            "SELECT name, parent_name, action_type, action_detail, signature, "
            "search_mae, validation_mae, status, defect_tags, core_contribution, "
            "rationale, component_explanations, edit_plan, skill_metrics, "
            "active_skill, root_domains FROM search_trace "
            "WHERE active_skill = ? ORDER BY created_at DESC LIMIT ?",
            (skill_name, limit),
        ).fetchall()
        return [TraceEntry(*r) for r in rows]

    def _query_recent_traces(self, limit: int = 10) -> list[TraceEntry]:
        rows = self.conn.execute(
            "SELECT name, parent_name, action_type, action_detail, signature, "
            "search_mae, validation_mae, status, defect_tags, core_contribution, "
            "rationale, component_explanations, edit_plan, skill_metrics, "
            "active_skill, root_domains FROM search_trace "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [TraceEntry(*r) for r in rows]

    def _query_relevant_ablation(
        self,
        current_xc: XCFunctional,
        defect_tags: list[str],
        limit: int = 12,
    ) -> list[AblationEntry]:
        families = {item.family for item in extract_component_families(current_xc)}
        if "self_interaction_error" in defect_tags:
            families.update({"hf_exchange.global", "hf_exchange.range_separated", "exchange.mgga"})
        if "validation_regression" in defect_tags:
            families.update({"exchange.gga", "exchange.mgga", "correlation.gga"})
        rows: list[tuple] = []
        for family in sorted(families):
            rows.extend(
                self.conn.execute(
                    "SELECT component, component_family, functional_name, result, metric, value, "
                    "analysis, confidence FROM ablation_evidence "
                    "WHERE component_family = ? OR component_family LIKE ? "
                    "ORDER BY confidence DESC, created_at DESC LIMIT ?",
                    (family, f"{family}.%", limit),
                ).fetchall()
            )
        if len(rows) < limit:
            rows.extend(
                self.conn.execute(
                    "SELECT component, component_family, functional_name, result, metric, value, "
                    "analysis, confidence FROM ablation_evidence "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit - len(rows),),
                ).fetchall()
            )
        deduped = []
        seen = set()
        for row in rows:
            key = (row[0], row[3], row[4], row[6])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= limit:
                break
        return [AblationEntry(*r) for r in deduped]

    def _query_relevant_traces(
        self,
        active_skill: str,
        defect_tags: list[str],
        limit: int = 5,
    ) -> list[TraceEntry]:
        rows: list[tuple] = []
        if active_skill:
            rows.extend(
                self.conn.execute(
                    "SELECT name, parent_name, action_type, action_detail, signature, "
                    "search_mae, validation_mae, status, defect_tags, core_contribution, "
                    "rationale, component_explanations, edit_plan, skill_metrics, "
                    "active_skill, root_domains FROM search_trace "
                    "WHERE active_skill = ? ORDER BY created_at DESC LIMIT ?",
                    (active_skill, limit),
                ).fetchall()
            )
        for defect in defect_tags:
            if len(rows) >= limit:
                break
            rows.extend(
                self.conn.execute(
                    "SELECT name, parent_name, action_type, action_detail, signature, "
                    "search_mae, validation_mae, status, defect_tags, core_contribution, "
                    "rationale, component_explanations, edit_plan, skill_metrics, "
                    "active_skill, root_domains FROM search_trace "
                    "WHERE defect_tags LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (f"%{defect}%", limit - len(rows)),
                ).fetchall()
            )
        if len(rows) < limit:
            rows.extend(
                self.conn.execute(
                    "SELECT name, parent_name, action_type, action_detail, signature, "
                    "search_mae, validation_mae, status, defect_tags, core_contribution, "
                    "rationale, component_explanations, edit_plan, skill_metrics, "
                    "active_skill, root_domains FROM search_trace "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit - len(rows),),
                ).fetchall()
            )
        seen = set()
        traces = []
        for row in rows:
            if row[4] in seen:
                continue
            seen.add(row[4])
            traces.append(TraceEntry(*row))
            if len(traces) >= limit:
                break
        return traces

    def _query_mechanism_cards(
        self,
        *,
        current_xc: XCFunctional,
        defect_tags: list[str],
        target_subsets: list[str],
        active_skill: str,
        limit: int,
    ) -> list[MechanismCardEntry]:
        current_families = {item.family for item in extract_component_families(current_xc)}
        rows = self.conn.execute(
            "SELECT mechanism_key, statement, target_subset, target_delta, validation_penalty, "
            "risk_score, support_count, target_defects_json, component_families_json, "
            "expected_effects_json, risk_subsets_json, guardrails_json, evidence_refs_json, "
            "confidence, status FROM mechanism_cards ORDER BY confidence DESC, updated_at DESC"
        ).fetchall()
        scored: list[tuple[float, MechanismCardEntry]] = []
        for row in rows:
            target_subset = str(row[2] or "")
            if target_subsets and target_subset and target_subset not in target_subsets:
                continue
            target_defects = json.loads(row[7] or "[]")
            families = json.loads(row[8] or "[]")
            score = float(row[13])
            score += 0.35 if target_subset in target_subsets else 0.0
            score += 0.25 * len(set(target_defects) & set(defect_tags))
            score += 0.15 * len(set(families) & current_families)
            if active_skill and active_skill in row[0]:
                score += 0.2
            score += max(-0.3, min(0.3, float(row[3] or 0.0) / 10.0))
            score -= 0.25 * float(row[5] or 0.0)
            scored.append((
                score,
                MechanismCardEntry(
                    mechanism_key=row[0],
                    statement=row[1],
                    target_subset=target_subset,
                    target_delta=float(row[3] or 0.0),
                    validation_penalty=float(row[4] or 0.0),
                    risk_score=float(row[5] or 0.0),
                    support_count=int(row[6] or 0),
                    target_defects=target_defects,
                    component_families=families,
                    expected_effects=json.loads(row[9] or "{}"),
                    risk_subsets=json.loads(row[10] or "[]"),
                    guardrails=json.loads(row[11] or "[]"),
                    evidence_refs=json.loads(row[12] or "[]"),
                    confidence=float(row[13]),
                    status=row[14],
                ),
            ))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def _query_failure_patterns(
        self,
        *,
        current_xc: XCFunctional,
        defect_tags: list[str],
        active_skill: str,
        hard_only: bool = False,
        limit: int,
    ) -> list[FailurePatternEntry]:
        del current_xc, defect_tags
        where = "WHERE hard_block=1" if hard_only else ""
        rows = self.conn.execute(
            "SELECT pattern_key, symptom, root_cause, repair_hint, action_type, component_family, "
            "hard_block, occurrences, confidence FROM failure_patterns "
            f"{where} ORDER BY hard_block DESC, confidence DESC, updated_at DESC LIMIT ?",
            (max(limit * 3, limit),),
        ).fetchall()
        entries = [
            FailurePatternEntry(
                pattern_key=row[0],
                symptom=row[1],
                root_cause=row[2],
                repair_hint=row[3],
                action_type=row[4],
                component_family=row[5],
                hard_block=bool(row[6]),
                occurrences=int(row[7]),
                confidence=float(row[8]),
            )
            for row in rows
            if not active_skill or active_skill in row[0] or not row[4]
        ]
        return entries[:limit]

    def _query_risk_patterns(
        self,
        *,
        target_subsets: list[str],
        active_skill: str,
        limit: int,
    ) -> list[RiskPatternEntry]:
        rows = self.conn.execute(
            "SELECT pattern_key, target_subset, risk_subset, symptom, action_type, active_skill, "
            "component_family, occurrences, avg_ratio, confidence FROM risk_patterns "
            "ORDER BY confidence DESC, updated_at DESC LIMIT ?",
            (max(limit * 4, limit),),
        ).fetchall()
        entries: list[RiskPatternEntry] = []
        for row in rows:
            if target_subsets and row[1] not in target_subsets:
                continue
            if active_skill and row[5] and active_skill != row[5]:
                continue
            entries.append(RiskPatternEntry(
                pattern_key=row[0],
                target_subset=row[1],
                risk_subset=row[2],
                symptom=row[3],
                action_type=row[4],
                active_skill=row[5],
                component_family=row[6],
                occurrences=int(row[7]),
                avg_ratio=float(row[8]),
                confidence=float(row[9]),
            ))
            if len(entries) >= limit:
                break
        return entries

    def _query_libxc_anchors(
        self,
        current_xc: XCFunctional,
        current_result: BenchmarkResult,
        defect_tags: list[str],
        target_subsets: list[str],
        limit: int,
    ) -> list[LibxcAnchorEntry]:
        rows = self.conn.execute(
            "SELECT functional_name, source, structural_signature, component_families_json, "
            "hf_exchange_json, subset_metrics_json, defect_tags_json "
            "FROM functional_evaluations WHERE source = 'libxc_prerun' "
            "ORDER BY updated_at DESC LIMIT 500"
        ).fetchall()
        current_families = {item.family for item in extract_component_families(current_xc)}
        current_hf = current_xc.hf_exchange
        targets = target_subsets or self._target_subsets_from_defects(defect_tags) or list(current_result.subsets)
        target_scored: list[tuple[float, LibxcAnchorEntry]] = []
        neighbor_scored: list[tuple[float, LibxcAnchorEntry]] = []
        for row in rows:
            families_payload = json.loads(row[3] or "[]")
            families = {item.get("family", "") for item in families_payload if item.get("family")}
            hf = json.loads(row[4] or "{}")
            metrics = json.loads(row[5] or "{}")
            flat_metrics = self._flatten_subset_metrics(metrics)
            if targets and not any(subset in flat_metrics for subset in targets):
                continue
            family_jaccard = len(current_families & families) / max(1, len(current_families | families))
            hf_dist = (
                abs(float(hf.get("alpha", 0.0)) - current_hf.alpha)
                + abs(float(hf.get("beta", 0.0)) - current_hf.beta)
                + abs(float(hf.get("omega", 0.0)) - current_hf.omega)
            )
            similarity = max(0.0, min(1.0, 0.75 * family_jaccard + 0.25 * (1.0 / (1.0 + hf_dist))))
            target_maes = [
                float(flat_metrics[subset].get("mae_kcal", 0.0) or 0.0)
                for subset in targets
                if subset in flat_metrics and float(flat_metrics[subset].get("mae_kcal", 0.0) or 0.0) > 0
            ]
            current_target_maes = [
                current_result.subsets[subset].mae_kcal
                for subset in targets
                if subset in current_result.subsets and current_result.subsets[subset].mae_kcal > 0
            ]
            target_gain = 0.0
            if target_maes and current_target_maes:
                target_gain = (sum(current_target_maes) / len(current_target_maes)) - (sum(target_maes) / len(target_maes))
            validation_penalty = 0.0
            for subset, payload in flat_metrics.items():
                if subset in targets:
                    continue
                baseline = current_result.subsets.get(subset)
                mae = float(payload.get("mae_kcal", 0.0) or 0.0)
                if baseline and baseline.mae_kcal > 0 and mae > baseline.mae_kcal:
                    validation_penalty += min(5.0, mae - baseline.mae_kcal)
                elif subset in {"BH76", "G21IP", "G21EA", "ISO34", "S22", "S66", "W4-11"}:
                    validation_penalty += max(0.0, min(10.0, mae - 5.0))
            failure_penalty = sum(float(payload.get("n_failures", 0) or 0) for payload in flat_metrics.values()) * 0.1
            target_score = target_gain - 0.7 * validation_penalty - 0.3 * failure_penalty + 0.1 * similarity
            target_scored.append((
                target_score,
                LibxcAnchorEntry(
                    functional_name=row[0],
                    source=row[1],
                    structural_signature=row[2],
                    similarity=similarity,
                    anchor_type="target_anchor",
                    target_score=target_score,
                    validation_penalty=validation_penalty,
                    subset_metrics=flat_metrics,
                    component_families=sorted(families),
                    defect_tags=json.loads(row[6] or "[]"),
                ),
            ))
            neighbor_scored.append((
                similarity,
                LibxcAnchorEntry(
                    functional_name=row[0],
                    source=row[1],
                    structural_signature=row[2],
                    similarity=similarity,
                    anchor_type="nearest_neighbor",
                    target_score=target_score,
                    validation_penalty=validation_penalty,
                    subset_metrics=flat_metrics,
                    component_families=sorted(families),
                    defect_tags=json.loads(row[6] or "[]"),
                ),
            ))
        target_scored.sort(key=lambda item: item[0], reverse=True)
        neighbor_scored.sort(key=lambda item: item[0], reverse=True)
        selected = [entry for _, entry in target_scored[:limit]]
        seen = {entry.structural_signature for entry in selected}
        for _, entry in neighbor_scored:
            if entry.structural_signature in seen:
                continue
            selected.append(entry)
            seen.add(entry.structural_signature)
            if len(selected) >= limit * 2:
                break
        return selected

    def component_priors_for(
        self,
        current_xc: XCFunctional,
        current_result: BenchmarkResult,
        *,
        target_subsets: list[str] | None = None,
        limit: int = 32,
    ) -> list[ComponentBenchmarkPriorEntry]:
        """Return only component benchmark priors, without broader prompt retrieval."""
        targets = target_subsets or list(current_result.subsets)
        return self._query_component_benchmark_priors(
            current_xc,
            current_result,
            targets,
            limit=max(1, int(limit)),
        )
    def _query_component_benchmark_priors(
        self,
        current_xc: XCFunctional,
        current_result: BenchmarkResult,
        target_subsets: list[str],
        limit: int,
    ) -> list[ComponentBenchmarkPriorEntry]:
        """Rank replace/add candidates by association with pre-run functional results.

        These are whole-functional associations, not causal component effects.
        """
        rows = self.conn.execute(
            "SELECT functional_name, xc_json, subset_metrics_json "
            "FROM functional_evaluations WHERE source = 'libxc_prerun' "
            "ORDER BY updated_at DESC LIMIT 1000"
        ).fetchall()
        targets = target_subsets or list(current_result.subsets)
        current_target_maes = [
            current_result.subsets[subset].mae_kcal
            for subset in targets
            if subset in current_result.subsets and current_result.subsets[subset].mae_kcal > 0
        ]
        current_target_mae = (
            sum(current_target_maes) / len(current_target_maes)
            if current_target_maes else 0.0
        )
        current_components = {component.name for component in current_xc.components}
        guard_names = {"BH76", "G21EA", "G21IP", "ISO34", "W4-11", "S22", "S66"}
        scoring_guard_names = {"BH76", "G21EA", "G21IP"}
        grouped: dict[str, dict[str, Any]] = {}
        for functional_name, xc_json, metrics_json in rows:
            try:
                xc_payload = json.loads(xc_json or "{}")
                components = xc_payload.get("components", [])
                metrics = self._flatten_subset_metrics(json.loads(metrics_json or "{}"))
            except (TypeError, json.JSONDecodeError):
                continue
            target_maes = [
                float(metrics[subset].get("mae_kcal", 0.0) or 0.0)
                for subset in targets
                if subset in metrics and float(metrics[subset].get("mae_kcal", 0.0) or 0.0) > 0
            ]
            if not target_maes:
                continue
            target_mae = sum(target_maes) / len(target_maes)
            target_gain = current_target_mae - target_mae if current_target_mae > 0 else -target_mae
            guard_metrics = {
                subset: float(payload.get("mae_kcal", 0.0) or 0.0)
                for subset, payload in metrics.items()
                if subset in guard_names and subset not in targets
                and float(payload.get("mae_kcal", 0.0) or 0.0) > 0
            }
            guard_penalty = sum(
                max(0.0, mae - 5.0)
                for subset, mae in guard_metrics.items()
                if subset in scoring_guard_names
            )
            failures = sum(int(payload.get("n_failures", 0) or 0) for payload in metrics.values())
            target_failures = sum(
                int(metrics[subset].get("n_failures", 0) or 0)
                for subset in targets if subset in metrics
            )
            association_score = (
                -999.0
                if target_failures
                else target_gain - 0.25 * guard_penalty - 0.5 * failures
            )
            for component in components:
                component_name = str(component.get("name", ""))
                role = PURE_COMPONENT_ROLES.get(component_name)
                if not role or component_name in current_components:
                    continue
                slot = grouped.setdefault(component_name, {
                    "role": role,
                    "occurrences": 0,
                    "best": None,
                })
                slot["occurrences"] += 1
                candidate = {
                    "functional_name": functional_name,
                    "association_score": association_score,
                    "target_mae": target_mae,
                    "target_gain": target_gain,
                    "guard_metrics": guard_metrics,
                    "failures": failures,
                    "target_failures": target_failures,
                }
                if slot["best"] is None or association_score > slot["best"]["association_score"]:
                    slot["best"] = candidate
        priors: list[ComponentBenchmarkPriorEntry] = []
        for component_name, payload in grouped.items():
            best = payload["best"]
            if best is None:
                continue
            priors.append(ComponentBenchmarkPriorEntry(
                component_name=component_name,
                role=payload["role"],
                occurrences=payload["occurrences"],
                best_functional=best["functional_name"],
                association_score=best["association_score"],
                target_mae=best["target_mae"],
                target_gain=best["target_gain"],
                guard_metrics=best["guard_metrics"],
                failures=best["failures"],
                target_failures=best["target_failures"],
            ))
        priors.sort(key=lambda item: (item.association_score, item.occurrences), reverse=True)
        promising = priors[:max(1, limit - 3)]
        selected_names = {item.component_name for item in promising}
        risky = [
            item for item in sorted(priors, key=lambda item: item.association_score)
            if item.component_name not in selected_names
        ][:3]
        return promising + risky

    def _compute_candidate_priors(
        self,
        *,
        active_skill: str,
        target_subsets: list[str],
        limit: int,
    ) -> list[CandidatePriorEntry]:
        rows = self.conn.execute(
            "SELECT ee.action_type, ee.active_skill, ee.generalization_risk, ee.support_refs_json, "
            "ee.counter_refs_json, ese.subset, ese.improvement_delta, ese.ratio, "
            "ese.before_mae, ese.after_mae "
            "FROM edit_effects ee JOIN edit_subset_effects ese ON ee.effect_id = ese.effect_id "
            "ORDER BY ee.created_at DESC LIMIT 400"
        ).fetchall()
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row[0], row[1] or "")
            slot = grouped.setdefault(key, {"n": 0, "delta": 0.0, "risk": 0.0, "refs": []})
            subset = str(row[5])
            delta = float(row[6] or 0.0)
            ratio = float(row[7] or 1.0)
            before_mae = float(row[8] or 0.0)
            after_mae = float(row[9] or 0.0)
            if after_mae <= 0.0 and before_mae > 0.0:
                delta = before_mae - 999.0
                ratio = 999.0 / max(before_mae, 1e-6)
            if target_subsets and subset not in target_subsets:
                if delta < -0.25 or ratio > 1.05:
                    slot["risk"] += min(1.0, max(ratio - 1.0, abs(delta) / 5.0))
                continue
            slot["n"] += 1
            slot["delta"] += delta
            slot["risk"] += float(row[2] or 0.0)
            slot["refs"].extend(json.loads(row[3] or "[]") + json.loads(row[4] or "[]"))
        priors: list[CandidatePriorEntry] = []
        for (action_type, skill), payload in grouped.items():
            if payload["n"] <= 0 and payload["risk"] <= 0:
                continue
            n = max(1, payload["n"])
            avg_delta = payload["delta"] / n
            avg_risk = payload["risk"] / n
            prior_score = max(-1.0, min(1.0, avg_delta / 5.0 - 0.25 * avg_risk))
            if n < 3 and prior_score > 0:
                prior_score = 0.0
            elif n < 3 and prior_score < 0:
                prior_score *= 0.5
            if active_skill and skill and active_skill != skill:
                prior_score *= 0.7
            priors.append(CandidatePriorEntry(
                key=f"{','.join(target_subsets) or 'any'}:{action_type}:{skill or 'any'}",
                skill=skill,
                action_type=action_type,
                prior_score=prior_score,
                risk_score=max(0.0, min(1.0, avg_risk)),
                evidence_refs=list(dict.fromkeys(payload["refs"]))[:8],
                summary=f"{action_type}/{skill or 'any'} on {','.join(target_subsets) or 'any'}: avg_delta={avg_delta:+.3f}, risk={avg_risk:.2f}, n={n}",
            ))
        priors.sort(key=lambda item: item.prior_score - item.risk_score, reverse=True)
        return priors[:limit]

    def _query_action_outcomes(
        self,
        *,
        target_subsets: list[str],
        active_skill: str,
        limit: int,
    ) -> list[ActionOutcomeEntry]:
        """Summarize concrete edit-pattern outcomes from parent->child evaluations."""
        rows = self.conn.execute(
            "SELECT ee.effect_id, ee.action_type, ee.active_skill, ee.action_detail_json, "
            "ee.component_families_json, ee.subset_effects_json, ee.generalization_risk, "
            "p.xc_json, c.xc_json "
            "FROM edit_effects ee "
            "JOIN functional_evaluations p ON ee.parent_eval_id = p.eval_id "
            "JOIN functional_evaluations c ON ee.child_eval_id = c.eval_id "
            "ORDER BY ee.created_at DESC LIMIT 500"
        ).fetchall()
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            action_type = str(row[1] or "")
            skill = str(row[2] or "")
            if active_skill and skill and skill != active_skill:
                # Keep cross-skill evidence for hard risks, but discount it below.
                skill_match = False
            else:
                skill_match = True
            try:
                action_detail = json.loads(row[3] or "{}")
                families = json.loads(row[4] or "[]")
                subset_effects = json.loads(row[5] or "{}")
                parent = {"xc_json": row[7]}
                child = {"xc_json": row[8]}
            except (TypeError, json.JSONDecodeError):
                continue
            pattern_key = self._action_pattern_key(
                parent=parent,
                child=child,
                action_type=action_type,
                action_detail=action_detail,
                component_families=families if isinstance(families, list) else [],
            )
            family = families[0] if isinstance(families, list) and families else ""
            outcome = self._summarize_action_outcome(
                pattern_key=pattern_key,
                subset_effects=subset_effects if isinstance(subset_effects, dict) else {},
                action_type=action_type,
                active_skill=skill,
                component_family=str(family),
                target_subsets=target_subsets,
            )
            slot = grouped.setdefault(
                pattern_key,
                {
                    "entry": outcome,
                    "target_delta": 0.0,
                    "risk": 0.0,
                    "worst": 1.0,
                    "n": 0,
                    "supports": {},
                    "regressions": {},
                    "skill_match": False,
                },
            )
            slot["target_delta"] += outcome.target_delta
            slot["risk"] += outcome.risk_score
            slot["worst"] = max(float(slot["worst"]), outcome.worst_regression_ratio)
            slot["n"] += 1
            slot["skill_match"] = bool(slot["skill_match"] or skill_match)
            for subset, delta in outcome.supports.items():
                slot["supports"][subset] = slot["supports"].get(subset, 0.0) + delta
            for subset, ratio in outcome.regressions.items():
                slot["regressions"][subset] = max(slot["regressions"].get(subset, 1.0), ratio)

        entries: list[ActionOutcomeEntry] = []
        for pattern_key, payload in grouped.items():
            n = max(1, int(payload["n"]))
            avg_target_delta = float(payload["target_delta"]) / n
            avg_risk = float(payload["risk"]) / n
            worst = float(payload["worst"])
            verdict = "risky" if avg_risk > 0.10 or worst > 1.10 else "promising"
            if verdict == "promising" and avg_target_delta <= 0.10:
                continue
            entry = payload["entry"]
            confidence = min(0.95, 0.45 + 0.10 * n + (0.15 if payload["skill_match"] else 0.0))
            entries.append(ActionOutcomeEntry(
                pattern_key=pattern_key,
                verdict=verdict,
                action_type=entry.action_type,
                active_skill=entry.active_skill,
                component_family=entry.component_family,
                target_delta=avg_target_delta,
                risk_score=max(0.0, min(1.0, avg_risk)),
                worst_regression_ratio=worst,
                occurrences=n,
                supports={k: v / n for k, v in sorted(payload["supports"].items())},
                regressions=dict(sorted(payload["regressions"].items())),
                recommendation=self._action_outcome_recommendation(
                    pattern_key=pattern_key,
                    verdict=verdict,
                    regressions=payload["regressions"],
                ),
                confidence=confidence,
            ))
        risky_entries = sorted(
            [item for item in entries if item.verdict == "risky"],
            key=lambda item: (-item.risk_score, -item.confidence, item.pattern_key),
        )
        promising_entries = sorted(
            [item for item in entries if item.verdict == "promising"],
            key=lambda item: (-item.target_delta, item.risk_score, item.pattern_key),
        )
        return risky_entries[:limit] + promising_entries[:limit]

    def _summarize_action_outcome(
        self,
        *,
        pattern_key: str,
        subset_effects: dict[str, dict[str, Any]],
        action_type: str,
        active_skill: str,
        component_family: str,
        target_subsets: list[str],
    ) -> ActionOutcomeEntry:
        targets = set(target_subsets)
        if not targets:
            targets = {
                subset for subset, payload in subset_effects.items()
                if isinstance(payload, dict) and payload.get("panel") == "search"
            }
        supports: dict[str, float] = {}
        regressions: dict[str, float] = {}
        target_delta = 0.0
        risk = 0.0
        worst_ratio = 1.0
        for subset, payload in subset_effects.items():
            if not isinstance(payload, dict):
                continue
            delta = float(payload.get("improvement_delta", 0.0) or 0.0)
            ratio = float(payload.get("ratio", 1.0) or 1.0)
            after_mae = float(payload.get("after_mae", 0.0) or 0.0)
            before_mae = float(payload.get("before_mae", 0.0) or 0.0)
            if after_mae <= 0.0 and before_mae > 0.0:
                after_mae = 999.0
                delta = before_mae - after_mae
                ratio = after_mae / max(before_mae, 1e-6)
            if subset in targets:
                target_delta += delta
                if delta > 0:
                    supports[subset] = delta
                elif delta < -0.25 or ratio > 1.05:
                    regressions[subset] = ratio
                    risk += min(1.0, max(ratio - 1.0, abs(delta) / 5.0))
                    worst_ratio = max(worst_ratio, ratio)
            elif delta < -0.25 or ratio > 1.05:
                regressions[subset] = ratio
                risk += min(1.0, max(ratio - 1.0, abs(delta) / 5.0))
                worst_ratio = max(worst_ratio, ratio)
        risk = max(0.0, min(1.0, risk))
        verdict = "risky" if risk > 0.10 or worst_ratio > 1.10 else "promising"
        return ActionOutcomeEntry(
            pattern_key=pattern_key,
            verdict=verdict,
            action_type=action_type,
            active_skill=active_skill,
            component_family=component_family,
            target_delta=target_delta,
            risk_score=risk,
            worst_regression_ratio=worst_ratio,
            supports=supports,
            regressions=regressions,
            recommendation=self._action_outcome_recommendation(
                pattern_key=pattern_key,
                verdict=verdict,
                regressions=regressions,
            ),
        )

    def _action_pattern_key(
        self,
        *,
        parent: dict[str, Any],
        child: dict[str, Any],
        action_type: str,
        action_detail: dict[str, Any],
        component_families: list[str],
    ) -> str:
        action_kind = action_type.replace("_component", "").replace("_constraint", "")
        detail = {k: v for k, v in action_detail.items() if k != "xc_dict"}
        if action_kind == "add_descriptor":
            descriptor = detail.get("descriptor", detail)
            if isinstance(descriptor, dict):
                template = descriptor.get("template") or descriptor.get("family") or "descriptor"
                params = descriptor.get("params", {}) if isinstance(descriptor.get("params"), dict) else descriptor
                scale = params.get("scale")
                direction = "positive" if isinstance(scale, (int, float)) and float(scale) > 0 else "negative"
                return f"add_descriptor.{template}.{direction}"
            return "add_descriptor.descriptor"
        if action_kind == "rewire":
            parent_xc = self._safe_xc_from_json(parent.get("xc_json", ""))
            child_xc = self._safe_xc_from_json(child.get("xc_json", ""))
            patterns: list[str] = []
            if parent_xc and child_xc:
                for key in ("alpha", "beta", "omega"):
                    before = float(getattr(parent_xc.hf_exchange, key, 0.0))
                    after = float(getattr(child_xc.hf_exchange, key, 0.0))
                    if abs(after - before) > 1e-8:
                        direction = "increase" if after > before else "decrease"
                        patterns.append(f"{key}.{direction}")
            for key, value in detail.items():
                if key in {"alpha", "beta", "omega"} and not any(item.startswith(f"{key}.") for item in patterns):
                    direction = "set"
                    if isinstance(value, (int, float)):
                        direction = "increase" if float(value) > 0 else "decrease"
                    patterns.append(f"{key}.{direction}")
                elif isinstance(key, str) and key.startswith("mixing:"):
                    patterns.append(f"mixing.{key.split(':', 1)[1]}")
                elif isinstance(key, str) and "." in key:
                    comp, param = key.split(".", 1)
                    patterns.append(f"{comp}.{param}")
            return "rewire." + "+".join(patterns or ["parameter"])
        if action_kind == "replace":
            old_name = self._component_name_from_action(detail.get("old_name") or detail.get("old_component"))
            new_name = self._component_name_from_action(detail.get("new_name") or detail.get("new_component"))
            old_name = old_name or "unknown"
            new_name = new_name or "unknown"
            return f"replace.{old_name}.to.{new_name}"
        if action_kind == "add":
            name = self._component_name_from_action(detail.get("name") or detail.get("component") or detail.get("type"))
            name = name or "component"
            return f"add.{name}"
        if action_kind == "remove":
            name = self._component_name_from_action(detail.get("name") or detail.get("component"))
            return f"remove.{name or detail.get('index') or 'component'}"
        if action_kind == "enforce":
            return f"enforce.{detail.get('constraint') or 'constraint'}"
        family = component_families[0] if component_families else "unknown"
        return f"{action_type}.{family}"

    def _component_name_from_action(self, value: Any) -> str:
        if isinstance(value, dict):
            for key in ("name", "component", "id"):
                if isinstance(value.get(key), str) and value[key]:
                    return value[key]
            return ""
        return str(value) if value is not None else ""

    def _safe_xc_from_json(self, raw: str) -> XCFunctional | None:
        try:
            payload = json.loads(raw or "{}")
            if isinstance(payload, dict) and "components" in payload:
                return XCFunctional.from_dict(payload)
        except Exception:
            return None
        return None

    def _action_outcome_recommendation(
        self,
        *,
        pattern_key: str,
        verdict: str,
        regressions: dict[str, Any],
    ) -> str:
        if verdict == "risky":
            if pattern_key.startswith("rewire.alpha.increase"):
                return "Avoid further global-alpha increases here; prefer range separation or a correlation-slot edit."
            if regressions:
                return "Do not repeat this edit without a guardrail for " + ", ".join(sorted(regressions)[:3]) + "."
            return "Treat this edit pattern as high risk in similar contexts."
        return "Prefer this pattern when it matches the current ablation evidence and constraints."

    def recent_evidence_refs(self, limit: int = 8) -> list[str]:
        refs: list[str] = []
        rows = self.conn.execute(
            "SELECT component, metric, result FROM ablation_evidence "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        refs.extend(f"ablation:{component}:{metric}:{result}" for component, metric, result in rows)
        rows = self.conn.execute(
            "SELECT key FROM knowledge ORDER BY created_at DESC LIMIT ?",
            (max(0, limit - len(refs)),),
        ).fetchall()
        refs.extend(f"knowledge:{key}" for (key,) in rows)
        return refs[:limit]

    def evidence_refs_for_defects(self, defect_tags: list[str], limit: int = 8) -> list[str]:
        """Return evidence biased toward the current failure mode, not merely recency."""
        refs: list[str] = []
        family_patterns: list[str] = []
        if "self_interaction_error" in defect_tags:
            family_patterns.extend(["hf_exchange.%", "exchange.%"])
        if "barrier_underestimation" in defect_tags:
            family_patterns.extend(["hf_exchange.%", "exchange.%"])
        if "noncovalent_error" in defect_tags:
            family_patterns.extend(["correlation.dispersion", "correlation.nlc"])
        if "overbinding" in defect_tags:
            family_patterns.extend(["exchange.%", "correlation.%"])

        for pattern in family_patterns:
            rows = self.conn.execute(
                "SELECT component, metric, result FROM ablation_evidence "
                "WHERE component_family LIKE ? ORDER BY confidence DESC, created_at DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
            refs.extend(
                f"ablation:{component}:{metric}:{result}" for component, metric, result in rows
            )
            if len(refs) >= limit:
                break

        if len(refs) < limit:
            refs.extend(self.recent_evidence_refs(limit - len(refs)))
        return list(dict.fromkeys(refs))[:limit]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def format_bundle_for_prompt(self, bundle: KnowledgeBundle) -> str:
        return format_knowledge_bundle(bundle)

    # ------------------------------------------------------------------
    # Knowledge aging and cleanup
    # ------------------------------------------------------------------

    def age_entries(self, max_age_days: int = 30) -> int:
        """Remove low-confidence, old entries that aren't from seed/literature sources.

        Keeps entries with source='literature' or source='seed' regardless of age.
        Removes entries with confidence < 0.3 that are older than max_age_days.

        Returns number of entries removed.
        """
        cutoff = datetime.now().timestamp() - max_age_days * 86400
        cutoff_str = datetime.fromtimestamp(cutoff).isoformat()
        cur = self.conn.execute(
            "DELETE FROM knowledge "
            "WHERE confidence < 0.3 "
            "AND source NOT IN ('literature', 'seed', 'mass_benchmark') "
            "AND created_at < ?",
            (cutoff_str,),
        )
        self.conn.commit()
        removed = cur.rowcount
        if removed > 0:
            logger.info(f"Aged out {removed} low-confidence knowledge entries older than {max_age_days} days")
        return removed

    def boost_entry(self, key: str, delta: float = 0.1) -> bool:
        """Increase confidence of a knowledge entry when it leads to a successful proposal.

        Returns True if entry was found and updated.
        """
        row = self.conn.execute(
            "SELECT id, confidence FROM knowledge WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return False
        new_confidence = min(row[1] + delta, 1.0)
        self.conn.execute(
            "UPDATE knowledge SET confidence = ? WHERE key = ?", (new_confidence, key)
        )
        self.conn.commit()
        return True

    def decay_entry(self, key: str, delta: float = 0.05) -> bool:
        """Decrease confidence of a knowledge entry when it leads to a failed proposal.

        Returns True if entry was found and updated.
        """
        row = self.conn.execute(
            "SELECT id, confidence FROM knowledge WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return False
        new_confidence = max(row[1] - delta, 0.0)
        self.conn.execute(
            "UPDATE knowledge SET confidence = ? WHERE key = ?", (new_confidence, key)
        )
        self.conn.commit()
        return True

    def _stable_id(self, prefix: str, *parts: object) -> str:
        payload = json.dumps([str(part) for part in parts], sort_keys=True)
        return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"

    def _normalize_key(self, *parts: object) -> str:
        raw = ".".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())
        normalized = "".join(ch if ch.isalnum() else "_" for ch in raw)
        return "_".join(part for part in normalized.split("_") if part)[:160] or "unknown"

    def _result_to_metrics(self, result: BenchmarkResult | None) -> dict[str, dict[str, float | int]]:
        if result is None:
            return {}
        return {
            name: {
                "mae_kcal": sr.mae_kcal if sr.n_reactions > 0 and sr.mae_kcal > 0 else 999.0,
                "rms_kcal": sr.rms_kcal if sr.n_reactions > 0 and sr.rms_kcal > 0 else 999.0,
                "max_error_kcal": sr.max_error_kcal if sr.n_reactions > 0 and sr.max_error_kcal > 0 else 999.0,
                "n_reactions": sr.n_reactions,
                "n_failures": sr.n_failures,
                "valid_measurement": int(sr.n_reactions > 0 and sr.mae_kcal > 0),
            }
            for name, sr in sorted(result.subsets.items())
        }

    def _valid_overall_mae(self, result: BenchmarkResult | None) -> float:
        if result is None:
            return 999.0
        if result.overall_mae <= 0:
            return 999.0
        if sum(sr.n_reactions for sr in result.subsets.values()) <= 0:
            return 999.0
        return result.overall_mae

    def _benchmark_protocol_hash(
        self,
        result: BenchmarkResult,
        validation_result: BenchmarkResult | None,
        benchmark_context: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "search_subsets": sorted(result.subsets),
            "validation_subsets": sorted(validation_result.subsets) if validation_result else [],
            "context": benchmark_context or {},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def _load_functional_eval(self, eval_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT eval_id, structural_signature, functional_name, source, xc_json, xc_code, "
            "component_families_json, hf_exchange_json, subset_metrics_json, "
            "benchmark_protocol_hash, benchmark_context_json, status, failure_reason, defect_tags_json "
            "FROM functional_evaluations WHERE eval_id=?",
            (eval_id,),
        ).fetchone()
        if not row:
            return None
        keys = [
            "eval_id", "structural_signature", "functional_name", "source", "xc_json", "xc_code",
            "component_families_json", "hf_exchange_json", "subset_metrics_json",
            "benchmark_protocol_hash", "benchmark_context_json", "status", "failure_reason", "defect_tags_json",
        ]
        return dict(zip(keys, row))

    def _flatten_subset_metrics(self, metrics_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        flat: dict[str, dict[str, Any]] = {}
        for panel in ("search", "validation"):
            panel_metrics = metrics_payload.get(panel, {})
            if not isinstance(panel_metrics, dict):
                continue
            for subset, payload in panel_metrics.items():
                if isinstance(payload, dict) and subset not in flat:
                    flat[subset] = {**payload, "panel": panel}
        return flat

    def _compute_subset_effects(
        self,
        parent: dict[str, Any],
        child: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], float, list[str], list[str]]:
        parent_metrics = self._flatten_subset_metrics(json.loads(parent["subset_metrics_json"] or "{}"))
        child_metrics = self._flatten_subset_metrics(json.loads(child["subset_metrics_json"] or "{}"))
        effects: dict[str, dict[str, Any]] = {}
        risk = 0.0
        support_refs: list[str] = []
        counter_refs: list[str] = []
        for subset in sorted(set(parent_metrics) & set(child_metrics)):
            before_valid = bool(parent_metrics[subset].get("valid_measurement", 1))
            after_valid = bool(child_metrics[subset].get("valid_measurement", 1))
            if not before_valid:
                continue
            before = float(parent_metrics[subset].get("mae_kcal", 0.0) or 0.0)
            after = float(child_metrics[subset].get("mae_kcal", 0.0) or 0.0)
            if before <= 0:
                continue
            if not after_valid:
                after = 999.0
            delta = before - after
            ratio = after / max(before, 1e-6)
            effects[subset] = {
                "before_mae": before,
                "after_mae": after,
                "improvement_delta": delta,
                "ratio": ratio,
                "panel": child_metrics[subset].get("panel", parent_metrics[subset].get("panel", "")),
            }
            ref = f"effect:{child['eval_id']}:{subset}"
            if delta >= 0:
                support_refs.append(ref)
            else:
                counter_refs.append(ref)
            if ratio > 1.05:
                risk += min(1.0, ratio - 1.0)
        risk = max(0.0, min(1.0, risk))
        return effects, risk, support_refs, counter_refs

    def _record_edit_subset_effects(
        self,
        effect_id: str,
        subset_effects: dict[str, dict[str, Any]],
        target_subsets: list[str],
    ) -> None:
        now = datetime.now().isoformat()
        targets = set(target_subsets)
        for subset, payload in subset_effects.items():
            delta = float(payload.get("improvement_delta", 0.0) or 0.0)
            ratio = float(payload.get("ratio", 1.0) or 1.0)
            self.conn.execute(
                "INSERT INTO edit_subset_effects "
                "(effect_id, subset, panel, before_mae, after_mae, improvement_delta, ratio, "
                "is_target, is_regression, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(effect_id, subset) DO UPDATE SET "
                "panel=excluded.panel, before_mae=excluded.before_mae, after_mae=excluded.after_mae, "
                "improvement_delta=excluded.improvement_delta, ratio=excluded.ratio, "
                "is_target=excluded.is_target, is_regression=excluded.is_regression",
                (
                    effect_id,
                    subset,
                    payload.get("panel", ""),
                    float(payload.get("before_mae", 0.0) or 0.0),
                    float(payload.get("after_mae", 0.0) or 0.0),
                    delta,
                    ratio,
                    int(subset in targets),
                    int(delta < -0.25 or ratio > 1.05),
                    now,
                ),
            )
        self.conn.commit()

    def _record_risk_patterns_for_effect(
        self,
        *,
        effect_id: str,
        subset_effects: dict[str, dict[str, Any]],
        action_type: str,
        active_skill: str,
        component_families: list[str],
        target_subsets: list[str],
    ) -> None:
        del effect_id
        targets = target_subsets or [
            subset for subset, payload in subset_effects.items()
            if payload.get("panel") == "search"
        ]
        component_family = component_families[0] if component_families else ""
        for target in targets:
            for subset, payload in subset_effects.items():
                if subset == target:
                    continue
                delta = float(payload.get("improvement_delta", 0.0) or 0.0)
                ratio = float(payload.get("ratio", 1.0) or 1.0)
                if payload.get("panel") == "validation" and (delta < -0.25 or ratio > 1.05):
                    self.record_risk_pattern(
                        target_subset=target,
                        risk_subset=subset,
                        symptom="validation_regression",
                        action_type=action_type,
                        active_skill=active_skill,
                        component_family=component_family,
                        ratio=ratio,
                    )

    def _defects_from_effects(self, expected_effects: dict[str, float]) -> list[str]:
        defects: list[str] = []
        if expected_effects.get("SIE4x4", 0.0) > 0:
            defects.append("self_interaction_error")
        if expected_effects.get("BH76", 0.0) > 0:
            defects.append("barrier_underestimation")
        if expected_effects.get("S22", 0.0) > 0 or expected_effects.get("S66", 0.0) > 0:
            defects.append("noncovalent_error")
        if expected_effects.get("G21IP", 0.0) < 0 or expected_effects.get("ISO34", 0.0) < 0:
            defects.append("validation_regression")
        return list(dict.fromkeys(defects))

    def _defects_from_target(self, target_subset: str, target_delta: float, risk_score: float) -> list[str]:
        defects = self._target_defects_for_subset(target_subset) if target_delta > 0 else []
        if risk_score > 0:
            defects.append("validation_regression")
        return list(dict.fromkeys(defects))

    def _target_defects_for_subset(self, subset: str) -> list[str]:
        if subset == "SIE4x4":
            return ["self_interaction_error"]
        if subset == "BH76":
            return ["barrier_underestimation"]
        if subset in {"S22", "S66"}:
            return ["noncovalent_error"]
        if subset in {"G21IP", "G21EA"}:
            return ["ionization_or_affinity_error"]
        if subset == "ISO34":
            return ["isomerization_error"]
        return []

    def _target_subsets_from_defects(self, defect_tags: list[str]) -> list[str]:
        targets: list[str] = []
        if "self_interaction_error" in defect_tags:
            targets.append("SIE4x4")
        if "barrier_underestimation" in defect_tags:
            targets.append("BH76")
        if "validation_regression" in defect_tags:
            targets.extend(["G21IP", "ISO34", "S22", "S66"])
        if "noncovalent_error" in defect_tags:
            targets.extend(["S22", "S66"])
        return list(dict.fromkeys(targets))

    def _default_repair_hint(self, symptom: str) -> str:
        lowered = symptom.lower()
        if "exchange sum" in lowered or "alpha+beta" in lowered:
            return "Renormalize exchange mixing so DFT exchange + alpha <= 1 and alpha + beta <= 1."
        if "evaluation_failure" in lowered or "all-failed" in lowered:
            return "Prefer a simpler libxc-backed edit and avoid unsupported custom/RSH combinations."
        if "validation_regression" in lowered:
            return "Keep the useful search-subset mechanism but add a guardrail for the regressed validation subset."
        return "Preserve the intended mechanism while removing the failing constraint."

    def _subset_from_payload(self, subset: str, payload: dict[str, Any]) -> SubsetResult:
        return SubsetResult(
            subset=subset,
            mae_kcal=float(payload.get("mae_kcal", 0.0)),
            rms_kcal=float(payload.get("rms_kcal", payload.get("mae_kcal", 0.0)) or 0.0),
            max_error_kcal=float(payload.get("max_error_kcal", payload.get("mae_kcal", 0.0)) or 0.0),
            n_reactions=int(payload.get("n_reactions", 0) or 0),
            n_failures=int(payload.get("n_failures", 0) or 0),
            walltime_s=float(payload.get("walltime_s", 0.0) or 0.0),
            per_reaction_errors=[],
            failure_details=list(payload.get("failure_details", []) or []),
        )

    def _xc_from_prerun_payload(
        self,
        name: str,
        raw: dict[str, Any],
        decomposition: dict[str, Any] | None = None,
    ) -> XCFunctional | None:
        meta = raw.get("__meta__") if isinstance(raw.get("__meta__"), dict) else {}
        xc_dict = raw.get("xc_dict") or raw.get("functional") or raw.get("xc") or meta.get("xc_dict")
        if isinstance(xc_dict, dict) and "components" in xc_dict:
            try:
                return XCFunctional.from_dict(xc_dict)
            except Exception:
                pass
        decomposed = self._xc_from_libxc_decomposition(name, decomposition)
        if decomposed is not None:
            return decomposed
        try:
            return XCFunctional.from_known(name)
        except Exception:
            pass
        try:
            catalogue = XCFunctional.enumerate_libxc()
            info = catalogue.get(name) or catalogue.get(name.upper())
        except Exception:
            info = None
        if not info:
            return None
        role = str(info.get("role", "unknown"))
        xctype = str(info.get("xc_type", "GGA"))
        components: list[ComponentSpec]
        if role == "exchange_correlation":
            components = [ComponentSpec(name=name, xctype=xctype, role="exchange_correlation")]
        elif role == "exchange":
            components = [
                ComponentSpec(name=name, xctype=xctype, role="exchange"),
                ComponentSpec(name="GGA_C_PBE", xctype="GGA", role="correlation"),
            ]
        elif role == "correlation":
            components = [
                ComponentSpec(name="GGA_X_PBE", xctype="GGA", role="exchange"),
                ComponentSpec(name=name, xctype=xctype, role="correlation"),
            ]
        else:
            return None
        rsh = info.get("rsh_coeff", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        alpha = float(info.get("hybrid_coeff", 0.0) or 0.0)
        beta = 0.0
        omega = 0.0
        if len(rsh) >= 3 and float(rsh[0] or 0.0) > 0:
            omega = float(rsh[0] or 0.0)
            alpha = float(rsh[2] or alpha)
            beta = max(0.0, float(rsh[1] or 0.0) - alpha)
        return XCFunctional(name=name, components=components, hf_exchange=HFExchange(alpha=alpha, beta=beta, omega=omega))

    def _xc_from_libxc_decomposition(
        self,
        name: str,
        decomposition: dict[str, Any] | None,
    ) -> XCFunctional | None:
        if not isinstance(decomposition, dict) or decomposition.get("status") != "parsed":
            return None
        id_to_name = {libxc_id: component_name for component_name, libxc_id in NAME_TO_ID.items()}
        components: list[ComponentSpec] = []
        seen: set[str] = set()
        for payload in decomposition.get("components", []):
            component_name = id_to_name.get(int(payload.get("libxc_id", -1) or -1))
            role = PURE_COMPONENT_ROLES.get(component_name or "")
            if not component_name or not role or component_name in seen:
                continue
            seen.add(component_name)
            components.append(ComponentSpec(
                name=component_name,
                xctype=str(payload.get("xc_type", "GGA") or "GGA"),
                role=role,
                mixing_factor=float(payload.get("mixing_factor", 1.0) or 1.0),
            ))
        if not components:
            return None
        rsh = decomposition.get("rsh_coeff", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        alpha = float(decomposition.get("hybrid_coeff", 0.0) or 0.0)
        beta = 0.0
        omega = 0.0
        if len(rsh) >= 3 and float(rsh[0] or 0.0) > 0:
            omega = float(rsh[0] or 0.0)
            alpha = float(rsh[2] or alpha)
            beta = max(0.0, float(rsh[1] or 0.0) - alpha)
        return XCFunctional(
            name=name,
            components=components,
            hf_exchange=HFExchange(alpha=alpha, beta=beta, omega=omega),
        )

    def _defect_tags_from_result(self, result: BenchmarkResult) -> list[str]:
        tags: list[str] = []
        sie = result.subsets.get("SIE4x4")
        if sie and sie.mae_kcal > 10:
            tags.append("self_interaction_error")
        bh = result.subsets.get("BH76")
        if bh and bh.mae_kcal > 5:
            tags.append("barrier_underestimation")
        if any(sr.n_reactions == 0 and sr.n_failures > 0 for sr in result.subsets.values()):
            tags.append("evaluation_failure")
        return tags

    def close(self):
        self.conn.close()


def format_knowledge_bundle(bundle: KnowledgeBundle) -> str:
    """Format a KnowledgeBundle into a string suitable for LLM prompt injection.

    Standalone function so prompts.py can call it without a KnowledgeEngine instance.
    """
    lines = []

    if bundle.mechanism_cards:
        lines.append("\n## Target Mechanism Cards")
        for card in bundle.mechanism_cards[:5]:
            effects = ", ".join(
                f"{subset}:{delta:+.2f}"
                for subset, delta in sorted(card.expected_effects.items())
            )
            risks = f" | risks={', '.join(card.risk_subsets)}" if card.risk_subsets else ""
            guards = f" | guardrails={'; '.join(card.guardrails[:3])}" if card.guardrails else ""
            lines.append(
                f"- [{card.mechanism_key}] {card.statement} "
                f"(target={card.target_subset or 'n/a'}, target_delta={card.target_delta:+.2f}, "
                f"risk={card.risk_score:.2f}, n={card.support_count}, conf={card.confidence:.2f}, "
                f"status={card.status}, effects={effects or 'n/a'}{risks}{guards})"
            )

    if bundle.risk_patterns:
        lines.append("\n## Validation Risk Guards")
        for pattern in bundle.risk_patterns[:5]:
            lines.append(
                f"- [{pattern.pattern_key}] targeting {pattern.target_subset} has risk on "
                f"{pattern.risk_subset}: {pattern.symptom} "
                f"(action={pattern.action_type or 'any'}, skill={pattern.active_skill or 'any'}, "
                f"family={pattern.component_family or 'any'}, ratio~{pattern.avg_ratio:.2f}, "
                f"seen={pattern.occurrences}, conf={pattern.confidence:.2f})"
            )

    if bundle.action_outcomes:
        risky_actions = [item for item in bundle.action_outcomes if item.verdict == "risky"]
        promising_actions = [item for item in bundle.action_outcomes if item.verdict == "promising"]
        if risky_actions:
            lines.append("\n## Action Outcome Memory - Do Not Repeat Blindly")
            for item in risky_actions[:5]:
                regressions = ", ".join(
                    f"{subset}x{ratio:.2f}"
                    for subset, ratio in sorted(item.regressions.items())
                ) or "n/a"
                supports = ", ".join(
                    f"{subset}:{delta:+.2f}"
                    for subset, delta in sorted(item.supports.items())
                ) or "n/a"
                lines.append(
                    f"- [{item.pattern_key}] action={item.action_type}, skill={item.active_skill or 'any'}, "
                    f"family={item.component_family or 'any'}, target_delta={item.target_delta:+.2f}, "
                    f"risk={item.risk_score:.2f}, worst_ratio={item.worst_regression_ratio:.2f}, "
                    f"seen={item.occurrences}, conf={item.confidence:.2f}, supports={supports}, "
                    f"regressions={regressions}; {item.recommendation}"
                )
        if promising_actions:
            lines.append("\n## Action Outcome Memory - Promising Patterns")
            for item in promising_actions[:5]:
                supports = ", ".join(
                    f"{subset}:{delta:+.2f}"
                    for subset, delta in sorted(item.supports.items())
                ) or "n/a"
                lines.append(
                    f"- [{item.pattern_key}] action={item.action_type}, skill={item.active_skill or 'any'}, "
                    f"target_delta={item.target_delta:+.2f}, risk={item.risk_score:.2f}, "
                    f"seen={item.occurrences}, conf={item.confidence:.2f}, supports={supports}; "
                    f"{item.recommendation}"
                )

    if bundle.failure_patterns:
        lines.append("\n## Known Failure Guards")
        for pattern in bundle.failure_patterns[:5]:
            block = "hard" if pattern.hard_block else "soft"
            lines.append(
                f"- [{pattern.pattern_key}] {block} guard: {pattern.symptom}; "
                f"repair={pattern.repair_hint} (seen={pattern.occurrences}, conf={pattern.confidence:.2f})"
            )

    if bundle.libxc_anchors:
        target_anchors = [item for item in bundle.libxc_anchors if item.anchor_type == "target_anchor"]
        neighbors = [item for item in bundle.libxc_anchors if item.anchor_type == "nearest_neighbor"]
        if target_anchors:
            lines.append("\n## Libxc Target Anchors")
        for anchor in target_anchors[:5]:
            metrics = []
            for subset, payload in sorted(anchor.subset_metrics.items()):
                if len(metrics) >= 4:
                    break
                metrics.append(f"{subset}={float(payload.get('mae_kcal', 0.0)):.2f}")
            lines.append(
                f"- {anchor.functional_name}: target_score={anchor.target_score:+.2f}, "
                f"validation_penalty={anchor.validation_penalty:.2f}, similarity={anchor.similarity:.2f}, "
                f"families={','.join(anchor.component_families[:4])}, metrics={', '.join(metrics)}"
            )
        if neighbors:
            lines.append("\n## Nearest Known Functionals")
        for anchor in neighbors[:5]:
            metrics = []
            for subset, payload in sorted(anchor.subset_metrics.items()):
                if len(metrics) >= 4:
                    break
                metrics.append(f"{subset}={float(payload.get('mae_kcal', 0.0)):.2f}")
            lines.append(
                f"- {anchor.functional_name}: similarity={anchor.similarity:.2f}, "
                f"families={','.join(anchor.component_families[:4])}, metrics={', '.join(metrics)}"
            )

    if bundle.component_benchmark_priors:
        lines.append("\n## Component Benchmark Priors (Associative, Not Causal)")
        lines.append(
            "These candidate components occur inside previously benchmarked whole functionals. "
            "Use them to choose hypotheses, but do not claim that the component alone caused the result."
        )
        for prior in bundle.component_benchmark_priors[:10]:
            guards = ", ".join(
                f"{subset}={mae:.2f}"
                for subset, mae in sorted(prior.guard_metrics.items())
            ) or "n/a"
            lines.append(
                f"- {prior.component_name} ({prior.role}): best_associated_functional={prior.best_functional}, "
                f"target_mae={prior.target_mae:.2f}, target_gain_vs_current={prior.target_gain:+.2f}, "
                f"association_score={prior.association_score:+.2f}, guards={guards}, "
                f"target_failures={prior.target_failures}, total_failures={prior.failures}, "
                f"observed_in={prior.occurrences} functionals"
            )

    if bundle.candidate_priors:
        lines.append("\n## Candidate Priors")
        for prior in bundle.candidate_priors[:5]:
            lines.append(
                f"- {prior.key}: prior={prior.prior_score:+.2f}, risk={prior.risk_score:.2f}; {prior.summary}"
            )

    if bundle.active_hypotheses:
        lines.append("\n## Active Hypotheses")
        for hyp in bundle.active_hypotheses[:4]:
            lines.append(
                f"- {hyp.hypothesis_id} ({hyp.status}, conf={hyp.confidence:.2f}): {hyp.statement}"
            )

    # Ablation evidence
    if bundle.ablation_evidence:
        lines.append("\n## Ablation Evidence (past component effects)")
        seen = set()
        for entry in bundle.ablation_evidence[:10]:
            key = f"{entry.component}.{entry.metric}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- {entry.component} ({entry.component_family}): "
                f"{entry.result} on {entry.metric}={entry.value:.2f} — {entry.analysis}"
            )

    if bundle.relevant_traces:
        lines.append("\n## Relevant Search Traces")
        for trace in bundle.relevant_traces[:6]:
            semantic = f" | idea={trace.core_contribution}" if trace.core_contribution else ""
            lines.append(
                f"- {trace.name}: action={trace.action_type}, skill={trace.active_skill or 'n/a'}, "
                f"status={trace.status}, mae={trace.search_mae:.2f}{semantic}"
            )

    # Relevant traces
    if bundle.relevant_traces:
        lines.append("\n## Recent Search History")
        seen_sigs = set()
        for entry in bundle.relevant_traces[:8]:
            if entry.signature in seen_sigs:
                continue
            seen_sigs.add(entry.signature)
            lines.append(
                f"- {entry.name} ({entry.action_type}): MAE={entry.search_mae:.2f}, "
                f"status={entry.status}, findings={entry.finding_tags}"
            )

    return "\n".join(lines) if lines else "No prior knowledge available."
