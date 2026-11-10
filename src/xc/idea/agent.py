"""IdeaAgent — instantiate skill-guided XC edits from structured evidence.

Benchmark interpretation happens upstream. This layer consumes the current
research brief, recent evidence, and one selected edit skill, then materializes
a concrete candidate plus the semantic state needed by MCTS and handoff.

Mirrors Xcientist's IdeaAgent: each proposal carries not just the functional
but also core_contribution, rationale, component_explanations, memory_refs,
edit_plan, and skill_metrics — enabling downstream consumers (MCTS, knowledge
engine, result store) to operate on rich semantic state.
"""

from __future__ import annotations

import json
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..xc_functional import XCFunctional
from ..benchmark.result import BenchmarkResult
from ..logging_utils import colorize, log_block, pretty_json_or_text
from .actions import ActionType, Action, apply_action, EditPlan, ComponentEdit, ValidationProtocol
from .knowledge_engine import KnowledgeEngine, KnowledgeBundle, format_knowledge_bundle
from .ablation import AblationReport, format_ablation_report
from .skills import SkillCatalog, EditSkill
from .search_state import ProposalState
from .prompts import build_proposal_prompt, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """One search step in the history."""
    name: str
    mae: float
    status: str  # "accepted" | "rejected" | "failed"
    skill: str
    finding_tags: list[str] = field(default_factory=list)
    core_contribution: str = ""
    rationale: str = ""


class ProposalFailure(RuntimeError):
    """Proposal failed with structured context for search reflection."""

    def __init__(self, message: str, context: dict[str, Any]):
        super().__init__(message)
        self.context = context


class IdeaAgent:
    """Proposes XC functional modifications with rich semantic content."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        target_problem: str = "reduce SIE4x4 MAE while keeping ISO34 ≤ baseline × 1.2",
        knowledge_engine: KnowledgeEngine | None = None,
        exploration_rate: float = 0.2,
        proposal_prior_path: str | Path | None = "knowledge/xc_component_coefficient_analysis_promptsafe.md",
        llm_timeout_s: float = 120.0,
        llm_retry_attempts: int = 4,
        llm_retry_base_delay_s: float = 1.0,
        llm_retry_max_delay_s: float = 12.0,
        llm_retry_backoff: float = 2.0,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.target_problem = target_problem
        self.knowledge = knowledge_engine
        self.exploration_rate = exploration_rate
        self.proposal_prior_path = Path(proposal_prior_path) if proposal_prior_path else None
        self.proposal_prior_text = self._load_proposal_prior_text(self.proposal_prior_path)
        self.llm_timeout_s = max(1.0, float(llm_timeout_s))
        self.llm_retry_attempts = int(llm_retry_attempts)
        self.llm_retry_base_delay_s = max(0.0, float(llm_retry_base_delay_s))
        self.llm_retry_max_delay_s = max(self.llm_retry_base_delay_s, float(llm_retry_max_delay_s))
        self.llm_retry_backoff = max(1.0, float(llm_retry_backoff))
        self.skill_catalog = SkillCatalog()
        self.history: list[HistoryEntry] = []
        self._last_ablation: AblationReport | None = None
        self._proposal_counter: int = 0

    def _load_proposal_prior_text(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            if not path.exists():
                logger.info("Proposal scientific prior not found at %s; skipping prompt injection", path)
                return ""
            text = path.read_text(encoding="utf-8").strip()
            if text:
                logger.info(
                    "Loaded proposal scientific prior from %s chars=%d",
                    path,
                    len(text),
                )
            return text
        except OSError as exc:
            logger.warning("Failed to load proposal scientific prior from %s: %s", path, exc)
            return ""

    def propose(
        self,
        current_best: XCFunctional,
        current_result: BenchmarkResult,
        ablation_report: AblationReport | None = None,
        preferred_skill_override: str | None = None,
        analysis=None,
        immediate_feedback: str | None = None,
    ) -> tuple[XCFunctional, ProposalState]:
        """Propose a candidate XC functional with rich semantic state.

        Returns:
            Tuple of (candidate XCFunctional, ProposalState with semantic content).
            The ProposalState carries core_contribution, rationale, component_explanations,
            edit_plan, skill_metrics, and memory_refs alongside the physical functional.
        """
        proposal_index, proposal_id = self._next_proposal_id()
        context = getattr(self, "_active_proposal_context", {}) or {}
        action_contract = context.get("action_contract") if isinstance(context.get("action_contract"), dict) else None
        logger.info(
            "Proposal %s start: parent=%s rollout=%s depth=%s requested_skill=%s action_family=%s has_result=%s",
            proposal_id,
            current_best.name,
            context.get("rollout_index"),
            context.get("node_depth"),
            preferred_skill_override or "auto",
            context.get("action_family", ""),
            current_result is not None,
        )
        if current_result is None:
            raise ValueError(
                f"Proposal {proposal_id} cannot proceed without benchmark context for parent {current_best.name}"
            )

        selected_skill: EditSkill | None = None
        action: Action | None = None
        semantic_content: dict[str, Any] = {}
        failure_stage = "skill_selection"
        self._last_proposal_debug = {
            "stage": failure_stage,
            "proposal_id": proposal_id,
            "parent": current_best.name,
            "requested_skill": preferred_skill_override or "auto",
            "action_family": context.get("action_family", ""),
            "action_contract": action_contract or {},
        }
        try:
            # 1. Select one action skill; benchmark interpretation already happened upstream.
            if preferred_skill_override and preferred_skill_override in self.skill_catalog.skills:
                selected_skill = self.skill_catalog.get(preferred_skill_override)
            else:
                findings = analysis.analysis_artifact.findings if analysis and analysis.analysis_artifact else []
                selected_skill = self.skill_catalog.select_skills(
                    findings=findings,
                    agenda=analysis.search_agenda if analysis else None,
                    limit=1,
                    xc=current_best,
                )[0]
            logger.info(
                "Proposal %s selected skill=%s current_subsets=%s",
                proposal_id,
                selected_skill.name,
                list(current_result.subsets),
            )

            # 2. Retrieve node-relevant memory for this skill, not just recent trace rows.
            knowledge_bundle = KnowledgeBundle()
            if self.knowledge:
                defect_tags: list[Any] = []
                if analysis and analysis.analysis_artifact:
                    for finding in analysis.analysis_artifact.findings:
                        defect_tags.extend(finding.metadata.get("target_defects", []) or [])
                        metric = str(finding.metadata.get("metric", ""))
                        if "SIE4x4" in metric:
                            defect_tags.append("self_interaction_error")
                        if "BH76" in metric:
                            defect_tags.append("barrier_underestimation")
                clean_defect_tags = self._normalize_string_list(defect_tags, "defect_tags", proposal_id)
                knowledge_bundle = self.knowledge.retrieve_for_node(
                    current_xc=current_best,
                    current_result=current_result,
                    defect_tags=list(dict.fromkeys(clean_defect_tags)),
                    target_subsets=list(current_result.subsets),
                    active_skill=selected_skill.name,
                )
                knowledge_bundle.ablation_report = ablation_report
                logger.info(
                    "Proposal %s knowledge bundle: traces=%d ablations=%d mechanisms=%d priors=%d "
                    "component_benchmark_priors=%d failures=%d risks=%d action_outcomes=%d",
                    proposal_id,
                    len(knowledge_bundle.relevant_traces),
                    len(knowledge_bundle.ablation_evidence),
                    len(knowledge_bundle.mechanism_cards),
                    len(knowledge_bundle.candidate_priors),
                    len(knowledge_bundle.component_benchmark_priors),
                    len(knowledge_bundle.failure_patterns),
                    len(knowledge_bundle.risk_patterns),
                    len(knowledge_bundle.action_outcomes),
                )
                if knowledge_bundle.component_benchmark_priors:
                    logger.info(
                        "Proposal %s component benchmark prior context: %s",
                        proposal_id,
                        "; ".join(
                            f"{item.component_name}:{item.role}:best={item.best_functional}:"
                            f"target_gain={item.target_gain:+.2f}:score={item.association_score:+.2f}:"
                            f"target_failures={item.target_failures}:failures={item.failures}"
                            for item in knowledge_bundle.component_benchmark_priors[:8]
                        ),
                    )
                if knowledge_bundle.action_outcomes:
                    logger.info(
                        "Proposal %s action outcome context: %s",
                        proposal_id,
                        "; ".join(
                            f"{item.pattern_key}:{item.verdict}:skill={item.active_skill or 'any'}:"
                            f"target_delta={item.target_delta:+.2f}:risk={item.risk_score:.2f}"
                            for item in knowledge_bundle.action_outcomes[:5]
                        ),
                    )

            # 3. Build prompt with structured evidence + Skill context
            prompt = build_proposal_prompt(
                current_best=current_best,
                current_result=current_result,
                target_problem=self.target_problem,
                knowledge_bundle=knowledge_bundle,
                scientific_prior_text=self.proposal_prior_text or None,
                preferred_action=selected_skill.name,
                ablation_report=ablation_report,
                selected_skill=selected_skill,
                analysis=analysis,
                immediate_feedback=immediate_feedback,
                search_feedback=context.get("duplicate_feedback") or None,
                action_contract=action_contract,
            )
            logger.info(
                "Proposal %s prompt prepared: chars=%d immediate_feedback=%s scientific_prior=%s prior_chars=%d",
                proposal_id,
                len(prompt),
                bool(immediate_feedback),
                bool(self.proposal_prior_text),
                len(self.proposal_prior_text or ""),
            )

            # 5. LLM instantiation
            failure_stage = "llm_parse"
            action, semantic_content = self._call_llm(
                prompt,
                selected_skill,
                current_best,
                proposal_id=proposal_id,
                action_contract=action_contract,
            )

            # 6. Apply action
            failure_stage = "apply_action"
            candidate = apply_action(current_best, action)
            candidate.name = f"{current_best.name}__{_format_action_suffix(action)}"
            candidate.metadata["parent"] = current_best.name
            candidate.metadata["action"] = action.action_type.value
            candidate.metadata["action_detail"] = action.changes
            candidate.metadata["skill"] = selected_skill.name
            candidate.metadata["action_family"] = context.get("action_family", "")
            candidate.metadata["proposal_id"] = proposal_id
            candidate.metadata["proposal_index"] = proposal_index
            candidate.metadata["rollout_index"] = context.get("rollout_index")

            memory_refs = self._normalize_string_list(
                semantic_content.get("memory_refs", []),
                "memory_refs",
                proposal_id,
            )
            target_findings = self._normalize_string_list(
                semantic_content.get("target_findings", []),
                "target_findings",
                proposal_id,
            )
            component_explanations = self._normalize_string_mapping(
                semantic_content.get("component_explanations", {}),
                "component_explanations",
                proposal_id,
            )

            # 7. Build ProposalState with rich semantic content
            state = ProposalState(
                xc=candidate,
                structural_sig=candidate.structural_signature(),
                core_contribution=str(semantic_content.get("core_contribution", "")),
                rationale=str(semantic_content.get("rationale", action.rationale)),
                memory_refs=memory_refs,
                component_explanations=component_explanations,
                paper_graph_context=str(semantic_content.get("paper_graph_context", "")),
                finding_tags=[],
                active_skill=selected_skill.name,
                action_taken=action.action_type,
                action_detail=action.changes,
                proposal_id=proposal_id,
                proposal_index=proposal_index,
                edit_plan=semantic_content.get("edit_plan"),
                skill_metrics=semantic_content.get("skill_metrics", {}),
                parent_sig=current_best.structural_signature(),
                tags=target_findings,
                root_domains=_infer_root_domains(selected_skill),
            )
            state.skill_metrics["action_family"] = context.get("action_family", "")
            logger.info(
                "Proposal %s built candidate=%s action=%s signature=%s",
                proposal_id,
                candidate.name,
                action.action_type.value,
                state.structural_sig,
            )
            return candidate, state
        except Exception as exc:
            failure_context = self._proposal_failure_context(
                proposal_id=proposal_id,
                current_best=current_best,
                selected_skill=selected_skill,
                action=action,
                stage=failure_stage,
                original_error=exc,
            )
            logger.exception(
                "Proposal %s failed: parent=%s rollout=%s depth=%s skill=%s",
                proposal_id,
                current_best.name,
                context.get("rollout_index"),
                context.get("node_depth"),
                selected_skill.name if selected_skill else preferred_skill_override or "unknown",
            )
            raise ProposalFailure(str(exc), failure_context) from exc

    def _proposal_failure_context(
        self,
        *,
        proposal_id: str,
        current_best: XCFunctional,
        selected_skill: EditSkill | None,
        action: Action | None,
        stage: str,
        original_error: Exception,
    ) -> dict[str, Any]:
        debug = dict(getattr(self, "_last_proposal_debug", {}) or {})
        context = getattr(self, "_active_proposal_context", {}) or {}
        payload: dict[str, Any] = {
            "proposal_id": proposal_id,
            "stage": stage,
            "parent": current_best.name,
            "rollout": context.get("rollout_index"),
            "depth": context.get("node_depth"),
            "skill": selected_skill.name if selected_skill else context.get("preferred_skill", ""),
            "skill_blueprint": [item.value for item in selected_skill.blueprint] if selected_skill else [],
            "error_type": original_error.__class__.__name__,
            "error": str(original_error),
        }
        payload.update(debug)
        if action is not None:
            payload["final_action_type"] = action.action_type.value
            payload["final_changes"] = _compact_failure_value(action.changes)
        return payload

    def record_result(
        self,
        name: str,
        mae: float,
        status: str,
        skill_name: str = "",
        finding_tags: list[str] | None = None,
        core_contribution: str = "",
        rationale: str = "",
    ) -> None:
        """Record a result and update skill priors."""
        self.history.append(HistoryEntry(
            name=name,
            mae=mae,
            status=status,
            skill=skill_name,
            finding_tags=list(finding_tags or []),
            core_contribution=core_contribution,
            rationale=rationale,
        ))
        if len(self.history) > 20:
            self.history = self.history[-20:]

        # Update skill prior
        reward = 1.0 / (1.0 + mae) if mae > 0 else 0.0
        success = status == "accepted"
        if skill_name:
            self.skill_catalog.update(skill_name, reward, success)

    def reflect_search(self, context: dict[str, Any]) -> dict[str, Any]:
        """Summarize search evidence into lessons, not proposals."""
        import openai
        import httpx

        client = openai.OpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url,
            timeout=self.llm_timeout_s,
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        )
        prompt = (
            "You are a scientific search summarizer for an XC-functional optimization run.\n"
            "Your job is to derive durable lessons from the evidence. Do NOT propose a next action.\n"
            "Use recent_evaluations, global_digest, and retrieved_related_experience_by_skill. "
            "Also use recent_rollout_events: these are failed/skipped expansion attempts that never reached benchmark evaluation. "
            "If evidence is weak, say it is uncertain.\n"
            "If previous_reflection is present, update or revise those lessons using the new evidence; "
            "do not blindly preserve old lessons when contradicted.\n"
            "Return STRICT JSON only with this schema:\n"
            "{\n"
            '  "lessons": ["short evidence-backed conclusions"],\n'
            '  "do_not_repeat": [{"pattern": "...", "reason": "...", "unless": "..."}],\n'
            '  "uncertainties": ["open causal questions"]\n'
            "}\n"
            "Do not include next_hypotheses, proposed_actions, or parameter recommendations unless framed as a lesson or prohibition.\n\n"
            "Reflection input context:\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True, indent=2)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You summarize scientific search trajectories. "
                    "You produce factual lessons and prohibitions, never proposals."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        content = self._with_content_retries(
            lambda: self._with_llm_retries(
                lambda: client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    stream=False,
                ),
                openai_module=openai,
                operation="search reflection",
            ).choices[0].message.content
            or "",
            operation="search reflection",
        )
        try:
            payload = self._parse_reflection_response(content)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Search reflection returned invalid JSON; requesting one JSON-only repair pass")
            repair = self._with_llm_retries(
                lambda: client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Repair malformed JSON. Return only one JSON object with keys "
                                "lessons, do_not_repeat, uncertainties. Do not add proposed actions."
                            ),
                        },
                        {"role": "user", "content": content},
                    ],
                    temperature=0.0,
                    stream=False,
                ),
                openai_module=openai,
                operation="search reflection JSON repair",
            )
            content = repair.choices[0].message.content or ""
            payload = self._parse_reflection_response(content)
        return payload

    def _parse_reflection_response(self, content: str) -> dict[str, Any]:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError("reflection response contains no JSON object")
        payload = json.loads(content[json_start:json_end])
        if not isinstance(payload, dict):
            raise ValueError("reflection response must be a JSON object")
        lessons = self._normalize_string_list(payload.get("lessons", []), "reflection.lessons", "reflection")
        uncertainties = self._normalize_string_list(
            payload.get("uncertainties", []),
            "reflection.uncertainties",
            "reflection",
        )
        do_not_repeat_raw = payload.get("do_not_repeat", [])
        do_not_repeat: list[dict[str, str]] = []
        if isinstance(do_not_repeat_raw, list):
            for item in do_not_repeat_raw[:8]:
                if isinstance(item, dict):
                    pattern = str(item.get("pattern", "")).strip()
                    reason = str(item.get("reason", "")).strip()
                    unless = str(item.get("unless", "")).strip()
                else:
                    pattern = str(item).strip()
                    reason = ""
                    unless = ""
                if pattern or reason:
                    do_not_repeat.append(
                        {
                            "pattern": pattern,
                            "reason": reason,
                            "unless": unless,
                        }
                    )
        return {
            "lessons": lessons[:8],
            "do_not_repeat": do_not_repeat[:8],
            "uncertainties": uncertainties[:8],
        }

    def _call_llm(
        self,
        prompt: str,
        skill: EditSkill,
        current_best: XCFunctional,
        *,
        proposal_id: str = "",
        action_contract: dict[str, Any] | None = None,
    ) -> tuple[Action, dict[str, Any]]:
        """Call LLM to instantiate an action within a Skill's blueprint.

        Returns:
            Tuple of (Action, semantic_content dict with core_contribution,
            rationale, component_explanations, edit_plan, skill_metrics, etc.)
        """
        import openai
        import httpx
        self._last_proposal_debug = {
            "stage": "llm_request",
            "skill": skill.name,
            "skill_blueprint": [item.value for item in skill.blueprint],
            "proposal_id": proposal_id,
            "action_contract": action_contract or {},
        }
        client = openai.OpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url,
            timeout=self.llm_timeout_s,
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        t0 = time.time()
        MAX_WALL_TIME = 180  # 3 min hard wall-clock limit for streaming
        chunks = []
        try:
            stream = self._with_llm_retries(
                lambda: client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    stream=True,
                ),
                openai_module=openai,
                operation="proposal stream",
            )
            for chunk in stream:
                if time.time() - t0 > MAX_WALL_TIME:
                    logger.warning(f"LLM streaming timed out after {MAX_WALL_TIME}s, using partial response")
                    break
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
        except self._retryable_llm_errors(openai) as exc:
            logger.warning(
                "LLM stream failed before usable content (%s); falling back to non-streaming",
                exc.__class__.__name__,
            )
        content = "".join(chunks)
        if not content.strip():
            logger.warning("LLM stream returned no content; retrying once without streaming")
            content = self._with_content_retries(
                lambda: self._with_llm_retries(
                    lambda: client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7,
                        stream=False,
                    ),
                    openai_module=openai,
                    operation="proposal fallback",
                ).choices[0].message.content
                or "",
                operation="proposal fallback",
            )
        self._last_proposal_debug.update(
            {
                "stage": "llm_response",
                "raw_response_excerpt": content[:1200],
                "raw_response_chars": len(content),
            }
        )
        logger.info(
            "%s",
            log_block(
                f"LLM raw response ({proposal_id or 'n/a'}, {self.model}, skill={skill.name}, chars={len(content)})",
                pretty_json_or_text(content),
                color="cyan",
            ),
        )
        try:
            action, semantic_content = self._parse_response(content, skill, action_contract=action_contract)
        except json.JSONDecodeError:
            logger.warning("Proposal %s returned invalid JSON; requesting one JSON-only repair pass", proposal_id or "n/a")
            repair = self._with_llm_retries(
                lambda: client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Repair malformed JSON. Return only one valid JSON object with no markdown.",
                        },
                        {"role": "user", "content": content},
                    ],
                    temperature=0.0,
                    stream=False,
                ),
                openai_module=openai,
                operation="JSON repair",
            )
            content = repair.choices[0].message.content or ""
            self._last_proposal_debug.update(
                {
                    "stage": "json_repair",
                    "repaired_response_excerpt": content[:1200],
                    "repaired_response_chars": len(content),
                }
            )
            logger.info(
                "%s",
                log_block(
                    f"LLM JSON repair response ({self.model}, chars={len(content)})",
                    pretty_json_or_text(content),
                    color="yellow",
                ),
            )
            action, semantic_content = self._parse_response(content, skill, action_contract=action_contract)
        if not action.changes:
            logger.warning("Proposal %s returned no compilable changes; requesting one schema repair pass", proposal_id or "n/a")
            repair = self._with_llm_retries(
                lambda: client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Convert the candidate into one valid executable XC edit. "
                                "Return only JSON. Use action_type in "
                                "[rewire, add_component, add_descriptor, remove_component, replace_component, enforce_constraint]. "
                                "For rewire, changes may use alpha/beta/omega or numeric component parameters. "
                                "For add_component, changes must include name or type. "
                                "For remove_component, changes must include name or index. "
                                "For replace_component, changes must include old_name and new_name. "
                                "Preserve the scientific intent, but fit this executable schema exactly."
                            ),
                        },
                        {"role": "user", "content": content},
                    ],
                    temperature=0.0,
                    stream=False,
                ),
                openai_module=openai,
                operation="schema repair",
            )
            content = repair.choices[0].message.content or ""
            self._last_proposal_debug.update(
                {
                    "stage": "schema_repair",
                    "schema_repair_response_excerpt": content[:1200],
                    "schema_repair_response_chars": len(content),
                }
            )
            logger.info(
                "%s",
                log_block(
                    f"LLM schema repair response ({self.model}, chars={len(content)})",
                    pretty_json_or_text(content),
                    color="yellow",
                ),
            )
            try:
                action, semantic_content = self._parse_response(content, skill, action_contract=action_contract)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Proposal %s schema repair returned non-JSON content; requesting JSON-only repair pass", proposal_id or "n/a")
                repair = self._with_llm_retries(
                    lambda: client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Return only one valid JSON object with no markdown or explanation.",
                            },
                            {"role": "user", "content": content},
                        ],
                        temperature=0.0,
                        stream=False,
                    ),
                    openai_module=openai,
                    operation="schema JSON repair",
                )
                content = repair.choices[0].message.content or ""
                self._last_proposal_debug.update(
                    {
                        "stage": "schema_json_repair",
                        "schema_json_repair_response_excerpt": content[:1200],
                        "schema_json_repair_response_chars": len(content),
                    }
                )
                logger.info(
                    "%s",
                    log_block(
                        f"LLM schema JSON repair response ({self.model}, chars={len(content)})",
                        pretty_json_or_text(content),
                        color="yellow",
                    ),
                )
                action, semantic_content = self._parse_response(content, skill, action_contract=action_contract)
            if not action.changes:
                raise ValueError(
                    f"LLM returned empty or unparseable changes after schema repair. "
                    f"Raw response: {content[:300]}"
                )
        if action_contract:
            ok, reason = _action_matches_contract(action, action_contract)
            if not ok:
                logger.warning(
                    "Proposal %s violates action contract family=%s: %s; requesting one contract repair pass",
                    proposal_id or "n/a",
                    action_contract.get("action_family"),
                    reason,
                )
                repair = self._with_llm_retries(
                    lambda: client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Repair this proposal so it obeys the Action Family Contract exactly. "
                                    "Return only one valid JSON object with no markdown."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Contract violation: {reason}\n"
                                    f"Action Family Contract:\n{json.dumps(action_contract, ensure_ascii=False, indent=2)}\n\n"
                                    f"Original response:\n{content}"
                                ),
                            },
                        ],
                        temperature=0.0,
                        stream=False,
                    ),
                    openai_module=openai,
                    operation="action contract repair",
                )
                content = repair.choices[0].message.content or ""
                self._last_proposal_debug.update(
                    {
                        "stage": "action_contract_repair",
                        "contract_violation": reason,
                        "contract_repair_response_excerpt": content[:1200],
                        "contract_repair_response_chars": len(content),
                    }
                )
                logger.info(
                    "%s",
                    log_block(
                        f"LLM action contract repair response ({self.model}, chars={len(content)})",
                        pretty_json_or_text(content),
                        color="yellow",
                    ),
                )
                action, semantic_content = self._parse_response(content, skill, action_contract=action_contract)
                ok, reason = _action_matches_contract(action, action_contract)
                if not ok:
                    raise ValueError(
                        f"LLM proposal violates action family contract after repair: {reason}"
                    )
        logger.info(
            "Proposal %s %s %s",
            proposal_id or "n/a",
            colorize(action.action_type.value, "magenta", bold=True),
            json.dumps(
                {
                    "changes": action.changes,
                    "core_contribution": semantic_content.get("core_contribution", ""),
                },
                ensure_ascii=False,
            ),
        )
        return action, semantic_content

    def _next_proposal_id(self) -> tuple[int, str]:
        self._proposal_counter += 1
        return self._proposal_counter, f"P{self._proposal_counter:05d}"

    def _normalize_string_list(self, values: Any, field_name: str, proposal_id: str) -> list[str]:
        if values is None:
            return []
        if not isinstance(values, list):
            logger.warning(
                "Proposal %s expected list for %s but got %s; coercing",
                proposal_id,
                field_name,
                type(values).__name__,
            )
            values = [values]
        normalized: list[str] = []
        for item in values:
            if item is None:
                continue
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = json.dumps(item, sort_keys=True)
            else:
                text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    def _normalize_string_mapping(self, mapping: Any, field_name: str, proposal_id: str) -> dict[str, str]:
        if mapping is None:
            return {}
        if not isinstance(mapping, dict):
            logger.warning(
                "Proposal %s expected dict for %s but got %s; dropping field",
                proposal_id,
                field_name,
                type(mapping).__name__,
            )
            return {}
        normalized: dict[str, str] = {}
        for key, value in mapping.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            if isinstance(value, str):
                value_text = value.strip()
            elif isinstance(value, dict):
                value_text = json.dumps(value, sort_keys=True)
            else:
                value_text = str(value).strip()
            if value_text:
                normalized[key_text] = value_text
        return normalized

    def _with_llm_retries(
        self,
        request: Callable[[], Any],
        *,
        openai_module: Any,
        operation: str,
        max_attempts: int | None = None,
        base_delay_s: float | None = None,
        max_delay_s: float | None = None,
        backoff: float | None = None,
    ) -> Any:
        """Retry transient LLM failures with configurable exponential backoff.

        Set ``llm_retry_attempts`` to 0 or a negative value for unlimited retries.
        That is useful for long HPC runs where a temporary API outage should pause
        proposal generation, not kill the already-evaluated search state.
        """
        attempts = self.llm_retry_attempts if max_attempts is None else int(max_attempts)
        base_delay = self.llm_retry_base_delay_s if base_delay_s is None else float(base_delay_s)
        max_delay = self.llm_retry_max_delay_s if max_delay_s is None else float(max_delay_s)
        backoff_factor = self.llm_retry_backoff if backoff is None else float(backoff)
        base_delay = max(0.0, base_delay)
        max_delay = max(base_delay, max_delay)
        backoff_factor = max(1.0, backoff_factor)
        retryable_errors = self._retryable_llm_errors(openai_module)
        attempt = 1
        while True:
            try:
                started = time.perf_counter()
                result = request()
                latency_s = time.perf_counter() - started
                usage = self._llm_response_usage(result)
                logger.info(
                    "LLM_API_SUCCESS source=idea_agent model=%s operation=%s attempt=%d latency_s=%.2f "
                    "response_chars=%d usage=%s",
                    self.model,
                    operation,
                    attempt,
                    latency_s,
                    self._llm_response_text_length(result),
                    json.dumps(usage, sort_keys=True) if usage else "{}",
                )
                return result
            except Exception as exc:
                if not self._is_retryable_llm_exception(exc, retryable_errors):
                    raise
                if attempts > 0 and attempt >= attempts:
                    raise
                delay = min(max_delay, base_delay * (backoff_factor ** max(0, attempt - 1)))
                delay *= 1.0 + random.uniform(0.0, 0.2)
                attempt_label = "unlimited" if attempts <= 0 else str(attempts)
                logger.warning(
                    "Transient LLM failure during %s (%s); retrying in %.1fs [%d/%s]: %s",
                    operation,
                    exc.__class__.__name__,
                    delay,
                    attempt,
                    attempt_label,
                    self._compact_llm_error(exc),
                )
                time.sleep(delay)
                attempt += 1

    def _retryable_llm_errors(self, openai_module: Any) -> tuple[type[BaseException], ...]:
        return tuple(
            exc
            for exc in (
                getattr(openai_module, "RateLimitError", None),
                getattr(openai_module, "APIConnectionError", None),
                getattr(openai_module, "APITimeoutError", None),
                getattr(openai_module, "InternalServerError", None),
                getattr(openai_module, "APIStatusError", None),
                getattr(openai_module, "APIError", None),
            )
            if exc is not None
        )

    def _is_retryable_llm_exception(
        self,
        exc: Exception,
        retryable_error_types: tuple[type[BaseException], ...] = (),
    ) -> bool:
        text = self._compact_llm_error(exc).lower()
        # Some OpenAI-compatible gateways intermittently return an empty
        # upstream 404 while their model router is unavailable.  A genuine
        # bad model name normally carries an invalid-request type and a
        # descriptive message, and remains fatal below.
        if self._is_transient_upstream_model_not_found(exc, text=text):
            return True
        fatal_tokens = (
            "invalid api key",
            "incorrect api key",
            "unauthorized",
            "authentication",
            "permission_denied",
            "model_not_found",
            "does not exist",
            "not found",
            "invalid_request_error",
        )
        if any(token in text for token in fatal_tokens):
            return False
        retry_tokens = (
            "timeout",
            "timed out",
            "read operation timed out",
            "request timed out",
            "rate limit",
            "ratelimit",
            "too many requests",
            "temporarily unavailable",
            "connection",
            "server error",
            "internalservererror",
            "internal error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "shell_api_error",
            "insufficient_balance",
            "insufficient account balance",
            "insufficient quota",
            "quota",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
        return isinstance(exc, retryable_error_types) or any(token in text for token in retry_tokens)

    def _is_transient_upstream_model_not_found(self, exc: Exception, *, text: str = "") -> bool:
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                code = str(error.get("code") or "").strip().lower()
                error_type = str(error.get("type") or "").strip().lower()
                message = str(error.get("message") or "").strip()
                localized = str(error.get("localized_message") or "").strip()
                if (
                    status_code == 404
                    and code == "model_not_found"
                    and error_type == "upstream_error"
                    and not message
                    and not localized
                ):
                    return True

        compact = text or self._compact_llm_error(exc).lower()
        return (
            "404" in compact
            and "model_not_found" in compact
            and "upstream_error" in compact
            and ("'message': ''" in compact or '"message": ""' in compact)
        )

    def _compact_llm_error(self, exc: Exception, *, max_chars: int = 500) -> str:
        text = str(exc).replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text

    def _llm_response_text_length(self, response: Any) -> int:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return 0
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        return len(content or "")

    def _llm_response_usage(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return {
                str(key): value
                for key, value in usage.items()
                if isinstance(value, (int, float, str, bool)) or value is None
            }
        result: dict[str, Any] = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        ):
            value = getattr(usage, key, None)
            if isinstance(value, (int, float, str, bool)) or value is None:
                result[key] = value
        return {key: value for key, value in result.items() if value is not None}

    def _with_content_retries(
        self,
        request: Callable[[], str],
        *,
        operation: str,
        max_attempts: int = 3,
        base_delay_s: float = 1.0,
        max_delay_s: float = 8.0,
    ) -> str:
        """Retry syntactically successful responses that carry no usable text."""
        for attempt in range(1, max_attempts + 1):
            content = request() or ""
            if content.strip():
                return content
            if attempt == max_attempts:
                return content
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay *= 1.0 + random.uniform(0.0, 0.2)
            logger.warning(
                "LLM returned empty content during %s; retrying in %.1fs [%d/%d]",
                operation,
                delay,
                attempt,
                max_attempts,
            )
            time.sleep(delay)
        return ""

    def _parse_response(
        self, content: str, skill: EditSkill, action_contract: dict[str, Any] | None = None
    ) -> tuple[Action, dict[str, Any]]:
        """Parse LLM response into an Action + semantic content dict.

        Handles both flat ({"alpha": 0.35}) and nested
        ({"parameters": {"alpha": {"to": 0.35}}}) JSON formats.
        Also handles LLM returning skill names as action_type
        (e.g. "add-range-separation") and composite changes
        (e.g. {"add": {...}, "rewire": {...}, "enforce": [...]}).

        Returns:
            Tuple of (Action, semantic_content dict).
        """
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError(
                f"LLM response contains no JSON object. "
                f"Response: {content[:300]}"
            )
        json_str = content[json_start:json_end]
        data = json.loads(json_str)
        action_type_str = data.get("action_type", skill.blueprint[0].value)
        parse_debug: dict[str, Any] = {
            "original_action_type": action_type_str,
            "skill": skill.name,
            "skill_blueprint": [item.value for item in skill.blueprint],
            "action_contract": action_contract or {},
        }

        # Map skill names to their first blueprint ActionType
        action_aliases = {
            "add_component": ActionType.ADD_COMPONENT,
            "add_descriptor": ActionType.ADD_DESCRIPTOR,
            "remove_component": ActionType.REMOVE_COMPONENT,
            "replace_component": ActionType.REPLACE_COMPONENT,
            "enforce_constraint": ActionType.ENFORCE_CONSTRAINT,
        }
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            if action_type_str in action_aliases:
                action_type = action_aliases[action_type_str]
            else:
                action_type = skill.blueprint[0]
        if action_type_str not in {item.value for item in ActionType} and action_type_str not in action_aliases:
            skill_map = {
                "increase-hf-screening": ActionType.REWIRE,
                "add-range-separation": ActionType.REWIRE,
                "add-dispersion-correction": ActionType.ADD_COMPONENT,
                "descriptor-synthesis": ActionType.ADD_DESCRIPTOR,
                "add-descriptor": ActionType.ADD_DESCRIPTOR,
                "replace-exchange-component": ActionType.REPLACE_COMPONENT,
            }
            action_type = skill_map.get(action_type_str, skill.blueprint[0])
            parse_debug["action_type_mapping"] = action_type.value
        allowed_blueprint = _allowed_action_types(action_contract) if action_contract else list(skill.blueprint)
        if action_type not in allowed_blueprint:
            parse_debug["blueprint_coercion"] = {
                "from": action_type.value,
                "to": allowed_blueprint[0].value,
                "reason": "action outside selected action contract" if action_contract else "action outside selected skill blueprint",
            }
            logger.warning(
                "LLM action %s is outside skill=%s allowed_actions=%s; coercing to %s",
                action_type.value,
                skill.name,
                [item.value for item in allowed_blueprint],
                allowed_blueprint[0].value,
            )
            action_type = allowed_blueprint[0]

        raw_changes = data.get("changes", {})
        parse_debug["raw_changes_excerpt"] = _compact_failure_value(raw_changes)
        changes = self._flatten_changes(raw_changes)
        if self._is_hf_exchange_rewire(action_type, changes, raw_changes):
            parse_debug["hf_rewire_coercion"] = {
                "from": action_type.value,
                "to": ActionType.REWIRE.value,
                "reason": "HF exchange schema compiles as parameter rewire",
            }
            action_type = ActionType.REWIRE

        # If changes are still empty or invalid, extract from skill context
        if not changes or not self._has_valid_keys(changes, action_type):
            changes = self._extract_from_composite(raw_changes, action_type)
        if self._is_hf_exchange_rewire(action_type, changes, raw_changes):
            parse_debug["hf_rewire_coercion"] = {
                "from": action_type.value,
                "to": ActionType.REWIRE.value,
                "reason": "HF exchange schema compiles as parameter rewire",
            }
            action_type = ActionType.REWIRE
        changes = self._sanitize_changes(changes, action_type)
        parse_debug["final_action_type"] = action_type.value
        parse_debug["final_changes"] = _compact_failure_value(changes)
        current_debug = dict(getattr(self, "_last_proposal_debug", {}) or {})
        current_debug.update(parse_debug)
        self._last_proposal_debug = current_debug

        # Extract semantic content (mirrors Xcientist's IdeaState fields)
        semantic_content = {
            "core_contribution": data.get("core_contribution", ""),
            "rationale": data.get("rationale", ""),
            "component_explanations": data.get("component_explanations", {}),
            "target_findings": data.get("target_findings", data.get("target_defects", [])),
            "memory_refs": data.get("edit_plan", {}).get("memory_refs", [])
                if isinstance(data.get("edit_plan"), dict) else [],
            "paper_graph_context": "",
            "skill_metrics": data.get("skill_metrics", {}),
        }

        # Parse edit_plan if present
        edit_plan_data = data.get("edit_plan")
        edit_plan = None
        if isinstance(edit_plan_data, dict):
            edit_plan = EditPlan.from_dict(edit_plan_data)
        semantic_content["edit_plan"] = edit_plan.to_dict() if edit_plan else None

        # Build Action with edit_plan attached
        action = Action(
            action_type=action_type,
            changes=changes,
            rationale=data.get("rationale", ""),
            edit_plan=edit_plan,
        )

        return action, semantic_content

    def _is_hf_exchange_rewire(
        self,
        action_type: ActionType,
        changes: dict,
        raw_changes: dict,
    ) -> bool:
        """Treat HF-exchange replacement schemas as executable HF parameter rewires."""
        if action_type not in {ActionType.ADD_COMPONENT, ActionType.REPLACE_COMPONENT}:
            return False
        has_hf_params = any(
            key in changes and isinstance(changes[key], (int, float))
            for key in ("alpha", "beta", "omega")
        )

        def _normalize_key(value: Any) -> str:
            return str(value).lower().replace(" ", "").replace("_", "").replace("-", "")

        markers = [
            changes.get("old_name", ""),
            changes.get("new_name", ""),
            changes.get("name", ""),
            changes.get("component", ""),
            *raw_changes.keys(),
        ]
        mentions_hf_exchange = any("hfexchange" in _normalize_key(marker) for marker in markers)
        return has_hf_params and mentions_hf_exchange

    def _sanitize_changes(self, changes: dict, action_type: ActionType) -> dict:
        """Drop explanatory fields that are not executable edit keys."""
        if action_type == ActionType.REPLACE_COMPONENT:
            return self._sanitize_replace_changes(changes)
        if action_type == ActionType.ADD_DESCRIPTOR:
            descriptor = changes.get("descriptor", changes)
            return {"descriptor": descriptor} if isinstance(descriptor, dict) else {}
        if action_type == ActionType.ADD_COMPONENT and _is_pseudo_hf_component(changes.get("name", "")):
            # SR_HF_EXCHANGE/LR_HF_EXCHANGE are not libxc/PySCF components in this
            # representation. Range-separated HF must compile as HF alpha/beta/omega
            # rewiring; otherwise the search evaluates invalid pseudo-components.
            return {
                key: float(value)
                for key, value in changes.items()
                if key in {"alpha", "beta", "omega"} and isinstance(value, (int, float))
            }
        if action_type != ActionType.REWIRE:
            return changes
        sanitized: dict[str, Any] = {}
        scalar_keys = {"alpha", "beta", "omega", "s6", "s8", "a1", "a2", "mixing_factor"}
        for key, value in changes.items():
            if key in scalar_keys and isinstance(value, (int, float)):
                sanitized[key] = float(value)
            elif key.startswith("mixing:") and isinstance(value, (int, float)):
                sanitized[key] = float(value)
            elif key.startswith("scale:") and isinstance(value, (int, float)):
                sanitized[key] = float(value)
            elif "." in key and isinstance(value, (int, float)):
                sanitized[key] = float(value)
        return sanitized

    def _sanitize_replace_changes(self, changes: dict) -> dict:
        """Normalize replace_component schemas before they reach apply_action."""
        sanitized = dict(changes or {})
        nested = sanitized.pop("replace", None) or sanitized.pop("replace_component", None)
        if isinstance(nested, dict):
            if "old_name" not in sanitized:
                sanitized["old_name"] = self._component_name(
                    nested.get("old_name") or nested.get("from") or nested.get("old_component") or ""
                )
            if "new_name" not in sanitized:
                self._merge_component_spec(
                    sanitized,
                    nested.get("new_name") or nested.get("to") or nested.get("new_component") or "",
                    prefix="new_",
                )
            for key in ("role", "mixing_factor", "new_xctype"):
                if key in nested and key not in sanitized:
                    sanitized[key] = nested[key]
            if "ext_params" in nested and isinstance(nested["ext_params"], dict):
                sanitized.setdefault("new_ext_params", nested["ext_params"])

        if "old_component" in sanitized and "old_name" not in sanitized:
            sanitized["old_name"] = self._component_name(sanitized.pop("old_component"))
        if "new_component" in sanitized and "new_name" not in sanitized:
            self._merge_component_spec(sanitized, sanitized.pop("new_component"), prefix="new_")
        if isinstance(sanitized.get("old_name"), dict):
            sanitized["old_name"] = self._component_name(sanitized["old_name"])
        if isinstance(sanitized.get("new_name"), dict):
            spec = sanitized.pop("new_name")
            self._merge_component_spec(sanitized, spec, prefix="new_")
        return sanitized

    def _has_valid_keys(self, changes: dict, action_type: ActionType) -> bool:
        """Check if changes dict has keys relevant to the action_type."""
        if action_type == ActionType.REWIRE:
            if any(k in changes for k in ("alpha", "beta", "omega", "s6", "s8", "mixing_factor")):
                return True
            if any(str(k).startswith("mixing:") for k in changes):
                return True
            if any(str(k).startswith("scale:") for k in changes):
                return True
            if any("." in str(k) for k in changes):
                return True
            # Also accept HF_EXCHANGE_RANGE_SEPARATED.alpha etc.
            return any(k.startswith("HF_EXCHANGE") and "." in k for k in changes)
        elif action_type == ActionType.ADD_COMPONENT:
            return "name" in changes or "type" in changes
        elif action_type == ActionType.ADD_DESCRIPTOR:
            return isinstance(changes.get("descriptor"), dict) or "template" in changes
        elif action_type == ActionType.REPLACE_COMPONENT:
            return "old_name" in changes or "new_name" in changes
        elif action_type == ActionType.REMOVE_COMPONENT:
            return "name" in changes or "index" in changes
        elif action_type == ActionType.ENFORCE_CONSTRAINT:
            return "constraint" in changes
        return bool(changes)

    def _extract_from_composite(self, raw: dict, action_type: ActionType) -> dict:
        """Extract relevant parameters from a composite LLM response.

        LLM may return: {"add": {...}, "rewire": {"alpha": 0.25, "omega": 0.11}, "enforce": [...]}
        We extract the parameters relevant to the chosen action_type.
        """
        extracted = {}

        def _normalize_key(k: str) -> str:
            return k.lower().replace(" ", "").replace("_", "")

        if action_type == ActionType.REWIRE:
            rewire = raw.get("rewire", {})
            if isinstance(rewire, dict):
                for k in ("alpha", "beta", "omega", "s6", "s8", "a1", "a2"):
                    if k in rewire:
                        v = rewire[k]
                        if isinstance(v, dict):
                            extracted[k] = v.get("to", v.get("value", next((x for x in v.values() if isinstance(x, (int, float))), None)))
                        elif isinstance(v, (int, float)):
                            extracted[k] = float(v)
                for key, value in rewire.items():
                    if str(key).startswith(("mixing:", "scale:")) and isinstance(value, (int, float)):
                        extracted[str(key)] = float(value)
            for key, value in raw.items():
                if str(key).startswith(("mixing:", "scale:")) and isinstance(value, (int, float)):
                    extracted[str(key)] = float(value)
            # Also check "HF Exchange" / "hf_exchange" keys (case/space-insensitive)
            for raw_key, raw_val in raw.items():
                norm = _normalize_key(raw_key)
                if norm == "hfexchange" and isinstance(raw_val, dict):
                    for k in ("alpha", "beta", "omega"):
                        if k not in extracted and k in raw_val:
                            v = raw_val[k]
                            if isinstance(v, (int, float)):
                                extracted[k] = float(v)
                # Handle dotted keys like HF_EXCHANGE_RANGE_SEPARATED.omega
                elif "." in raw_key and isinstance(raw_val, (int, float)):
                    prefix, leaf = raw_key.split(".", 1)
                    prefix_norm = _normalize_key(prefix)
                    if prefix_norm.startswith("hfexchange") and leaf in ("alpha", "beta", "omega"):
                        if leaf not in extracted:
                            extracted[leaf] = float(raw_val)

        elif action_type == ActionType.ADD_COMPONENT:
            add = raw.get("add", {})
            if isinstance(add, dict):
                component = add.get("component", "")
                if component in ("RSH", "range-separated"):
                    extracted["name"] = "HYB_GGA_XC_WB97X"
                    extracted["xctype"] = "HYB_GGA"
                    extracted["role"] = "exchange"
                elif component in ("D3", "D3BJ", "dispersion"):
                    extracted["type"] = "D3BJ"
                elif "name" in add:
                    extracted["name"] = add["name"]

        elif action_type == ActionType.ADD_DESCRIPTOR:
            descriptor = raw.get("descriptor") or raw.get("add_descriptor") or raw.get("custom_descriptor")
            if isinstance(descriptor, dict):
                extracted["descriptor"] = descriptor
            elif raw.get("template") or raw.get("family"):
                extracted["descriptor"] = dict(raw)

        elif action_type == ActionType.REPLACE_COMPONENT:
            for key in ("old_name", "new_name"):
                if key in raw:
                    extracted[key] = raw[key]
            # Also check old_component/new_component aliases
            if "old_component" in raw and "old_name" not in extracted:
                extracted["old_name"] = self._component_name(raw["old_component"])
            if "new_component" in raw and "new_name" not in extracted:
                self._merge_component_spec(extracted, raw["new_component"], prefix="new_")
            for nested_key in ("replace_component", "replace"):
                nested_replace = raw.get(nested_key)
                if isinstance(nested_replace, dict):
                    extracted.setdefault(
                        "old_name",
                        self._component_name(
                            nested_replace.get("old_name")
                            or nested_replace.get("from")
                            or nested_replace.get("old_component")
                            or ""
                        ),
                    )
                    new_spec = (
                        nested_replace.get("new_name")
                        or nested_replace.get("to")
                        or nested_replace.get("new_component")
                        or ""
                    )
                    if "new_name" not in extracted:
                        self._merge_component_spec(extracted, new_spec, prefix="new_")
                    # Extract role, mixing_factor, ext_params from nested replace dict
                    for rk in ("role", "mixing_factor", "new_xctype"):
                        if rk in nested_replace and rk not in extracted:
                            extracted[rk] = nested_replace[rk]
                    if "ext_params" in nested_replace and isinstance(nested_replace["ext_params"], dict):
                        extracted["new_ext_params"] = nested_replace["ext_params"]
            if not extracted:
                for key, value in raw.items():
                    if isinstance(value, dict) and value.get("replace_with"):
                        extracted["old_name"] = key
                        self._merge_component_spec(extracted, value["replace_with"], prefix="new_")
                        if "xctype" in value:
                            extracted["new_xctype"] = value["xctype"]
                        break

        elif action_type == ActionType.REMOVE_COMPONENT:
            remove = raw.get("remove") or raw.get("remove_component")
            if isinstance(remove, dict):
                if "name" in remove:
                    extracted["name"] = self._component_name(remove["name"])
                elif "component" in remove:
                    extracted["name"] = self._component_name(remove["component"])
                elif "index" in remove:
                    extracted["index"] = remove["index"]
            elif "name" in raw:
                extracted["name"] = self._component_name(raw["name"])
            elif "component" in raw:
                extracted["name"] = self._component_name(raw["component"])
            elif "index" in raw:
                extracted["index"] = raw["index"]

        return extracted

    def _flatten_changes(self, raw: dict) -> dict:
        """Flatten nested LLM response into the format apply_action expects.

        Handles multiple LLM response formats:
          {"parameters": {"alpha": {"from": 0.25, "to": 0.35}}} -> {"alpha": 0.35}
          {"alpha": {"from": 0.25, "to": 0.35}} -> {"alpha": 0.35}
          {"alpha": 0.35} -> {"alpha": 0.35}
          {"HF Exchange": {"alpha": 0.30, "beta": 0.0}} -> {"alpha": 0.30, "beta": 0.0}
          {"HF Exchange.alpha": 0.30} -> {"alpha": 0.30}
          {"GGA_X_PBE.mixing": 0.70} -> {"mixing:GGA_X_PBE": 0.70}
          {"GGA_X_PBE": {"_kappa": 0.804, "_mu": 0.22}} -> {"GGA_X_PBE._kappa": 0.804, ...}
          {"enforce_exchange_sum_rule": true} -> (skipped, non-numeric annotation)
          {"name": "GGA_X_PBE", "xctype": "GGA"} -> {"name": "GGA_X_PBE", "xctype": "GGA"}
        """
        flat = {}
        params = raw.get("parameters", raw)

        def _normalize_key(k: str) -> str:
            return k.lower().replace(" ", "").replace("_", "")

        _SUBGROUP_KEYS = {"hf_exchange", "exchange", "correlation", "dispersion", "rewire", "add", "parameters", "unchanged"}
        _SUBGROUP_KEYS_NORM = {_normalize_key(k) for k in _SUBGROUP_KEYS}

        _ANNOTATION_KEYS = {
            "enforce_exchange_sum_rule", "enforce_one_electron_sie_check",
            "enforce_one_electron_SIE_check", "enforce_lda_limit",
            "exchange_sum_rule", "constraint_enforced", "constraint",
            "enforced_constraints", "component_type", "scheme",
            "short_range_hf_fraction", "long_range_hf_fraction",
            "slot", "preserve_exchange", "preserve_correlation",
            "preserve_hf_exchange", "preserve_mixing",
        }

        for key, value in params.items():
            if key in {"descriptor", "add_descriptor", "custom_descriptor"} and isinstance(value, dict):
                flat["descriptor"] = value
                continue
            if key == "old_component":
                flat["old_name"] = self._component_name(value)
                continue
            if key == "new_component":
                self._merge_component_spec(flat, value, prefix="new_")
                continue
            if key in {"replace_component", "replace"} and isinstance(value, dict):
                flat["old_name"] = self._component_name(
                    value.get("old_name") or value.get("from") or value.get("old_component") or ""
                )
                self._merge_component_spec(
                    flat,
                    value.get("new_name") or value.get("to") or value.get("new_component") or "",
                    prefix="new_",
                )
                # Also extract role, mixing_factor, ext_params from replace dict
                for rk in ("role", "mixing_factor", "new_xctype"):
                    if rk in value and rk not in flat:
                        flat[rk] = value[rk]
                if "ext_params" in value and isinstance(value["ext_params"], dict):
                    flat["new_ext_params"] = value["ext_params"]
                continue
            if _normalize_key(key) in _ANNOTATION_KEYS:
                continue
            if isinstance(value, bool):
                continue

            # Handle dotted keys like "HF Exchange.alpha" -> "alpha"
            if "." in key and isinstance(value, (int, float)):
                prefix, leaf = key.split(".", 1)
                prefix_norm = _normalize_key(prefix)
                # HF exchange variants: HF_EXCHANGE, HF_EXCHANGE_GLOBAL,
                # HF_EXCHANGE_RANGE_SEPARATED, etc.
                if (prefix_norm in ("hfexchange", "dftexchange", "exchange", "correlation")
                        or prefix_norm.startswith("hfexchange")):
                    flat[leaf] = float(value)
                    continue
                if leaf in ("mixing", "mixing_factor"):
                    flat[f"mixing:{prefix}"] = float(value)
                    continue
                flat[key] = float(value)
                continue
            if "." in key and isinstance(value, str):
                continue

            # Check if this key is a subgroup (case-insensitive, space-insensitive)
            key_norm = _normalize_key(key)
            if key_norm == "hfexchange" and isinstance(value, dict):
                # Handle HF_EXCHANGE with from/to structure:
                # {"HF_EXCHANGE": {"from": {...}, "to": {...}}}
                if "to" in value and isinstance(value["to"], dict):
                    target = value["to"]
                    for param in ("alpha", "beta", "omega"):
                        if param in target and isinstance(target[param], (int, float)):
                            flat[param] = float(target[param])
                else:
                    # Simple dict: {"HF Exchange": {"alpha": 0.25, "beta": 0.75, "omega": 0.2}}
                    self._unwrap_subgroup(value, flat)
            elif key_norm in _SUBGROUP_KEYS_NORM and isinstance(value, dict):
                self._unwrap_subgroup(value, flat)
            elif isinstance(value, dict):
                # Check for "replace_with" pattern
                if "replace_with" in value:
                    flat["old_name"] = key
                    self._merge_component_spec(flat, value["replace_with"], prefix="new_")
                    if "xctype" in value:
                        flat["new_xctype"] = value["xctype"]
                    continue
                # Check for "from"/"to" pattern (e.g. "exchange_component": {"from": "GGA_X_PBE", "to": "GGA_X_HJS_PBE"})
                if "from" in value and "to" in value:
                    flat["old_name"] = self._component_name(value["from"])
                    self._merge_component_spec(flat, value["to"], prefix="new_")
                    # Also extract any other keys from this dict (mixing, ext_params, etc.)
                    for k, v in value.items():
                        if k in ("from", "to"):
                            continue
                        if isinstance(v, (int, float)):
                            flat[k] = float(v)
                        elif isinstance(v, dict):
                            # Nested dicts like ext_params, preserve_hf_exchange
                            self._unwrap_subgroup(v, flat)
                    continue
                # Check if this is a component-specific dict (e.g. "GGA_X_PBE": {...})
                is_component_dict = key.isupper() and "_" in key
                if is_component_dict:
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            if k in ("mixing", "mixing_factor", "mixing_sr", "mixing_lr"):
                                flat[f"mixing:{key}"] = float(v)
                            elif k.startswith("_"):
                                flat[f"{key}.{k}"] = float(v)
                            else:
                                flat[f"{key}.{k}"] = float(v)
                        elif isinstance(v, str):
                            flat[f"{key}.{k}"] = v
                    continue
                # Preserve dicts that are parameter sets (e.g. new_ext_params, ext_params)
                if key in ("new_ext_params", "ext_params", "new_ext_param_bounds", "ext_param_bounds"):
                    flat[key] = value
                    continue
                # Standard nested dict with "to"/"value"/"new" keys
                if "to" in value:
                    flat[key] = value["to"]
                elif "value" in value:
                    flat[key] = value["value"]
                elif "new" in value:
                    flat[key] = value["new"]
                else:
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            flat[key] = float(v)
                            break
            elif isinstance(value, (int, float)):
                if key == "mixing":
                    flat["mixing_factor"] = float(value)
                else:
                    flat[key] = float(value)
            elif isinstance(value, str):
                flat[key] = value
            elif isinstance(value, list):
                flat[key] = value

        for key in ("name", "xctype", "role", "type", "old_name", "new_name",
                     "new_xctype", "mixing_factor", "constraint", "index"):
            if key in raw and key not in flat:
                flat[key] = raw[key]

        # Normalize old/new name aliases from various LLM formats
        for old_key, new_key in (("old_component", "old_name"), ("new_component", "new_name"),
                                  ("from", "old_name"), ("to", "new_name")):
            if old_key in flat and new_key not in flat:
                value = flat.pop(old_key)
                if new_key == "new_name":
                    self._merge_component_spec(flat, value, prefix="new_")
                else:
                    flat[new_key] = self._component_name(value)
            elif old_key in raw and new_key not in flat:
                if new_key == "new_name":
                    self._merge_component_spec(flat, raw[old_key], prefix="new_")
                else:
                    flat[new_key] = self._component_name(raw[old_key])

        return flat

    def _component_name(self, value: Any) -> str:
        """Extract a component name from either a plain string or component spec."""
        if isinstance(value, dict):
            for key in ("name", "component", "id"):
                if isinstance(value.get(key), str):
                    return value[key]
            return ""
        return str(value) if value is not None else ""

    def _merge_component_spec(self, target: dict, value: Any, *, prefix: str = "") -> None:
        """Merge a plain component name or nested component spec into flat changes."""
        if isinstance(value, dict):
            name = self._component_name(value)
            if name:
                target[f"{prefix}name"] = name
            if isinstance(value.get("xctype"), str):
                target[f"{prefix}xctype"] = value["xctype"]
            if isinstance(value.get("role"), str):
                target["role"] = value["role"]
            if isinstance(value.get("mixing_factor"), (int, float)):
                target["mixing_factor"] = float(value["mixing_factor"])
            if isinstance(value.get("ext_params"), dict):
                target[f"{prefix}ext_params"] = value["ext_params"]
            if isinstance(value.get("ext_param_bounds"), dict):
                target[f"{prefix}ext_param_bounds"] = value["ext_param_bounds"]
            return
        name = self._component_name(value)
        if name:
            target[f"{prefix}name"] = name

    def _unwrap_subgroup(self, sub: dict, flat: dict) -> None:
        """Recursively unwrap a nested sub-group dict into flat key-value pairs."""
        _NESTED_SUBGROUPS = {"hf_exchange", "exchange", "correlation", "dispersion", "rewire", "add", "parameters", "unchanged"}

        def _normalize_key(k: str) -> str:
            return k.lower().replace(" ", "").replace("_", "")

        for key, value in sub.items():
            if key in _NESTED_SUBGROUPS and isinstance(value, dict):
                self._unwrap_subgroup(value, flat)
            elif isinstance(value, dict):
                if "to" in value:
                    flat[key] = value["to"]
                elif "value" in value:
                    flat[key] = value["value"]
                elif "new" in value:
                    flat[key] = value["new"]
                else:
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            flat[key] = float(v)
                            break
            elif isinstance(value, (int, float)):
                # Handle dotted keys like HF_EXCHANGE_RANGE_SEPARATED.omega
                if "." in key:
                    prefix, leaf = key.split(".", 1)
                    prefix_norm = _normalize_key(prefix)
                    if (prefix_norm in ("hfexchange", "dftexchange", "exchange", "correlation")
                            or prefix_norm.startswith("hfexchange")):
                        flat[leaf] = float(value)
                        continue
                    if leaf in ("mixing", "mixing_factor"):
                        flat[f"mixing:{prefix}"] = float(value)
                        continue
                flat[key] = float(value)
            elif isinstance(value, str):
                flat[key] = value


def _infer_root_domains(skill: EditSkill) -> list[str]:
    """Coarse provenance domain for downstream handoff and trace browsing."""
    if skill.control_arm:
        return ["control_arm"]
    if "theory" in skill.name:
        return ["theory_transfer"]
    if "simplification" in skill.name or "replacement" in skill.name:
        return ["component_architecture"]
    return ["mechanism_search"]


def _action_type_from_contract_token(token: Any) -> ActionType | None:
    aliases = {
        "rewire": ActionType.REWIRE,
        "add": ActionType.ADD_COMPONENT,
        "add_component": ActionType.ADD_COMPONENT,
        "add_descriptor": ActionType.ADD_DESCRIPTOR,
        "remove": ActionType.REMOVE_COMPONENT,
        "remove_component": ActionType.REMOVE_COMPONENT,
        "replace": ActionType.REPLACE_COMPONENT,
        "replace_component": ActionType.REPLACE_COMPONENT,
        "enforce": ActionType.ENFORCE_CONSTRAINT,
        "enforce_constraint": ActionType.ENFORCE_CONSTRAINT,
    }
    text = str(token)
    if text in aliases:
        return aliases[text]
    try:
        return ActionType(text)
    except ValueError:
        return None


def _allowed_action_types(action_contract: dict[str, Any] | None) -> list[ActionType]:
    if not action_contract:
        return []
    values: list[ActionType] = []
    for token in action_contract.get("allowed_action_types", []) or []:
        action_type = _action_type_from_contract_token(token)
        if action_type is not None and action_type not in values:
            values.append(action_type)
    return values or [ActionType.REWIRE]


def _action_matches_contract(action: Action, action_contract: dict[str, Any]) -> tuple[bool, str]:
    allowed = _allowed_action_types(action_contract)
    if action.action_type not in allowed:
        return (
            False,
            f"action_type {action.action_type.value} not allowed; allowed={[item.value for item in allowed]}",
        )
    family = str(action_contract.get("action_family", ""))
    changes = action.changes if isinstance(action.changes, dict) else {}
    if family == "global_rewire":
        if action.action_type != ActionType.REWIRE:
            return False, "global_rewire requires action_type=rewire"
        bad = [key for key in changes if key not in {"alpha", "beta", "omega"}]
        if bad:
            return False, f"global_rewire may only change alpha/beta/omega; disallowed keys={bad}"
        if not changes:
            return False, "global_rewire requires at least one of alpha, beta, omega"
    elif family == "component_rewire":
        if action.action_type != ActionType.REWIRE:
            return False, "component_rewire requires action_type=rewire"
        bad = [
            key for key in changes
            if key in {"alpha", "beta", "omega"} or str(key).upper().startswith("HF_EXCHANGE.")
        ]
        if bad:
            return False, f"component_rewire must not change global HF parameters; disallowed keys={bad}"
        if not any(
            str(key).startswith(("mixing:", "scale:")) or "." in str(key)
            for key in changes
        ):
            return False, "component_rewire requires mixing:COMP, scale:ROLE, or COMP.PARAM"
    elif family == "component_structure":
        if action.action_type not in {ActionType.ADD_COMPONENT, ActionType.REMOVE_COMPONENT, ActionType.REPLACE_COMPONENT}:
            return False, "component_structure requires add_component, remove_component, or replace_component"
        if action.action_type == ActionType.ADD_COMPONENT and not (changes.get("name") or changes.get("type")):
            return False, "add_component requires changes.name or changes.type"
        if action.action_type == ActionType.REMOVE_COMPONENT and not ("name" in changes or "index" in changes):
            return False, "remove_component requires changes.name or changes.index"
        if action.action_type == ActionType.REPLACE_COMPONENT and not (changes.get("old_name") and changes.get("new_name")):
            return False, "replace_component requires old_name and new_name"
    elif family == "descriptor_synthesis":
        if action.action_type != ActionType.ADD_DESCRIPTOR:
            return False, "descriptor_synthesis requires add_descriptor"
        descriptor = changes.get("descriptor") if isinstance(changes, dict) else None
        if not isinstance(descriptor, dict):
            return False, "add_descriptor requires changes.descriptor"
    return True, ""


def _compact_failure_value(value: Any, *, max_text: int = 1200) -> Any:
    """Compact proposal failure context for reflection logs."""
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key == "xc_dict":
                continue
            compact[str(key)] = _compact_failure_value(item, max_text=max_text)
            if len(compact) >= 20:
                compact["..."] = "truncated"
                break
        return compact
    if isinstance(value, list):
        return [_compact_failure_value(item, max_text=max_text) for item in value[:12]]
    if isinstance(value, str):
        return value[:max_text]
    return value


def _is_pseudo_hf_component(name: Any) -> bool:
    normalized = str(name).lower().replace(" ", "").replace("_", "").replace("-", "")
    return normalized in {
        "srhfexchange",
        "lrhfexchange",
        "hfexchange",
        "shortrangehfexchange",
        "longrangehfexchange",
    }


def _format_action_suffix(action: Action) -> str:
    """Build a compact, stable candidate-name suffix from executable changes."""
    digest = hashlib.sha256(
        json.dumps(action.changes, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    if action.action_type == ActionType.ADD_DESCRIPTOR:
        descriptor = action.changes.get("descriptor", action.changes)
        if isinstance(descriptor, dict):
            template = str(descriptor.get("template") or descriptor.get("family") or "descriptor")
            params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else descriptor
            scale = params.get("scale") if isinstance(params, dict) else None
            if isinstance(scale, (int, float)):
                return _safe_name_part(f"add_descriptor_{template}_scale={float(scale):.3f}_{digest}")
            return _safe_name_part(f"add_descriptor_{template}_{digest}")
    if not action.changes:
        return _safe_name_part(f"{action.action_type.value}_{digest}")
    key = next(iter(action.changes))
    value = action.changes[key]
    if isinstance(value, float):
        raw = f"{action.action_type.value}_{key}={value:.3f}_{digest}"
    elif isinstance(value, (int, str)):
        raw = f"{action.action_type.value}_{key}={value}_{digest}"
    else:
        raw = f"{action.action_type.value}_{key}_{digest}"
    return _safe_name_part(raw)


def _safe_name_part(value: str, *, max_len: int = 96) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"-", "_", ".", "="}:
            keep.append(char)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return (cleaned[:max_len].rstrip("_") or "action")
