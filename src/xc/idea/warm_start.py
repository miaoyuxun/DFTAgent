"""LLM structure exploration followed by local parameter warm-start tuning."""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..benchmark.result import BenchmarkResult
from ..logging_utils import log_block, pretty_json_or_text
from ..xc_functional import XCFunctional
from ..zno_component_registry import component_roles_for_pool
from .actions import Action, ActionType, apply_action
from .agent import IdeaAgent, SYSTEM_PROMPT
from .evolutionary import EvolutionCandidate, EvolutionarySearch, EvolutionConfig
from .knowledge_engine import KnowledgeEngine, format_knowledge_bundle

logger = logging.getLogger(__name__)


@dataclass
class WarmStartConfig:
    """Configuration for LLM-generated EA warm-start seeds."""

    n_structures: int = 5
    tuning_steps: int = 6
    reward_threshold: float = 0.3
    high_reward_direction_threshold: float = 0.7
    max_structure_actions: int = 3
    max_structure_batches: int = 4


@dataclass
class WarmStartRecord:
    """One evaluated warm-start candidate."""

    stage: str
    scaffold_index: int
    tuning_step: int
    scaffold_name: str
    candidate: EvolutionCandidate
    action_detail: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "scaffold_index": self.scaffold_index,
            "tuning_step": self.tuning_step,
            "scaffold_name": self.scaffold_name,
            "action_detail": self.action_detail,
            "rationale": self.rationale,
            **self.candidate.to_dict(),
        }


@dataclass
class WarmStartResult:
    records: list[WarmStartRecord]
    selected: list[WarmStartRecord]
    baseline: EvolutionCandidate

    @property
    def best(self) -> EvolutionCandidate:
        return max(self.records, key=lambda item: _rank_key(item.candidate)).candidate

    @property
    def summary(self) -> dict[str, Any]:
        finite = [
            item.candidate.fitness
            for item in self.records
            if math.isfinite(item.candidate.fitness)
        ]
        return {
            "evaluated_nodes": len(self.records),
            "selected_seeds": len(self.selected),
            "best": self.best.to_dict(),
            "baseline": self.baseline.to_dict(),
            "best_raw_reward": max(finite) if finite else float("-inf"),
            "selected_raw_rewards": [item.candidate.fitness for item in self.selected],
        }


class LLMWarmStartSearch:
    """Generate structure proposals, locally tune them, and emit EA seeds."""

    def __init__(
        self,
        *,
        config: WarmStartConfig | None = None,
        scorer_config: EvolutionConfig | None = None,
    ):
        self.config = config or WarmStartConfig()
        self.scorer = EvolutionarySearch(scorer_config or EvolutionConfig())

    def search(
        self,
        *,
        root_xc: XCFunctional,
        root_result: BenchmarkResult,
        baseline_validation_result: BenchmarkResult,
        runner,
        validation_runner,
        idea_agent: IdeaAgent,
        knowledge: KnowledgeEngine | None,
        output_dir: Path | str,
    ) -> WarmStartResult:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        baseline = EvolutionCandidate(
            xc=copy.deepcopy(root_xc),
            parameters=self._read_parameters(root_xc),
            generation=0,
            search_result=root_result,
            validation_result=baseline_validation_result,
        )
        self.scorer._score(baseline, root_result, baseline_validation_result)

        baseline_signatures = self._baseline_signatures(knowledge)
        scaffolds = self._propose_structures(
            root_xc=root_xc,
            root_result=root_result,
            idea_agent=idea_agent,
            knowledge=knowledge,
            baseline_signatures=baseline_signatures,
        )
        scaffolds.insert(0, ("baseline", copy.deepcopy(root_xc), "unchanged baseline scaffold"))

        records: list[WarmStartRecord] = []
        evaluated_by_sig: dict[str, WarmStartRecord] = {}
        for scaffold_index, (label, scaffold, rationale) in enumerate(scaffolds):
            logger.info(
                "Warm-start scaffold %d/%d label=%s name=%s sig=%s effective_sig=%s",
                scaffold_index + 1,
                len(scaffolds),
                label,
                scaffold.name,
                scaffold.structural_signature(),
                scaffold.effective_signature(),
            )
            initial = self._evaluate_candidate(
                xc=scaffold,
                generation=0,
                root_result=root_result,
                baseline_validation=baseline_validation_result,
                runner=runner,
                validation_runner=validation_runner,
            )
            record = WarmStartRecord(
                stage="structure_initial",
                scaffold_index=scaffold_index,
                tuning_step=0,
                scaffold_name=label,
                candidate=initial,
                action_detail=scaffold.metadata.get("action_detail", {}),
                rationale=rationale,
            )
            records.append(record)
            evaluated_by_sig[scaffold.effective_signature()] = record

            best_for_scaffold = initial
            banned_directions: set[tuple[str, str]] = set()
            trajectory = [record]
            parameters = self.scorer._parameter_space(scaffold)
            if not parameters:
                logger.info("Warm-start scaffold %s has no tunable parameters; skipping tuning", label)
                continue

            for step in range(1, self.config.tuning_steps + 1):
                proposal = self._propose_tuning(
                    scaffold=scaffold,
                    current=best_for_scaffold,
                    trajectory=trajectory,
                    parameters=parameters,
                    banned_directions=banned_directions,
                    idea_agent=idea_agent,
                    step=step,
                )
                if proposal is None:
                    logger.warning("Warm-start tuning step skipped: scaffold=%s step=%d", label, step)
                    continue
                vector, action_detail, rationale = proposal
                candidate_xc = self.scorer._build_candidate(scaffold, vector, step)
                candidate_xc.name = (
                    f"{scaffold.name}__warm_s{scaffold_index:02d}_t{step:02d}_"
                    f"{candidate_xc.effective_signature()}"
                )
                candidate_xc.metadata.update(
                    {
                        "action": "warm_start_rewire",
                        "action_detail": action_detail,
                        "search_algorithm": "llm_warm_start",
                        "warm_start_scaffold": label,
                        "warm_start_step": step,
                    }
                )
                cached = evaluated_by_sig.get(candidate_xc.effective_signature())
                if cached is not None:
                    logger.info(
                        "Warm-start duplicate tuning candidate skipped: scaffold=%s step=%d effective_sig=%s",
                        label,
                        step,
                        candidate_xc.effective_signature(),
                    )
                    trajectory.append(cached)
                    continue
                candidate = self._evaluate_candidate(
                    xc=candidate_xc,
                    generation=step,
                    root_result=root_result,
                    baseline_validation=baseline_validation_result,
                    runner=runner,
                    validation_runner=validation_runner,
                )
                record = WarmStartRecord(
                    stage="tuning",
                    scaffold_index=scaffold_index,
                    tuning_step=step,
                    scaffold_name=label,
                    candidate=candidate,
                    action_detail=action_detail,
                    rationale=rationale,
                )
                records.append(record)
                trajectory.append(record)
                evaluated_by_sig[candidate_xc.effective_signature()] = record
                changed = _parameter_directions(best_for_scaffold.parameters, candidate.parameters)
                if candidate.fitness > self.config.high_reward_direction_threshold:
                    banned_directions.update(changed)
                    logger.info(
                        "Warm-start high-reward direction lock: scaffold=%s step=%d reward=%.4f banned=%s",
                        label,
                        step,
                        candidate.fitness,
                        sorted(f"{key}:{direction}" for key, direction in changed),
                    )
                if _rank_key(candidate) > _rank_key(best_for_scaffold):
                    best_for_scaffold = candidate

        selected = self._select(records)
        result = WarmStartResult(records=records, selected=selected, baseline=baseline)
        self.write_artifacts(result, output_path)
        self._log_ranked_records(records)
        return result

    def _propose_structures(
        self,
        *,
        root_xc: XCFunctional,
        root_result: BenchmarkResult,
        idea_agent: IdeaAgent,
        knowledge: KnowledgeEngine | None,
        baseline_signatures: set[str],
    ) -> list[tuple[str, XCFunctional, str]]:
        accepted: list[tuple[str, XCFunctional, str]] = []
        seen = {root_xc.structural_signature(), root_xc.effective_signature(), *baseline_signatures}
        feedback: list[str] = []
        for batch in range(1, self.config.max_structure_batches + 1):
            missing = self.config.n_structures - len(accepted)
            if missing <= 0:
                break
            payload = self._call_structure_llm(
                root_xc=root_xc,
                root_result=root_result,
                idea_agent=idea_agent,
                knowledge=knowledge,
                count=missing,
                feedback=feedback,
                batch=batch,
            )
            for index, spec in enumerate(payload.get("structures", []) or [], start=1):
                if len(accepted) >= self.config.n_structures:
                    break
                try:
                    candidate, action_detail = self._apply_structure_spec(
                        root_xc,
                        spec,
                        idea_agent,
                    )
                except Exception as exc:
                    msg = f"batch {batch} structure {index} invalid: {exc}"
                    logger.warning("Warm-start %s", msg)
                    feedback.append(msg)
                    continue
                hard = [item for item in candidate.validate() if item.startswith("HARD:")]
                if hard:
                    logger.info(
                        "Warm-start scaffold %s compiles but needs parameter tuning before benchmarking: %s",
                        candidate.name,
                        "; ".join(hard),
                    )
                sigs = {candidate.structural_signature(), candidate.effective_signature()}
                if seen & sigs:
                    msg = (
                        f"{candidate.name} duplicates an existing scaffold or one of the "
                        "benchmark-library baselines"
                    )
                    logger.warning("Warm-start %s", msg)
                    feedback.append(msg)
                    continue
                candidate.metadata.update(
                    {
                        "action": "warm_start_structure",
                        "action_detail": action_detail,
                        "search_algorithm": "llm_warm_start",
                    }
                )
                seen.update(sigs)
                label = str(spec.get("label") or f"structure_{len(accepted) + 1}")
                rationale = str(spec.get("rationale") or "")
                accepted.append((label, candidate, rationale))
        if len(accepted) < self.config.n_structures:
            logger.warning(
                "Warm-start accepted only %d/%d structure proposals after validation",
                len(accepted),
                self.config.n_structures,
            )
        return accepted

    def _call_structure_llm(
        self,
        *,
        root_xc: XCFunctional,
        root_result: BenchmarkResult,
        idea_agent: IdeaAgent,
        knowledge: KnowledgeEngine | None,
        count: int,
        feedback: list[str],
        batch: int,
    ) -> dict[str, Any]:
        knowledge_text = ""
        if knowledge is not None:
            bundle = knowledge.retrieve_for_node(
                current_xc=root_xc,
                current_result=root_result,
                defect_tags=["self_interaction_error"],
                target_subsets=list(root_result.subsets),
                active_skill="component-replacement",
            )
            knowledge_text = format_knowledge_bundle(bundle)
        prompt = "\n".join(
            [
                "## Task",
                f"Propose {count} distinct component-structure scaffolds for later parameter tuning.",
                "Start from the baseline functional shown below. You may use 1-3 component actions per scaffold.",
                "Allowed actions: add_component, remove_component, replace_component.",
                "Forbidden now: rewire, alpha/beta/omega edits, mixing edits, ext_param edits, descriptors.",
                "Prefer trying at least some add_component scaffolds: for pure exchange/correlation add_component, use only the component name/role; the compiler will split about 0.10 mixing from the largest existing same-role component automatically.",
                "Do not remove the only active exchange or correlation component. remove_component is only valid when another active same-role component remains after removal.",
                "Do not duplicate any standard benchmark-library baseline functional.",
                "",
                "## Baseline Functional",
                root_xc.to_json(),
                "",
                "## Baseline Search Result",
                _format_result(root_result),
                "",
                "## Component Benchmark Priors",
                knowledge_text or "No component benchmark priors available.",
                "",
                "## Approved Pure Component Catalogue",
                _format_component_catalogue(),
                "",
                "## Rejection Feedback",
                "\n".join(f"- {item}" for item in feedback[-10:]) or "None.",
                "",
                "## Required JSON",
                "Return only JSON:",
                "{",
                '  "structures": [',
                "    {",
                '      "label": "short_name",',
                '      "rationale": "why this scaffold may help SIE after tuning",',
                '      "actions": [',
                '        {"action_type": "replace_component", "changes": {"old_name": "GGA_C_PBE", "new_name": "GGA_C_APBE"}}',
                "      ]",
                "    }",
                "  ]",
                "}",
            ]
        )
        return self._llm_json(idea_agent, prompt, f"warm-start structure batch {batch}", temperature=0.55)

    def _apply_structure_spec(
        self,
        root_xc: XCFunctional,
        spec: dict[str, Any],
        idea_agent: IdeaAgent,
    ) -> tuple[XCFunctional, dict[str, Any]]:
        actions_payload = spec.get("actions")
        if not isinstance(actions_payload, list) or not actions_payload:
            raise ValueError("structure spec requires a non-empty actions list")
        if len(actions_payload) > self.config.max_structure_actions:
            raise ValueError(f"too many structure actions ({len(actions_payload)})")
        skill = idea_agent.skill_catalog.get("component-replacement")
        contract = {
            "action_family": "component_structure",
            "allowed_action_types": ["add_component", "remove_component", "replace_component"],
        }
        candidate = copy.deepcopy(root_xc)
        normalized_actions: list[dict[str, Any]] = []
        for payload in actions_payload:
            action, _semantic = idea_agent._parse_response(
                json.dumps(payload),
                skill,
                action_contract=contract,
            )
            if action.action_type not in {
                ActionType.ADD_COMPONENT,
                ActionType.REMOVE_COMPONENT,
                ActionType.REPLACE_COMPONENT,
            }:
                raise ValueError(f"non-structure action {action.action_type.value} is not allowed")
            parameter_keys = _structure_parameter_keys(action.changes)
            if parameter_keys:
                raise ValueError(
                    "structure-stage actions must not include parameter rewires; "
                    f"disallowed keys={parameter_keys}"
                )
            candidate = apply_action(candidate, action)
            normalized_actions.append(
                {
                    "action_type": action.action_type.value,
                    "changes": action.changes,
                    "rationale": action.rationale,
                }
            )
        candidate.name = (
            f"{root_xc.name}__warm_structure_{candidate.structural_signature()}"
        )
        return candidate, {"actions": normalized_actions}

    def _propose_tuning(
        self,
        *,
        scaffold: XCFunctional,
        current: EvolutionCandidate,
        trajectory: list[WarmStartRecord],
        parameters,
        banned_directions: set[tuple[str, str]],
        idea_agent: IdeaAgent,
        step: int,
    ) -> tuple[dict[str, float], dict[str, Any], str] | None:
        allowed = {item.key: item for item in parameters}
        prompt = "\n".join(
            [
                "## Task",
                "Tune only numerical parameters for this fixed component scaffold.",
                "Do not add, remove, or replace components. Do not propose descriptors.",
                "You may change multiple parameters in one step.",
                "The optimization score is raw_reward; higher is better. SIE does not need to beat baseline at every step.",
                "",
                "## Fixed Scaffold",
                scaffold.to_json(),
                "",
                "## Current Best Tuning State",
                json.dumps(
                    {
                        "name": current.xc.name,
                        "raw_reward": current.fitness,
                        "parameters": current.parameters,
                        "search": _result_metrics(current.search_result),
                        "guard": _result_metrics(current.validation_result),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
                "## Full Trajectory For This Scaffold",
                json.dumps([_record_brief(item) for item in trajectory], indent=2, ensure_ascii=False),
                "",
                "## Allowed Parameter Keys And Bounds",
                json.dumps(
                    {
                        item.key: {
                            "lower": item.lower,
                            "upper": item.upper,
                            "kind": item.kind,
                            "role": item.role,
                        }
                        for item in parameters
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
                "## Banned Directions",
                (
                    "\n".join(f"- {key}: do not {direction}" for key, direction in sorted(banned_directions))
                    if banned_directions
                    else "None."
                ),
                "",
                "## Required JSON",
                "Return only JSON:",
                "{",
                '  "action_type": "rewire",',
                '  "changes": {"omega": 0.45, "mixing:GGA_X_HJS_PBE": 0.55},',
                '  "rationale": "short explanation of why these parameter moves are worth one test"',
                "}",
            ]
        )
        skill = idea_agent.skill_catalog.get("parameter-control")
        for attempt in range(1, 3):
            payload = self._llm_json(
                idea_agent,
                prompt,
                f"warm-start tuning scaffold={scaffold.name} step={step} attempt={attempt}",
                temperature=0.45 if attempt == 1 else 0.1,
            )
            try:
                action, semantic = idea_agent._parse_response(
                    json.dumps(payload),
                    skill,
                    action_contract={"action_family": "warm_start_parameter_tuning", "allowed_action_types": ["rewire"]},
                )
            except Exception as exc:
                logger.warning("Warm-start tuning parse failed: %s", exc)
                continue
            changes = {
                key: float(value)
                for key, value in action.changes.items()
                if key in allowed and isinstance(value, (int, float))
            }
            unknown = [key for key in action.changes if key not in allowed]
            if unknown:
                prompt += "\n\n## Repair Feedback\nUnknown or disallowed parameter keys: " + ", ".join(map(str, unknown))
                continue
            if not changes:
                prompt += "\n\n## Repair Feedback\nNo valid numeric parameter changes were proposed."
                continue
            violation = _banned_direction_violation(current.parameters, changes, banned_directions)
            if violation:
                prompt += "\n\n## Repair Feedback\nYou violated banned directions: " + "; ".join(violation)
                continue
            vector = dict(current.parameters)
            vector.update(changes)
            vector = self.scorer._repair_vector(vector, parameters)
            if all(abs(vector.get(key, current.parameters.get(key, 0.0)) - current.parameters.get(key, 0.0)) < 1e-10 for key in allowed):
                prompt += "\n\n## Repair Feedback\nThe repaired parameter vector is identical to the current best state."
                continue
            return vector, {"changes": changes, "repaired_parameters": vector}, str(
                semantic.get("rationale") or payload.get("rationale") or action.rationale
            )
        return None

    def _evaluate_candidate(
        self,
        *,
        xc: XCFunctional,
        generation: int,
        root_result: BenchmarkResult,
        baseline_validation: BenchmarkResult,
        runner,
        validation_runner,
    ) -> EvolutionCandidate:
        candidate = EvolutionCandidate(
            xc=xc,
            parameters=self._read_parameters(xc),
            generation=generation,
        )
        hard = [item for item in xc.validate() if item.startswith("HARD:")]
        if hard:
            candidate.failure_reason = "hard_constraint_violation"
            candidate.fitness = float("-inf")
            candidate.tree_reward = 0.0
            return candidate
        candidate.search_result = runner.evaluate(xc)
        candidate.validation_result = validation_runner.evaluate(xc)
        self.scorer._score(candidate, root_result, baseline_validation)
        logger.info(
            "Warm-start evaluated stage generation=%d name=%s raw_reward=%.6f tree_reward=%.6f "
            "promotable=%s failure=%s SIE=%.4f guard=%.4f params=%s",
            generation,
            xc.name,
            candidate.fitness,
            candidate.tree_reward,
            candidate.promotable,
            candidate.failure_reason,
            candidate.search_result.overall_mae if candidate.search_result else float("nan"),
            candidate.validation_result.overall_mae if candidate.validation_result else float("nan"),
            json.dumps(candidate.parameters, sort_keys=True),
        )
        return candidate

    def _read_parameters(self, xc: XCFunctional) -> dict[str, float]:
        parameters = self.scorer._parameter_space(xc)
        if not parameters:
            return {}
        return self.scorer._read_parameters(xc, parameters)

    def _select(self, records: list[WarmStartRecord]) -> list[WarmStartRecord]:
        selected: list[WarmStartRecord] = []
        seen: set[str] = set()
        for record in sorted(records, key=lambda item: _rank_key(item.candidate), reverse=True):
            candidate = record.candidate
            if not candidate.promotable or candidate.fitness < self.config.reward_threshold:
                continue
            sig = candidate.xc.effective_signature()
            if sig in seen:
                continue
            seen.add(sig)
            selected.append(record)
        return selected

    def write_artifacts(self, result: WarmStartResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_payload = [record.to_dict() for record in result.records]
        (output_dir / "warm_start_all_candidates.json").write_text(
            json.dumps(all_payload, indent=2, ensure_ascii=False)
        )
        (output_dir / "warm_start_population.json").write_text(
            json.dumps([record.to_dict() for record in result.selected], indent=2, ensure_ascii=False)
        )
        (output_dir / "warm_start_summary.json").write_text(
            json.dumps(result.summary, indent=2, ensure_ascii=False)
        )
        self._write_csv(result.records, output_dir / "warm_start_candidates.csv")

    def _write_csv(self, records: list[WarmStartRecord], path: Path) -> None:
        subset_names = sorted({
            subset
            for record in records
            for benchmark in (record.candidate.search_result, record.candidate.validation_result)
            if benchmark is not None
            for subset in benchmark.subsets
        })
        fields = [
            "rank",
            "stage",
            "scaffold_index",
            "tuning_step",
            "scaffold_name",
            "name",
            "signature",
            "effective_signature",
            "raw_reward",
            "tree_reward",
            "promotable",
            "failure_reason",
            "guard_penalty",
            "worst_validation_ratio",
        ]
        for subset in subset_names:
            fields.extend([f"{subset}_mae", f"{subset}_failures"])
        fields.extend(["parameters_json", "action_detail_json", "xc_json"])
        ranked = sorted(records, key=lambda item: _rank_key(item.candidate), reverse=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rank, record in enumerate(ranked, start=1):
                candidate = record.candidate
                row: dict[str, Any] = {
                    "rank": rank,
                    "stage": record.stage,
                    "scaffold_index": record.scaffold_index,
                    "tuning_step": record.tuning_step,
                    "scaffold_name": record.scaffold_name,
                    "name": candidate.xc.name,
                    "signature": candidate.xc.structural_signature(),
                    "effective_signature": candidate.xc.effective_signature(),
                    "raw_reward": candidate.fitness,
                    "tree_reward": candidate.tree_reward,
                    "promotable": candidate.promotable,
                    "failure_reason": candidate.failure_reason,
                    "guard_penalty": candidate.guard_penalty,
                    "worst_validation_ratio": candidate.worst_validation_ratio,
                    "parameters_json": json.dumps(candidate.parameters, sort_keys=True),
                    "action_detail_json": json.dumps(record.action_detail, sort_keys=True, default=str),
                    "xc_json": json.dumps(candidate.xc.to_dict(), sort_keys=True, default=str),
                }
                for subset in subset_names:
                    subset_result = _subset_result(candidate, subset)
                    row[f"{subset}_mae"] = subset_result.mae_kcal if subset_result else None
                    row[f"{subset}_failures"] = subset_result.n_failures if subset_result else None
                writer.writerow(row)

    def _log_ranked_records(self, records: list[WarmStartRecord]) -> None:
        lines = [
            "rank  raw_reward  stage              scaffold  step  SIE4x4  BH76  G21EA  G21IP  failures  name",
            "----  ----------  -----------------  --------  ----  ------  ----  -----  -----  --------  ----",
        ]
        for rank, record in enumerate(sorted(records, key=lambda item: _rank_key(item.candidate), reverse=True), start=1):
            candidate = record.candidate
            failures = 0
            for benchmark in (candidate.search_result, candidate.validation_result):
                failures += benchmark.total_failures if benchmark else 0
            lines.append(
                f"{rank:>4}  {_fmt(candidate.fitness):>10}  {record.stage:<17}  "
                f"{record.scaffold_index:>8}  {record.tuning_step:>4}  "
                f"{_subset_mae(candidate, 'SIE4x4'):>6}  {_subset_mae(candidate, 'BH76'):>4}  "
                f"{_subset_mae(candidate, 'G21EA'):>5}  {_subset_mae(candidate, 'G21IP'):>5}  "
                f"{failures:>8}  {candidate.xc.name}"
            )
        logger.info("%s", log_block("Warm-start candidates ranked before seed selection", "\n".join(lines), color="magenta"))

    def _baseline_signatures(self, knowledge: KnowledgeEngine | None) -> set[str]:
        if knowledge is None:
            return set()
        rows = knowledge.conn.execute(
            "SELECT structural_signature, xc_json FROM functional_evaluations WHERE source='libxc_prerun'"
        ).fetchall()
        sigs: set[str] = set()
        for structural_sig, xc_json in rows:
            if structural_sig:
                sigs.add(str(structural_sig))
            try:
                xc = XCFunctional.from_dict(json.loads(xc_json or "{}"))
                sigs.add(xc.structural_signature())
                sigs.add(xc.effective_signature())
            except Exception:
                continue
        return sigs

    def _llm_json(
        self,
        idea_agent: IdeaAgent,
        prompt: str,
        label: str,
        *,
        temperature: float,
    ) -> dict[str, Any]:
        import httpx
        import openai

        client = openai.OpenAI(
            api_key=idea_agent.api_key or "dummy",
            base_url=idea_agent.base_url,
            timeout=getattr(idea_agent, "llm_timeout_s", 120.0),
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        )
        started = time.time()
        response = idea_agent._with_llm_retries(
            lambda: client.chat.completions.create(
                model=idea_agent.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=False,
            ),
            openai_module=openai,
            operation=label,
        )
        content = response.choices[0].message.content or ""
        logger.info(
            "%s",
            log_block(
                f"LLM warm-start response ({label}, chars={len(content)}, wall={time.time() - started:.1f}s)",
                pretty_json_or_text(content),
                color="cyan",
            ),
        )
        return _extract_json_object(content)


def _extract_json_object(content: str) -> dict[str, Any]:
    start = content.find("{")
    end = content.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"LLM response contains no JSON object: {content[:300]}")
    payload = json.loads(content[start:end])
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON response must be an object")
    return payload


def _format_component_catalogue(component_pool: str = "expanded") -> str:
    roles = component_roles_for_pool(component_pool)
    exchange = sorted(name for name, role in roles.items() if role == "exchange")
    correlation = sorted(name for name, role in roles.items() if role == "correlation")
    return "Pure exchange: " + ", ".join(exchange) + "\nPure correlation: " + ", ".join(correlation)


def _format_result(result: BenchmarkResult) -> str:
    return json.dumps(_result_metrics(result), indent=2, ensure_ascii=False)


def _result_metrics(result: BenchmarkResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        name: {
            "mae_kcal": subset.mae_kcal,
            "n_failures": subset.n_failures,
            "n_reactions": subset.n_reactions,
        }
        for name, subset in result.subsets.items()
    }


def _record_brief(record: WarmStartRecord) -> dict[str, Any]:
    candidate = record.candidate
    return {
        "stage": record.stage,
        "step": record.tuning_step,
        "raw_reward": candidate.fitness,
        "promotable": candidate.promotable,
        "failure_reason": candidate.failure_reason,
        "search": _result_metrics(candidate.search_result),
        "guard": _result_metrics(candidate.validation_result),
        "parameters": candidate.parameters,
    }


def _subset_result(candidate: EvolutionCandidate, subset: str):
    if candidate.search_result and subset in candidate.search_result.subsets:
        return candidate.search_result.subsets[subset]
    if candidate.validation_result and subset in candidate.validation_result.subsets:
        return candidate.validation_result.subsets[subset]
    return None


def _subset_mae(candidate: EvolutionCandidate, subset: str) -> str:
    sr = _subset_result(candidate, subset)
    return f"{sr.mae_kcal:.3f}" if sr else "n/a"


def _fmt(value: float) -> str:
    if value == float("-inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def _parameter_directions(
    before: dict[str, float],
    after: dict[str, float],
    *,
    tol: float = 1e-8,
) -> set[tuple[str, str]]:
    directions: set[tuple[str, str]] = set()
    for key, value in after.items():
        old = before.get(key)
        if old is None:
            continue
        if value > old + tol:
            directions.add((key, "increase"))
        elif value < old - tol:
            directions.add((key, "decrease"))
    return directions


def _banned_direction_violation(
    current: dict[str, float],
    changes: dict[str, float],
    banned: set[tuple[str, str]],
    *,
    tol: float = 1e-8,
) -> list[str]:
    violations: list[str] = []
    for key, value in changes.items():
        old = current.get(key)
        if old is None:
            continue
        direction = "increase" if value > old + tol else "decrease" if value < old - tol else ""
        if direction and (key, direction) in banned:
            violations.append(f"{key}:{direction}")
    return violations


def _structure_parameter_keys(changes: dict[str, Any]) -> list[str]:
    disallowed: list[str] = []
    parameter_like = {
        "alpha",
        "beta",
        "omega",
        "mixing",
        "mixing_factor",
        "ext_params",
        "ext_param_bounds",
        "new_ext_params",
        "new_ext_param_bounds",
        "params",
    }
    for key in changes:
        text = str(key)
        if text in parameter_like or text.startswith("mixing:") or "." in text:
            disallowed.append(text)
    return disallowed


def _rank_key(candidate: EvolutionCandidate) -> tuple[float, float, float]:
    return EvolutionarySearch._rank_key(candidate)
