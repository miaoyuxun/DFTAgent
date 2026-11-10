"""Prompts for artifact-first XC idea generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .ablation import AblationReport, format_ablation_report
from .knowledge_engine import KnowledgeBundle, format_knowledge_bundle
from .skills import EditSkill
from ..xc_functional import PURE_COMPONENT_ROLES

SYSTEM_PROMPT = """You are an expert DFT functional designer specializing in exchange-correlation (XC) functional optimization. Your task is to propose modifications to XC functionals to improve accuracy on benchmark datasets, guided by physical constraints and domain knowledge.

## Available Operations

You have 6 atomic operations:
- `rewire`: Tune parameters (alpha, omega, mu, kappa, mixing_factor, s6, s8, etc.)
- `add_component`: Add a libxc component (including dispersion corrections D3/D3BJ)
- `add_descriptor`: Add one restricted GGA-level descriptor correction from an approved template
- `remove_component`: Remove an existing component
- `replace_component`: Swap one component for another (e.g., PBE → SCAN exchange)
- `enforce_constraint`: Apply a physical constraint (Lieb-Oxford, one-electron SIE, LDA limit)

## Skills

The selected Skill is provided in the user prompt as scientific framing. If an
Action Family Contract is present, it is the hard executable constraint and
overrides the Skill blueprint for action_type selection.

## How to Interpret Ablation Results (component-level causal evidence)

- `result=positive`: removing this component HURT performance → it is **helpful**. Keep it or strengthen it.
- `result=negative`: removing this component HELPED performance → it is **harmful** or redundant. Consider removing or replacing it.
- `result=inconclusive`: removing had no significant effect → the component is negligible. Avoid overcommitting to changes based on this evidence.

## Anti-Pattern Guards (DO NOT do these)

- Do NOT default to a parameter-only tweak unless the selected Skill explicitly marks the proposal as a control arm.
- Do NOT add a component without specifying its physical role and how it interacts with existing components.
- Do NOT add more than one custom descriptor correction.
- Do NOT write arbitrary Python code for descriptors; use only the approved descriptor schema.
- Do NOT let alpha exceed 0.50 — this hurts thermochemistry significantly.
- Do NOT add more than 6 components — the functional becomes uninterpretable and overfitting-prone.
- Do NOT directly copy parameters from another functional without checking sum rules and constraint compatibility.
- Do NOT add a second dispersion correction if one is already present.
- Do NOT propose changes that violate the exchange/correlation sum rules (mixing factors must sum to 1.0).
- Do NOT introduce a gate/router/threshold mechanism — DFT functionals are deterministic energy expressions, not conditional architectures.

## Key Principles

- Small, incremental changes are preferred over large jumps, but each change must be physically motivated.
- Respect physical constraints (Lieb-Oxford bound, sum rules, LDA limit).
- Consider trade-offs: improving one subset may hurt another.
- Use ablation evidence as your primary signal for component decisions.
- Use domain knowledge to guide parameter choices and replacement selections.
- When replacing a component, specify the physical reason (e.g., "SCAN uses kinetic density tau to distinguish covalent from weak bonds, which PBE cannot do").
- When tuning a parameter, specify the target value and the expected effect (e.g., "Increasing omega from 0 to 0.30 introduces long-range HF exchange, which should reduce SIE4x4 MAE by ~2 kcal/mol").

## Response Format

Return STRICT JSON (no prose, no Markdown outside the JSON):

```json
{
  "action_type": "<one of: rewire, add_component, add_descriptor, remove_component, replace_component, enforce_constraint>",
  "changes": {
    "<parameter_or_component_key>": <value>,
    ...
  },
  "rationale": "<detailed explanation referencing: (1) which finding this addresses, (2) why this change was chosen over alternatives, (3) what observation or ablation evidence supports it, (4) expected effect on each benchmark subset>",
  "core_contribution": "<one-sentence statement of what this modification achieves physically>",
  "component_explanations": {
    "<component_name>": "<role description for any new or modified component>"
  },
  "target_findings": ["<finding_id_or_note>"],
  "edit_plan": {
    "skill_name": "<skill_name>",
    "objective": "<what this plan aims to achieve>",
    "target_findings": ["<finding_id_or_note>"],
    "component_edits": [
      {
        "op": "<action_type>",
        "component": "<component_or_parameter_name>",
        "target": {"<key>": "<value>"},
        "condition": "<when this edit applies>",
        "details": "<what specifically this edit does>",
        "reason": "<why this edit was chosen>"
      }
    ],
    "guardrails": ["<guardrail_from_skill>"],
    "memory_refs": ["<knowledge_key_referenced>"]
  },
  "skill_metrics": {
    "idea_taste_mode": "<taste_mode_or_default>",
    "guardrails_applied": ["<guardrail>"],
    "constraints": ["<constraint>"]
  }
}
```

The `rationale` field must be at least 100 characters and reference specific evidence. The `core_contribution` must be a precise physical statement, not a vague description. The `edit_plan.component_edits` must detail each atomic operation with a `reason` field explaining why it was chosen.
"""


def build_proposal_prompt(
    current_best,
    current_result,
    target_problem: str = "improve overall MAE",
    knowledge_bundle=None,
    scientific_prior_text: str | None = None,
    preferred_action=None,
    ablation_report: AblationReport | None = None,
    selected_skill: EditSkill | None = None,
    analysis=None,
    immediate_feedback: str | None = None,
    search_feedback: list[str] | None = None,
    action_contract: dict | None = None,
) -> str:
    """Build the user prompt from structured artifacts and one edit skill."""
    parts = []

    # 1. Target problem
    parts.append(f"## Target Problem\n{target_problem}\n")

    if analysis is not None:
        parts.append("## Structured Research Brief")
        parts.append(analysis.research_brief)
        if analysis.analysis_artifact:
            if analysis.analysis_artifact.research_notes:
                parts.append("### Research Notes")
                parts.extend(f"- {item}" for item in analysis.analysis_artifact.research_notes)
            if analysis.analysis_artifact.open_questions:
                parts.append("### Open Questions")
                parts.extend(f"- {item}" for item in analysis.analysis_artifact.open_questions)
        if getattr(analysis, "search_reflection", ""):
            parts.append("### Search Reflection")
            parts.append(analysis.search_reflection)
        parts.append("")

    # 3. Current functional (with component-level detail)
    parts.append("## Current Best Functional")
    parts.append(f"Name: {current_best.name}")
    parts.append(f"Components: {_format_components(current_best)}")
    parts.append(f"HF Exchange: alpha={current_best.hf_exchange.alpha}, "
                 f"beta={current_best.hf_exchange.beta}, omega={current_best.hf_exchange.omega}")
    if current_best.metadata:
        parts.append(f"Metadata: {current_best.metadata}")
        if "last_raw_reward" in current_best.metadata:
            parts.append(
                "Score interpretation: last_raw_reward is the uncapped optimization score "
                "used to compare incumbents; last_reward is capped to [0, 1] only for MCTS tree search."
            )
    parts.append("")

    # 4. Current results (with per-subset detail)
    parts.append("## Current Benchmark Results")
    for subset_name, sr in current_result.subsets.items():
        parts.append(f"- {subset_name}: MAE={sr.mae_kcal:.2f}, RMS={sr.rms_kcal:.2f}, "
                     f"max={sr.max_error_kcal:.2f}, n={sr.n_reactions}, failures={sr.n_failures}")
    parts.append(f"Overall MAE: {current_result.overall_mae:.2f} kcal/mol\n")

    # 6. Ablation report (PRIMARY signal — mirrors Xcientist's re_analysis_replan)
    if ablation_report and ablation_report.ablation_results:
        parts.append("## Component Ablation Results (PRIMARY signal for your decision)")
        parts.append(format_ablation_report(ablation_report))
        parts.append("")
        parts.append("Use these ablation results as your PRIMARY evidence for deciding which "
                      "components to keep, strengthen, remove, or replace. Every component "
                      "decision must reference specific ablation evidence.")
        parts.append("")

    # 7. Scientific prior document (prompt-safe empirical background)
    if scientific_prior_text:
        parts.append("## Scientific Prior")
        parts.append(
            "The following document is background scientific prior only. Treat it as empirical "
            "evidence and mechanism intuition. Do NOT let it override the action family contract, "
            "runtime benchmark evidence, ablation evidence, or other hard executable constraints."
        )
        parts.append(scientific_prior_text)
        parts.append("")

    # 8. Knowledge context (includes domain knowledge, anti-patterns, traces)
    if knowledge_bundle:
        knowledge_text = format_knowledge_bundle(knowledge_bundle)
        if knowledge_text and knowledge_text != "No prior knowledge available.":
            parts.append("## Knowledge Context")
            parts.append(knowledge_text)
            parts.append("")

    # 9. Selected Skill (with guardrails and execution logic)
    if selected_skill:
        parts.append("## Selected Skill: " + selected_skill.name)
        parts.append(
            "Note: this Skill provides scientific framing. The Action Family Contract below, "
            "if present, is the HARD executable constraint."
        )
        parts.append(f"Description: {selected_skill.description}")
        parts.append(f"Blueprint: [{', '.join(b.value for b in selected_skill.blueprint)}]")
        if selected_skill.applicability:
            parts.append(f"Applicability: {selected_skill.applicability}")
        if selected_skill.control_arm:
            parts.append("Role: control arm")
        if selected_skill.guardrails:
            parts.append("Guardrails:")
            for g in selected_skill.guardrails:
                parts.append(f"  - {g}")
        if selected_skill.execution_logic:
            parts.append("Execution logic:")
            for step in selected_skill.execution_logic:
                parts.append(f"  {step}")
        if selected_skill.required_protocols:
            parts.append(f"Required validation protocols: {selected_skill.required_protocols}")
        parts.append("")
    elif preferred_action:
        parts.append(f"## Suggested Action\n{preferred_action}\n")

    if action_contract:
        parts.append("## Action Family Contract (HARD EXECUTABLE CONSTRAINT)")
        parts.append(f"Family: {action_contract.get('action_family')}")
        parts.append(f"Description: {action_contract.get('description', '')}")
        parts.append(
            "Allowed action_type values: "
            + ", ".join(str(item) for item in action_contract.get("allowed_action_types", []))
        )
        parts.append("Requirements:")
        for item in action_contract.get("requirements", []):
            parts.append(f"- {item}")
        examples = action_contract.get("examples", [])
        if examples:
            parts.append("Valid schema examples:")
            for example in examples[:3]:
                parts.append(f"- {example}")
        parts.append(
            "If your preferred scientific idea does not fit this family, do NOT switch families; "
            "choose a different idea that fits this contract."
        )
        parts.append("")

    parts.append("## Approved Pure Component Catalogue")
    parts.append(
        "Only use these entries for slot-level add/replace operations. Pure exchange "
        "components may only occupy exchange slots; pure correlation components may "
        "only occupy correlation slots. Use LibXC defaults for entries without exposed parameters."
    )
    exchange_components = sorted(
        name for name, role in PURE_COMPONENT_ROLES.items() if role == "exchange"
    )
    correlation_components = sorted(
        name for name, role in PURE_COMPONENT_ROLES.items() if role == "correlation"
    )
    parts.append("Pure exchange: " + ", ".join(exchange_components))
    parts.append("Pure correlation: " + ", ".join(correlation_components))
    parts.append("")

    if immediate_feedback:
        parts.append("## Immediate Feedback From The Just-Rejected Attempt")
        parts.append(immediate_feedback)
        parts.append(
            "Revise the next proposal immediately so it preserves the useful intent "
            "while removing the violation. Do not repeat the rejected edit unchanged."
        )
        parts.append("")

    if search_feedback:
        parts.append("## Search Feedback From Duplicate Proposals")
        parts.append(
            "The following recent proposals compiled to benchmark-equivalent content and were skipped. "
            "Avoid repeating these same effective edits; change the actual LibXC expression, HF tuple, "
            "or active ext_params if you want a distinct candidate."
        )
        for item in search_feedback[-5:]:
            parts.append(f"- {item}")
        parts.append("")

    # 10. Anti-pattern guards (explicit negative constraints)
    parts.append("## Anti-Pattern Guards (DO NOT violate these)")
    parts.append("- Do NOT propose shallow parameter tweaks without physical justification")
    parts.append("- Do NOT add components without specifying their physical role")
    parts.append("- Do NOT add more than one custom descriptor correction")
    parts.append("- Do NOT write arbitrary Python code for descriptors; use only the approved descriptor schema")
    parts.append("- Do NOT violate sum rules (exchange/correlation mixing must sum to 1.0)")
    parts.append("- Do NOT set alpha > 0.50")
    parts.append("- Do NOT add more than 6 components total")
    parts.append("- `add_component` is allowed and useful for structure exploration: for pure exchange/correlation components, the compiler uses split-and-add, carving about 0.10 mixing from the largest existing same-role component instead of adding a full-weight component")
    parts.append("- Do NOT remove the only active exchange or correlation component; use replace_component, or add another same-role component first and remove only during later pruning")
    parts.append("- Do NOT add a second dispersion correction if one exists")
    parts.append("- Do NOT propose conditional/gate/router mechanisms — DFT is deterministic")
    parts.append("")

    # 11. Instruction with strict JSON schema
    parts.append("## Your Task")
    parts.append("Propose the next modification to improve the functional. "
                 "If ablation results are available, use them as your PRIMARY signal "
                 "for component decisions. If an Action Family Contract is specified, obey it exactly. "
                 "Your response must be STRICT JSON with all fields filled.")
    parts.append("")
    parts.append("Required JSON fields:")
    parts.append("- `action_type`: one of rewire, add_component, add_descriptor, remove_component, replace_component, enforce_constraint")
    parts.append("- `changes`: dict of parameter/component changes")
    parts.append("- For `add_descriptor`, `changes.descriptor` must include template, scale, gradient_sensitivity, density_decay, rho_power, intended_effect")
    parts.append("- `rationale`: ≥100 chars, referencing findings, ablation evidence, and expected effects")
    parts.append("- `core_contribution`: one-sentence physical statement of what this achieves")
    parts.append("- `component_explanations`: dict mapping component names to role descriptions")
    parts.append("- `target_findings`: list of finding IDs or research-note anchors")
    parts.append("- `edit_plan`: structured plan with skill_name, objective, component_edits (each with op, component, target, condition, details, reason), guardrails")
    parts.append("- `skill_metrics`: dict with idea_taste_mode, guardrails_applied, constraints")

    return "\n".join(parts)


def format_parent_pool_functional_graph_evidence(
    subgraph: dict[str, Any] | None,
) -> str:
    """Render one shared permanent-memory graph for a Basin parent pool."""
    if not isinstance(subgraph, Mapping):
        return ""
    nodes = [item for item in subgraph.get("nodes", []) if isinstance(item, Mapping)]
    edges = [item for item in subgraph.get("edges", []) if isinstance(item, Mapping)]
    anchors = [
        item
        for item in subgraph.get("parent_anchors", [])
        if isinstance(item, Mapping)
    ]
    if not nodes:
        return ""

    labels = {
        str(node.get("node_id", "")): f"G{index}"
        for index, node in enumerate(nodes)
        if node.get("node_id")
    }
    lines = [
        "## Shared Parent-Pool Evolution Subgraph",
        "This is evaluated historical decision evidence related to the current successful parent pool. "
        "Original one-action edges are replay-validated; Basin composite edges preserve the evaluated action sequence. "
        "Parent-to-anchor links express structural relatedness and are not executable actions. "
        "Positive target_gain means lower MAE.",
        "",
        "### Parent-to-Historical Anchors",
    ]
    for anchor in anchors:
        anchor_label = labels.get(str(anchor.get("anchor_node_id", "")))
        if anchor_label is None:
            continue
        lines.append(
            f"- Basin parent {_graph_text(anchor.get('parent_id'), 40)} "
            f"({_graph_text(anchor.get('parent_name'), 60)}) -> {anchor_label} "
            f"(match={_graph_number(anchor.get('match_score'), 2)})"
        )

    lines.extend(["", "### Historical Functional Nodes"])
    lines.extend(
        f"- {labels.get(str(node.get('node_id', '')), '?')} "
        f"[{_graph_text(node.get('node_id'), 80)}]: {_format_graph_node(node)}"
        for node in nodes
    )

    lines.extend(["", "### Directed Historical Action Edges"])
    for edge in edges:
        source = labels.get(str(edge.get("source", "")))
        target = labels.get(str(edge.get("target", "")))
        if source is None or target is None:
            continue
        lines.append(f"- {_format_graph_edge(edge, source, target)}")
    return "\n".join(lines)


def _format_graph_node(node: Mapping[str, Any]) -> str:
    components = []
    for component in node.get("components", []) or []:
        if not isinstance(component, Mapping):
            continue
        text = (
            f"{_graph_text(component.get('name'), 50)}"
            f"[{_graph_text(component.get('role'), 20)},mix={_graph_number(component.get('mixing_factor'), 4)}]"
        )
        components.append(text)

    hybrid = node.get("hybrid", {})
    hybrid = hybrid if isinstance(hybrid, Mapping) else {}
    hf = hybrid.get("hf_exchange", {})
    hf = hf if isinstance(hf, Mapping) else {}
    backend = hybrid.get("backend_coefficients", {})
    backend = backend if isinstance(backend, Mapping) else {}
    rsh = backend.get("rsh_coeff", [])
    rsh_text = ",".join(_graph_number(value, 5) for value in rsh) if isinstance(rsh, list) else "unknown"

    metrics = []
    for subset, payload in (node.get("metrics", {}) or {}).items():
        if isinstance(payload, Mapping):
            metrics.append(
                f"{_graph_text(subset, 20)}={_graph_number(payload.get('mae_kcal'), 3)}"
                f"(fail={_graph_text(payload.get('n_failures'), 12) or 'unknown'})"
            )
    data = node.get("data_status", {})
    data = data if isinstance(data, Mapping) else {}
    return (
        f"name={_graph_text(node.get('name'), 60)}; rung={_graph_text(node.get('rung'), 20)}; "
        f"role={_graph_text(node.get('role'), 30)}; components={','.join(components) or 'none'}; "
        f"hf=(alpha={_graph_number(hf.get('alpha'), 5)},beta={_graph_number(hf.get('beta'), 5)},"
        f"omega={_graph_number(hf.get('omega'), 5)}); "
        f"backend=(hybrid_coeff={_graph_number(backend.get('hybrid_coeff'), 5)},rsh=[{rsh_text}]); "
        f"metrics={','.join(metrics) or 'unknown'}; "
        f"data=operability:{_graph_text(data.get('operability'), 20)},"
        f"conversion:{_graph_text(data.get('conversion_status'), 20)},"
        f"completeness:{_graph_number(data.get('metric_completeness'), 2)}"
    )


def _format_graph_edge(edge: Mapping[str, Any], source: str, target: str) -> str:
    operation = edge.get("operation", {})
    operation = operation if isinstance(operation, Mapping) else {}
    structural = operation.get("structural_change", {})
    structural = structural if isinstance(structural, Mapping) else {}
    action_type = _graph_text(operation.get("action_type"), 24) or "unknown"
    if action_type == "composite":
        step_texts = []
        steps = operation.get("steps", []) or []
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            changes = step.get("changes", {})
            changes_text = _graph_text(
                json.dumps(changes, sort_keys=True, ensure_ascii=False, default=str),
                180,
            )
            step_texts.append(
                f"{_graph_text(step.get('index'), 8)}:"
                f"{_graph_text(step.get('action_type'), 24)}({changes_text})"
            )
        action = (
            f"composite#{_graph_text(operation.get('primitive_action_count'), 8)}"
            f"[{';'.join(step_texts)}]"
        )
    elif action_type == "replace":
        structural_text = (
            f"{_graph_text(structural.get('old_name'), 45)}"
            f"->{_graph_text(structural.get('new_name'), 45)}"
        )
        action = f"{action_type}({structural_text})"
    else:
        structural_text = _graph_text(structural.get("name"), 50)
        action = f"{action_type}({structural_text})" if structural_text else action_type

    changes = []
    parameter_changes = operation.get("parameter_changes", []) or []
    for item in parameter_changes[:8]:
        if not isinstance(item, Mapping):
            continue
        changes.append(
            f"{_graph_text(item.get('parameter'), 60)}:"
            f"{_graph_text(item.get('source_value'), 24) or 'unknown'}"
            f"->{_graph_text(item.get('target_value'), 24) or 'unknown'}"
        )
    if len(parameter_changes) > 8:
        changes.append(f"...(+{len(parameter_changes) - 8})")
    if operation.get("composite") and action_type != "composite":
        action += "+rewire"
    if changes:
        action += "[" + ",".join(changes) + "]"

    effects = []
    for subset, payload in (edge.get("metric_delta", {}) or {}).items():
        if not isinstance(payload, Mapping):
            continue
        delta = payload.get("delta_mae")
        try:
            value = float(delta)
            effect = f"target_gain={-value:+.3f}" if subset == "SIE4x4" else f"delta={value:+.3f}"
        except (TypeError, ValueError):
            effect = "delta=unknown"
        failure = payload.get("failure_delta")
        if failure is not None:
            effect += f",failure_delta={_graph_text(failure, 16)}"
        effects.append(f"{_graph_text(subset, 20)}({effect})")
    data = edge.get("data_status", {})
    data = data if isinstance(data, Mapping) else {}
    return (
        f"edge={_graph_text(edge.get('edge_id'), 80)}: {source} -> {target}; "
        f"depth={_graph_text(edge.get('path_depth'), 8)}; "
        f"relevance={_graph_text(edge.get('relevance'), 16)}; risk={bool(edge.get('risk'))}; "
        f"action={action}; effects={','.join(effects) or 'unknown'}; "
        f"data=completeness:{_graph_number(data.get('metric_completeness'), 2)},"
        f"llm_analysis:{_graph_text(data.get('llm_analysis_status'), 24) or 'unknown'}"
    )


def _graph_text(value: Any, limit: int) -> str:
    text = " ".join(("" if value is None else str(value)).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _graph_number(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "unknown"


def _format_components(xc) -> str:
    """Format component list for prompt with role descriptions."""
    lines = []
    for c in xc.components:
        params = f", ext_params={c.ext_params}" if c.ext_params else ""
        bounds = f", bounds={c.ext_param_bounds}" if c.ext_param_bounds else ""
        formula = f", formula={c.formula}" if c.formula else ""
        lines.append(f"  {c.name} ({c.role}, mixing={c.mixing_factor}{params}{bounds}{formula})")
    return "\n".join(lines)
