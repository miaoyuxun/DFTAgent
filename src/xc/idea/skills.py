"""Directory-backed XC edit skills used only during idea generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .actions import ActionType


@dataclass
class EditSkill:
    name: str
    description: str
    applicability: list[str]
    blueprint: list[ActionType]
    guardrails: list[str] = field(default_factory=list)
    required_protocols: list[str] = field(default_factory=list)
    execution_logic: list[str] = field(default_factory=list)
    control_arm: bool = False
    source_path: str = ""
    attempts: int = 0
    successes: int = 0
    reward_ema: float = 0.0

    @property
    def prior_score(self) -> float:
        if self.attempts == 0:
            return 0.5
        # Laplace-smoothed success rate: (successes + 1) / (attempts + 2)
        # This prevents extreme swings with few data points:
        #   0/1 → 0.33 (not 0.0), 1/1 → 0.67 (not 1.0)
        # With enough data, converges to the raw rate.
        smoothed_rate = (self.successes + 1) / (self.attempts + 2)
        # Blend with reward EMA, but weight raw rate more as data accumulates
        # (min 50% weight on smoothed_rate once we have ≥3 attempts)
        rate_weight = max(0.5, 1.0 - 2.0 / (self.attempts + 2))
        return rate_weight * smoothed_rate + (1.0 - rate_weight) * min(self.reward_ema, 1.0)

    def to_prompt_line(self) -> str:
        return (
            f"{self.name}: {self.description} | applicability={self.applicability} | "
            f"blueprint={[item.value for item in self.blueprint]}"
        )


class SkillCatalog:
    def __init__(self, skill_root: Path | None = None):
        self.skill_root = skill_root or Path(__file__).with_name("skills")
        self.skills: dict[str, EditSkill] = {}
        self._selection_cursor = 0
        self._selection_credit: dict[str, float] = {}
        self.last_selection_debug: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        self.skills.clear()
        if not self.skill_root.exists():
            return
        for path in sorted(self.skill_root.glob("*/SKILL.md")):
            skill = self._parse(path)
            if skill is not None:
                self.skills[skill.name] = skill

    def _parse(self, path: Path) -> EditSkill | None:
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        sections = _parse_sections(body)
        name = frontmatter.get("name") or path.parent.name
        description = frontmatter.get("description", "").strip()
        if str(frontmatter.get("disabled", "false")).lower() == "true":
            return None
        blueprint: list[ActionType] = []
        for raw in sections.get("atomic_blueprint", []):
            token = raw.split("(", 1)[0].strip().lower()
            try:
                blueprint.append(ActionType(token))
            except ValueError:
                continue
        if not description or not blueprint:
            return None
        control_arm = str(frontmatter.get("control_arm", "false")).lower() == "true"
        return EditSkill(
            name=name,
            description=description,
            applicability=sections.get("applicability", []),
            blueprint=blueprint,
            guardrails=sections.get("guardrails", []),
            required_protocols=sections.get("required_protocols", []),
            execution_logic=sections.get("execution_logic", []),
            control_arm=control_arm,
            source_path=str(path),
        )

    def get(self, name: str) -> EditSkill:
        return self.skills[name]

    def update(self, name: str, reward: float, success: bool) -> None:
        skill = self.skills.get(name)
        if skill is None:
            return
        skill.attempts += 1
        skill.successes += int(success)
        skill.reward_ema = 0.8 * skill.reward_ema + 0.2 * max(0.0, min(1.0, reward))

    def select_skills(
        self,
        *,
        findings: Iterable,
        agenda: list[str] | None,
        limit: int,
        xc=None,
        taste_bias: dict[str, float] | None = None,
        adaptive_bias: dict[str, float] | None = None,
        defect_tags: Iterable[str] | None = None,
        excluded_skills: Iterable[str] | None = None,
    ) -> list[EditSkill]:
        del findings, agenda, taste_bias, defect_tags
        excluded = set(excluded_skills or [])
        eligible: list[EditSkill] = []
        filtered: dict[str, str] = {}
        penalties: dict[str, float] = {}
        for skill in sorted(self.skills.values(), key=lambda item: item.name):
            if skill.name in excluded:
                filtered[skill.name] = "excluded by scheduler gate"
                continue
            structure_fit = _structure_fit(skill, xc)
            if structure_fit == 0.0:
                filtered[skill.name] = _hard_filter_reason(skill, xc)
                continue
            eligible.append(skill)
            penalties[skill.name] = _soft_penalty_factor(skill, adaptive_bias or {})
        if not eligible:
            self.last_selection_debug = {
                "mode": "uniform_round_robin",
                "eligible": [],
                "filtered": filtered,
                "penalties": {},
                "probabilities": {},
                "selected": [],
            }
            return []

        total_weight = sum(penalties.get(skill.name, 1.0) for skill in eligible) or 1.0
        active_names = {skill.name for skill in eligible}
        self._selection_credit = {
            name: credit for name, credit in self._selection_credit.items()
            if name in active_names
        }
        for skill in eligible:
            self._selection_credit[skill.name] = (
                self._selection_credit.get(skill.name, 0.0)
                + penalties.get(skill.name, 1.0)
            )
        offset = self._selection_cursor % len(eligible)
        order = {
            skill.name: idx
            for idx, skill in enumerate(eligible[offset:] + eligible[:offset])
        }
        selected = sorted(
            eligible,
            key=lambda skill: (-self._selection_credit.get(skill.name, 0.0), order[skill.name]),
        )[: max(1, limit)]
        for skill in selected:
            self._selection_credit[skill.name] = self._selection_credit.get(skill.name, 0.0) - total_weight / len(eligible)
        self._selection_cursor = (self._selection_cursor + max(1, len(selected))) % len(eligible)
        probabilities = {
            skill.name: penalties.get(skill.name, 1.0) / total_weight
            for skill in eligible
        }
        self.last_selection_debug = {
            "mode": "uniform_round_robin",
            "eligible": [skill.name for skill in eligible],
            "filtered": filtered,
            "penalties": penalties,
            "probabilities": probabilities,
            "selected": [skill.name for skill in selected],
        }
        return selected


def _structure_fit(skill: EditSkill, xc) -> float:
    """Cheap XC-aware compatibility score for skill selection."""
    if xc is None:
        return 0.5
    n_components = len(getattr(xc, "components", []) or [])
    has_exchange = any(getattr(c, "role", "") == "exchange" for c in xc.components)
    has_correlation = any(getattr(c, "role", "") == "correlation" for c in xc.components)
    has_range_sep = bool(
        getattr(getattr(xc, "hf_exchange", None), "omega", 0.0)
        or any("RSH" in getattr(c, "name", "").upper() for c in xc.components)
    )
    if skill.name == "component-simplification":
        return 1.0 if n_components > 2 else 0.0
    if skill.name == "component-replacement":
        return 1.0 if has_exchange and has_correlation else 0.0
    if skill.name == "range-separation":
        return 0.25 if has_range_sep else 1.0
    if skill.name == "descriptor-synthesis":
        has_descriptor = any(
            getattr(c, "role", "") == "descriptor_correction" or getattr(c, "descriptor_spec", None)
            for c in xc.components
        )
        return 0.0 if has_descriptor else 0.8
    return 0.5


def _soft_penalty_factor(skill: EditSkill, adaptive_bias: dict[str, float]) -> float:
    """Only negative feedback can reduce a skill's turn priority."""
    factor = 1.0
    bias = float(adaptive_bias.get(skill.name, 0.0))
    if bias <= -1.0:
        factor *= 0.5
    elif bias <= -0.5:
        factor *= 0.7
    if skill.attempts >= 4:
        failure_rate = 1.0 - (skill.successes / max(1, skill.attempts))
        if failure_rate >= 0.8:
            factor *= 0.7
    return max(0.35, min(1.0, factor))


def _hard_filter_reason(skill: EditSkill, xc) -> str:
    if xc is None:
        return "incompatible"
    n_components = len(getattr(xc, "components", []) or [])
    has_exchange = any(getattr(c, "role", "") == "exchange" for c in xc.components)
    has_correlation = any(getattr(c, "role", "") == "correlation" for c in xc.components)
    has_descriptor = any(
        getattr(c, "role", "") == "descriptor_correction" or getattr(c, "descriptor_spec", None)
        for c in getattr(xc, "components", [])
    )
    if skill.name == "component-simplification" and n_components <= 2:
        return "needs more than two components"
    if skill.name == "component-replacement" and not (has_exchange and has_correlation):
        return "needs exchange and correlation components"
    if skill.name == "descriptor-synthesis" and has_descriptor:
        return "descriptor already present"
    return "incompatible"


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    _, raw, body = text.split("---\n", 2)
    data: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip('"')
    return data, body


def _parse_sections(body: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections[current] = []
        elif current and line:
            if line.startswith("- "):
                sections[current].append(line[2:].strip())
            elif current == "execution_logic":
                sections[current].append(line.lstrip("0123456789. ").strip())
    return sections
