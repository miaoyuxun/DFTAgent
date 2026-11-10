from __future__ import annotations

from examples.custom_reward_task import CallbackRewardTask, example_reward
from scripts.run_mean_mae_basin_agent import _expand_environment
from xc.idea.agent import IdeaAgent
from xc.idea.basin_agent import BasinAgentConfig, LLMBasinAgentSearch
from xc.idea.diverse_bo_basin import DiverseBOBasinConfig, DiverseBOBasinSearch
from xc.xc_functional import XCFunctional


def test_public_search_classes_construct() -> None:
    assert LLMBasinAgentSearch(config=BasinAgentConfig(cycles=1))
    assert DiverseBOBasinSearch(config=DiverseBOBasinConfig(cycles=1))


def test_basin_agent_loads_public_skills() -> None:
    agent = IdeaAgent()
    assert "component-replacement" in agent.skill_catalog.skills
    assert "parameter-control" in agent.skill_catalog.skills


def test_callback_reward_task_separates_objective_and_diagnostics() -> None:
    def evaluator(xcs, labels, stage):
        assert labels == ["candidate"]
        assert stage == "smoke"
        return [
            {
                "valid": True,
                "objective": {"target_error": 0.2, "guard_error": 0.3},
                "diagnostics": {"scf_cycles": 7},
                "walltime_s": 1.5,
            }
            for _ in xcs
        ]

    task = CallbackRewardTask(
        evaluator=evaluator,
        reward_function=example_reward,
        instructions="Minimize target error while respecting the guard.",
    )
    result = task.evaluate_many(
        [XCFunctional.from_known("PBE")],
        labels=["candidate"],
        stage="smoke",
    )[0]
    assert result.valid
    assert abs(result.reward - 0.7) < 1e-12
    assert result.objective["target_error"] == 0.2
    assert result.proposal_feedback == {"scf_cycles": 7}


def test_yaml_environment_placeholders_are_expanded(monkeypatch) -> None:
    monkeypatch.setenv("DFTAGENT_PUBLIC_TEST_TOKEN", "configured")
    payload = {
        "llm": {
            "api_key": "${DFTAGENT_PUBLIC_TEST_TOKEN}",
            "base_url": "${DFTAGENT_PUBLIC_MISSING_VALUE}",
        }
    }

    expanded = _expand_environment(payload)
    assert expanded["llm"]["api_key"] == "configured"
    assert expanded["llm"]["base_url"] == ""
