#!/usr/bin/env python3
"""Run the basin agent against an equal-weight mean-MAE benchmark task."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xc.benchmark.vc_manifest_runner import (  # noqa: E402
    VCManifestBenchmarkRunner,
    VCManifestRunnerConfig,
)
from xc.idea.agent import IdeaAgent  # noqa: E402
from xc.idea.basin_agent import BasinAgentConfig, LLMBasinAgentSearch  # noqa: E402
from xc.idea.evolutionary import EvolutionConfig  # noqa: E402
from xc.search_tasks import MeanMAEBenchmarkTask, MeanMAETaskConfig  # noqa: E402
from xc.xc_functional import XCFunctional  # noqa: E402
from xc.zno_component_registry import validate_component_pool_lock  # noqa: E402


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return _expand_environment(payload)


def _expand_environment(value):
    """Expand shell-style environment placeholders throughout a YAML tree."""

    if isinstance(value, str):
        pattern = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")
        return pattern.sub(
            lambda match: os.environ.get(
                match.group("braced") or match.group("plain"), ""
            ),
            value,
        )
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _resolve_env_value(value: str, *, fallback_env: str) -> str:
    value = str(value or "").strip()
    if value.startswith("$"):
        return os.environ.get(value[1:].strip("{}"), "")
    return os.path.expandvars(value) or os.environ.get(fallback_env, "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "mean_mae_basin.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-nodes", type=Path, default=None)
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    output_dir = args.output_dir or Path(config.get("output_dir", "results/mean_mae_basin"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, str(config.get("log_level", "INFO")).upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    for noisy in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    search_values = dict(config.get("search") or {})
    if args.seed is not None:
        search_values["seed"] = int(args.seed)
    if args.resume_nodes is not None:
        resume_path = args.resume_nodes.expanduser()
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        search_values["resume_nodes_path"] = str(resume_path.resolve())
    core_dir = str(output_dir) + "_core" if args.output_dir is not None else str(
        config.get("core_output_dir") or (str(output_dir) + "_core")
    )
    if not Path(core_dir).is_absolute():
        core_dir = str(ROOT / core_dir)
    search_values["core_output_dir"] = core_dir
    search_config = BasinAgentConfig(**search_values)

    task_values = dict(config.get("task") or {})
    subsets = tuple(str(name) for name in task_values.get("subsets", ()))
    task_component_pool = str(task_values.get("component_pool", search_config.component_pool))
    if task_component_pool != search_config.component_pool:
        raise ValueError(
            "search.component_pool and task.component_pool must match: "
            f"{search_config.component_pool!r} != {task_component_pool!r}"
        )
    pool_signature = validate_component_pool_lock(
        search_config.component_pool,
        config.get("component_pool_lock"),
    )
    logging.getLogger(__name__).info("COMPONENT_POOL_LOCK_OK %s", pool_signature)

    root_name = str(task_values.get("root_functional", "PBE0"))
    root_xc = XCFunctional.from_known(root_name)
    task_config = MeanMAETaskConfig(
        subsets=subsets,
        component_pool=task_component_pool,
        max_failures_per_subset=int(task_values.get("max_failures_per_subset", 0)),
        allow_hf_exchange=bool(task_values.get("allow_hf_exchange", True)),
        allow_ext_param_rewire=bool(task_values.get("allow_ext_param_rewire", False)),
    )
    # Validation is independent of the remote runner, which keeps smoke tests cheap.
    task = MeanMAEBenchmarkTask(runner=None, config=task_config)
    task.validate_functional(root_xc)
    _validate_split_pbe0(root_xc)

    common_eval = dict(config.get("evaluation") or {})
    benchmark_values = dict(config.get("benchmark_evaluation") or {})
    persistent_queue = _resolve_env_value(
        common_eval.get("persistent_queue_dir", "$DFTAGENT_PERSISTENT_VC_QUEUE"),
        fallback_env="DFTAGENT_PERSISTENT_VC_QUEUE",
    )
    vc_prefix = _resolve_env_value(
        common_eval.get("vc_prefix", "$DFTAGENT_VC_PREFIX"),
        fallback_env="DFTAGENT_VC_PREFIX",
    )
    evaluation_backend = str(common_eval.get("backend", "persistent_vc")).strip().lower()
    effective_config = {
        "output_dir": str(output_dir),
        "core_output_dir": core_dir,
        "search": search_values,
        "task": task_values,
        "component_pool_lock": config.get("component_pool_lock") or {},
        "component_pool_signature": pool_signature,
        "evaluation": {
            **common_eval,
            "persistent_queue_dir": persistent_queue,
            "vc_prefix_configured": bool(vc_prefix),
        },
        "benchmark_evaluation": benchmark_values,
        "root_effective_spec": root_xc.effective_spec(),
    }
    effective_path = output_dir / "effective_mean_mae_basin_config.json"
    effective_path.write_text(
        json.dumps(effective_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(core_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(effective_path, Path(core_dir) / effective_path.name)
    if args.validate_config:
        print(json.dumps(effective_config, indent=2, ensure_ascii=False), flush=True)
        print("Mean-MAE Basin Agent configuration is valid; no evaluator call was made.", flush=True)
        return 0

    if evaluation_backend not in {"vc", "persistent_vc"}:
        raise ValueError("evaluation.backend must be vc or persistent_vc")
    if evaluation_backend == "persistent_vc" and not persistent_queue:
        raise ValueError(
            "evaluation.persistent_queue_dir or DFTAGENT_PERSISTENT_VC_QUEUE is required"
        )
    if evaluation_backend == "vc" and not vc_prefix:
        raise ValueError("evaluation.vc_prefix or DFTAGENT_VC_PREFIX is required")
    runner = VCManifestBenchmarkRunner(
        VCManifestRunnerConfig(
            output_dir=output_dir / "benchmark_evaluations",
            vc_prefix=vc_prefix if evaluation_backend == "vc" else "",
            persistent_queue_dir=(
                Path(persistent_queue) if evaluation_backend == "persistent_vc" else None
            ),
            container_repo_root=str(common_eval.get("container_repo_root", "/workspace/DFTAgent")),
            python_executable=str(common_eval.get("python_executable", "python3")),
            conda_sh=str(common_eval.get("conda_sh", "")),
            conda_env=str(common_eval.get("conda_env", "")),
            pre_cmd=str(common_eval.get("pre_cmd", "")),
            gmtkn55_dir=Path(benchmark_values.get("gmtkn55_dir", ROOT / "benchmark" / "GMTKN55")),
            subsets=task_config.subsets,
            basis=str(benchmark_values.get("basis", "def2-TZVP")),
            scf_max_cycle=int(benchmark_values.get("scf_max_cycle", 200)),
            scf_conv_tol=float(benchmark_values.get("scf_conv_tol", 1e-8)),
            n_workers=int(benchmark_values.get("n_workers", 16)),
            n_threads=int(benchmark_values.get("n_threads", 1)),
            molecules_per_task=int(benchmark_values.get("molecules_per_task", 8)),
            aggregate_mode=str(benchmark_values.get("aggregate_mode", "reaction_mean")),
            concurrent_subset_jobs=int(common_eval.get("max_concurrent", 8)),
            vc_tasks_per_submit=int(common_eval.get("vc_tasks_per_submit", 0)),
            poll_seconds=float(common_eval.get("queue_monitor_poll_s", 2)),
            wait_timeout_seconds=float(common_eval.get("wait_timeout_seconds", 0)),
        ),
        repo_root=ROOT,
    )
    task.runner = runner

    llm_file = Path(config.get("llm_config", ROOT / "config" / "llm.yaml"))
    if not llm_file.is_absolute():
        llm_file = ROOT / llm_file
    llm_values = dict((_load_yaml(llm_file).get("llm") or {}))
    llm_values.update(config.get("llm_overrides") or {})
    if not str(llm_values.get("api_key") or "").strip():
        raise ValueError(
            "No LLM API key configured. Set OPENAI_API_KEY or edit config/llm.yaml."
        )
    idea = IdeaAgent(
        model=str(llm_values.get("model", "gpt-5.6-sol")),
        base_url=llm_values.get("base_url"),
        api_key=llm_values.get("api_key"),
        llm_timeout_s=float(llm_values.get("timeout_s", 600)),
        llm_retry_attempts=int(llm_values.get("retry_attempts", 0)),
        llm_retry_base_delay_s=float(llm_values.get("retry_initial_delay_s", 5)),
        llm_retry_max_delay_s=float(llm_values.get("retry_max_delay_s", 60)),
        llm_retry_backoff=float(llm_values.get("retry_backoff", 1.4)),
        target_problem=(
            "Minimize the equal-weight arithmetic mean MAE of ALKBDE10, DIPCS10, and G2RC. "
            "The maximized raw reward is the negative mean MAE."
        ),
        knowledge_engine=None,
        proposal_prior_path=None,
    )

    scorer_config = EvolutionConfig(
        seed=search_config.seed,
        component_pool=search_config.component_pool,
        include_global_hf=True,
        alpha_bounds=tuple(float(x) for x in task_values.get("alpha_bounds", [0.0, 0.5])),
        beta_bounds=tuple(float(x) for x in task_values.get("beta_bounds", [0.0, 1.0])),
        omega_bounds=tuple(float(x) for x in task_values.get("omega_bounds", [0.0, 0.5])),
        include_role_scales=True,
        role_scale_bounds=tuple(float(x) for x in task_values.get("role_scale_bounds", [0.8, 1.2])),
        include_component_mixing=True,
        include_component_ext_params=False,
        exact_mixing_sum=True,
        evaluation_concurrency=search_config.proposal_evaluation_concurrency,
    )
    search = LLMBasinAgentSearch(config=search_config, scorer_config=scorer_config)
    result = search.search(
        root_xc=root_xc,
        idea_agent=idea,
        knowledge=None,
        output_dir=output_dir,
        task=task,
    )
    print(json.dumps(result.summary, indent=2, default=str), flush=True)
    return 0


def _validate_split_pbe0(xc: XCFunctional) -> None:
    components = {component.name: component for component in xc.components}
    if set(components) != {"GGA_X_PBE", "GGA_C_PBE"}:
        raise ValueError(f"PBE0 root is not split into editable PBE X/C components: {set(components)}")
    if abs(float(xc.hf_exchange.alpha) - 0.25) > 1e-12:
        raise ValueError("PBE0 root must start at alpha=0.25")
    if abs(float(components["GGA_X_PBE"].mixing_factor) - 0.75) > 1e-12:
        raise ValueError("PBE0 root must contain 0.75 GGA_X_PBE")
    if abs(float(components["GGA_C_PBE"].mixing_factor) - 1.0) > 1e-12:
        raise ValueError("PBE0 root must contain 1.0 GGA_C_PBE")


if __name__ == "__main__":
    raise SystemExit(main())
