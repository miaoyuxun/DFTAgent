#!/usr/bin/env python3
"""Run a pure diverse-BO baseline on an equal-weight mean-MAE task."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_mean_mae_basin_agent import (  # noqa: E402
    _expand_environment,
    _validate_split_pbe0,
)
from scripts.run_sie_crystal_guard_diverse_bo import TaskDiverseBOBasinSearch  # noqa: E402
from xc.benchmark.vc_manifest_runner import VCManifestBenchmarkRunner, VCManifestRunnerConfig  # noqa: E402
from xc.idea.diverse_bo_basin import DiverseBOBasinConfig  # noqa: E402
from xc.idea.evolutionary import EvolutionConfig  # noqa: E402
from xc.search_tasks import MeanMAEBenchmarkTask, MeanMAETaskConfig  # noqa: E402
from xc.search_tasks.base import SearchTaskEvaluation  # noqa: E402
from xc.xc_functional import XCFunctional  # noqa: E402
from xc.zno_component_registry import validate_component_pool_lock  # noqa: E402


logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return _expand_environment(payload)


def _resolve_env_value(value: str, *, fallback_env: str) -> str:
    value = str(value or "").strip()
    if value.startswith("$"):
        return os.environ.get(value[1:].strip("{}"), "")
    return os.path.expandvars(value) or os.environ.get(fallback_env, "")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_root_evaluation(path: Path, task: MeanMAEBenchmarkTask) -> SearchTaskEvaluation:
    evaluation = SearchTaskEvaluation.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
    if evaluation is None or not evaluation.valid:
        raise ValueError(f"Root evaluation is missing or invalid: {path}")
    if evaluation.task_name != task.name:
        raise ValueError(f"Root task mismatch: {evaluation.task_name!r} != {task.name!r}")
    subset_payload = dict(evaluation.objective.get("subsets") or {})
    if set(subset_payload) != set(task.config.subsets):
        raise ValueError(
            "Root subsets do not match the configured task: "
            f"{sorted(subset_payload)} != {sorted(task.config.subsets)}"
        )
    expected_reward = -sum(
        float(subset_payload[name]["mae_kcal"]) for name in task.config.subsets
    ) / len(task.config.subsets)
    if abs(evaluation.reward - expected_reward) > 1e-10:
        raise ValueError(
            "Root reward is inconsistent with equal-weight mean MAE: "
            f"{evaluation.reward} != {expected_reward}"
        )
    return evaluation


class MeanMAEDiverseBOBasinSearch(TaskDiverseBOBasinSearch):
    """Task-aware BO with compact per-subset CSV output."""

    def _write_csv(self, path: Path) -> None:
        subset_names = tuple(getattr(self, "_subset_names", ()))
        fields = [
            "node_id", "cycle", "stage", "parent_id", "name", "raw_reward",
            "mean_mae_kcal", "promotable", "success_vs_parent", "failure_reason",
            "eval_duration_s", "nearest_node_id", "nearest_delta",
            *[f"{name}_mae_kcal" for name in subset_names],
            *[f"{name}_failures" for name in subset_names],
            "action_detail_json", "parameters_json", "task_evaluation_json", "xc_json",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for node in self.nodes:
                evaluation = node.task_evaluation
                objective = evaluation.objective if evaluation else {}
                subsets = dict(objective.get("subsets") or {})
                row = {
                    "node_id": node.node_id,
                    "cycle": node.cycle,
                    "stage": node.stage,
                    "parent_id": node.parent_id,
                    "name": node.xc.name,
                    "raw_reward": node.raw_reward,
                    "mean_mae_kcal": objective.get("mean_mae_kcal"),
                    "promotable": node.candidate.promotable,
                    "success_vs_parent": node.is_success,
                    "failure_reason": node.candidate.failure_reason,
                    "eval_duration_s": node.eval_duration_s,
                    "nearest_node_id": node.nearest_node_id,
                    "nearest_delta": node.nearest_delta,
                    "action_detail_json": json.dumps(node.action_detail, sort_keys=True, default=str),
                    "parameters_json": json.dumps(node.candidate.parameters, sort_keys=True, default=str),
                    "task_evaluation_json": json.dumps(
                        evaluation.to_dict() if evaluation else {}, sort_keys=True, default=str
                    ),
                    "xc_json": json.dumps(node.xc.to_dict(), sort_keys=True, default=str),
                }
                for name in subset_names:
                    payload = dict(subsets.get(name) or {})
                    row[f"{name}_mae_kcal"] = payload.get("mae_kcal")
                    row[f"{name}_failures"] = payload.get("n_failures")
                writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "mean_mae_bo.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-nodes", type=Path, default=None)
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    output_dir = _resolve_path(
        args.output_dir or Path(config.get("output_dir", "results/mean_mae_bo"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, str(config.get("log_level", "INFO")).upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    search_values = dict(config.get("search") or {})
    if args.seed is not None:
        search_values["seed"] = int(args.seed)
    if args.resume_nodes is not None:
        search_values["resume_nodes_path"] = str(_resolve_path(args.resume_nodes).resolve())
    core_dir = Path(str(output_dir) + "_core") if args.output_dir is not None else _resolve_path(
        config.get("core_output_dir") or (str(output_dir) + "_core")
    )
    search_values["core_output_dir"] = str(core_dir)
    search_config = DiverseBOBasinConfig(**search_values)

    task_values = dict(config.get("task") or {})
    task_component_pool = str(task_values.get("component_pool", search_config.component_pool))
    if task_component_pool != search_config.component_pool:
        raise ValueError("search.component_pool and task.component_pool must match")
    pool_signature = validate_component_pool_lock(
        search_config.component_pool, config.get("component_pool_lock")
    )
    logger.info("COMPONENT_POOL_LOCK_OK %s", pool_signature)

    task_config = MeanMAETaskConfig(
        subsets=tuple(str(name) for name in task_values.get("subsets", ())),
        component_pool=task_component_pool,
        max_failures_per_subset=int(task_values.get("max_failures_per_subset", 0)),
        allow_hf_exchange=bool(task_values.get("allow_hf_exchange", True)),
        allow_ext_param_rewire=bool(task_values.get("allow_ext_param_rewire", False)),
    )
    task = MeanMAEBenchmarkTask(runner=None, config=task_config)
    root_xc = XCFunctional.from_known(str(task_values.get("root_functional", "PBE0")))
    task.validate_functional(root_xc)
    _validate_split_pbe0(root_xc)
    root_evaluation_path = _resolve_path(task_values["root_evaluation_path"])
    root_evaluation = _load_root_evaluation(root_evaluation_path, task)

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
        "core_output_dir": str(core_dir),
        "search": search_values,
        "task": task_values,
        "component_pool_signature": pool_signature,
        "evaluation": {
            **common_eval,
            "persistent_queue_dir": persistent_queue,
            "vc_prefix_configured": bool(vc_prefix),
        },
        "benchmark_evaluation": benchmark_values,
        "root_effective_spec": root_xc.effective_spec(),
        "budget": {
            "root_excluded": True,
            "candidate_evaluations": int(search_config.cycles)
            * int(search_config.proposals_per_cycle),
        },
    }
    effective_path = output_dir / "effective_mean_mae_bo_config.json"
    effective_path.write_text(
        json.dumps(effective_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    core_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(effective_path, core_dir / effective_path.name)
    shutil.copy2(root_evaluation_path, core_dir / "root_evaluation_source.json")
    if args.validate_config:
        print(json.dumps(effective_config, indent=2, ensure_ascii=False), flush=True)
        print("Mean-MAE BO configuration is valid; no evaluator call was made.", flush=True)
        return 0

    if evaluation_backend not in {"vc", "persistent_vc"}:
        raise ValueError("evaluation.backend must be vc or persistent_vc")
    if evaluation_backend == "persistent_vc" and not persistent_queue:
        raise ValueError("DFTAGENT_PERSISTENT_VC_QUEUE is required")
    if evaluation_backend == "vc" and not vc_prefix:
        raise ValueError("DFTAGENT_VC_PREFIX is required")
    task.runner = VCManifestBenchmarkRunner(
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
            molecules_per_task=int(benchmark_values.get("molecules_per_task", 16)),
            aggregate_mode=str(benchmark_values.get("aggregate_mode", "reaction_mean")),
            concurrent_subset_jobs=int(common_eval.get("max_concurrent", 8)),
            vc_tasks_per_submit=int(common_eval.get("vc_tasks_per_submit", 0)),
            poll_seconds=float(common_eval.get("queue_monitor_poll_s", 2)),
            wait_timeout_seconds=float(common_eval.get("wait_timeout_seconds", 0)),
        ),
        repo_root=ROOT,
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
    search = MeanMAEDiverseBOBasinSearch(config=search_config, scorer_config=scorer_config)
    search._subset_names = task_config.subsets
    result = search.search_task(
        root_xc=root_xc,
        task=task,
        root_evaluation=root_evaluation,
        output_dir=output_dir,
    )
    for name in (
        "diverse_bo_basin_nodes.csv",
        "diverse_bo_basin_nodes.json",
        "diverse_bo_basin_summary.json",
    ):
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, core_dir / name)
    print(json.dumps(result.summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
