#!/usr/bin/env python3
"""Run a semilocal basin agent with SIE4x4 reward and configurable crystal guards."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xc.benchmark.runner import BenchmarkRunner  # noqa: E402
from xc.benchmark.vc_manifest_runner import (  # noqa: E402
    VCManifestBenchmarkRunner,
    VCManifestRunnerConfig,
)
from xc.idea.agent import IdeaAgent  # noqa: E402
from xc.idea.basin_agent import BasinAgentConfig, LLMBasinAgentSearch  # noqa: E402
from xc.idea.evolutionary import EvolutionConfig  # noqa: E402
from xc.search_tasks import (  # noqa: E402
    CrystalGuardSlurmDockerEvaluator,
    CrystalGuardSlurmEvaluatorConfig,
    CrystalGuardVCEvaluator,
    CrystalGuardVCEvaluatorConfig,
    CrystalGuardPersistentVCEvaluator,
    CrystalGuardPersistentVCEvaluatorConfig,
    SIECrystalGuardSearchTask,
    SIECrystalGuardTaskConfig,
    make_mtask_pw92_functional,
)
from xc.xc_functional import XCFunctional  # noqa: E402
from xc.zno_component_registry import validate_component_pool_lock  # noqa: E402


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _resolve_image(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("$"):
        return os.environ.get(value[1:].strip("{}"), "")
    return os.path.expandvars(value) or os.environ.get("DFTAGENT_IMAGE", "")


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
        default=ROOT / "config" / "sie_crystal_guard_basin_agent.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override search.seed for repeat runs")
    parser.add_argument(
        "--resume-nodes",
        type=Path,
        default=None,
        help="Resume from an existing basin_agent_nodes.json without repeating completed evaluations",
    )
    parser.add_argument(
        "--resume-through-cycle",
        type=int,
        default=None,
        help="Load only through this cycle and run the configured number of additional cycles",
    )
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    output_dir = args.output_dir or Path(config.get("output_dir", "results/run_sie_crystal_guard_basin_agent"))
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
    if args.resume_through_cycle is not None:
        search_values["resume_through_cycle"] = int(args.resume_through_cycle)
    core_dir = None if args.output_dir is not None else (
        search_values.get("core_output_dir") or config.get("core_output_dir")
    )
    if not core_dir:
        core_dir = str(output_dir) + "_core"
    elif not Path(core_dir).is_absolute():
        core_dir = str(ROOT / core_dir)
    search_values["core_output_dir"] = core_dir
    search_config = BasinAgentConfig(**search_values)

    task_values = dict(config.get("task") or {})
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
    root_functional_name = str(task_values.get("root_functional", "mTASK-PW92"))
    if root_functional_name.upper().replace("_", "-") == "MTASK-PW92":
        root_xc = make_mtask_pw92_functional()
    else:
        root_xc = XCFunctional.from_known(root_functional_name)
    protected_components = tuple(str(name) for name in task_values.get("protected_components", ()))
    minimum_component_mixing = {
        str(name): float(value)
        for name, value in dict(task_values.get("minimum_component_mixing") or {}).items()
    }
    root_components = {component.name: component for component in root_xc.components}
    missing_protected = sorted(set(protected_components) - set(root_components))
    if missing_protected:
        raise ValueError(
            f"Protected components are absent from root {root_functional_name!r}: {missing_protected}"
        )
    fixed_component_ext_params = {
        name: dict(root_components[name].ext_params)
        for name in protected_components
    }
    root_crystal_paths = {}
    for material, value in dict(task_values.get("root_crystal_result_paths") or {}).items():
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        root_crystal_paths[str(material)] = str(path)
    task_config = SIECrystalGuardTaskConfig(
        component_pool=task_component_pool,
        baseline_sie_mae_kcal=float(task_values.get("baseline_sie_mae_kcal", 10.0)),
        baseline_sie_failures=int(task_values.get("baseline_sie_failures", 0)),
        baseline_sie_reactions=int(task_values.get("baseline_sie_reactions", 16)),
        zns_reference_gap_ev=float(task_values.get("zns_reference_gap_ev", 3.83)),
        mgo_reference_gap_ev=float(task_values.get("mgo_reference_gap_ev", 7.77)),
        zns_free_tolerance_ev=float(task_values.get("zns_free_tolerance_ev", 0.50)),
        mgo_free_tolerance_ev=float(task_values.get("mgo_free_tolerance_ev", 0.20)),
        guard_scale_ev=float(task_values.get("guard_scale_ev", 0.30)),
        guard_mean_weight=float(task_values.get("guard_mean_weight", 0.25)),
        guard_max_weight=float(task_values.get("guard_max_weight", 0.10)),
        sie_fail_penalties=tuple(float(x) for x in task_values.get("sie_fail_penalties", [0.15, 0.15, 0.25, 0.25])),
        extra_sie_fail_penalty=float(task_values.get("extra_sie_fail_penalty", 0.50)),
        forbid_hf_exchange=bool(task_values.get("forbid_hf_exchange", True)),
        allow_ext_param_rewire=bool(task_values.get("allow_ext_param_rewire", False)),
        protected_components=protected_components,
        minimum_component_mixing=minimum_component_mixing,
        fixed_component_ext_params=fixed_component_ext_params,
        root_label=str(task_values.get("root_label", root_functional_name)),
        crystal_guard_materials=tuple(task_values.get("crystal_guard_materials", ["ZnS", "MgO"])),
        root_crystal_result_paths=root_crystal_paths,
    )

    common_eval = dict(config.get("evaluation") or {})
    evaluation_backend = str(common_eval.get("backend", "slurm_docker")).strip().lower()
    if evaluation_backend not in {"slurm_docker", "vc", "persistent_vc"}:
        raise ValueError("evaluation.backend must be slurm_docker, vc, or persistent_vc")
    image = _resolve_image(common_eval.get("image", "$DFTAGENT_IMAGE"))
    vc_prefix = _resolve_env_value(
        common_eval.get("vc_prefix", "$DFTAGENT_VC_PREFIX"),
        fallback_env="DFTAGENT_VC_PREFIX",
    )
    if evaluation_backend == "slurm_docker" and not image:
        raise ValueError("evaluation.image or DFTAGENT_IMAGE is required")
    if evaluation_backend == "vc" and not vc_prefix:
        raise ValueError("evaluation.vc_prefix or DFTAGENT_VC_PREFIX is required")
    persistent_queue = _resolve_env_value(
        common_eval.get("persistent_queue_dir", "$DFTAGENT_PERSISTENT_VC_QUEUE"),
        fallback_env="DFTAGENT_PERSISTENT_VC_QUEUE",
    )
    if evaluation_backend == "persistent_vc" and not persistent_queue:
        raise ValueError(
            "evaluation.persistent_queue_dir or DFTAGENT_PERSISTENT_VC_QUEUE is required"
        )
    benchmark_values = dict(config.get("benchmark_evaluation") or {})
    crystal_values = dict(config.get("crystal_evaluation") or {})

    benchmark_output_dir = output_dir / "sie_evaluations"
    if evaluation_backend in {"vc", "persistent_vc"}:
        container_repo_root = str(common_eval.get("container_repo_root", "/code/remote/DFTAgent"))
        sie_runner = VCManifestBenchmarkRunner(
            VCManifestRunnerConfig(
                output_dir=benchmark_output_dir,
                vc_prefix=vc_prefix if evaluation_backend == "vc" else "",
                persistent_queue_dir=Path(persistent_queue) if evaluation_backend == "persistent_vc" else None,
                container_repo_root=container_repo_root,
                python_executable=str(common_eval.get("python_executable", "python3")),
                conda_sh=str(common_eval.get("conda_sh", "")),
                conda_env=str(common_eval.get("conda_env", "")),
                pre_cmd=str(common_eval.get("pre_cmd", "")),
                gmtkn55_dir=Path(benchmark_values.get("gmtkn55_dir", ROOT / "benchmark" / "GMTKN55")),
                subsets=("SIE4x4",),
                basis=str(benchmark_values.get("basis", "def2-TZVP")),
                scf_max_cycle=int(benchmark_values.get("scf_max_cycle", 200)),
                scf_conv_tol=float(benchmark_values.get("scf_conv_tol", 1e-8)),
                n_workers=int(benchmark_values.get("n_workers", 8)),
                n_threads=int(benchmark_values.get("n_threads", 1)),
                molecules_per_task=int(benchmark_values.get("molecules_per_task", 8)),
                aggregate_mode=str(benchmark_values.get("aggregate_mode", "reaction_mean")),
                concurrent_subset_jobs=int(common_eval.get("max_concurrent", 32)),
                vc_tasks_per_submit=int(common_eval.get("vc_tasks_per_submit", 0)),
                poll_seconds=float(common_eval.get("queue_monitor_poll_s", 10)),
                wait_timeout_seconds=float(common_eval.get("wait_timeout_seconds", 0)),
            ),
            repo_root=ROOT,
        )
        if evaluation_backend == "persistent_vc":
            crystal_evaluator = CrystalGuardPersistentVCEvaluator(
                CrystalGuardPersistentVCEvaluatorConfig(
                    output_dir=output_dir / "crystal_guard_evaluations",
                    queue_dir=Path(persistent_queue),
                    cpus_per_candidate=int(common_eval.get("cpus_per_candidate", 8)),
                    hard_timeout_s=int(common_eval.get("hard_timeout_s", 7200)),
                    poll_seconds=float(common_eval.get("queue_monitor_poll_s", 2)),
                    wait_timeout_seconds=float(common_eval.get("wait_timeout_seconds", 0)),
                    band_mesh=tuple(int(x) for x in crystal_values.get("band_mesh", [2, 2, 2])),
                    materials=tuple(task_config.crystal_guard_materials),
                ),
                repo_root=ROOT,
            )
        else:
            crystal_evaluator = CrystalGuardVCEvaluator(
                CrystalGuardVCEvaluatorConfig(
                    output_dir=output_dir / "crystal_guard_evaluations",
                    vc_prefix=vc_prefix,
                    container_repo_root=container_repo_root,
                    python_executable=str(common_eval.get("python_executable", "python3")),
                    conda_sh=str(common_eval.get("conda_sh", "")),
                    conda_env=str(common_eval.get("conda_env", "")),
                    pre_cmd=str(common_eval.get("pre_cmd", "")),
                    cpus_per_candidate=int(common_eval.get("cpus_per_candidate", 8)),
                    max_concurrent_submissions=int(common_eval.get("max_concurrent", 32)),
                    hard_timeout_s=int(common_eval.get("hard_timeout_s", 7200)),
                    band_mesh=tuple(int(x) for x in crystal_values.get("band_mesh", [2, 2, 2])),
                    materials=tuple(task_config.crystal_guard_materials),
                ),
                repo_root=ROOT,
            )
    else:
        sie_runner = BenchmarkRunner(
            gmtkn55_dir=Path(benchmark_values.get("gmtkn55_dir", ROOT / "benchmark" / "GMTKN55")),
            basis=str(benchmark_values.get("basis", "def2-TZVP")),
            subsets=["SIE4x4"],
            n_workers=int(benchmark_values.get("n_workers", 16)),
            n_threads=int(benchmark_values.get("n_threads", 1)),
            execution_backend="slurm_docker_queue",
            slurm_partition=str(common_eval.get("partition", "compute")),
            slurm_mem=str(common_eval.get("memory", "120G")),
            slurm_docker_image=image,
            slurm_docker_container_repo_root=str(common_eval.get("container_repo_root", "/workspace/DFTAgent")),
            slurm_docker_time=str(common_eval.get("slurm_time", "02:10:00")),
            slurm_docker_total_cpus=int(common_eval.get("total_cpus", 512)),
            slurm_docker_cpus_per_worker=int(common_eval.get("cpus_per_candidate", 16)),
            slurm_docker_worker_count=int(common_eval.get("max_concurrent", 32)),
            slurm_docker_cpus_per_node=int(common_eval.get("cpus_per_node", 128)),
            slurm_docker_submit_mode=str(common_eval.get("slurm_submit_mode", "allocation")),
            slurm_docker_array_max_running=int(common_eval.get("array_max_running", 0)),
            slurm_docker_queue_dispatch_mode=str(common_eval.get("queue_dispatch_mode", "candidate_dedicated")),
            slurm_docker_thread_policy=str(benchmark_values.get("thread_policy", "atom_adaptive")),
            slurm_docker_atom_thread_bins=str(benchmark_values.get("atom_thread_bins", "2:1,6:2,10:4,999:16")),
            slurm_docker_max_threads_per_sample=int(benchmark_values.get("max_threads_per_sample", 16)),
            slurm_docker_molecules_per_task=int(benchmark_values.get("molecules_per_task", 1)),
            slurm_docker_output_dir=benchmark_output_dir,
            aggregate_mode=str(benchmark_values.get("aggregate_mode", "reaction_mean")),
            scf_max_cycle=int(benchmark_values.get("scf_max_cycle", 200)),
            scf_conv_tol=float(benchmark_values.get("scf_conv_tol", 1e-8)),
        )
        crystal_cfg = CrystalGuardSlurmEvaluatorConfig(
            output_dir=output_dir / "crystal_guard_evaluations",
            image=image,
            container_repo_root=str(common_eval.get("container_repo_root", "/workspace/DFTAgent")),
            partition=str(common_eval.get("partition", "compute")),
            cpus_per_candidate=int(common_eval.get("cpus_per_candidate", 16)),
            max_concurrent=int(common_eval.get("max_concurrent", 32)),
            memory=str(common_eval.get("memory", "120G")),
            hard_timeout_s=int(common_eval.get("hard_timeout_s", 7200)),
            slurm_time=str(common_eval.get("slurm_time", "02:10:00")),
            band_mesh=tuple(int(x) for x in crystal_values.get("band_mesh", [2, 2, 2])),
            materials=tuple(task_config.crystal_guard_materials),
            adaptive_straggler_cutoff=bool(common_eval.get("adaptive_straggler_cutoff", True)),
            straggler_cutoff_start_fraction=float(common_eval.get("straggler_cutoff_start_fraction", 0.75)),
            straggler_cutoff_min_multiplier=float(common_eval.get("straggler_cutoff_min_multiplier", 1.5)),
            straggler_cutoff_mad_multiplier=float(common_eval.get("straggler_cutoff_mad_multiplier", 3.0)),
            straggler_cutoff_max_fraction=float(common_eval.get("straggler_cutoff_max_fraction", 0.05)),
            poll_s=float(common_eval.get("queue_monitor_poll_s", 30)),
        )
        crystal_evaluator = CrystalGuardSlurmDockerEvaluator(crystal_cfg, repo_root=ROOT)
    task = SIECrystalGuardSearchTask(
        sie_runner=sie_runner,
        crystal_evaluator=crystal_evaluator,
        config=task_config,
    )

    llm_file = Path(config.get("llm_config", ROOT / "config" / "llm.yaml"))
    if not llm_file.is_absolute():
        llm_file = ROOT / llm_file
    llm_values = dict((_load_yaml(llm_file).get("llm") or {}))
    llm_values.update(config.get("llm_overrides") or {})
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
            "Reduce SIE4x4 MAE for a semilocal crystal-oriented functional while "
            f"keeping {', '.join(task_config.crystal_guard_materials)} band gaps within hard guard limits."
        ),
        knowledge_engine=None,
        proposal_prior_path=None,
    )

    effective_config = {
        "output_dir": str(output_dir),
        "core_output_dir": str(core_dir),
        "search": search_values,
        "task": task_values,
        "component_pool_lock": config.get("component_pool_lock") or {},
        "component_pool_signature": pool_signature,
        "evaluation": {
            **common_eval,
            "backend": evaluation_backend,
            "image": image,
            "vc_prefix": vc_prefix,
            "persistent_queue_dir": persistent_queue,
        },
        "benchmark_evaluation": benchmark_values,
        "crystal_evaluation": crystal_values,
        "model": idea.model,
    }
    (output_dir / "effective_sie_crystal_guard_run_config.json").write_text(
        json.dumps(effective_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(core_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        output_dir / "effective_sie_crystal_guard_run_config.json",
        Path(core_dir) / "effective_sie_crystal_guard_run_config.json",
    )
    print(json.dumps(effective_config, indent=2, ensure_ascii=False), flush=True)
    if args.validate_config:
        print("SIE+crystal basin-agent configuration is valid; no LLM or PySCF call was made.", flush=True)
        return 0

    root_evaluation = task.root_evaluation_from_existing()
    (output_dir / "root_evaluation_source.json").write_text(
        json.dumps(root_evaluation.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(output_dir / "root_evaluation_source.json", Path(core_dir) / "root_evaluation_source.json")
    logging.getLogger(__name__).info(
        "ROOT_EVALUATION_LOADED task=%s reward=%.6f sie_mae=%.3f; root will not be submitted",
        root_evaluation.task_name,
        root_evaluation.reward,
        root_evaluation.objective.get("sie4x4_mae_kcal"),
    )

    scorer_config = EvolutionConfig(
        seed=search_config.seed,
        component_pool=search_config.component_pool,
        include_global_hf=False,
        include_role_scales=True,
        role_scale_bounds=(0.8, 1.2),
        include_component_mixing=True,
        component_mixing_bounds={
            name: (minimum, 1.0)
            for name, minimum in minimum_component_mixing.items()
        },
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
        root_evaluation=root_evaluation,
    )
    print(json.dumps(result.summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
