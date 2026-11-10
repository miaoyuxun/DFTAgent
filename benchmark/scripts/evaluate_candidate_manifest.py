"""Evaluate XC candidates from a manifest on selected GMTKN55 subsets.

The manifest format is produced by ``select_reward_band_candidates.py``.  This
script is intentionally source-agnostic: it only needs each candidate's
``xc_dict`` and metadata labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xc.benchmark.result import BenchmarkResult  # noqa: E402
from xc.benchmark.runner import BenchmarkRunner, SlurmAllocationPool  # noqa: E402
from xc.xc_functional import XCFunctional  # noqa: E402


DEFAULT_SUBSETS = ["W4-11", "ISO34", "S66"]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected manifest JSON list: {path}")
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("xc_dict"), dict):
            raise ValueError(f"Manifest entry {index} is missing xc_dict")
        candidates.append(item)
    return candidates


def _subset_payload(result: BenchmarkResult, subset: str) -> dict[str, Any]:
    sr = result.subsets.get(subset)
    if sr is None:
        return {
            f"{subset}_mae": "",
            f"{subset}_failures": "",
            f"{subset}_n": "",
            f"{subset}_wall_s": "",
        }
    return {
        f"{subset}_mae": sr.mae_kcal,
        f"{subset}_failures": sr.n_failures,
        f"{subset}_n": sr.n_reactions,
        f"{subset}_wall_s": sr.walltime_s,
    }


def _benchmark_result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "xc_name": result.xc_name,
        "subsets": {
            name: subset.to_dict()
            for name, subset in result.subsets.items()
        },
        "overall_mae": result.overall_mae,
        "overall_rms": result.overall_rms,
        "total_walltime_s": result.total_walltime_s,
        "total_failures": result.total_failures,
        "timestamp": result.timestamp,
    }


def write_outputs(
    output_dir: Path,
    candidates: list[dict[str, Any]],
    results: list[BenchmarkResult],
    subsets: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for candidate, result in zip(candidates, results):
        payload.append(
            {
                "candidate": candidate,
                "benchmark_result": _benchmark_result_to_dict(result),
            }
        )
    (output_dir / "manifest_benchmark_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = [
        "rank",
        "label",
        "source",
        "source_id",
        "stage",
        "generation",
        "source_raw_reward",
        "source_SIE4x4_mae",
        "source_BH76_mae",
        "source_G21EA_mae",
        "source_G21IP_mae",
        "overall_mae",
        "overall_rms",
        "total_failures",
        "total_walltime_s",
        "structural_signature",
        "effective_signature",
    ]
    for subset in subsets:
        fields.extend([f"{subset}_mae", f"{subset}_failures", f"{subset}_n", f"{subset}_wall_s"])
    fields.append("xc_json")

    with (output_dir / "manifest_benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate, result in zip(candidates, results):
            xc = XCFunctional.from_dict(candidate["xc_dict"])
            metrics = candidate.get("metrics") or {}
            row: dict[str, Any] = {
                "rank": candidate.get("rank"),
                "label": candidate.get("label"),
                "source": candidate.get("source"),
                "source_id": candidate.get("source_id"),
                "stage": candidate.get("stage"),
                "generation": candidate.get("generation"),
                "source_raw_reward": candidate.get("raw_reward"),
                "source_SIE4x4_mae": metrics.get("SIE4x4_mae"),
                "source_BH76_mae": metrics.get("BH76_mae"),
                "source_G21EA_mae": metrics.get("G21EA_mae"),
                "source_G21IP_mae": metrics.get("G21IP_mae"),
                "overall_mae": result.overall_mae,
                "overall_rms": result.overall_rms,
                "total_failures": result.total_failures,
                "total_walltime_s": result.total_walltime_s,
                "structural_signature": xc.structural_signature(),
                "effective_signature": xc.effective_signature(),
                "xc_json": json.dumps(candidate["xc_dict"], sort_keys=True),
            }
            for subset in subsets:
                row.update(_subset_payload(result, subset))
            writer.writerow(row)


def _make_runner(args: argparse.Namespace, output_dir: Path) -> BenchmarkRunner:
    slurm_allocation = None
    if args.benchmark_execution_backend == "srun":
        allocation_kwargs = dict(
            partition=args.slurm_partition,
            nodes=args.slurm_nodes,
            cpus_per_task=args.slurm_cpus_per_task,
            memory=args.slurm_mem,
            extra_args=list(args.slurm_extra_args or []),
        )
        slurm_allocation = (
            SlurmAllocationPool(count=max(1, args.slurm_allocation_count), **allocation_kwargs)
            if args.slurm_allocation_count > 1
            else None
        )

    return BenchmarkRunner(
        gmtkn55_dir=args.gmtkn55_dir,
        basis=args.basis,
        subsets=list(args.subsets),
        scf_max_cycle=args.scf_max_cycle,
        scf_conv_tol=args.scf_conv_tol,
        n_workers=args.n_workers,
        n_threads=args.n_threads,
        cache_dir=args.benchmark_cache,
        execution_backend=args.benchmark_execution_backend,
        slurm_launcher=args.slurm_launcher,
        slurm_nodes=args.slurm_nodes,
        slurm_cpus_per_task=args.slurm_cpus_per_task,
        slurm_mem=args.slurm_mem,
        slurm_partition=args.slurm_partition,
        slurm_output_dir=args.slurm_output_dir or str(output_dir / "slurm_logs"),
        slurm_extra_args=args.slurm_extra_args,
        slurm_allocation=slurm_allocation,
        slurm_allocations_per_evaluation=args.slurm_allocations_per_evaluation,
        aggregate_mode=args.aggregate_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="", help="Candidate manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark outputs")
    parser.add_argument("--gmtkn55-dir", default=str(REPO_ROOT / "benchmark" / "GMTKN55"))
    parser.add_argument("--benchmark-cache", default=str(REPO_ROOT / "benchmark" / "results"))
    parser.add_argument("--basis", default="def2-TZVP")
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N manifest entries")
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-8)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--aggregate-mode", choices=["reaction_mean", "subset_mean"], default="reaction_mean")
    parser.add_argument(
        "--benchmark-execution-backend",
        choices=["local", "srun", "slurm_docker", "slurm_docker_queue"],
        default="local",
    )
    parser.add_argument("--slurm-launcher", default="srun")
    parser.add_argument("--slurm-nodes", type=int, default=1)
    parser.add_argument("--slurm-cpus-per-task", type=int, default=8)
    parser.add_argument("--slurm-allocation-count", type=int, default=1)
    parser.add_argument("--slurm-allocations-per-evaluation", type=int, default=None)
    parser.add_argument("--slurm-mem", default="10G")
    parser.add_argument("--slurm-partition", default="dcu_htc")
    parser.add_argument("--slurm-time", default="24:00:00")
    parser.add_argument("--slurm-output-dir", default=None)
    parser.add_argument("--slurm-extra-args", nargs="*", default=None)
    parser.add_argument("--slurm-docker-image", default=None, help="Docker image for slurm_docker backend")
    parser.add_argument("--slurm-docker-container-repo-root", default="/workspace/DFTAgent")
    parser.add_argument("--slurm-docker-array-concurrency", type=int, default=32)
    parser.add_argument("--slurm-docker-molecules-per-task", type=int, default=8)
    parser.add_argument("--slurm-docker-queue-molecules-per-task", type=int, default=1)
    parser.add_argument("--slurm-docker-candidate-window", type=int, default=4)
    parser.add_argument("--slurm-docker-prepare-only", action="store_true")
    parser.add_argument("--slurm-docker-submit", action="store_true")
    parser.add_argument("--slurm-docker-aggregate-only", action="store_true")
    parser.add_argument("--slurm-docker-clean-output", action="store_true")
    parser.add_argument("--slurm-docker-profile-timing", action="store_true")
    parser.add_argument("--slurm-docker-total-cpus", type=int, default=128)
    parser.add_argument("--slurm-docker-cpus-per-worker", type=int, default=128)
    parser.add_argument("--slurm-docker-worker-count", type=int, default=0)
    parser.add_argument("--slurm-docker-nodes", type=int, default=0)
    parser.add_argument("--slurm-docker-ntasks-per-node", type=int, default=0)
    parser.add_argument("--slurm-docker-thread-policy", choices=["fixed", "atom_adaptive"], default="atom_adaptive")
    parser.add_argument("--slurm-docker-atom-thread-bins", default="2:1,6:2,10:4,999:8")
    parser.add_argument("--slurm-docker-max-threads-per-sample", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _run_slurm_docker_backend(args: argparse.Namespace) -> None:
    """Delegate manifest evaluation to the Slurm+Docker array-job runner."""

    if not args.slurm_docker_image and not args.slurm_docker_aggregate_only:
        raise SystemExit("--benchmark-execution-backend slurm_docker requires --slurm-docker-image")
    mode_args: list[str] = []
    if args.slurm_docker_aggregate_only:
        mode_args.append("--aggregate-only")
    elif args.slurm_docker_submit:
        mode_args.append("--submit")
    else:
        mode_args.append("--prepare-only")
    if args.slurm_docker_clean_output:
        mode_args.append("--clean-output")
    if args.slurm_docker_profile_timing:
        mode_args.append("--profile-timing")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "benchmark" / "scripts" / "evaluate_manifest_slurm_docker.py"),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--gmtkn55-dir",
        str(args.gmtkn55_dir),
        "--subsets",
        *list(args.subsets),
        "--basis",
        str(args.basis),
        "--scf-max-cycle",
        str(args.scf_max_cycle),
        "--scf-conv-tol",
        str(args.scf_conv_tol),
        "--aggregate-mode",
        str(args.aggregate_mode),
        "--molecules-per-task",
        str(args.slurm_docker_molecules_per_task),
        "--n-workers",
        str(args.n_workers),
        "--n-threads",
        str(args.n_threads),
        "--candidate-window",
        str(args.slurm_docker_candidate_window),
        "--limit",
        str(args.limit),
        "--image",
        str(args.slurm_docker_image or "unused-for-aggregate-only"),
        "--container-repo-root",
        str(args.slurm_docker_container_repo_root),
        "--partition",
        str(args.slurm_partition),
        "--time",
        str(args.slurm_time),
        "--mem",
        str(args.slurm_mem),
        "--cpus-per-task",
        str(args.slurm_cpus_per_task),
        "--array-concurrency",
        str(args.slurm_docker_array_concurrency),
        "--log-level",
        str(args.log_level),
        *mode_args,
    ]
    if args.slurm_extra_args:
        cmd.extend(["--extra-sbatch", *list(args.slurm_extra_args)])
    logging.info("Delegating manifest evaluation to Slurm+Docker runner")
    logging.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _run_slurm_docker_queue_backend(args: argparse.Namespace) -> None:
    """Delegate manifest evaluation to the multi-worker Slurm+Docker queue runner."""

    if not args.slurm_docker_image and not args.slurm_docker_aggregate_only:
        raise SystemExit("--benchmark-execution-backend slurm_docker_queue requires --slurm-docker-image")
    mode_args: list[str] = []
    if args.slurm_docker_aggregate_only:
        mode_args.append("--aggregate-only")
    elif args.slurm_docker_submit:
        mode_args.append("--submit")
    else:
        mode_args.append("--prepare-only")
    if args.slurm_docker_clean_output:
        mode_args.append("--clean-output")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "benchmark" / "scripts" / "evaluate_manifest_slurm_docker_queue.py"),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--gmtkn55-dir",
        str(args.gmtkn55_dir),
        "--subsets",
        *list(args.subsets),
        "--basis",
        str(args.basis),
        "--scf-max-cycle",
        str(args.scf_max_cycle),
        "--scf-conv-tol",
        str(args.scf_conv_tol),
        "--aggregate-mode",
        str(args.aggregate_mode),
        "--molecules-per-task",
        str(args.slurm_docker_queue_molecules_per_task),
        "--limit",
        str(args.limit),
        "--image",
        str(args.slurm_docker_image or "unused-for-aggregate-only"),
        "--container-repo-root",
        str(args.slurm_docker_container_repo_root),
        "--partition",
        str(args.slurm_partition),
        "--time",
        str(args.slurm_time),
        "--mem",
        str(args.slurm_mem),
        "--total-cpus",
        str(args.slurm_docker_total_cpus),
        "--cpus-per-worker",
        str(args.slurm_docker_cpus_per_worker),
        "--worker-count",
        str(args.slurm_docker_worker_count),
        "--nodes",
        str(args.slurm_docker_nodes),
        "--ntasks-per-node",
        str(args.slurm_docker_ntasks_per_node),
        "--thread-policy",
        str(args.slurm_docker_thread_policy),
        "--n-threads",
        str(args.n_threads),
        "--atom-thread-bins",
        str(args.slurm_docker_atom_thread_bins),
        "--max-threads-per-sample",
        str(args.slurm_docker_max_threads_per_sample),
        "--log-level",
        str(args.log_level),
        *mode_args,
    ]
    if args.slurm_extra_args:
        cmd.extend(["--extra-sbatch", *list(args.slurm_extra_args)])
    logging.info("Delegating manifest evaluation to Slurm+Docker queue runner")
    logging.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    if (
        not str(args.manifest).strip()
        and not (
            args.benchmark_execution_backend == "slurm_docker_queue"
            and args.slurm_docker_aggregate_only
        )
    ):
        raise SystemExit("--manifest is required unless using slurm_docker_queue with --slurm-docker-aggregate-only")
    if args.benchmark_execution_backend == "slurm_docker":
        _run_slurm_docker_backend(args)
        return
    if args.benchmark_execution_backend == "slurm_docker_queue":
        _run_slurm_docker_queue_backend(args)
        return

    candidates = load_manifest(Path(args.manifest))
    if args.limit > 0:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("No manifest candidates selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluated_manifest.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    runner = _make_runner(args, output_dir)
    results: list[BenchmarkResult] = []
    try:
        allocation = getattr(runner, "slurm_allocation", None)
        if allocation is not None and hasattr(allocation, "start"):
            allocation.start()
        for idx, candidate in enumerate(candidates, start=1):
            xc = XCFunctional.from_dict(candidate["xc_dict"])
            xc.metadata["followup_benchmark_source"] = {
                "label": candidate.get("label"),
                "source": candidate.get("source"),
                "source_id": candidate.get("source_id"),
                "raw_reward": candidate.get("raw_reward"),
            }
            logging.info(
                "Evaluating manifest candidate %d/%d label=%s reward=%s name=%s",
                idx,
                len(candidates),
                candidate.get("label"),
                candidate.get("raw_reward"),
                xc.name,
            )
            results.append(runner.evaluate(xc))
            write_outputs(output_dir, candidates[: len(results)], results, list(args.subsets))
    finally:
        if hasattr(runner, "close"):
            runner.close()

    write_outputs(output_dir, candidates, results, list(args.subsets))
    logging.info("Wrote manifest benchmark outputs to %s", output_dir)


if __name__ == "__main__":
    main()
