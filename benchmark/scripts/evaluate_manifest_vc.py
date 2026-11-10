"""Evaluate manifest candidates with vc submit array jobs.

This driver targets schedulers where one task can request at most 8 CPUs but
many tasks may be submitted.  It splits each candidate/subset into molecule
shards, each shard containing at most ``--molecules-per-task`` molecules and
using one CPU per molecule via ``n_workers=8, n_threads=1`` by default.

The default order is reward-descending and candidate-major.  Use
``--schedule-order subset_major`` when the goal is to finish one subset across
all selected candidates before moving to the next subset.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xc.benchmark.result import BenchmarkResult  # noqa: E402
from xc.benchmark.runner import (  # noqa: E402
    _molecule_names_from_reactions,
    _subset_result_from_molecule_energies,
    parse_res_file,
)
from xc.persistent_vc_queue import PersistentVCQueue  # noqa: E402
from xc.xc_functional import XCFunctional  # noqa: E402


DEFAULT_SUBSETS = ["W4-11", "ISO34", "S66"]

_STALE_HANDLE_RETRY_ATTEMPTS = 30
_STALE_HANDLE_RETRY_INITIAL_DELAY_S = 0.5
_STALE_HANDLE_RETRY_MAX_DELAY_S = 5.0
_T = TypeVar("_T")


def _retry_stale_handle(
    func: Callable[[], _T],
    *,
    description: str,
    path: Path | None = None,
) -> _T:
    """Retry transient ESTALE errors from the shared VC filesystem."""

    delay = _STALE_HANDLE_RETRY_INITIAL_DELAY_S
    for attempt in range(1, _STALE_HANDLE_RETRY_ATTEMPTS + 1):
        try:
            return func()
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt >= _STALE_HANDLE_RETRY_ATTEMPTS:
                raise
            logging.warning(
                "VC_FILESYSTEM_ESTALE operation=%s path=%s attempt=%d/%d retry_in_s=%.2f",
                description,
                path or "",
                attempt,
                _STALE_HANDLE_RETRY_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, _STALE_HANDLE_RETRY_MAX_DELAY_S)


def _path_exists(path: Path) -> bool:
    return _retry_stale_handle(
        path.exists,
        description="check path existence",
        path=path,
    )


def _mkdir(path: Path) -> None:
    _retry_stale_handle(
        lambda: path.mkdir(parents=True, exist_ok=True),
        description="create directory",
        path=path,
    )


def _read_text(path: Path, *, errors: str | None = None) -> str:
    kwargs: dict[str, Any] = {"encoding": "utf-8"}
    if errors is not None:
        kwargs["errors"] = errors
    return _retry_stale_handle(
        lambda: path.read_text(**kwargs),
        description="read text file",
        path=path,
    )


def _write_text(path: Path, text: str) -> None:
    _retry_stale_handle(
        lambda: path.write_text(text, encoding="utf-8"),
        description="write text file",
        path=path,
    )


def _float_or_neg_inf(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def load_manifest(path: Path, *, keep_order: bool) -> list[dict[str, Any]]:
    data = json.loads(_read_text(path))
    if not isinstance(data, list):
        raise ValueError(f"Expected manifest JSON list: {path}")
    candidates = [item for item in data if isinstance(item, dict) and isinstance(item.get("xc_dict"), dict)]
    if not keep_order:
        candidates.sort(key=lambda item: _float_or_neg_inf(item.get("raw_reward")), reverse=True)
    return candidates


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def _candidate_label(candidate: dict[str, Any], index: int) -> str:
    return str(candidate.get("label") or f"candidate_{index:02d}")


def _safe_name(value: str) -> str:
    keep = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value]
    return "_".join("".join(keep).split("_"))[:120] or "candidate"


def _task_path(task_dir: Path, task_index: int) -> Path:
    return task_dir / f"task_{task_index}.json"


def _write_candidate_tasks(
    *,
    candidate: dict[str, Any],
    candidate_index: int,
    output_dir: Path,
    subsets: list[str],
    gmtkn55_dir: Path,
    basis: str,
    scf_conv_tol: float,
    scf_max_cycle: int,
    n_workers: int,
    n_threads: int,
    molecules_per_task: int,
    density_output_root: Path | None = None,
    save_density_for_labels: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    label = _candidate_label(candidate, candidate_index)
    candidate_dir = output_dir / f"{candidate_index:02d}_{_safe_name(label)}"
    task_dir = candidate_dir / "tasks"
    result_dir = candidate_dir / "shard_results"
    _mkdir(task_dir)
    _mkdir(result_dir)
    xc_dict = candidate["xc_dict"]
    save_density_for_labels = save_density_for_labels or set()
    should_save_density = label in save_density_for_labels or str(candidate.get("source_id") or "") in save_density_for_labels

    all_tasks: list[dict[str, Any]] = []
    reactions_by_subset: dict[str, list[dict[str, Any]]] = {}
    task_index = 1
    for subset in subsets:
        subset_dir = gmtkn55_dir / subset
        reactions = parse_res_file(subset_dir / ".res")
        reactions_by_subset[subset] = reactions
        mol_names = _molecule_names_from_reactions(reactions)
        shards = _chunks(mol_names, max(1, int(molecules_per_task)))
        for shard_index, mols in enumerate(shards, start=1):
            result_path = result_dir / f"{subset}_shard_{shard_index:03d}_of_{len(shards):03d}.json"
            task = {
                "candidate_index": candidate_index,
                "candidate_label": label,
                "candidate": candidate,
                "subset_name": subset,
                "subset_dir": str(subset_dir),
                "mol_names": mols,
                "xc_dict": xc_dict,
                "basis": basis,
                "scf_conv_tol": scf_conv_tol,
                "scf_max_cycle": scf_max_cycle,
                "n_workers": min(int(n_workers), max(1, len(mols))),
                "n_threads": int(n_threads),
                "shard_index": shard_index,
                "n_shards": len(shards),
                "result_path": str(result_path),
            }
            if should_save_density:
                root = density_output_root or (output_dir / "density_cache")
                task["density_output_dir"] = str(root / _safe_name(label) / _safe_name(subset))
            path = _task_path(task_dir, task_index)
            task["task_path"] = str(path)
            _write_text(path, json.dumps(task, indent=2, ensure_ascii=False))
            all_tasks.append(task)
            task_index += 1
    _write_text(
        candidate_dir / "candidate_tasks.json",
        json.dumps(all_tasks, indent=2, ensure_ascii=False),
    )
    return all_tasks, reactions_by_subset


def _write_submit_script(
    *,
    script_path: Path,
    task_count: int,
    vc_prefix: str,
    python_executable: str,
    container_repo_root: str,
    container_task_dir: str,
    conda_sh: str,
    conda_env: str,
    pre_cmd: str,
    job_name: str,
    vc_log_path: str,
    vc_debug: bool,
) -> None:
    setup_parts = []
    if conda_sh:
        setup_parts.append(f"source {conda_sh}")
    if conda_env:
        setup_parts.append(f"conda activate {conda_env}")
    if pre_cmd:
        setup_parts.append(pre_cmd)
    setup_parts.append(f"cd {container_repo_root}")
    setup_parts.append("export PYTHONPATH=$PWD/src:${PYTHONPATH:-}")
    setup_parts.append(
        f"{python_executable} benchmark/scripts/vc_benchmark_worker.py "
        f"--task {container_task_dir}/task_$JOB.json"
    )
    container_cmd = " && ".join(setup_parts)
    debug_flag = " --debug" if vc_debug else ""
    script = f"""#!/usr/bin/env bash
set -euo pipefail
cd "{REPO_ROOT}"
{vc_prefix} --sync{debug_flag} --job "{job_name}" JOB=1:{task_count} "{vc_log_path}" --cmd '{container_cmd}'
"""
    _write_text(script_path, script)


def _container_path(path: Path, *, container_repo_root: str) -> str:
    rel = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    return f"{container_repo_root.rstrip('/')}/{rel}"

def _run_submit_script(script_path: Path) -> None:
    completed = subprocess.run(["bash", str(script_path)], text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"vc submit failed with return code {completed.returncode}: {script_path}")


def _tail_text(path: Path, *, max_lines: int = 40) -> str:
    try:
        lines = _read_text(path, errors="replace").splitlines()
    except OSError as exc:
        return f"<unable to read {path}: {exc}>"
    return "\n".join(lines[-max_lines:])


def _job_log_paths(job: dict[str, Any]) -> list[Path]:
    paths = [Path(job["submit_log_path"])]
    task_log_path = Path(job["task_log_path"])
    paths.append(task_log_path)
    if "JOB" in task_log_path.name:
        pattern = task_log_path.name.replace("JOB", "*")
        paths.extend(
            _retry_stale_handle(
                lambda: list(task_log_path.parent.glob(pattern)),
                description="enumerate VC task logs",
                path=task_log_path.parent,
            )
        )
    return list(dict.fromkeys(paths))


def _remove_job_logs(job: dict[str, Any]) -> None:
    for path in _job_log_paths(job):
        _retry_stale_handle(
            lambda path=path: path.unlink(missing_ok=True),
            description="remove successful VC log",
            path=path,
        )


def _cap_job_logs(job: dict[str, Any], *, max_bytes: int) -> None:
    max_bytes = max(1024, int(max_bytes))
    for path in _job_log_paths(job):
        if not _path_exists(path):
            continue

        def cap_one(path: Path = path) -> None:
            size = path.stat().st_size
            if size <= max_bytes:
                return
            marker = f"<log truncated; original_bytes={size}>\n".encode()
            tail_bytes = max(0, max_bytes - len(marker))
            with path.open("rb") as handle:
                handle.seek(-tail_bytes, os.SEEK_END)
                tail = handle.read(tail_bytes)
            with path.open("wb") as handle:
                handle.write(marker)
                handle.write(tail)

        _retry_stale_handle(
            cap_one,
            description="truncate failed VC log",
            path=path,
        )


def _wait_for_results(tasks: list[dict[str, Any]], *, poll_seconds: float, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    while True:
        missing = [task for task in tasks if not _path_exists(Path(task["result_path"]))]
        if not missing:
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {len(missing)} shard results")
        logging.info("Waiting for %d/%d shard results", len(missing), len(tasks))
        time.sleep(max(1.0, poll_seconds))


def _aggregate_subset(
    *,
    candidate: dict[str, Any],
    candidate_index: int,
    subset: str,
    subset_tasks: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    mol_energies: dict[str, float | None] = {}
    mol_errors: dict[str, str] = {}
    walltime = 0.0
    for task in subset_tasks:
        result_path = Path(task["result_path"])
        if not _path_exists(result_path):
            for mol_name in task["mol_names"]:
                mol_energies[mol_name] = None
                mol_errors[mol_name] = f"missing_shard_result:{result_path}"
            continue
        shard = json.loads(_read_text(result_path))
        mol_energies.update(shard.get("mol_energies") or {})
        mol_errors.update(shard.get("mol_errors") or {})
        walltime = max(walltime, float(shard.get("walltime_s") or 0.0))
    return _subset_result_from_molecule_energies(
        subset_name=subset,
        reactions=reactions,
        mol_energies=mol_energies,
        mol_errors=mol_errors,
        walltime_s=walltime,
        log_label=str(candidate.get("label") or candidate_index),
    )


def _candidate_result_from_subsets(
    candidate: dict[str, Any],
    subset_results: dict[str, Any],
    *,
    aggregate_mode: str,
) -> BenchmarkResult:
    xc = XCFunctional.from_dict(candidate["xc_dict"])
    result = BenchmarkResult(xc_name=xc.name, subsets=dict(subset_results))
    result.compute_overall(aggregate_mode=aggregate_mode)
    return result


def _aggregate_candidate(
    *,
    candidate: dict[str, Any],
    candidate_index: int,
    tasks: list[dict[str, Any]],
    reactions_by_subset: dict[str, list[dict[str, Any]]],
    subsets: list[str],
    aggregate_mode: str,
) -> BenchmarkResult:
    subset_results = {}
    for subset in subsets:
        subset_tasks = [task for task in tasks if task["subset_name"] == subset]
        mol_energies: dict[str, float | None] = {}
        mol_errors: dict[str, str] = {}
        walltime = 0.0
        for task in subset_tasks:
            result_path = Path(task["result_path"])
            if not _path_exists(result_path):
                for mol_name in task["mol_names"]:
                    mol_energies[mol_name] = None
                    mol_errors[mol_name] = f"missing_shard_result:{result_path}"
                continue
            shard = json.loads(_read_text(result_path))
            mol_energies.update(shard.get("mol_energies") or {})
            mol_errors.update(shard.get("mol_errors") or {})
            walltime = max(walltime, float(shard.get("walltime_s") or 0.0))
        subset_result, reaction_rows = _subset_result_from_molecule_energies(
            subset_name=subset,
            reactions=reactions_by_subset[subset],
            mol_energies=mol_energies,
            mol_errors=mol_errors,
            walltime_s=walltime,
            log_label=str(candidate.get("label") or candidate_index),
        )
        subset_results[subset] = subset_result
        out_dir = Path(tasks[0]["task_path"]).parent.parent
        _write_text(
            out_dir / f"{subset}_reaction_rows.json",
            json.dumps(reaction_rows, indent=2, ensure_ascii=False),
        )

    xc = XCFunctional.from_dict(candidate["xc_dict"])
    result = BenchmarkResult(xc_name=xc.name, subsets=subset_results)
    result.compute_overall(aggregate_mode=aggregate_mode)
    return result


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


def _missing_shards(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [task for task in tasks if not _path_exists(Path(task["result_path"]))]


def _write_outputs(output_dir: Path, payloads: list[dict[str, Any]], subsets: list[str]) -> None:
    _write_text(
        output_dir / "vc_manifest_benchmark_results.json",
        json.dumps(payloads, indent=2, ensure_ascii=False),
    )
    fields = [
        "rank",
        "label",
        "source",
        "source_id",
        "source_raw_reward",
        "overall_mae",
        "overall_rms",
        "total_failures",
        "total_walltime_s",
    ]
    for subset in subsets:
        fields.extend([f"{subset}_mae", f"{subset}_failures", f"{subset}_n", f"{subset}_wall_s"])
    summary_path = output_dir / "vc_manifest_benchmark_summary.csv"
    handle = _retry_stale_handle(
        lambda: summary_path.open("w", encoding="utf-8", newline=""),
        description="open benchmark summary",
        path=summary_path,
    )
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for payload in payloads:
            candidate = payload["candidate"]
            result = payload["benchmark_result"]
            row: dict[str, Any] = {
                "rank": candidate.get("rank"),
                "label": candidate.get("label"),
                "source": candidate.get("source"),
                "source_id": candidate.get("source_id"),
                "source_raw_reward": candidate.get("raw_reward"),
                "overall_mae": result.get("overall_mae"),
                "overall_rms": result.get("overall_rms"),
                "total_failures": result.get("total_failures"),
                "total_walltime_s": result.get("total_walltime_s"),
            }
            for subset in subsets:
                sr = (result.get("subsets") or {}).get(subset) or {}
                row.update(
                    {
                        f"{subset}_mae": sr.get("mae_kcal", ""),
                        f"{subset}_failures": sr.get("n_failures", ""),
                        f"{subset}_n": sr.get("n_reactions", ""),
                        f"{subset}_wall_s": sr.get("walltime_s", ""),
                    }
                )
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gmtkn55-dir", default=str(REPO_ROOT / "benchmark" / "GMTKN55"))
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--basis", default="def2-TZVP")
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-8)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--molecules-per-task", type=int, default=8)
    parser.add_argument("--aggregate-mode", choices=["reaction_mean", "subset_mean"], default="reaction_mean")
    parser.add_argument("--keep-manifest-order", action="store_true")
    parser.add_argument("--schedule-order", choices=["candidate_major", "subset_major"], default="candidate_major")
    parser.add_argument("--subset-barrier", action="store_true", help="With subset_major scheduling, finish all submitted jobs for one subset before starting the next subset")
    parser.add_argument("--save-density-for-labels", nargs="*", default=[], help="Candidate labels or source_id values whose converged SCF densities should be saved")
    parser.add_argument("--density-output-dir", default="", help="Optional root directory for saved density matrices; default is output_dir/density_cache")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vc-prefix", default="", help="Prefix before JOB=1:N, e.g. \"$cpu_cmd\" or full vc submit options")
    parser.add_argument(
        "--persistent-queue-dir",
        default="",
        help="Reuse a run-level persistent VC worker queue instead of submitting new VC jobs",
    )
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--container-repo-root", default="/code/remote/DFTAgent")
    parser.add_argument("--conda-sh", default="/code/remote/miniforge3/etc/profile.d/conda.sh")
    parser.add_argument("--conda-env", default="dftagent")
    parser.add_argument("--pre-cmd", default="", help="Extra shell snippet to run inside the container before cd/worker")
    parser.add_argument("--prepare-only", action="store_true", help="Write task files and submit scripts but do not run vc submit")
    parser.add_argument("--submit", action="store_true", help="Run generated vc submit scripts")
    parser.add_argument("--concurrent-subset-jobs", type=int, default=1, help="How many vc submit jobs to keep running at once")
    parser.add_argument("--vc-tasks-per-submit", type=int, default=0, help="Maximum 8CPU shard tasks per vc submit job; 0 means submit the whole subset as one array")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--vc-debug", action="store_true", help="Enable verbose vc submit diagnostics; disabled by default to control log volume")
    parser.add_argument("--keep-success-logs", action="store_true", help="Retain successful vc submit/task logs instead of deleting them")
    parser.add_argument("--max-retained-log-bytes", type=int, default=65536, help="Maximum bytes retained for each failed VC log")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    output_dir = Path(args.output_dir)
    if args.submit and not args.persistent_queue_dir and not str(args.vc_prefix).strip():
        raise SystemExit("--vc-prefix is required for --submit unless --persistent-queue-dir is set")
    if args.clean_output and _path_exists(output_dir):
        _retry_stale_handle(
            lambda: shutil.rmtree(output_dir),
            description="clean output directory",
            path=output_dir,
        )
    _mkdir(output_dir)

    candidates = load_manifest(Path(args.manifest), keep_order=args.keep_manifest_order)
    if args.limit > 0:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("No candidates selected")

    gmtkn55_dir = Path(args.gmtkn55_dir)
    density_output_root = Path(args.density_output_dir) if args.density_output_dir else output_dir / "density_cache"
    save_density_for_labels = {str(item) for item in (args.save_density_for_labels or []) if str(item)}
    payloads: list[dict[str, Any]] = []
    latest_payload_by_candidate: dict[int, dict[str, Any]] = {}
    plan: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    cached_subset_jobs: list[dict[str, Any]] = []

    # Prepare all candidate/subset array jobs in reward order.  Each job is a vc
    # array over molecule shards for exactly one candidate and one subset.
    for candidate_index, candidate in enumerate(candidates, start=1):
        label = _candidate_label(candidate, candidate_index)
        logging.info(
            "Preparing candidate %d/%d label=%s reward=%s",
            candidate_index,
            len(candidates),
            label,
            candidate.get("raw_reward"),
        )
        tasks, reactions_by_subset = _write_candidate_tasks(
            candidate=candidate,
            candidate_index=candidate_index,
            output_dir=output_dir,
            subsets=list(args.subsets),
            gmtkn55_dir=gmtkn55_dir,
            basis=args.basis,
            scf_conv_tol=args.scf_conv_tol,
            scf_max_cycle=args.scf_max_cycle,
            n_workers=args.n_workers,
            n_threads=args.n_threads,
            molecules_per_task=args.molecules_per_task,
            density_output_root=density_output_root,
            save_density_for_labels=save_density_for_labels,
        )
        task_dir = Path(tasks[0]["task_path"]).parent
        candidate_dir = task_dir.parent
        subset_results: dict[str, Any] = {}
        candidate_state = {
            "candidate_index": candidate_index,
            "candidate": candidate,
            "label": label,
            "task_dir": task_dir,
            "candidate_dir": candidate_dir,
            "reactions_by_subset": reactions_by_subset,
            "subset_results": subset_results,
            "completed_subsets": set(),
        }
        for subset in list(args.subsets):
            subset_tasks = [task for task in tasks if task["subset_name"] == subset]
            if not subset_tasks:
                continue
            candidate_state.setdefault("all_subset_tasks", {})[subset] = subset_tasks
            missing_subset_tasks = _missing_shards(subset_tasks)
            if not missing_subset_tasks:
                cached_job = {
                    "candidate_state": candidate_state,
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "label": label,
                    "subset": subset,
                    "subset_tasks": [],
                    "all_subset_tasks": subset_tasks,
                    "part_index": 0,
                    "part_count": 0,
                    "script_path": None,
                    "submit_log_path": None,
                    "task_count": 0,
                    "process": None,
                    "submit_log_handle": None,
                    "cached": True,
                }
                cached_subset_jobs.append(cached_job)
                plan.append(
                    {
                        "candidate_index": candidate_index,
                        "label": label,
                        "raw_reward": candidate.get("raw_reward"),
                        "subset": subset,
                        "part_index": 0,
                        "part_count": 0,
                        "task_count": 0,
                        "submit_script": "",
                        "status": "cached_all_shards_present",
                    }
                )
                logging.info(
                    "Reusing cached shard results candidate=%s subset=%s shards=%d",
                    label,
                    subset,
                    len(subset_tasks),
                )
                continue
            cached_count = len(subset_tasks) - len(missing_subset_tasks)
            if cached_count:
                logging.info(
                    "Reusing %d/%d cached shard results candidate=%s subset=%s; submitting %d missing shards",
                    cached_count,
                    len(subset_tasks),
                    label,
                    subset,
                    len(missing_subset_tasks),
                )
            max_per_submit = int(args.vc_tasks_per_submit or 0)
            if max_per_submit <= 0:
                max_per_submit = len(missing_subset_tasks)
            chunks = [
                missing_subset_tasks[i : i + max_per_submit]
                for i in range(0, len(missing_subset_tasks), max_per_submit)
            ]
            for part_index, part_tasks in enumerate(chunks, start=1):
                subset_task_dir = task_dir / f"subset_{_safe_name(subset)}_part{part_index:03d}"
                _mkdir(subset_task_dir)
                for idx, task in enumerate(part_tasks, start=1):
                    task_payload = dict(task)
                    subset_task_path = subset_task_dir / f"task_{idx}.json"
                    task_payload["task_path"] = str(subset_task_path)
                    _write_text(
                        subset_task_path,
                        json.dumps(task_payload, indent=2, ensure_ascii=False),
                    )
                part_suffix = f"{_safe_name(subset)}_part{part_index:03d}"
                script_path = candidate_dir / f"run_vc_submit_{part_suffix}.sh"
                _write_submit_script(
                    script_path=script_path,
                    task_count=len(part_tasks),
                    vc_prefix=args.vc_prefix,
                    python_executable=args.python_executable,
                    container_repo_root=args.container_repo_root,
                    container_task_dir=_container_path(subset_task_dir, container_repo_root=args.container_repo_root),
                    conda_sh=args.conda_sh,
                    conda_env=args.conda_env,
                    pre_cmd=args.pre_cmd,
                    job_name=f"gmtkn55_{candidate_index:02d}_{_safe_name(label)[:24]}_{part_suffix}",
                    vc_log_path=str(candidate_dir / "vc_logs" / f"{part_suffix}.JOB.log"),
                    vc_debug=bool(args.vc_debug),
                )
                task_log_path = candidate_dir / "vc_logs" / f"{part_suffix}.JOB.log"
                job = {
                    "candidate_state": candidate_state,
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "label": label,
                    "subset": subset,
                    "subset_tasks": part_tasks,
                    "all_subset_tasks": subset_tasks,
                    "part_index": part_index,
                    "part_count": len(chunks),
                    "script_path": script_path,
                    "submit_log_path": candidate_dir / "vc_logs" / f"submit_{part_suffix}.log",
                    "task_log_path": task_log_path,
                    "task_count": len(part_tasks),
                    "process": None,
                    "submit_log_handle": None,
                }
                prepared.append(job)
                plan.append(
                    {
                        "candidate_index": candidate_index,
                        "label": label,
                        "raw_reward": candidate.get("raw_reward"),
                        "subset": subset,
                        "part_index": part_index,
                        "part_count": len(chunks),
                        "task_count": len(part_tasks),
                        "cached_task_count": cached_count,
                        "submit_script": str(script_path),
                        "status": "submit_missing_shards",
                    }
                )

    if args.schedule_order == "subset_major":
        subset_rank = {subset: idx for idx, subset in enumerate(list(args.subsets))}

        def job_order(job: dict[str, Any]) -> tuple[int, int, int]:
            return (
                subset_rank.get(str(job.get("subset")), 10**9),
                int(job.get("candidate_index") or 0),
                int(job.get("part_index") or 0),
            )

        prepared.sort(key=job_order)
        cached_subset_jobs.sort(key=job_order)
        plan.sort(key=lambda row: (
            subset_rank.get(str(row.get("subset")), 10**9),
            int(row.get("candidate_index") or 0),
            int(row.get("part_index") or 0),
        ))

    _write_text(
        output_dir / "vc_execution_plan.json",
        json.dumps(plan, indent=2, ensure_ascii=False),
    )

    if args.prepare_only:
        logging.info("Prepared vc tasks/scripts only; plan=%s", output_dir / "vc_execution_plan.json")
        return
    if not args.submit:
        logging.info("Not submitting because --submit was not set; plan=%s", output_dir / "vc_execution_plan.json")
        return

    if args.persistent_queue_dir:
        queue_path = Path(args.persistent_queue_dir)
        if not queue_path.is_absolute():
            queue_path = REPO_ROOT / queue_path
        queue = PersistentVCQueue(queue_path, repo_root=REPO_ROOT)
        queue_tasks: list[dict[str, Any]] = []
        for job in prepared:
            for shard in job["subset_tasks"]:
                payload = dict(shard)
                payload["subset_dir"] = queue.repo_relative(payload["subset_dir"])
                if payload.get("density_output_dir"):
                    payload["density_output_dir"] = queue.repo_relative(payload["density_output_dir"])
                result_path = queue.repo_relative(payload["result_path"])
                payload["result_path"] = result_path
                queue_tasks.append(
                    {
                        "kind": "benchmark_shard",
                        "payload": payload,
                        "result_path": result_path,
                    }
                )
        submitted = queue.submit(
            queue_tasks,
            batch_id=f"benchmark_{output_dir.name}_{int(time.time())}",
        )
        logging.info(
            "PERSISTENT_VC_BENCHMARK_ENQUEUED tasks=%d queue=%s",
            len(submitted),
            queue_path,
        )
        queue.wait(
            submitted,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.wait_timeout_seconds,
        )
        # Reuse the normal aggregation path, but do not launch the generated
        # one-shot VC scripts now that all missing shards are complete.
        cached_subset_jobs.extend(prepared)
        prepared = []
        logging.info(
            "PERSISTENT_VC_BENCHMARK_DONE tasks=%d queue=%s",
            len(submitted),
            queue_path,
        )

    active: list[dict[str, Any]] = []
    max_active = max(1, int(args.concurrent_subset_jobs))
    finished_candidate_indices: set[int] = set()
    deferred_pending = list(prepared)
    pending: list[dict[str, Any]] = []

    def load_next_pending_subset() -> None:
        if pending or active or not deferred_pending:
            return
        if not (args.subset_barrier and args.schedule_order == "subset_major"):
            pending.extend(deferred_pending)
            deferred_pending.clear()
            return
        next_subset = str(deferred_pending[0].get("subset"))
        while deferred_pending and str(deferred_pending[0].get("subset")) == next_subset:
            pending.append(deferred_pending.pop(0))
        logging.info("Starting subset barrier group subset=%s jobs=%d", next_subset, len(pending))

    load_next_pending_subset()

    def finalize_subset(job: dict[str, Any]) -> None:
        nonlocal payloads
        state = job["candidate_state"]
        candidate = state["candidate"]
        candidate_index = state["candidate_index"]
        label = state["label"]
        subset = job["subset"]
        all_subset_tasks = job.get("all_subset_tasks") or job["subset_tasks"]
        missing = [task for task in all_subset_tasks if not _path_exists(Path(task["result_path"]))]
        candidate_dir = state["candidate_dir"]
        if missing:
            logging.info(
                "Shard part complete label=%s subset=%s part=%s/%s; waiting for %d remaining shard results",
                label,
                subset,
                job.get("part_index", 1),
                job.get("part_count", 1),
                len(missing),
            )
            return
        if subset in state["completed_subsets"]:
            return
        subset_result, reaction_rows = _aggregate_subset(
            candidate=candidate,
            candidate_index=candidate_index,
            subset=subset,
            subset_tasks=all_subset_tasks,
            reactions=state["reactions_by_subset"][subset],
        )
        state["subset_results"][subset] = subset_result
        state["completed_subsets"].add(subset)
        _write_text(
            candidate_dir / f"{subset}_result.json",
            json.dumps(subset_result.to_dict(), indent=2, ensure_ascii=False),
        )
        _write_text(
            candidate_dir / f"{subset}_reaction_rows.json",
            json.dumps(reaction_rows, indent=2, ensure_ascii=False),
        )
        partial_result = _candidate_result_from_subsets(
            candidate,
            state["subset_results"],
            aggregate_mode=args.aggregate_mode,
        )
        latest_payload_by_candidate[candidate_index] = {
            "candidate": candidate,
            "benchmark_result": _benchmark_result_to_dict(partial_result),
        }
        _write_outputs(
            output_dir,
            [latest_payload_by_candidate[idx] for idx in sorted(latest_payload_by_candidate)],
            list(args.subsets),
        )
        logging.info(
            "Subset complete label=%s subset=%s mae=%.4f failures=%d; partial_overall=%.4f",
            label,
            subset,
            subset_result.mae_kcal,
            subset_result.n_failures,
            partial_result.overall_mae,
        )
        if (
            len(state["completed_subsets"]) == len(list(args.subsets))
            and candidate_index not in finished_candidate_indices
        ):
            result = _candidate_result_from_subsets(
                candidate,
                state["subset_results"],
                aggregate_mode=args.aggregate_mode,
            )
            latest_payload_by_candidate[candidate_index] = {
                "candidate": candidate,
                "benchmark_result": _benchmark_result_to_dict(result),
            }
            payloads = [latest_payload_by_candidate[idx] for idx in sorted(latest_payload_by_candidate)]
            finished_candidate_indices.add(candidate_index)
            _write_outputs(output_dir, payloads, list(args.subsets))
            logging.info(
                "Candidate complete label=%s overall_mae=%.4f failures=%d",
                label,
                result.overall_mae,
                result.total_failures,
            )

    for job in cached_subset_jobs:
        logging.info(
            "Aggregating cached subset candidate=%s subset=%s shards=%d",
            job["label"],
            job["subset"],
            len(job.get("all_subset_tasks") or []),
        )
        finalize_subset(job)

    while pending or active or deferred_pending:
        while pending and len(active) < max_active:
            job = pending.pop(0)
            logging.info(
                "Submitting subset job candidate=%s subset=%s shards=%d script=%s",
                job["label"],
                job["subset"],
                job["task_count"],
                job["script_path"],
            )
            submit_log_path = Path(job["submit_log_path"])
            _mkdir(submit_log_path.parent)
            submit_log_handle = _retry_stale_handle(
                lambda: submit_log_path.open("w", encoding="utf-8"),
                description="open VC submit log",
                path=submit_log_path,
            )
            job["submit_log_handle"] = submit_log_handle
            job["process"] = subprocess.Popen(
                ["bash", str(job["script_path"])],
                stdout=submit_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append(job)
        still_active: list[dict[str, Any]] = []
        for job in active:
            proc = job["process"]
            rc = proc.poll()
            if rc is None:
                still_active.append(job)
                continue
            handle = job.get("submit_log_handle")
            if handle is not None:
                handle.close()
                job["submit_log_handle"] = None
            if rc != 0:
                _cap_job_logs(job, max_bytes=args.max_retained_log_bytes)
                tail = _tail_text(Path(job["submit_log_path"]))
                raise RuntimeError(
                    f"vc subset job failed rc={rc}: candidate={job['label']} subset={job['subset']} "
                    f"script={job['script_path']} submit_log={job['submit_log_path']}\n--- submit log tail ---\n{tail}"
                )
            logging.info(
                "Subset submit finished candidate=%s subset=%s submit_log=%s",
                job["label"],
                job["subset"],
                job["submit_log_path"],
            )
            try:
                _wait_for_results(
                    job["subset_tasks"],
                    poll_seconds=args.poll_seconds,
                    timeout_seconds=args.wait_timeout_seconds,
                )
                finalize_subset(job)
            except Exception:
                _cap_job_logs(job, max_bytes=args.max_retained_log_bytes)
                raise
            if not args.keep_success_logs:
                _remove_job_logs(job)
        active = still_active
        if not pending and not active and deferred_pending:
            load_next_pending_subset()
        if pending or active:
            time.sleep(max(1.0, args.poll_seconds))

    if latest_payload_by_candidate:
        payloads = [latest_payload_by_candidate[idx] for idx in sorted(latest_payload_by_candidate)]
        _write_outputs(output_dir, payloads, list(args.subsets))
    elif not payloads:
        logging.info("No completed payloads; plan=%s", output_dir / "vc_execution_plan.json")
    else:
        _write_outputs(output_dir, payloads, list(args.subsets))


if __name__ == "__main__":
    main()
