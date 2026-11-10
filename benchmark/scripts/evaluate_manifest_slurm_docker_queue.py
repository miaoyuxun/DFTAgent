"""Evaluate a candidate manifest with a multi-worker Slurm+Docker file queue.

This runner is designed for high-throughput searches on clusters where each
Slurm task launches one Docker container through ``slurm-docker-run``.  All
containers consume molecule-level tasks from a shared queue directory, so tasks
can span multiple nodes without nested Slurm submissions.
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xc.benchmark.result import BenchmarkResult  # noqa: E402
from xc.slurm_env import clean_slurm_submit_env  # noqa: E402
from xc.benchmark.runner import (  # noqa: E402
    _molecule_names_from_reactions,
    _subset_result_from_molecule_energies,
    parse_res_file,
)
from xc.xc_functional import XCFunctional  # noqa: E402


DEFAULT_SUBSETS = ["SIE4x4", "BH76", "G21EA", "G21IP"]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    out = Path(args.output_dir)
    if args.clean_output and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_outputs(out, subsets=args.subsets, aggregate_mode=args.aggregate_mode)
        return
    if not str(args.manifest).strip():
        raise SystemExit("--manifest is required unless --aggregate-only is used")
    if not str(args.image).strip():
        raise SystemExit("--image is required unless --aggregate-only is used")

    candidates = load_manifest(Path(args.manifest), keep_order=args.keep_manifest_order)
    if args.limit > 0:
        candidates = candidates[: int(args.limit)]
    if not candidates:
        raise SystemExit("No candidates selected from manifest")

    config = scheduler_config(args, candidate_count=len(candidates))
    tasks = prepare_queue_tasks(
        candidates=candidates,
        output_dir=out,
        subsets=args.subsets,
        gmtkn55_dir=Path(args.gmtkn55_dir),
        container_repo_root=args.container_repo_root,
        basis=args.basis,
        scf_conv_tol=args.scf_conv_tol,
        scf_max_cycle=args.scf_max_cycle,
        thread_policy=args.thread_policy,
        fixed_threads=args.n_threads,
        atom_thread_bins=parse_thread_bins(args.atom_thread_bins),
        max_threads_per_sample=args.max_threads_per_sample,
        molecules_per_task=args.molecules_per_task,
        max_molecules=args.max_molecules,
        keep_existing_done=not args.clean_output,
        queue_dispatch_mode=args.queue_dispatch_mode,
    )
    (out / "scheduler_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_sbatch_script(args, out, config=config)
    logging.info(
        "Prepared queue runner: tasks=%d pending=%d output=%s workers=%d cpus_per_worker=%d total_cpus=%d",
        len(tasks),
        count_pending(out),
        out,
        config["worker_count"],
        config["cpus_per_worker"],
        config["allocated_cpus"],
    )
    if args.prepare_only:
        return
    if args.submit:
        subprocess.run(["sbatch", str(out / "run_slurm_docker_queue.sbatch")], check=True, env=clean_sbatch_env())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gmtkn55-dir", default=str(REPO_ROOT / "benchmark" / "GMTKN55"))
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--basis", default="def2-TZVP")
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-8)
    parser.add_argument("--aggregate-mode", choices=["reaction_mean", "subset_mean"], default="reaction_mean")
    parser.add_argument("--molecules-per-task", type=int, default=1)
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=0,
        help="Optional per-subset molecule cap for profiling/debugging. 0 means use the full subset.",
    )
    parser.add_argument("--keep-manifest-order", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--image", default="")
    parser.add_argument("--container-repo-root", default="/workspace/DFTAgent")
    parser.add_argument("--partition", default="compute")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--mem", default="0")
    parser.add_argument("--slurm-nodelist", default="", help="Optional Slurm nodelist, e.g. n09,n10")
    parser.add_argument("--total-cpus", type=int, default=128)
    parser.add_argument("--cpus-per-worker", type=int, default=128)
    parser.add_argument("--worker-count", type=int, default=0)
    parser.add_argument(
        "--slurm-submit-mode",
        choices=["allocation", "array"],
        default="allocation",
        help=(
            "allocation: one multi-task Slurm job with --ntasks-per-node "
            "enforcing --cpus-per-node. array: many small array tasks, each "
            "one worker, but Slurm may pack more than --cpus-per-node total "
            "CPU onto a physical node."
        ),
    )
    parser.add_argument(
        "--array-max-running",
        type=int,
        default=0,
        help="Optional %% concurrency cap for --slurm-submit-mode=array. 0 means no explicit cap.",
    )
    parser.add_argument(
        "--queue-dispatch-mode",
        choices=["global", "candidate_dedicated"],
        default="global",
        help=(
            "global: all workers consume one shared molecule queue. "
            "candidate_dedicated: workers are assigned to candidate queues "
            "and only consume that candidate's tasks."
        ),
    )
    parser.add_argument(
        "--workers-per-candidate",
        type=int,
        default=1,
        help=(
            "Only used with --queue-dispatch-mode=candidate_dedicated. "
            "Assign this many Slurm workers to each candidate queue; for example "
            "8 workers with --cpus-per-worker=4 gives each candidate a 32-CPU pool."
        ),
    )
    parser.add_argument("--nodes", type=int, default=0, help="Optional fixed node count. Omit to let Slurm place tasks.")
    parser.add_argument("--ntasks-per-node", type=int, default=0)
    parser.add_argument(
        "--cpus-per-node",
        type=int,
        default=128,
        help=(
            "Maximum CPU packing target per physical node for automatic Slurm "
            "layout. When --nodes and --ntasks-per-node are omitted, the "
            "runner uses this value to choose the minimum node count but does "
            "not require every allocated node to have this many CPUs."
        ),
    )
    parser.add_argument("--thread-policy", choices=["fixed", "atom_adaptive"], default="atom_adaptive")
    parser.add_argument("--n-threads", type=int, default=2, help="Fixed mode threads per molecule")
    parser.add_argument("--atom-thread-bins", default="2:1,6:2,10:4,999:8")
    parser.add_argument("--max-threads-per-sample", type=int, default=8)
    parser.add_argument("--extra-sbatch", action="append", default=[])
    parser.add_argument("--docker-extra-args", action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def scheduler_config(args: argparse.Namespace, *, candidate_count: int = 0) -> dict[str, Any]:
    cpus_per_worker = max(1, int(args.cpus_per_worker))
    if args.worker_count > 0:
        worker_count = int(args.worker_count)
    else:
        worker_count = max(1, math.ceil(max(1, int(args.total_cpus)) / cpus_per_worker))
    submit_mode = str(args.slurm_submit_mode)
    dispatch_mode = str(args.queue_dispatch_mode)
    workers_per_candidate = max(1, int(getattr(args, "workers_per_candidate", 1) or 1))
    nodes = int(args.nodes)
    ntasks_per_node = int(args.ntasks_per_node)
    cpus_per_node = max(cpus_per_worker, int(args.cpus_per_node))
    max_tasks_per_node = max(1, cpus_per_node // cpus_per_worker)
    if dispatch_mode == "candidate_dedicated":
        worker_count = max(1, worker_count, int(candidate_count or 0) * workers_per_candidate)
    if submit_mode != "array" and nodes > 0 and ntasks_per_node > 0:
        fixed_worker_count = nodes * ntasks_per_node
        required_candidate_workers = int(candidate_count or 0) * workers_per_candidate
        if dispatch_mode == "candidate_dedicated" and candidate_count and required_candidate_workers > fixed_worker_count:
            raise SystemExit(
                f"candidate_dedicated needs enough workers for candidate pools: "
                f"candidates={candidate_count}, workers_per_candidate={workers_per_candidate}, "
                f"required_workers={required_candidate_workers}, fixed workers={fixed_worker_count} "
                f"from --nodes={nodes} --ntasks-per-node={ntasks_per_node}"
            )
        worker_count = fixed_worker_count
    nodes_min = 0
    nodes_max = 0
    nodes_spec = ""
    if submit_mode == "array":
        # Each array element is a small independent Slurm job step.  This lets
        # Slurm place workers on fragmented CPU resources instead of requiring
        # a large contiguous multi-node allocation.  Because array elements are
        # separate Slurm allocations, cpus_per_node is documented in the config
        # but cannot strictly cap how many array elements Slurm packs onto one
        # physical node.  Use allocation mode when a hard per-node cap matters.
        nodes = 0
        ntasks_per_node = 0
        max_tasks_per_node = 0
    elif nodes <= 0 and ntasks_per_node <= 0:
        nodes_min = max(1, math.ceil(worker_count / max_tasks_per_node))
        nodes_max = max(nodes_min, worker_count)
        nodes = nodes_min
        nodes_spec = f"{nodes_min}-{nodes_max}" if nodes_max > nodes_min else str(nodes_min)
    elif nodes <= 0 and ntasks_per_node > 0:
        if ntasks_per_node > max_tasks_per_node:
            raise SystemExit(
                f"--ntasks-per-node={ntasks_per_node} would request "
                f"{ntasks_per_node * cpus_per_worker} CPUs per node, exceeding "
                f"--cpus-per-node={cpus_per_node}"
            )
        nodes_min = max(1, math.ceil(worker_count / ntasks_per_node))
        nodes_max = max(nodes_min, worker_count)
        nodes = nodes_min
        nodes_spec = f"{nodes_min}-{nodes_max}" if nodes_max > nodes_min else str(nodes_min)
    elif nodes > 0 and ntasks_per_node <= 0:
        if nodes * max_tasks_per_node < worker_count:
            raise SystemExit(
                f"--nodes={nodes} is too small for {worker_count} workers under "
                f"--cpus-per-node={cpus_per_node}; need at least "
                f"{math.ceil(worker_count / max_tasks_per_node)} nodes"
            )
        nodes_min = nodes
        nodes_max = nodes
        nodes_spec = str(nodes)
    else:
        if ntasks_per_node > max_tasks_per_node:
            raise SystemExit(
                f"--ntasks-per-node={ntasks_per_node} would request "
                f"{ntasks_per_node * cpus_per_worker} CPUs per node, exceeding "
                f"--cpus-per-node={cpus_per_node}"
            )
        nodes_min = nodes
        nodes_max = nodes
        nodes_spec = str(nodes)
    return {
        "worker_count": worker_count,
        "cpus_per_worker": cpus_per_worker,
        "allocated_cpus": worker_count * cpus_per_worker,
        "requested_total_cpus": int(args.total_cpus),
        "slurm_submit_mode": submit_mode,
        "queue_dispatch_mode": dispatch_mode,
        "workers_per_candidate": workers_per_candidate,
        "array_max_running": int(args.array_max_running),
        "requested_mem": str(args.mem),
        "slurm_nodelist": str(args.slurm_nodelist or ""),
        "effective_mem": effective_mem_request(str(args.mem), submit_mode),
        "nodes": nodes,
        "nodes_min": nodes_min,
        "nodes_max": nodes_max,
        "nodes_spec": nodes_spec,
        "ntasks_per_node": ntasks_per_node,
        "max_tasks_per_node": max_tasks_per_node,
        "cpus_per_node": cpus_per_node,
        "cpus_per_node_enforced": submit_mode != "array",
        "task_distribution": "cyclic" if submit_mode != "array" else "",
        "thread_policy": str(args.thread_policy),
        "n_threads": int(args.n_threads),
        "atom_thread_bins": str(args.atom_thread_bins),
        "max_threads_per_sample": int(args.max_threads_per_sample),
        "molecules_per_task": int(args.molecules_per_task),
    }


def prepare_queue_tasks(
    *,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    subsets: list[str],
    gmtkn55_dir: Path,
    container_repo_root: str,
    basis: str,
    scf_conv_tol: float,
    scf_max_cycle: int,
    thread_policy: str,
    fixed_threads: int,
    atom_thread_bins: list[tuple[int, int]],
    max_threads_per_sample: int,
    molecules_per_task: int,
    max_molecules: int,
    keep_existing_done: bool,
    queue_dispatch_mode: str,
) -> list[dict[str, Any]]:
    queue_dir = output_dir / "queue"
    for dirname in ("pending", "running", "done", "failed", "workers"):
        (queue_dir / dirname).mkdir(parents=True, exist_ok=True)
    stale_files = list((queue_dir / "pending").rglob("task_*.json")) + list((queue_dir / "running").rglob("task_*.json"))
    for stale in stale_files:
        stale.unlink()

    metadata: list[dict[str, Any]] = []
    reactions_index: dict[str, list[dict[str, Any]]] = {}
    tasks: list[dict[str, Any]] = []
    task_gmtkn55_dir = container_path(gmtkn55_dir, container_repo_root)
    task_index = 0
    for candidate_index, candidate in enumerate(candidates, start=1):
        label = candidate_label(candidate, candidate_index)
        metadata.append(
            {
                "candidate_index": candidate_index,
                "label": label,
                "raw_reward": candidate.get("raw_reward"),
                "candidate": candidate,
            }
        )
        candidate_pending_dir = queue_dir / "pending"
        if str(queue_dispatch_mode) == "candidate_dedicated":
            candidate_pending_dir = queue_dir / "pending" / f"candidate_{candidate_index:06d}"
            candidate_pending_dir.mkdir(parents=True, exist_ok=True)
        for subset in subsets:
            subset_dir = gmtkn55_dir / subset
            reactions = parse_res_file(subset_dir / ".res")
            reactions_index[subset] = reactions
            mol_names = _molecule_names_from_reactions(reactions)
            if int(max_molecules) > 0:
                mol_names = mol_names[: int(max_molecules)]
            for mols in chunks(mol_names, max(1, int(molecules_per_task))):
                task_index += 1
                natoms_values = [count_atoms(subset_dir / mol_name) for mol_name in mols]
                natoms = max(value for value in natoms_values if value is not None) if any(v is not None for v in natoms_values) else None
                assigned_threads = assign_threads(
                    natoms=natoms,
                    policy=thread_policy,
                    fixed_threads=fixed_threads,
                    bins=atom_thread_bins,
                    max_threads=max_threads_per_sample,
                )
                task_name = f"task_{task_index:06d}.json"
                result_name = f"task_{task_index:06d}.result.json"
                if keep_existing_done and (
                    (queue_dir / "done" / result_name).exists()
                    or (queue_dir / "failed" / result_name).exists()
                ):
                    pending_path = ""
                else:
                    pending_path = str(candidate_pending_dir / task_name)
                task = {
                    "task_index": task_index,
                    "candidate_index": candidate_index,
                    "candidate_label": label,
                    "subset_name": subset,
                    "gmtkn55_dir": task_gmtkn55_dir,
                    "subset_dir": f"{task_gmtkn55_dir.rstrip('/')}/{subset}",
                    "mol_names": mols,
                    "natoms": natoms,
                    "natoms_values": dict(zip(mols, natoms_values)),
                    "assigned_threads": assigned_threads,
                    "xc_dict": candidate["xc_dict"],
                    "basis": basis,
                    "scf_conv_tol": scf_conv_tol,
                    "scf_max_cycle": scf_max_cycle,
                    "n_threads": assigned_threads,
                    "profile": True,
                    "queued_at": "",
                    "task_path": pending_path,
                    "result_path": str(queue_dir / "done" / result_name),
                    "failed_result_path": str(queue_dir / "failed" / result_name),
                    "skip_existing": True,
                }
                if pending_path:
                    task["queued_at"] = time.time()
                    Path(pending_path).write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
                tasks.append(task)

    (output_dir / "candidate_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "reactions_index.json").write_text(json.dumps(reactions_index, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "task_manifest.json").write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    return tasks


def write_sbatch_script(args: argparse.Namespace, output_dir: Path, *, config: dict[str, Any]) -> None:
    logs = output_dir / "slurm_logs"
    logs.mkdir(parents=True, exist_ok=True)
    queue_container = container_path(output_dir / "queue", args.container_repo_root)
    worker_count = int(config["worker_count"])
    cpus_per_worker = int(config["cpus_per_worker"])
    submit_mode = str(config.get("slurm_submit_mode") or "allocation")
    dispatch_mode = str(config.get("queue_dispatch_mode") or "global")
    workers_per_candidate = max(1, int(config.get("workers_per_candidate") or 1))
    lines = [
        "#!/bin/bash",
        f"#SBATCH --partition={args.partition}",
        "#SBATCH --job-name=dftagent-docker-queue",
        f"#SBATCH --mem={config.get('effective_mem') or args.mem}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --output={logs.as_posix()}/%x-%j.out",
        f"#SBATCH --error={logs.as_posix()}/%x-%j.err",
    ]
    if str(config.get("slurm_nodelist") or ""):
        lines.append(f"#SBATCH --nodelist={config['slurm_nodelist']}")
    if submit_mode == "array":
        array_spec = f"0-{max(0, worker_count - 1)}"
        if int(config.get("array_max_running") or 0) > 0:
            array_spec += f"%{int(config['array_max_running'])}"
        lines.insert(3, f"#SBATCH --array={array_spec}")
        lines.insert(4, "#SBATCH --ntasks=1")
        lines.insert(5, f"#SBATCH --cpus-per-task={cpus_per_worker}")
    else:
        lines.insert(3, f"#SBATCH --ntasks={worker_count}")
        lines.insert(4, f"#SBATCH --cpus-per-task={cpus_per_worker}")
        lines.insert(5, "#SBATCH --distribution=cyclic")
        nodes_spec = str(config.get("nodes_spec") or "")
        if nodes_spec:
            lines.insert(3, f"#SBATCH --nodes={nodes_spec}")
        elif int(config.get("nodes") or 0) > 0:
            lines.insert(3, f"#SBATCH --nodes={int(config['nodes'])}")
        if int(config.get("ntasks_per_node") or 0) > 0:
            lines.insert(4, f"#SBATCH --ntasks-per-node={int(config['ntasks_per_node'])}")
    for extra in args.extra_sbatch:
        lines.append(f"#SBATCH {extra}")
    docker_extra = " ".join(args.docker_extra_args)
    if docker_extra:
        docker_extra += " "
    worker_policy = "task" if args.thread_policy == "atom_adaptive" else "fixed"
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"export DFTAGENT_IMAGE={shell_quote(args.image)}",
            f"export DFTAGENT_REPO={shell_quote(args.container_repo_root)}",
            f"export DFTAGENT_QUEUE={shell_quote(queue_container)}",
            f"export DFTAGENT_CPUS_PER_WORKER={cpus_per_worker}",
            f"export DFTAGENT_THREAD_POLICY={shell_quote(worker_policy)}",
            f"export DFTAGENT_FIXED_THREADS={int(args.n_threads)}",
            f"export DFTAGENT_QUEUE_DISPATCH_MODE={shell_quote(dispatch_mode)}",
            f"export DFTAGENT_WORKERS_PER_CANDIDATE={workers_per_candidate}",
            "",
            "echo \"Slurm allocation: job=${SLURM_JOB_ID:-unset} ntasks=${SLURM_NTASKS:-unset} cpus_per_task=${SLURM_CPUS_PER_TASK:-unset} tres_per_task=${SLURM_TRES_PER_TASK:-unset} job_cpus_per_node=${SLURM_JOB_CPUS_PER_NODE:-unset}\"",
            "echo \"=== scontrol allocation (one line) ===\"",
            "scontrol show job -o \"${SLURM_JOB_ID}\" || true",
            "echo \"=== scontrol allocation fields ===\"",
            "scontrol show job \"${SLURM_JOB_ID}\" | grep -E \"JobState|NumNodes|NumCPUs|NumTasks|CPUs/Task|ReqTRES|AllocTRES|MinCPUsNode|MinMemory\" || true",
            "job_record=$(scontrol show job -o \"${SLURM_JOB_ID}\")",
            "allocation_cpus_per_task=$(printf '%s\\n' \"${job_record}\" | tr ' ' '\\n' | sed -n 's/^CPUs\\/Task=//p' | head -n 1)",
            "allocation_num_cpus=$(printf '%s\\n' \"${job_record}\" | tr ' ' '\\n' | sed -n 's/^NumCPUs=//p' | head -n 1)",
            "tres_cpus_per_task=$(printf '%s\\n' \"${SLURM_TRES_PER_TASK:-}\" | tr ',' '\\n' | sed -n 's/^cpu[=:]//p' | head -n 1)",
            "required_cpus=$(( ${SLURM_NTASKS:-1} * DFTAGENT_CPUS_PER_WORKER ))",
            "if [ -n \"${allocation_cpus_per_task}\" ] && [ \"${allocation_cpus_per_task}\" != \"${DFTAGENT_CPUS_PER_WORKER}\" ]; then",
            "  if [ \"${tres_cpus_per_task:-}\" = \"${DFTAGENT_CPUS_PER_WORKER}\" ] && [ \"${allocation_num_cpus:-0}\" -ge \"${required_cpus}\" ]; then",
            "    echo \"WARNING: Slurm legacy CPUs/Task=${allocation_cpus_per_task}, but TRES/task=cpu=${tres_cpus_per_task} and NumCPUs=${allocation_num_cpus}; treating the legacy field as stale\" >&2",
            "  else",
            "    echo \"ERROR: inconsistent Slurm allocation: CPUs/Task=${allocation_cpus_per_task:-unset}, TRES/task=cpu=${tres_cpus_per_task:-unset}, NumCPUs=${allocation_num_cpus:-unset}, required=${required_cpus}\" >&2",
            "    exit 2",
            "  fi",
            "fi",
            "if [ \"${SLURM_CPUS_PER_TASK:-}\" != \"${DFTAGENT_CPUS_PER_WORKER}\" ]; then",
            "  echo \"WARNING: correcting stale SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset} to ${DFTAGENT_CPUS_PER_WORKER}; allocation CPUs/Task=${allocation_cpus_per_task:-unknown}\" >&2",
            "  export SLURM_CPUS_PER_TASK=\"${DFTAGENT_CPUS_PER_WORKER}\"",
            "fi",
            "",
            "echo \"=== allocated hostnames ===\"",
            "scontrol show hostnames \"${SLURM_JOB_NODELIST:-}\" | sort | uniq -c || true",
            "if [ \"${SLURM_ARRAY_TASK_ID:-}\" = \"\" ]; then",
            "  echo \"=== per-task host placement before docker ===\"",
            "  srun --ntasks=${SLURM_NTASKS:-1} --cpus-per-task=${DFTAGENT_CPUS_PER_WORKER} --cpu-bind=cores bash -lc 'echo \"host=$(hostname) proc=${SLURM_PROCID:-NA} localid=${SLURM_LOCALID:-NA} cpus_on_node=${SLURM_CPUS_ON_NODE:-NA} cpus_per_task=${SLURM_CPUS_PER_TASK:-NA}\"' | sort || true",
            "  echo \"=== per-node task counts ===\"",
            "  srun --ntasks=${SLURM_NTASKS:-1} --cpus-per-task=${DFTAGENT_CPUS_PER_WORKER} --cpu-bind=cores bash -lc 'hostname' | sort | uniq -c || true",
            "fi",
            _srun_launcher(submit_mode),
            "  set -euo pipefail",
            _worker_id_line(submit_mode),
            "  candidate_arg=\"\"",
            "  if [ \"${DFTAGENT_QUEUE_DISPATCH_MODE}\" = \"candidate_dedicated\" ]; then",
            _candidate_index_line(submit_mode),
            "    candidate_arg=\"--candidate-index ${candidate_index}\"",
            "  fi",
            f"  sudo -n slurm-docker-run --pull missing --network host {docker_extra}--workdir \"${{DFTAGENT_REPO}}\" \"${{DFTAGENT_IMAGE}}\" bash -lc \"",
            "    set -euo pipefail",
            "    export PYTHONPATH=${DFTAGENT_REPO}/src:\\${PYTHONPATH:-}",
            "    export DFTAGENT_COLOR=0",
            "    python3 -m xc.benchmark.file_queue_worker \\",
            "      --queue-dir \\\"${DFTAGENT_QUEUE}\\\" \\",
            "      --worker-id \\\"${worker_id}\\\" \\",
            "      --allocation-id \\\"${SLURM_JOB_ID}.${SLURM_PROCID}\\\" \\",
            "      --local-cpus \\\"${DFTAGENT_CPUS_PER_WORKER}\\\" \\",
            "      --local-workers \\\"${DFTAGENT_CPUS_PER_WORKER}\\\" \\",
            "      --thread-policy \\\"${DFTAGENT_THREAD_POLICY}\\\" \\",
            "      --n-threads \\\"${DFTAGENT_FIXED_THREADS}\\\" \\",
            "      ${candidate_arg}",
            "  \"",
            "'",
        ]
    )
    (output_dir / "run_slurm_docker_queue.sbatch").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _srun_launcher(submit_mode: str) -> str:
    if submit_mode == "array":
        return "srun --ntasks=1 --cpus-per-task=${DFTAGENT_CPUS_PER_WORKER} --kill-on-bad-exit=1 bash -lc '"
    return (
        "srun --ntasks=${SLURM_NTASKS} --cpus-per-task=${DFTAGENT_CPUS_PER_WORKER} "
        "--distribution=cyclic --kill-on-bad-exit=1 bash -lc '"
    )


def _worker_id_line(submit_mode: str) -> str:
    if submit_mode == "array":
        return '  worker_id="worker_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}_${SLURMD_NODENAME:-node}"'
    return '  worker_id="worker_${SLURM_JOB_ID}_${SLURM_PROCID}_${SLURMD_NODENAME:-node}"'


def _candidate_index_line(submit_mode: str) -> str:
    if submit_mode == "array":
        return "    candidate_index=$(( (${SLURM_ARRAY_TASK_ID:-0} / ${DFTAGENT_WORKERS_PER_CANDIDATE:-1}) + 1 ))"
    return "    candidate_index=$(( (${SLURM_PROCID:-0} / ${DFTAGENT_WORKERS_PER_CANDIDATE:-1}) + 1 ))"


def effective_mem_request(mem: str, submit_mode: str) -> str:
    text = str(mem).strip()
    if submit_mode == "array" and text in {"", "0", "0G", "0M", "0GB", "0MB"}:
        return "16G"
    return text or "16G"


def clean_sbatch_env() -> dict[str, str]:
    """Remove parent allocation settings before calling ``sbatch``."""

    env = clean_slurm_submit_env()
    removed = sorted(set(os.environ) - set(env))
    if removed:
        logging.warning("Cleared inherited Slurm submit environment: %s", ",".join(removed))
    return env


def aggregate_outputs(output_dir: Path, *, subsets: list[str], aggregate_mode: str) -> None:
    metadata = json.loads((output_dir / "candidate_metadata.json").read_text(encoding="utf-8"))
    reactions_index = json.loads((output_dir / "reactions_index.json").read_text(encoding="utf-8"))
    task_manifest = json.loads((output_dir / "task_manifest.json").read_text(encoding="utf-8"))
    scheduler = json.loads((output_dir / "scheduler_config.json").read_text(encoding="utf-8"))
    cutoff_candidates = load_candidate_cutoffs(output_dir)
    payloads: list[dict[str, Any]] = []
    task_rows, molecule_rows = collect_timing_rows(output_dir, task_manifest)

    for item in metadata:
        candidate_index = int(item["candidate_index"])
        candidate = item["candidate"]
        subset_results = {}
        candidate_tasks = [task for task in task_manifest if int(task["candidate_index"]) == candidate_index]
        candidate_dir = output_dir / f"candidate_{candidate_index:03d}_{safe_name(item['label'])}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for subset in subsets:
            tasks = [task for task in candidate_tasks if task["subset_name"] == subset]
            subset_result, reaction_rows = aggregate_subset(
                subset=subset,
                tasks=tasks,
                reactions=reactions_index[subset],
                label=item["label"],
            )
            subset_results[subset] = subset_result
            (candidate_dir / f"{subset}_result.json").write_text(
                json.dumps(subset_result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (candidate_dir / f"{subset}_reaction_rows.json").write_text(
                json.dumps(reaction_rows, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        xc = XCFunctional.from_dict(candidate["xc_dict"])
        result = BenchmarkResult(xc_name=xc.name, subsets=subset_results)
        result.compute_overall(aggregate_mode=aggregate_mode)
        result_dict = benchmark_result_to_dict(result)
        if candidate_index in cutoff_candidates:
            result_dict["runtime_cutoff"] = True
            result_dict["runtime_cutoff_reason"] = cutoff_candidates[candidate_index].get(
                "reason",
                "candidate_tail_cutoff",
            )
            result_dict["runtime_cutoff_detail"] = cutoff_candidates[candidate_index]
        payloads.append({"candidate": candidate, "benchmark_result": result_dict})

    write_csv(output_dir / "queue_task_profile.csv", task_rows)
    write_csv(output_dir / "molecule_profile.csv", molecule_rows)
    write_scheduler_summary(output_dir, task_rows, molecule_rows, scheduler)
    write_summary(output_dir, payloads, subsets)


def load_candidate_cutoffs(output_dir: Path) -> dict[int, dict[str, Any]]:
    path = output_dir / "candidate_tail_cutoff.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    result: dict[int, dict[str, Any]] = {}
    for item in payload.get("cutoff_candidates", []) if isinstance(payload, dict) else []:
        try:
            result[int(item["candidate_index"])] = dict(item)
        except Exception:
            continue
    return result


def aggregate_subset(
    *,
    subset: str,
    tasks: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
    label: str,
):
    mol_energies: dict[str, float | None] = {}
    mol_errors: dict[str, str] = {}
    starts: list[float] = []
    ends: list[float] = []
    for task in tasks:
        result_path = Path(task["result_path"])
        failed_path = Path(task["failed_result_path"])
        if not result_path.exists():
            error = f"missing_queue_result:{result_path}"
            if failed_path.exists():
                try:
                    error = json.loads(failed_path.read_text(encoding="utf-8")).get("error", error)
                except Exception:
                    pass
            for mol_name in task["mol_names"]:
                mol_energies[mol_name] = None
                mol_errors[mol_name] = error
            continue
        record = json.loads(result_path.read_text(encoding="utf-8"))
        payload = record.get("payload") or {}
        mol_energies.update(payload.get("mol_energies") or {})
        mol_errors.update(payload.get("mol_errors") or {})
        for mol in payload.get("molecule_timings") or []:
            if mol.get("started_at") is not None:
                starts.append(float(mol["started_at"]))
            if mol.get("ended_at") is not None:
                ends.append(float(mol["ended_at"]))
    walltime = max(0.0, max(ends) - min(starts)) if starts and ends else 0.0
    return _subset_result_from_molecule_energies(
        subset_name=subset,
        reactions=reactions,
        mol_energies=mol_energies,
        mol_errors=mol_errors,
        walltime_s=walltime,
        log_label=label,
    )


def collect_timing_rows(output_dir: Path, task_manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    molecule_rows: list[dict[str, Any]] = []
    for task in task_manifest:
        result_path = Path(task["result_path"])
        failed_path = Path(task["failed_result_path"])
        record = {}
        status = "missing"
        if result_path.exists():
            record = json.loads(result_path.read_text(encoding="utf-8"))
            status = "done"
        elif failed_path.exists():
            record = json.loads(failed_path.read_text(encoding="utf-8"))
            status = "failed"
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        assigned_threads = int(record.get("assigned_threads") or task.get("assigned_threads") or payload.get("assigned_threads") or 1)
        wall = float(payload.get("walltime_s") or 0.0)
        process_cpu = float(payload.get("process_cpu_time_s") or 0.0)
        task_rows.append(
            {
                "task_index": task["task_index"],
                "status": status,
                "candidate_index": task["candidate_index"],
                "candidate_label": task["candidate_label"],
                "subset_name": task["subset_name"],
                "mol_names": "|".join(task["mol_names"]),
                "natoms": task.get("natoms"),
                "assigned_threads": assigned_threads,
                "hostname": record.get("hostname", ""),
                "worker_id": record.get("worker_id", ""),
                "allocation_id": record.get("allocation_id", ""),
                "local_worker_index": record.get("local_worker_index", ""),
                "claimed_at": record.get("claimed_at", ""),
                "started_at": payload.get("started_at", ""),
                "ended_at": payload.get("ended_at", ""),
                "finished_at": record.get("finished_at", ""),
                "walltime_s": wall,
                "cpu_walltime_s": float(payload.get("cpu_walltime_s") or (wall * assigned_threads)),
                "process_cpu_time_s": process_cpu,
                "cpu_efficiency_vs_assigned": payload.get("cpu_efficiency_vs_assigned", ""),
                "error": record.get("error", ""),
            }
        )
        for mol in payload.get("molecule_timings") or []:
            molecule_rows.append(
                {
                    "task_index": task["task_index"],
                    "candidate_index": task["candidate_index"],
                    "candidate_label": task["candidate_label"],
                    "subset_name": task["subset_name"],
                    "mol_name": mol.get("mol_name"),
                    "natoms": (task.get("natoms_values") or {}).get(mol.get("mol_name")),
                    "assigned_threads": mol.get("assigned_threads", assigned_threads),
                    "ok": mol.get("ok"),
                    "walltime_s": mol.get("walltime_s"),
                    "cpu_walltime_s": mol.get("cpu_walltime_s"),
                    "process_cpu_time_s": mol.get("process_cpu_time_s"),
                    "cpu_efficiency_vs_assigned": mol.get("cpu_efficiency_vs_assigned"),
                    "started_at": mol.get("started_at"),
                    "ended_at": mol.get("ended_at"),
                    "hostname": record.get("hostname", ""),
                    "worker_id": record.get("worker_id", ""),
                    "allocation_id": record.get("allocation_id", ""),
                    "error": mol.get("error"),
                }
            )
    return task_rows, molecule_rows


def write_scheduler_summary(output_dir: Path, task_rows: list[dict[str, Any]], molecule_rows: list[dict[str, Any]], scheduler: dict[str, Any]) -> None:
    starts = [float(row["started_at"]) for row in task_rows if row.get("started_at") not in ("", None)]
    ends = [float(row["ended_at"]) for row in task_rows if row.get("ended_at") not in ("", None)]
    global_wall = max(0.0, max(ends) - min(starts)) if starts and ends else 0.0
    allocated_cpus = int(scheduler.get("allocated_cpus") or 0)
    used_cpu_wall = sum(float(row.get("cpu_walltime_s") or 0.0) for row in task_rows)
    process_cpu_time = sum(float(row.get("process_cpu_time_s") or 0.0) for row in task_rows)
    thread_hist: dict[str, int] = {}
    for row in task_rows:
        key = str(row.get("assigned_threads") or "")
        thread_hist[key] = thread_hist.get(key, 0) + 1
    summary = {
        "n_tasks": len(task_rows),
        "n_done": sum(1 for row in task_rows if row.get("status") == "done"),
        "n_failed": sum(1 for row in task_rows if row.get("status") == "failed"),
        "n_missing": sum(1 for row in task_rows if row.get("status") == "missing"),
        "n_molecules": len(molecule_rows),
        "global_wall_s": global_wall,
        "allocated_cpus": allocated_cpus,
        "allocated_cpu_wall_s": global_wall * allocated_cpus,
        "used_cpu_wall_s": used_cpu_wall,
        "cpu_utilization": used_cpu_wall / max(1e-12, global_wall * allocated_cpus) if global_wall and allocated_cpus else None,
        "process_cpu_time_s": process_cpu_time,
        "process_cpu_utilization": process_cpu_time / max(1e-12, global_wall * allocated_cpus) if global_wall and allocated_cpus else None,
        "thread_histogram": thread_hist,
        "scheduler_config": scheduler,
    }
    (output_dir / "queue_scheduler_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(output_dir: Path, payloads: list[dict[str, Any]], subsets: list[str]) -> None:
    (output_dir / "slurm_docker_queue_benchmark_results.json").write_text(
        json.dumps(payloads, indent=2, ensure_ascii=False),
        encoding="utf-8",
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
        "runtime_cutoff",
        "runtime_cutoff_reason",
    ]
    for subset in subsets:
        fields.extend([f"{subset}_mae", f"{subset}_failures", f"{subset}_n", f"{subset}_wall_s"])
    with (output_dir / "slurm_docker_queue_benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for payload in payloads:
            candidate = payload["candidate"]
            result = payload["benchmark_result"]
            row = {
                "rank": candidate.get("rank"),
                "label": candidate.get("label"),
                "source": candidate.get("source"),
                "source_id": candidate.get("source_id"),
                "source_raw_reward": candidate.get("raw_reward"),
                "overall_mae": result.get("overall_mae"),
                "overall_rms": result.get("overall_rms"),
                "total_failures": result.get("total_failures"),
                "total_walltime_s": result.get("total_walltime_s"),
                "runtime_cutoff": result.get("runtime_cutoff", False),
                "runtime_cutoff_reason": result.get("runtime_cutoff_reason", ""),
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


def load_manifest(path: Path, *, keep_order: bool) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected manifest JSON list: {path}")
    candidates = [item for item in payload if isinstance(item, dict) and isinstance(item.get("xc_dict"), dict)]
    if not keep_order:
        candidates.sort(key=lambda item: _float_or_neg_inf(item.get("raw_reward")), reverse=True)
    return candidates


def count_atoms(mol_dir: Path) -> int | None:
    xyz = mol_dir / "struc.xyz"
    if xyz.exists():
        try:
            return int(xyz.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip())
        except Exception:
            pass
    coord = mol_dir / "coord"
    if coord.exists():
        count = 0
        for line in coord.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("$") and len(stripped.split()) >= 4:
                count += 1
        return count
    return None


def parse_thread_bins(value: str) -> list[tuple[int, int]]:
    bins: list[tuple[int, int]] = []
    for item in str(value).split(","):
        if not item.strip():
            continue
        upper, threads = item.split(":", 1)
        bins.append((int(upper), int(threads)))
    if not bins:
        bins = [(999, 1)]
    return sorted(bins, key=lambda pair: pair[0])


def assign_threads(
    *,
    natoms: int | None,
    policy: str,
    fixed_threads: int,
    bins: list[tuple[int, int]],
    max_threads: int,
) -> int:
    if policy == "fixed":
        return min(max(1, int(fixed_threads)), max(1, int(max_threads)))
    if natoms is None:
        return 1
    for upper, threads in bins:
        if int(natoms) <= int(upper):
            return min(max(1, int(threads)), max(1, int(max_threads)))
    return min(max(1, int(bins[-1][1])), max(1, int(max_threads)))


def benchmark_result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "xc_name": result.xc_name,
        "subsets": {name: subset.to_dict() for name, subset in result.subsets.items()},
        "overall_mae": result.overall_mae,
        "overall_rms": result.overall_rms,
        "total_walltime_s": result.total_walltime_s,
        "total_failures": result.total_failures,
        "timestamp": result.timestamp,
    }


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def candidate_label(candidate: dict[str, Any], index: int) -> str:
    return str(candidate.get("label") or candidate.get("source_id") or f"candidate_{index:03d}")


def safe_name(value: str) -> str:
    keep = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value)]
    return "_".join("".join(keep).split("_"))[:120] or "item"


def container_path(path: Path, repo_root: str) -> str:
    rel = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    return f"{repo_root.rstrip('/')}/{rel}"


def shell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_pending(output_dir: Path) -> int:
    return len(list((output_dir / "queue" / "pending").rglob("task_*.json")))


def _float_or_neg_inf(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


if __name__ == "__main__":
    main()
