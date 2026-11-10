"""Persistent VC worker that consumes benchmark and crystal tasks until STOP."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .persistent_vc_queue import _atomic_write_json, _read_json, initialise_queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue_dir = initialise_queue(args.queue_dir.resolve())
    worker_id = _safe(args.worker_id)
    cpus = max(1, int(args.cpus))
    heartbeat = queue_dir / "workers" / f"{worker_id}.json"
    _atomic_write_json(
        heartbeat,
        {
            "event": "worker_start",
            "worker_id": worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "cpus": cpus,
            "started_at": time.time(),
        },
    )
    completed = 0
    failed = 0
    try:
        while True:
            # STOP means the controller is finished (or failed). Complete the
            # currently running task, but do not claim more queued work.
            if (queue_dir / "STOP").exists():
                break
            claim = _claim(queue_dir, worker_id)
            if claim is None:
                time.sleep(max(0.1, float(args.poll_seconds)))
                continue
            original_name, running_path, task = claim
            task_id = original_name[len("task_") : -len(".json")]
            started = time.time()
            try:
                detail = _execute(task, cpus=cpus)
                _atomic_write_json(
                    queue_dir / "done" / f"task_{task_id}.result.json",
                    {
                        "ok": True,
                        "task_id": task_id,
                        "kind": task.get("kind"),
                        "worker_id": worker_id,
                        "hostname": socket.gethostname(),
                        "started_at": started,
                        "finished_at": time.time(),
                        "detail": detail,
                    },
                )
                completed += 1
            except Exception as exc:
                _atomic_write_json(
                    queue_dir / "failed" / f"task_{task_id}.result.json",
                    {
                        "ok": False,
                        "task_id": task_id,
                        "worker_id": worker_id,
                        "hostname": socket.gethostname(),
                        "started_at": started,
                        "finished_at": time.time(),
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                failed += 1
            finally:
                running_path.unlink(missing_ok=True)
    finally:
        _atomic_write_json(
            heartbeat,
            {
                "event": "worker_done",
                "worker_id": worker_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "cpus": cpus,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "ended_at": time.time(),
            },
        )
    return 0


def _claim(
    queue_dir: Path, worker_id: str
) -> tuple[str, Path, dict[str, Any]] | None:
    for source in sorted((queue_dir / "pending").glob("task_*.json")):
        target = queue_dir / "running" / f"{source.stem}__{worker_id}.json"
        try:
            # Read before the atomic rename. Some shared filesystems briefly
            # hide the destination path after a successful cross-client
            # rename; execution must not depend on immediately reopening it.
            # Ownership is still decided exclusively by source.replace(): if
            # another worker wins, this worker discards the payload and moves
            # on without executing it.
            task = _read_json(source)
            source.replace(target)
            return source.name, target, task
        except (FileNotFoundError, PermissionError):
            continue
        except OSError:
            continue
    return None


def _execute(task: dict[str, Any], *, cpus: int) -> dict[str, Any]:
    kind = str(task.get("kind") or "")
    if kind == "benchmark_shard":
        return _run_benchmark(task, cpus=cpus)
    if kind == "crystal_guard":
        return _run_crystal(task, cpus=cpus)
    if kind == "smoke":
        result_path = _resolve_repo_path(str(task["result_path"]))
        _atomic_write_json(result_path, {"ok": True, "value": task.get("value")})
        return {"result_path": str(result_path)}
    raise ValueError(f"Unsupported persistent VC task kind: {kind!r}")


def _run_benchmark(task: dict[str, Any], *, cpus: int) -> dict[str, Any]:
    from .benchmark.runner import compute_molecule_batch
    from .xc_functional import XCFunctional

    payload = dict(task["payload"])
    result_path = _resolve_repo_path(str(task["result_path"]))
    subset_dir = _resolve_repo_path(str(payload["subset_dir"]))
    density_output = payload.get("density_output_dir")
    n_threads = max(1, int(payload.get("n_threads") or 1))
    max_workers = max(1, cpus // n_threads)
    result = compute_molecule_batch(
        subset_dir=subset_dir,
        mol_names=[str(item) for item in payload["mol_names"]],
        xc=XCFunctional.from_dict(payload["xc_dict"]),
        basis=str(payload["basis"]),
        scf_conv_tol=float(payload["scf_conv_tol"]),
        scf_max_cycle=int(payload["scf_max_cycle"]),
        n_workers=min(max_workers, max(1, int(payload.get("n_workers") or max_workers))),
        n_threads=n_threads,
        density_output_dir=_resolve_repo_path(str(density_output)) if density_output else None,
    )
    result.update(
        {
            "candidate_label": payload.get("candidate_label"),
            "subset_name": payload.get("subset_name"),
            "shard_index": payload.get("shard_index"),
            "n_shards": payload.get("n_shards"),
        }
    )
    _atomic_write_json(result_path, result)
    return {"result_path": str(result_path), "walltime_s": result.get("walltime_s")}


def _run_crystal(task: dict[str, Any], *, cpus: int) -> dict[str, Any]:
    task_path = _resolve_repo_path(str(task["task_json"]))
    result_path = _resolve_repo_path(str(task["result_path"]))
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(cpus)
    command = [
        sys.executable,
        str(_repo_root() / "case_study/crystal/scripts/python/crystal_guard_task_worker.py"),
        "--task-json",
        str(task_path),
    ]
    timeout_s = max(60, int(task.get("hard_timeout_s") or 7200))
    try:
        completed = subprocess.run(
            command,
            cwd=_repo_root(),
            env=env,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(command + ["--mark-timeout"], cwd=_repo_root(), env=env, check=False)
        return {"result_path": str(result_path), "timed_out": True}
    # Candidate-level SCF failures are valid terminal outcomes if the worker
    # wrote result.json. Only a missing result is an infrastructure failure.
    if not result_path.exists():
        detail = (completed.stderr or "")[-2000:]
        raise RuntimeError(f"Crystal worker rc={completed.returncode} produced no result: {detail}")
    return {"result_path": str(result_path), "returncode": completed.returncode}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:160]


if __name__ == "__main__":
    raise SystemExit(main())
