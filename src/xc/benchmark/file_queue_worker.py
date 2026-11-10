"""Long-lived Slurm worker for file-backed dynamic molecule queues."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .runner import _worker_compute_molecule, _worker_init


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--allocation-id", default="")
    parser.add_argument("--local-workers", type=int, default=8)
    parser.add_argument("--local-cpus", type=int, default=None)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--thread-policy", choices=["fixed", "task"], default="fixed")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="Optional 1-based candidate index. When set, this worker only consumes that candidate's tasks.",
    )
    parser.add_argument("--poll-interval", type=float, default=0.2)
    args = parser.parse_args()
    local_cpus = max(1, int(args.local_cpus or (int(args.local_workers) * int(args.n_threads))))
    candidate_index = max(0, int(args.candidate_index or 0))

    queue_dir = Path(args.queue_dir)
    for name in ("pending", "running", "done", "failed"):
        (queue_dir / name).mkdir(parents=True, exist_ok=True)

    heartbeat = queue_dir / "workers" / f"{_safe_name(args.worker_id)}.json"
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.write_text(
        json.dumps(
            {
                "event": "worker_start",
                "worker_id": args.worker_id,
                "allocation_id": args.allocation_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "local_workers": int(args.local_workers),
                "local_cpus": local_cpus,
                "n_threads": int(args.n_threads),
                "thread_policy": str(args.thread_policy),
                "candidate_index": candidate_index,
                "started_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if str(args.thread_policy) == "task":
        _run_task_thread_queue(
            queue_dir=queue_dir,
            worker_id=str(args.worker_id),
            allocation_id=str(args.allocation_id),
            local_cpus=local_cpus,
            candidate_index=candidate_index,
            poll_interval=max(0.05, float(args.poll_interval)),
        )
    else:
        with ProcessPoolExecutor(
            max_workers=max(1, int(args.local_workers)),
            initializer=_worker_init,
            initargs=(max(1, int(args.n_threads)),),
        ) as pool:
            futures = [
                pool.submit(
                    _queue_process_loop,
                    str(queue_dir),
                    str(args.worker_id),
                    str(args.allocation_id),
                    worker_index,
                    max(1, int(args.n_threads)),
                    candidate_index,
                )
                for worker_index in range(1, max(1, int(args.local_workers)) + 1)
            ]
            for future in as_completed(futures):
                future.result()

    heartbeat.write_text(
        json.dumps(
            {
                "event": "worker_done",
                "worker_id": args.worker_id,
                "allocation_id": args.allocation_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "local_workers": int(args.local_workers),
                "local_cpus": local_cpus,
                "n_threads": int(args.n_threads),
                "thread_policy": str(args.thread_policy),
                "candidate_index": candidate_index,
                "ended_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def _queue_process_loop(
    queue_dir_text: str,
    worker_id: str,
    allocation_id: str,
    local_worker_index: int,
    n_threads: int,
    candidate_index: int,
) -> int:
    queue_dir = Path(queue_dir_text)
    pending_dir = _pending_dir_for_candidate(queue_dir, candidate_index)
    running_dir = queue_dir / "running"
    done_dir = queue_dir / "done"
    failed_dir = queue_dir / "failed"

    while True:
        claim = _claim_next_task(
            pending_dir,
            running_dir,
            worker_id,
            local_worker_index,
            candidate_index=candidate_index,
        )
        if claim is None:
            return 0
        original_name, running_path = claim
        claimed_at = time.time()
        try:
            task = json.loads(running_path.read_text(encoding="utf-8"))
            payload = _execute_task(task, n_threads)
            record = {
                "ok": True,
                "task_file": original_name,
                "worker_id": worker_id,
                "allocation_id": allocation_id,
                "local_worker_index": int(local_worker_index),
                "assigned_threads": int(n_threads),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "claimed_at": claimed_at,
                "finished_at": time.time(),
                "task": task,
                "payload": payload,
            }
            _write_result(done_dir / _result_name(original_name), record)
        except Exception as exc:
            record = {
                "ok": False,
                "task_file": original_name,
                "worker_id": worker_id,
                "allocation_id": allocation_id,
                "local_worker_index": int(local_worker_index),
                "assigned_threads": int(n_threads),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "claimed_at": claimed_at,
                "finished_at": time.time(),
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            _write_result(failed_dir / _result_name(original_name), record)
        finally:
            try:
                running_path.unlink()
            except FileNotFoundError:
                pass


def _run_task_thread_queue(
    *,
    queue_dir: Path,
    worker_id: str,
    allocation_id: str,
    local_cpus: int,
    candidate_index: int,
    poll_interval: float,
) -> None:
    pending_dir = _pending_dir_for_candidate(queue_dir, candidate_index)
    running_dir = queue_dir / "running"
    available_cpus = max(1, int(local_cpus))
    active: list[dict[str, Any]] = []
    task_counter = 0
    with mp.Pool(processes=max(1, int(local_cpus)), maxtasksperchild=1) as pool:
        while True:
            still_active: list[dict[str, Any]] = []
            for item in active:
                result = item["async_result"]
                if result.ready():
                    result.get()
                    available_cpus += int(item["threads"])
                else:
                    still_active.append(item)
            active = still_active

            launched = False
            while available_cpus > 0:
                claim = _claim_next_fitting_task(
                    pending_dir=pending_dir,
                    running_dir=running_dir,
                    worker_id=worker_id,
                    local_worker_index=task_counter + 1,
                    max_threads=available_cpus,
                    local_cpus=local_cpus,
                    candidate_index=candidate_index,
                )
                if claim is None:
                    break
                original_name, running_path, threads = claim
                task_counter += 1
                available_cpus -= threads
                async_result = pool.apply_async(
                    _run_claimed_task,
                    (
                        str(queue_dir),
                        original_name,
                        str(running_path),
                        worker_id,
                        allocation_id,
                        task_counter,
                        threads,
                    ),
                )
                active.append({"async_result": async_result, "threads": threads})
                launched = True

            if not active and not _has_pending_tasks(pending_dir, candidate_index=candidate_index):
                return
            if not launched:
                time.sleep(poll_interval)


def _run_claimed_task(
    queue_dir_text: str,
    original_name: str,
    running_path_text: str,
    worker_id: str,
    allocation_id: str,
    local_worker_index: int,
    n_threads: int,
) -> int:
    queue_dir = Path(queue_dir_text)
    running_path = Path(running_path_text)
    done_dir = queue_dir / "done"
    failed_dir = queue_dir / "failed"
    claimed_at = time.time()
    try:
        task = json.loads(running_path.read_text(encoding="utf-8"))
        payload = _execute_task(task, n_threads)
        record = {
            "ok": True,
            "task_file": original_name,
            "worker_id": worker_id,
            "allocation_id": allocation_id,
            "local_worker_index": int(local_worker_index),
            "assigned_threads": int(n_threads),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_at": claimed_at,
            "finished_at": time.time(),
            "task": task,
            "payload": payload,
        }
        _write_result(done_dir / _result_name(original_name), record)
    except Exception as exc:
        record = {
            "ok": False,
            "task_file": original_name,
            "worker_id": worker_id,
            "allocation_id": allocation_id,
            "local_worker_index": int(local_worker_index),
            "assigned_threads": int(n_threads),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_at": claimed_at,
            "finished_at": time.time(),
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        _write_result(failed_dir / _result_name(original_name), record)
    finally:
        try:
            running_path.unlink()
        except FileNotFoundError:
            pass
    return 0


def _claim_next_task(
    pending_dir: Path,
    running_dir: Path,
    worker_id: str,
    local_worker_index: int,
    candidate_index: int = 0,
) -> tuple[str, Path] | None:
    for task_path in sorted(pending_dir.glob("task_*.json")):
        if not _task_matches_candidate(task_path, candidate_index):
            continue
        running_name = f"{task_path.stem}__{_safe_name(worker_id)}__w{local_worker_index:02d}.json"
        running_path = running_dir / running_name
        try:
            task_path.replace(running_path)
            return task_path.name, running_path
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return None


def _claim_next_fitting_task(
    *,
    pending_dir: Path,
    running_dir: Path,
    worker_id: str,
    local_worker_index: int,
    max_threads: int,
    local_cpus: int,
    candidate_index: int = 0,
) -> tuple[str, Path, int] | None:
    candidates: list[tuple[int, str, Path]] = []
    for task_path in sorted(pending_dir.glob("task_*.json")):
        if not _task_matches_candidate(task_path, candidate_index):
            continue
        threads = _threads_for_task_file(task_path, local_cpus=local_cpus)
        if threads <= max_threads:
            candidates.append((-threads, task_path.name, task_path))
    for _, _, task_path in sorted(candidates):
        threads = _threads_for_task_file(task_path, local_cpus=local_cpus)
        running_name = f"{task_path.stem}__{_safe_name(worker_id)}__w{local_worker_index:04d}.json"
        running_path = running_dir / running_name
        try:
            task_path.replace(running_path)
            return task_path.name, running_path, min(max(1, threads), max(1, int(local_cpus)))
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return None


def _threads_for_task_file(task_path: Path, *, local_cpus: int) -> int:
    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
        value = payload.get("assigned_threads") or payload.get("n_threads") or 1
        return min(max(1, int(value)), max(1, int(local_cpus)))
    except Exception:
        return 1


def _has_pending_tasks(pending_dir: Path, *, candidate_index: int = 0) -> bool:
    if int(candidate_index or 0) <= 0:
        return any(pending_dir.rglob("task_*.json"))
    return any(_task_matches_candidate(path, candidate_index) for path in pending_dir.glob("task_*.json"))


def _pending_dir_for_candidate(queue_dir: Path, candidate_index: int) -> Path:
    if int(candidate_index or 0) <= 0:
        return queue_dir / "pending"
    candidate_dir = queue_dir / "pending" / f"candidate_{int(candidate_index):06d}"
    if candidate_dir.exists():
        return candidate_dir
    # Backward-compatible fallback for queues prepared before candidate-local
    # pending directories were introduced.
    return queue_dir / "pending"


def _task_matches_candidate(task_path: Path, candidate_index: int) -> bool:
    if int(candidate_index or 0) <= 0:
        return True
    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
        return int(payload.get("candidate_index") or 0) == int(candidate_index)
    except Exception:
        return False


def _execute_task(task: dict[str, Any], n_threads: int) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(n_threads)))
    os.environ["MKL_NUM_THREADS"] = str(max(1, int(n_threads)))
    t0 = time.time()
    cpu0 = time.process_time()
    mol_energies: dict[str, float | None] = {}
    mol_errors: dict[str, str] = {}
    molecule_timings: list[dict[str, Any]] = []
    subset_dir = Path(task["gmtkn55_dir"]) / str(task["subset_name"])
    density_output_dir = task.get("density_output_dir")
    for mol_name in [str(name) for name in task.get("mol_names", [])]:
        mol_cpu0 = time.process_time()
        outcome = _worker_compute_molecule(
            (
                str(subset_dir),
                mol_name,
                task["xc_dict"],
                task["basis"],
                float(task["scf_conv_tol"]),
                int(task["scf_max_cycle"]),
                str(density_output_dir) if density_output_dir else None,
            )
        )
        mol_process_cpu = max(0.0, time.process_time() - mol_cpu0)
        assigned_cpu_wall = float(outcome.walltime_s) * int(n_threads)
        if outcome.ok:
            mol_energies[mol_name] = outcome.energy
        else:
            mol_energies[mol_name] = None
            mol_errors[mol_name] = outcome.error
        molecule_timings.append(
            {
                "mol_name": mol_name,
                "ok": bool(outcome.ok),
                "walltime_s": float(outcome.walltime_s),
                "started_at": float(outcome.started_at),
                "ended_at": float(outcome.ended_at),
                "assigned_threads": int(n_threads),
                "cpu_walltime_s": assigned_cpu_wall,
                "process_cpu_time_s": mol_process_cpu,
                "cpu_efficiency_vs_assigned": mol_process_cpu / max(1e-12, assigned_cpu_wall),
                "error": outcome.error,
            }
        )
    t1 = time.time()
    process_cpu = max(0.0, time.process_time() - cpu0)
    assigned_cpu_wall = (t1 - t0) * int(n_threads)
    return {
        "mol_energies": mol_energies,
        "mol_errors": mol_errors,
        "walltime_s": t1 - t0,
        "started_at": t0,
        "ended_at": t1,
        "n_molecules": len(mol_energies),
        "molecule_timings": molecule_timings,
        "n_workers": 1,
        "n_threads": int(n_threads),
        "assigned_threads": int(n_threads),
        "cpu_walltime_s": assigned_cpu_wall,
        "process_cpu_time_s": process_cpu,
        "cpu_efficiency_vs_assigned": process_cpu / max(1e-12, assigned_cpu_wall),
    }


def _write_result(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _result_name(task_file_name: str) -> str:
    return f"{Path(task_file_name).stem}.result.json"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:160]


if __name__ == "__main__":
    raise SystemExit(main())
