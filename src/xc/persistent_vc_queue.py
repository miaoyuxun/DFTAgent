"""File-backed queue shared by a run-level persistent VC worker fleet."""

from __future__ import annotations

import errno
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


_ESTALE_ATTEMPTS = 30
_RESULT_VISIBILITY_GRACE_SECONDS = 120.0


class PersistentVCQueue:
    """Submit work to workers that remain allocated for an entire search run."""

    def __init__(
        self,
        queue_dir: Path | str,
        *,
        repo_root: Path | str,
        result_visibility_grace_seconds: float = _RESULT_VISIBILITY_GRACE_SECONDS,
    ) -> None:
        self.queue_dir = Path(queue_dir).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.result_visibility_grace_seconds = max(
            0.0, float(result_visibility_grace_seconds)
        )
        for name in ("pending", "running", "done", "failed", "workers"):
            _retry(lambda name=name: (self.queue_dir / name).mkdir(parents=True, exist_ok=True))

    def submit(self, tasks: Iterable[dict[str, Any]], *, batch_id: str) -> list[dict[str, Any]]:
        submitted: list[dict[str, Any]] = []
        safe_batch = _safe(batch_id)
        for index, task in enumerate(tasks, start=1):
            payload = dict(task)
            task_id = str(payload.get("task_id") or f"{safe_batch}_{index:05d}_{uuid.uuid4().hex[:12]}")
            payload["task_id"] = task_id
            payload["batch_id"] = safe_batch
            pending_path = self.queue_dir / "pending" / f"task_{task_id}.json"
            _atomic_write_json(pending_path, payload)
            submitted.append(payload)
        return submitted

    def wait(
        self,
        tasks: Iterable[dict[str, Any]],
        *,
        poll_seconds: float = 2.0,
        timeout_seconds: float = 0.0,
    ) -> None:
        pending = {str(task["task_id"]): dict(task) for task in tasks}
        done_without_result_since: dict[str, float] = {}
        deadline = time.monotonic() + float(timeout_seconds) if timeout_seconds > 0 else None
        while pending:
            for task_id, task in list(pending.items()):
                failure_path = self.queue_dir / "failed" / f"task_{task_id}.result.json"
                if _exists(failure_path):
                    failure = _read_json(failure_path)
                    raise RuntimeError(
                        f"Persistent VC task failed task_id={task_id}: "
                        f"{failure.get('error', 'unspecified worker failure')}"
                    )
                # Crystal workers write progress snapshots to result.json at
                # every stage.  File existence therefore does not mean that a
                # task is complete; only the worker-owned done marker is a
                # terminal acknowledgement.
                done_path = self.queue_dir / "done" / f"task_{task_id}.result.json"
                if _exists(done_path):
                    result_path = self.resolve_repo_path(str(task["result_path"]))
                    if not _exists(result_path):
                        # Network filesystems do not guarantee visibility order
                        # across directories.  A worker writes result_path before
                        # its done marker, but the controller may observe the done
                        # marker first.  Allow that result to become visible while
                        # still failing a genuinely missing result after a finite
                        # grace period.
                        missing_since = done_without_result_since.setdefault(
                            task_id, time.monotonic()
                        )
                        if (
                            time.monotonic() - missing_since
                            >= self.result_visibility_grace_seconds
                        ):
                            raise RuntimeError(
                                "Persistent VC task completed without result after "
                                f"{self.result_visibility_grace_seconds:.1f}s visibility grace "
                                f"task_id={task_id}: {result_path}"
                            )
                        continue
                    done_without_result_since.pop(task_id, None)
                    pending.pop(task_id, None)
                    continue
            if not pending:
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for {len(pending)} persistent VC tasks: "
                    f"{list(pending)[:5]}"
                )
            time.sleep(max(0.2, float(poll_seconds)))

    def resolve_repo_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path

    def repo_relative(self, value: Path | str) -> str:
        path = Path(value)
        if not path.is_absolute():
            return path.as_posix()
        return path.resolve().relative_to(self.repo_root).as_posix()


def initialise_queue(queue_dir: Path | str) -> Path:
    path = Path(queue_dir)
    for name in ("pending", "running", "done", "failed", "workers"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)

    _retry(write)


def _read_json(path: Path) -> dict[str, Any]:
    return _retry(lambda: json.loads(path.read_text(encoding="utf-8-sig")))


def _exists(path: Path) -> bool:
    return bool(_retry(path.exists))


def _retry(func):
    delay = 0.1
    for attempt in range(1, _ESTALE_ATTEMPTS + 1):
        try:
            return func()
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt >= _ESTALE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(2.0, delay * 1.5)


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:100]

