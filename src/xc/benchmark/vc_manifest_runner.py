"""Batch BenchmarkRunner-compatible adapter for the VC array scheduler."""

from __future__ import annotations

import errno
import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from .result import BenchmarkResult, SubsetResult
from ..xc_functional import XCFunctional


logger = logging.getLogger(__name__)


_STALE_HANDLE_RETRY_ATTEMPTS = 30
_STALE_HANDLE_RETRY_INITIAL_DELAY_S = 0.25
_STALE_HANDLE_RETRY_MAX_DELAY_S = 2.0
_CONTROLLER_STALE_RETRY_ATTEMPTS = 4
_CONTROLLER_STALE_RETRY_INITIAL_DELAY_S = 15.0
_CONTROLLER_STALE_RETRY_MAX_DELAY_S = 60.0
_MAX_CONTROLLER_LOG_BYTES = 65536
_T = TypeVar("_T")


@dataclass
class VCManifestRunnerConfig:
    output_dir: Path
    vc_prefix: str = ""
    persistent_queue_dir: Path | None = None
    container_repo_root: str = "/code/remote/DFTAgent"
    python_executable: str = "python3"
    conda_sh: str = ""
    conda_env: str = ""
    pre_cmd: str = ""
    gmtkn55_dir: Path = Path("benchmark/GMTKN55")
    subsets: tuple[str, ...] = ("SIE4x4",)
    basis: str = "def2-TZVP"
    scf_max_cycle: int = 200
    scf_conv_tol: float = 1e-8
    n_workers: int = 8
    n_threads: int = 1
    molecules_per_task: int = 8
    aggregate_mode: str = "reaction_mean"
    concurrent_subset_jobs: int = 32
    vc_tasks_per_submit: int = 0
    poll_seconds: float = 10.0
    wait_timeout_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.gmtkn55_dir = Path(self.gmtkn55_dir)
        self.vc_prefix = str(self.vc_prefix or "").strip()
        if self.persistent_queue_dir is not None:
            self.persistent_queue_dir = Path(self.persistent_queue_dir)
        if not self.vc_prefix and self.persistent_queue_dir is None:
            raise ValueError("VC benchmark runner requires vc_prefix or persistent_queue_dir")
        self.concurrent_subset_jobs = max(1, int(self.concurrent_subset_jobs))
        self.molecules_per_task = max(1, int(self.molecules_per_task))


class VCManifestBenchmarkRunner:
    """Expose ``evaluate_many`` while delegating compute to VC array jobs."""

    def __init__(self, config: VCManifestRunnerConfig, *, repo_root: Path) -> None:
        self.config = config
        self.repo_root = Path(repo_root).resolve()
        self._batch_index = 0
        self._batch_index_lock = threading.Lock()

    def evaluate_many(
        self,
        xcs: Sequence[XCFunctional],
        *,
        label_prefix: str = "candidate",
    ) -> list[BenchmarkResult]:
        if not xcs:
            return []
        # Generic search runners may submit several candidates concurrently.
        # Reserve a unique directory without serialising the remote evaluations.
        with self._batch_index_lock:
            self._batch_index += 1
            batch_index = self._batch_index
        safe = _safe(label_prefix)
        batch_dir = self.config.output_dir / f"{batch_index:05d}_{safe}_{int(time.time())}"
        _retry_stale_handle(
            lambda: batch_dir.mkdir(parents=True, exist_ok=True),
            description="create VC benchmark batch directory",
            path=batch_dir,
        )
        manifest = []
        labels = []
        for index, xc in enumerate(xcs, start=1):
            label = f"{safe}_{index:04d}"
            labels.append(label)
            manifest.append(
                {
                    "rank": index,
                    "label": label,
                    "source": "search_task",
                    "source_id": xc.name,
                    "raw_reward": 0.0,
                    "xc_dict": xc.to_dict(),
                }
            )
        manifest_path = batch_dir / "functional_manifest.json"
        _retry_stale_handle(
            lambda: manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            ),
            description="write VC benchmark manifest",
            path=manifest_path,
        )
        output_dir = batch_dir / "evaluation"
        command = [
            sys.executable,
            str(self.repo_root / "benchmark/scripts/evaluate_manifest_vc.py"),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--gmtkn55-dir",
            str(self.config.gmtkn55_dir),
            "--subsets",
            *self.config.subsets,
            "--basis",
            self.config.basis,
            "--scf-max-cycle",
            str(self.config.scf_max_cycle),
            "--scf-conv-tol",
            str(self.config.scf_conv_tol),
            "--n-workers",
            str(self.config.n_workers),
            "--n-threads",
            str(self.config.n_threads),
            "--molecules-per-task",
            str(self.config.molecules_per_task),
            "--aggregate-mode",
            self.config.aggregate_mode,
            "--keep-manifest-order",
            "--python-executable",
            self.config.python_executable,
            "--container-repo-root",
            self.config.container_repo_root,
            "--conda-sh",
            self.config.conda_sh,
            "--conda-env",
            self.config.conda_env,
            "--pre-cmd",
            self.config.pre_cmd,
            "--concurrent-subset-jobs",
            str(self.config.concurrent_subset_jobs),
            "--vc-tasks-per-submit",
            str(self.config.vc_tasks_per_submit),
            "--poll-seconds",
            str(self.config.poll_seconds),
            "--wait-timeout-seconds",
            str(self.config.wait_timeout_seconds),
            "--clean-output",
            "--submit",
        ]
        if self.config.persistent_queue_dir is not None:
            command.extend(["--persistent-queue-dir", str(self.config.persistent_queue_dir)])
        else:
            command.extend(["--vc-prefix", self.config.vc_prefix])
        logger.info(
            "VC_BENCHMARK_BATCH_START candidates=%d subsets=%s output=%s persistent_queue=%s",
            len(xcs),
            ",".join(self.config.subsets),
            output_dir,
            self.config.persistent_queue_dir or "",
        )
        controller_command = list(command)
        controller_outputs: list[str] = []
        delay = _CONTROLLER_STALE_RETRY_INITIAL_DELAY_S
        for attempt in range(1, _CONTROLLER_STALE_RETRY_ATTEMPTS + 1):
            completed = subprocess.run(
                controller_command,
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            controller_outputs.append(
                f"=== controller attempt {attempt}/{_CONTROLLER_STALE_RETRY_ATTEMPTS} "
                f"rc={completed.returncode} ===\n{completed.stdout or ''}\n{completed.stderr or ''}"
            )
            failure_text = f"{completed.stdout or ''}\n{completed.stderr or ''}"
            if completed.returncode == 0 or "stale file handle" not in failure_text.lower():
                break
            if attempt >= _CONTROLLER_STALE_RETRY_ATTEMPTS:
                break
            logger.warning(
                "VC_CONTROLLER_ESTALE output=%s attempt=%d/%d retry_in_s=%.1f; "
                "completed shards will be reused",
                output_dir,
                attempt,
                _CONTROLLER_STALE_RETRY_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, _CONTROLLER_STALE_RETRY_MAX_DELAY_S)
            # The first attempt deliberately starts clean.  Retried controllers
            # must preserve completed shard files and submit only missing work.
            controller_command = [item for item in controller_command if item != "--clean-output"]
        _write_diagnostic_log(
            batch_dir / "controller.stdout.log",
            "\n\n".join(controller_outputs),
        )
        _write_diagnostic_log(
            batch_dir / "controller.stderr.log",
            completed.stderr or "",
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "")[-4000:]
            raise RuntimeError(f"VC benchmark batch failed rc={completed.returncode}: {detail}")

        result_path = output_dir / "vc_manifest_benchmark_results.json"
        result_text = _retry_stale_handle(
            lambda: result_path.read_text(encoding="utf-8-sig"),
            description="read VC benchmark result",
            path=result_path,
        )
        payloads = json.loads(result_text)
        by_label = {
            str((item.get("candidate") or {}).get("label")): item.get("benchmark_result") or {}
            for item in payloads
        }
        missing = [label for label in labels if label not in by_label]
        if missing:
            raise RuntimeError(f"VC benchmark output is missing candidates: {missing}")
        results = [_benchmark_result_from_dict(by_label[label]) for label in labels]
        logger.info("VC_BENCHMARK_BATCH_DONE candidates=%d output=%s", len(results), output_dir)
        return results


def _retry_stale_handle(
    func: Callable[[], _T],
    *,
    description: str,
    path: Path | None = None,
) -> _T:
    """Retry a single shared-filesystem operation after transient ESTALE.

    VC jobs write to network storage.  A completed remote evaluation must not
    be resubmitted merely because the controller briefly sees a stale file
    handle while persisting or reading its small bookkeeping files.
    """
    delay = _STALE_HANDLE_RETRY_INITIAL_DELAY_S
    for attempt in range(1, _STALE_HANDLE_RETRY_ATTEMPTS + 1):
        try:
            return func()
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt >= _STALE_HANDLE_RETRY_ATTEMPTS:
                raise
            logger.warning(
                "VC_FILESYSTEM_ESTALE operation=%s path=%s attempt=%d/%d retry_in_s=%.2f",
                description,
                path or "",
                attempt,
                _STALE_HANDLE_RETRY_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, _STALE_HANDLE_RETRY_MAX_DELAY_S)


def _write_diagnostic_log(path: Path, text: str) -> None:
    """Persist controller output without turning a log-only ESTALE into failure."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) > _MAX_CONTROLLER_LOG_BYTES:
        marker = (
            f"<controller log truncated; original_bytes={len(encoded)}>\n"
        ).encode()
        tail = encoded[-max(0, _MAX_CONTROLLER_LOG_BYTES - len(marker)):]
        text = (marker + tail).decode("utf-8", errors="replace")
    try:
        _retry_stale_handle(
            lambda: path.write_text(text, encoding="utf-8"),
            description="write VC controller diagnostic log",
            path=path,
        )
    except OSError as exc:
        if exc.errno != errno.ESTALE:
            raise
        logger.error(
            "VC_DIAGNOSTIC_LOG_SKIPPED stale file handle persisted after retries path=%s",
            path,
        )


def _benchmark_result_from_dict(payload: dict) -> BenchmarkResult:
    result = BenchmarkResult(
        xc_name=str(payload.get("xc_name", "")),
        subsets={
            str(name): SubsetResult.from_dict(dict(value))
            for name, value in dict(payload.get("subsets") or {}).items()
        },
        overall_mae=float(payload.get("overall_mae") or 0.0),
        overall_rms=float(payload.get("overall_rms") or 0.0),
        total_walltime_s=float(payload.get("total_walltime_s") or 0.0),
        total_failures=int(payload.get("total_failures") or 0),
        timestamp=str(payload.get("timestamp") or ""),
    )
    return result


def _safe(value: str) -> str:
    text = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))
    return "_".join(text.split("_"))[:80] or "batch"
