"""VC evaluator for independently submitted periodic crystal guards."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..xc_functional import XCFunctional
from .crystal_slurm import CrystalGuardSlurmDockerEvaluator


logger = logging.getLogger(__name__)


@dataclass
class CrystalGuardVCEvaluatorConfig:
    output_dir: Path
    vc_prefix: str
    container_repo_root: str = "/code/remote/DFTAgent"
    python_executable: str = "python3"
    conda_sh: str = ""
    conda_env: str = ""
    pre_cmd: str = ""
    cpus_per_candidate: int = 8
    max_concurrent_submissions: int = 32
    hard_timeout_s: int = 7200
    band_mesh: tuple[int, int, int] = (2, 2, 2)
    materials: tuple[str, ...] = ("ZnS",)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.vc_prefix = str(self.vc_prefix or "").strip()
        self.cpus_per_candidate = max(1, int(self.cpus_per_candidate))
        self.max_concurrent_submissions = max(1, int(self.max_concurrent_submissions))
        self.hard_timeout_s = max(60, int(self.hard_timeout_s))
        if not self.vc_prefix:
            raise ValueError("Crystal VC evaluator requires a non-empty vc_prefix")


class CrystalGuardVCEvaluator:
    """Prepare shared task payloads and submit each candidate independently."""

    def __init__(self, config: CrystalGuardVCEvaluatorConfig, *, repo_root: Path) -> None:
        self.config = config
        self.repo_root = Path(repo_root).resolve()
        self._batch_index = 0

    def __call__(
        self,
        xcs: Sequence[XCFunctional],
        *,
        labels: Sequence[str],
        stage: str,
    ) -> list[dict[str, Any]]:
        if len(xcs) != len(labels):
            raise ValueError("Crystal guard labels must match candidate count")
        if not xcs:
            return []
        self._batch_index += 1
        helper = CrystalGuardSlurmDockerEvaluator.__new__(CrystalGuardSlurmDockerEvaluator)
        helper.repo_root = self.repo_root
        helper._batch_index = self._batch_index - 1
        helper.config = _PreparationConfig(self.config)
        batch_dir, tasks = helper.prepare_batch(xcs, labels=labels, stage=stage)
        self._batch_index = helper._batch_index

        task_dir = batch_dir / "tasks"
        for one_based, task in enumerate(tasks, start=1):
            source = task_dir / f"task_{int(task['task_index']) - 1}.json"
            target = task_dir / f"vc_task_{one_based}.json"
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        log_dir = batch_dir / "vc_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        relative_batch = batch_dir.relative_to(self.repo_root).as_posix()
        setup = []
        if self.config.conda_sh:
            setup.append(f"source {shlex.quote(self.config.conda_sh)}")
        if self.config.conda_env:
            setup.append(f"conda activate {shlex.quote(self.config.conda_env)}")
        if self.config.pre_cmd:
            setup.append(self.config.pre_cmd)
        common_setup = setup + [
            f"cd {shlex.quote(self.config.container_repo_root)}",
            "export PYTHONPATH=$PWD/src:${PYTHONPATH:-}",
            f"export OMP_NUM_THREADS={self.config.cpus_per_candidate}",
            f"export MKL_NUM_THREADS={self.config.cpus_per_candidate}",
            f"export OPENBLAS_NUM_THREADS={self.config.cpus_per_candidate}",
        ]
        submissions: list[tuple[int, Path, Path, Path, dict[str, Any]]] = []
        for one_based in range(1, len(tasks) + 1):
            task_in = (
                f"{self.config.container_repo_root}/{relative_batch}/tasks/"
                f"vc_task_{one_based}.json"
            )
            worker = (
                f"{shlex.quote(self.config.python_executable)} "
                "case_study/crystal/scripts/python/crystal_guard_task_worker.py "
                f"--task-json {shlex.quote(task_in)}"
            )
            container_command = " && ".join(
                common_setup
                + [
                    (
                        f"rc=0; timeout --signal=TERM --kill-after=60s "
                        f"{self.config.hard_timeout_s}s {worker} || rc=$?; "
                        "if [ \"$rc\" -eq 124 ] || [ \"$rc\" -eq 137 ] || "
                        "[ \"$rc\" -eq 143 ]; then "
                        f"{worker} --mark-timeout; exit 0; fi; exit \"$rc\""
                    )
                ]
            )
            script = batch_dir / f"run_crystal_guard_vc_task_{one_based:04d}.sh"
            task_log = log_dir / f"crystal_guard_task_{one_based:04d}.JOB.log"
            submit_log = log_dir / f"submit_task_{one_based:04d}.log"
            script.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"cd {shlex.quote(str(self.repo_root))}\n"
                f"{self.config.vc_prefix} --sync --debug "
                f"--job crystal_guard_{self._batch_index:05d}_{one_based:04d} "
                f"JOB=1:1 {shlex.quote(str(task_log))} "
                f"--cmd {shlex.quote(container_command)} "
                f"> {shlex.quote(str(submit_log))} 2>&1\n",
                encoding="utf-8",
                newline="\n",
            )
            submissions.append((one_based, script, submit_log, task_log, tasks[one_based - 1]))

        logger.info(
            "CRYSTAL_GUARD_VC_SUBMIT stage=%s candidates=%d tasks=%d mode=independent "
            "tasks_per_submit=1 max_concurrent_submissions=%d batch_dir=%s",
            stage,
            len(xcs),
            len(tasks),
            min(self.config.max_concurrent_submissions, len(submissions)),
            batch_dir,
        )
        failures: list[str] = []
        with ThreadPoolExecutor(
            max_workers=min(self.config.max_concurrent_submissions, len(submissions))
        ) as executor:
            futures = {
                executor.submit(
                    subprocess.run,
                    ["bash", str(script)],
                    cwd=self.repo_root,
                    text=True,
                    check=False,
                ): (one_based, submit_log, task_log, task)
                for one_based, script, submit_log, task_log, task in submissions
            }
            for future in as_completed(futures):
                one_based, submit_log, task_log, task = futures[future]
                try:
                    completed = future.result()
                except Exception as exc:  # pragma: no cover - defensive subprocess boundary
                    failures.append(f"task={one_based} launch_error={exc}")
                    continue
                if completed.returncode != 0:
                    result_path = self.repo_root / str(task["result_path"])
                    payload = _read_json_dict(result_path)
                    if payload is not None and _is_terminal_payload(payload):
                        logger.warning(
                            "CRYSTAL_GUARD_VC_CANDIDATE_FAILED task=%d candidate=%s material=%s "
                            "stage=%s error=%s result=%s",
                            one_based,
                            task.get("candidate_index"),
                            task.get("material"),
                            payload.get("failed_stage") or payload.get("stage"),
                            _brief_error(payload.get("error")),
                            result_path,
                        )
                        continue
                    task_detail = _task_log_tail(task_log)
                    submit_detail = _file_tail(submit_log, limit=1000)
                    failures.append(
                        f"task={one_based} rc={completed.returncode} "
                        f"task_log_tail={task_detail or '<missing>'} "
                        f"submit_log_tail={submit_detail or '<missing>'}"
                    )
        if failures:
            failure_report = batch_dir / "vc_infrastructure_failures.json"
            failure_report.write_text(
                json.dumps({"failures": failures}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"Crystal guard VC infrastructure failures: {len(failures)}/{len(submissions)}; "
                f"report={failure_report}; samples=" + " | ".join(failures[:3])
            )
        results = helper.collect_results(tasks, candidate_count=len(xcs))
        failed_results = [item for item in results if not item.get("ok")]
        logger.info(
            "CRYSTAL_GUARD_VC_DONE stage=%s candidates=%d ok=%d failed=%d batch_dir=%s",
            stage,
            len(results),
            len(results) - len(failed_results),
            len(failed_results),
            batch_dir,
        )
        return results


class _PreparationConfig:
    """Minimal config surface consumed by the shared task preparation code."""

    def __init__(self, source: CrystalGuardVCEvaluatorConfig) -> None:
        self.output_dir = source.output_dir
        self.band_mesh = source.band_mesh
        self.materials = source.materials
        self.hard_timeout_s = source.hard_timeout_s
        # ``prepare_batch`` also renders an unused Slurm script. Supplying a
        # complete inert surface lets VC and Slurm share one task schema.
        self.max_concurrent = 1
        self.image = "unused-by-vc"
        self.container_repo_root = source.container_repo_root
        self.docker_extra_args = []
        self.slurm_nodelist = ""
        self.extra_sbatch = []
        self.partition = "unused"
        self.cpus_per_candidate = source.cpus_per_candidate
        self.memory = "1G"
        self.slurm_time = "00:01:00"


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_terminal_payload(payload: dict[str, Any]) -> bool:
    return bool(payload.get("timed_out") or payload.get("runtime_cutoff")) or str(
        payload.get("stage") or ""
    ) in {"complete", "failed", "timeout"}


def _brief_error(value: Any) -> str:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    return lines[-1][:500] if lines else "unspecified candidate failure"


def _file_tail(path: Path, *, limit: int) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()


def _task_log_tail(template: Path) -> str:
    matches = sorted(
        template.parent.glob(template.name.replace(".JOB.", ".*.")),
        key=lambda path: path.stat().st_mtime,
    )
    if matches:
        return _file_tail(matches[-1], limit=2000)
    return _file_tail(template, limit=2000)
