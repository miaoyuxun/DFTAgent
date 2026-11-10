"""Slurm+Docker batch evaluator for crystal band-gap guard calculations."""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..slurm_env import clean_slurm_submit_env
from ..xc_functional import XCFunctional

logger = logging.getLogger(__name__)


@dataclass
class CrystalGuardSlurmEvaluatorConfig:
    output_dir: Path
    image: str
    container_repo_root: str = "/workspace/DFTAgent"
    partition: str = "compute"
    cpus_per_candidate: int = 16
    max_concurrent: int = 32
    memory: str = "120G"
    hard_timeout_s: int = 7200
    slurm_time: str = "02:10:00"
    slurm_nodelist: str = ""
    band_mesh: tuple[int, int, int] = (2, 2, 2)
    materials: tuple[str, ...] = ("ZnS", "MgO")
    adaptive_straggler_cutoff: bool = True
    straggler_cutoff_start_fraction: float = 0.75
    straggler_cutoff_min_multiplier: float = 1.5
    straggler_cutoff_mad_multiplier: float = 3.0
    straggler_cutoff_max_fraction: float = 0.05
    poll_s: float = 30.0
    docker_extra_args: list[str] = field(default_factory=list)
    extra_sbatch: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.cpus_per_candidate = max(1, int(self.cpus_per_candidate))
        self.max_concurrent = max(1, int(self.max_concurrent))
        self.hard_timeout_s = max(60, int(self.hard_timeout_s))
        self.poll_s = max(5.0, float(self.poll_s))
        self.slurm_nodelist = str(self.slurm_nodelist or "").strip()
        self.extra_sbatch = [str(item).strip() for item in self.extra_sbatch if str(item).strip()]
        if not self.image:
            raise ValueError("Crystal guard evaluator requires a Docker image")
        if not self.memory:
            raise ValueError("Crystal guard evaluator requires explicit per-task memory")


class CrystalGuardSlurmDockerEvaluator:
    """Evaluate several semilocal functionals on MgO/ZnS gap guards."""

    def __init__(self, config: CrystalGuardSlurmEvaluatorConfig, *, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
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
        batch_dir, tasks = self.prepare_batch(xcs, labels=labels, stage=stage)
        script = batch_dir / "run_crystal_guard_array.sbatch"
        logger.info(
            "CRYSTAL_GUARD_SUBMIT stage=%s candidates=%d materials=%s tasks=%d concurrency=%d cpus_each=%d mem_each=%s timeout_s=%d script=%s",
            stage,
            len(xcs),
            ",".join(self.config.materials),
            len(tasks),
            min(self.config.max_concurrent, len(tasks)),
            self.config.cpus_per_candidate,
            self.config.memory,
            self.config.hard_timeout_s,
            script,
        )
        job_id = self._submit(script)
        self._monitor(batch_dir=batch_dir, tasks=tasks, job_id=job_id, candidate_count=len(xcs))
        return self.collect_results(tasks, candidate_count=len(xcs))

    def prepare_batch(
        self,
        xcs: Sequence[XCFunctional],
        *,
        labels: Sequence[str],
        stage: str,
    ) -> tuple[Path, list[dict[str, Any]]]:
        self._batch_index += 1
        safe_stage = _safe(stage)
        batch_dir = self.config.output_dir / f"{self._batch_index:04d}_{safe_stage}_{int(time.time())}"
        task_dir = batch_dir / "tasks"
        result_dir = batch_dir / "results"
        log_dir = batch_dir / "slurm_logs"
        for path in (task_dir, result_dir, log_dir):
            path.mkdir(parents=True, exist_ok=True)
        records = []
        for index, (xc, label) in enumerate(zip(xcs, labels), start=1):
            records.append(
                {
                    "id": f"candidate_{index:06d}",
                    "label": str(label),
                    "source": "basin_agent",
                    "source_id": xc.name,
                    "xc_dict": xc.to_dict(),
                }
            )
        manifest_path = batch_dir / "functional_manifest.json"
        manifest_path.write_text(
            json.dumps({"records": records}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tasks: list[dict[str, Any]] = []
        task_index = 0
        for candidate_index, label in enumerate(labels, start=1):
            functional_id = f"candidate_{candidate_index:06d}"
            for material in self.config.materials:
                task_index += 1
                material_dir = result_dir / f"candidate_{candidate_index:06d}" / str(material)
                task = {
                    "task_index": task_index,
                    "candidate_index": candidate_index,
                    "label": str(label),
                    "functional_id": functional_id,
                    "material": str(material),
                    "output_dir": str(material_dir.relative_to(self.repo_root)),
                    "result_path": str((material_dir / "result.json").relative_to(self.repo_root)),
                    "functional_manifest": str(manifest_path.relative_to(self.repo_root)),
                    "band_mesh": list(self.config.band_mesh),
                    "cpus": int(self.config.cpus_per_candidate),
                    "hard_timeout_s": self.config.hard_timeout_s,
                }
                task_path = task_dir / f"task_{task_index - 1}.json"
                task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
                tasks.append(task)
        (batch_dir / "batch_manifest.json").write_text(
            json.dumps({"stage": stage, "tasks": tasks}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (batch_dir / "run_crystal_guard_array.sbatch").write_text(
            self._render_script(batch_dir=batch_dir, task_count=len(tasks)),
            encoding="utf-8",
            newline="\n",
        )
        return batch_dir, tasks

    def collect_results(self, tasks: list[dict[str, Any]], *, candidate_count: int) -> list[dict[str, Any]]:
        by_candidate: list[dict[str, Any]] = [
            {"ok": True, "materials": {}, "cutoff": False} for _ in range(candidate_count)
        ]
        for task in tasks:
            candidate_index = int(task["candidate_index"]) - 1
            material = str(task["material"])
            result_path = self.repo_root / task["result_path"]
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
                except Exception as exc:
                    payload = {"ok": False, "stage": "failed", "error": f"unreadable_result:{exc}"}
            else:
                payload = {"ok": False, "stage": "failed", "error": "missing_result_after_slurm_completion"}
            by_candidate[candidate_index]["materials"][material] = payload
            if not payload.get("ok"):
                by_candidate[candidate_index]["ok"] = False
            if payload.get("timed_out") or payload.get("runtime_cutoff"):
                by_candidate[candidate_index]["cutoff"] = True
        return by_candidate

    def _render_script(self, *, batch_dir: Path, task_count: int) -> str:
        relative_batch = batch_dir.relative_to(self.repo_root).as_posix()
        maximum = task_count - 1
        concurrency = min(self.config.max_concurrent, task_count)
        image = shlex.quote(self.config.image)
        repo = shlex.quote(self.config.container_repo_root)
        docker_extra = " ".join(shlex.quote(str(item)) for item in self.config.docker_extra_args)
        nodelist_line = f"#SBATCH --nodelist={self.config.slurm_nodelist}\n" if self.config.slurm_nodelist else ""
        extra_sbatch_lines = "".join(f"#SBATCH {item}\n" for item in self.config.extra_sbatch)
        return f"""#!/bin/bash
#SBATCH --partition={self.config.partition}
#SBATCH --job-name=crystal-guard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={self.config.cpus_per_candidate}
#SBATCH --mem={self.config.memory}
#SBATCH --time={self.config.slurm_time}
{nodelist_line}{extra_sbatch_lines}#SBATCH --array=0-{maximum}%{concurrency}
#SBATCH --output={relative_batch}/slurm_logs/task_%A_%a.out
#SBATCH --error={relative_batch}/slurm_logs/task_%A_%a.err

set -u
task_rel={relative_batch}/tasks/task_${{SLURM_ARRAY_TASK_ID}}.json
task_in={self.config.container_repo_root}/$task_rel

sudo -n slurm-docker-run --pull missing --network host {docker_extra} \\
  --workdir {repo} {image} \\
  bash -lc "export PYTHONPATH={self.config.container_repo_root}/src:\\${{PYTHONPATH:-}}; \\
    export OMP_NUM_THREADS={self.config.cpus_per_candidate}; \\
    export MKL_NUM_THREADS={self.config.cpus_per_candidate}; \\
    export OPENBLAS_NUM_THREADS={self.config.cpus_per_candidate}; \\
    timeout --signal=TERM --kill-after=60s {self.config.hard_timeout_s}s \\
      python3 case_study/crystal/scripts/python/crystal_guard_task_worker.py --task-json '$task_in'"
rc=$?

if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] || [ "$rc" -eq 143 ]; then
  sudo -n slurm-docker-run --pull missing --network host {docker_extra} \\
    --workdir {repo} {image} \\
    bash -lc "python3 case_study/crystal/scripts/python/crystal_guard_task_worker.py --task-json '$task_in' --mark-timeout"
fi

exit 0
"""

    def _submit(self, script: Path) -> str:
        completed = subprocess.run(
            ["sbatch", "--parsable", str(script)],
            cwd=self.repo_root,
            env=clean_slurm_submit_env(),
            text=True,
            capture_output=True,
            check=False,
        )
        (script.parent / "submit.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        (script.parent / "submit.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Crystal guard sbatch failed rc={completed.returncode}: "
                f"{(completed.stderr or completed.stdout)[-1000:]}"
            )
        return (completed.stdout or "").strip().splitlines()[-1].split(";", 1)[0]

    def _monitor(self, *, batch_dir: Path, tasks: list[dict[str, Any]], job_id: str, candidate_count: int) -> None:
        started = time.time()
        already_cutoff: set[int] = set()
        while True:
            state = self._runtime_state(tasks, started=started, candidate_count=candidate_count)
            logger.info(
                "CRYSTAL_GUARD_MONITOR job=%s complete=%d/%d threshold=%s cutoff=%s",
                job_id,
                state["complete"],
                state["total"],
                state.get("threshold_s", "-"),
                sorted(already_cutoff),
            )
            if state["complete"] >= state["total"]:
                subprocess.run(["scancel", job_id], cwd=self.repo_root, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            if self.config.adaptive_straggler_cutoff:
                decision = self._adaptive_decision(state, already_cutoff=already_cutoff)
                threshold = decision.get("threshold_s")
                if threshold is not None:
                    state["threshold_s"] = round(float(threshold), 1)
                if decision.get("cutoff_candidates"):
                    self._mark_cutoff_candidates(
                        batch_dir=batch_dir,
                        tasks=tasks,
                        decision=decision,
                        job_id=job_id,
                    )
                    for item in decision["cutoff_candidates"]:
                        already_cutoff.add(int(item["candidate_index"]))
            if not _slurm_job_is_active(job_id):
                # Give shared storage a short grace period.
                time.sleep(min(10.0, self.config.poll_s))
                state = self._runtime_state(tasks, started=started, candidate_count=candidate_count)
                if state["complete"] >= state["total"]:
                    return
                self._mark_all_unfinished(tasks, reason="slurm_job_ended_before_results")
                return
            time.sleep(self.config.poll_s)

    def _runtime_state(self, tasks: list[dict[str, Any]], *, started: float, candidate_count: int) -> dict[str, Any]:
        by_candidate: dict[int, list[dict[str, Any]]] = {index: [] for index in range(1, candidate_count + 1)}
        now = time.time()
        for task in tasks:
            candidate_index = int(task["candidate_index"])
            result_path = self.repo_root / task["result_path"]
            completed = False
            walltime = None
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
                    completed = _crystal_payload_is_terminal(payload)
                    walltime = _finite_or_none(payload.get("walltime_s"))
                except Exception:
                    completed = False
                    walltime = None
            by_candidate[candidate_index].append(
                {
                    "task": task,
                    "complete": completed,
                    "walltime_s": walltime,
                }
            )
        candidates = []
        for candidate_index, items in by_candidate.items():
            complete = all(item["complete"] for item in items)
            durations = [item["walltime_s"] for item in items if item["walltime_s"] is not None]
            duration = max(durations) if complete and durations else None
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "complete": complete,
                    "duration_s": duration,
                    "elapsed_s": None if complete else now - started,
                }
            )
        return {
            "total": candidate_count,
            "complete": sum(1 for item in candidates if item["complete"]),
            "candidates": candidates,
        }

    def _adaptive_decision(self, state: dict[str, Any], *, already_cutoff: set[int]) -> dict[str, Any]:
        total = int(state["total"])
        start_keep = max(1, math.ceil(total * min(1.0, max(0.0, self.config.straggler_cutoff_start_fraction))))
        complete = int(state["complete"])
        if complete < start_keep:
            return {"reason": "waiting_for_enough_completed_candidates", "cutoff_candidates": []}
        durations = [
            float(item["duration_s"])
            for item in state["candidates"]
            if item.get("complete") and item.get("duration_s") is not None and float(item["duration_s"]) > 0
        ]
        if len(durations) < max(2, start_keep // 2):
            return {"reason": "insufficient_completed_duration_samples", "cutoff_candidates": []}
        median_s = _median(durations)
        mad_s = _median([abs(value - median_s) for value in durations])
        threshold_s = max(
            median_s * max(1.0, float(self.config.straggler_cutoff_min_multiplier)),
            median_s + max(0.0, float(self.config.straggler_cutoff_mad_multiplier)) * mad_s,
        )
        max_cutoff = max(0, math.floor(total * min(1.0, max(0.0, self.config.straggler_cutoff_max_fraction))))
        remaining = max(0, max_cutoff - len(already_cutoff))
        slow = []
        if remaining > 0:
            for item in state["candidates"]:
                candidate_index = int(item["candidate_index"])
                if candidate_index in already_cutoff or item.get("complete"):
                    continue
                elapsed = item.get("elapsed_s")
                if elapsed is not None and float(elapsed) > threshold_s:
                    slow.append({**item, "threshold_s": threshold_s})
        slow.sort(key=lambda item: float(item.get("elapsed_s") or 0), reverse=True)
        return {
            "reason": "adaptive_straggler_cutoff",
            "complete_candidates": complete,
            "total_candidates": total,
            "median_s": median_s,
            "mad_s": mad_s,
            "threshold_s": threshold_s,
            "max_cutoff": max_cutoff,
            "already_cutoff": sorted(already_cutoff),
            "cutoff_candidates": slow[:remaining],
        }

    def _mark_cutoff_candidates(
        self,
        *,
        batch_dir: Path,
        tasks: list[dict[str, Any]],
        decision: dict[str, Any],
        job_id: str,
    ) -> None:
        cutoff = {int(item["candidate_index"]) for item in decision.get("cutoff_candidates", [])}
        now = time.time()
        marked = []
        for task in tasks:
            if int(task["candidate_index"]) not in cutoff:
                continue
            result_path = self.repo_root / task["result_path"]
            if result_path.exists():
                continue
            _write_crystal_timeout_payload(result_path, task, reason="adaptive_straggler_cutoff", now=now)
            array_index = int(task["task_index"]) - 1
            subprocess.run(
                ["scancel", f"{job_id}_{array_index}"],
                cwd=self.repo_root,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            marked.append({"candidate_index": task["candidate_index"], "task_index": task["task_index"]})
        report = batch_dir / "candidate_tail_cutoff.json"
        report.write_text(
            json.dumps(
                {
                    "reason": "adaptive_straggler_cutoff",
                    "created_at": now,
                    "decision": decision,
                    "marked": marked,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _mark_all_unfinished(self, tasks: list[dict[str, Any]], *, reason: str) -> None:
        now = time.time()
        for task in tasks:
            result_path = self.repo_root / task["result_path"]
            if not result_path.exists():
                _write_crystal_timeout_payload(result_path, task, reason=reason, now=now)


def _write_crystal_timeout_payload(path: Path, task: dict[str, Any], *, reason: str, now: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": False,
        "stage": "timeout",
        "timed_out": True,
        "runtime_cutoff": True,
        "runtime_cutoff_reason": reason,
        "material": task.get("material"),
        "functional_id": task.get("functional_id"),
        "functional_label": task.get("label"),
        "error": reason,
        "walltime_s": float(task.get("hard_timeout_s") or 0.0),
        "timeout_marked_at_epoch_s": now,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _crystal_payload_is_terminal(payload: dict[str, Any]) -> bool:
    if payload.get("timed_out") or payload.get("runtime_cutoff"):
        return True
    return str(payload.get("stage") or "") in {"complete", "failed", "timeout"}


def _slurm_job_is_active(job_id: str) -> bool:
    completed = subprocess.run(
        ["squeue", "-j", str(job_id), "-h"],
        text=True,
        capture_output=True,
        check=False,
    )
    return bool((completed.stdout or "").strip())


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _median(values: Sequence[float]) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])
