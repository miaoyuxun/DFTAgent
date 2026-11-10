"""Slurm+Docker batch evaluator for self-consistent ZnO candidates."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..slurm_env import clean_slurm_submit_env
from ..xc_functional import XCFunctional

logger = logging.getLogger(__name__)


@dataclass
class ZnOSlurmEvaluatorConfig:
    output_dir: Path
    image: str
    container_repo_root: str = "/workspace/DFTAgent"
    partition: str = "compute"
    cpus_per_candidate: int = 16
    max_concurrent: int = 32
    memory: str = "120G"
    max_memory_mb: int = 110000
    hard_timeout_s: int = 7200
    slurm_time: str = "02:10:00"
    basis: str = "gth-dzvp-molopt-sr"
    pseudo: str = "gth-pbe"
    kmesh: tuple[int, int, int] = (2, 2, 2)
    precision: float = 1e-5
    smearing_sigma: float = 0.005
    precondition_conv_tol: float = 1e-6
    final_conv_tol: float = 1e-7
    precondition_max_cycle: int = 60
    final_max_cycle: int = 100
    precondition_damping: float = 0.2
    precondition_level_shift: float = 0.2
    verbose: int = 4
    cif_path: str = "case_study/crystal/structure/ZnO/ZnO_PBE_optimized.cif"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.cpus_per_candidate = max(1, int(self.cpus_per_candidate))
        self.max_concurrent = max(1, int(self.max_concurrent))
        self.hard_timeout_s = max(60, int(self.hard_timeout_s))
        self.max_memory_mb = max(1000, int(self.max_memory_mb))
        if not self.image:
            raise ValueError("ZnO Slurm evaluator requires a Docker image")
        if not self.memory:
            raise ValueError("ZnO Slurm evaluator requires explicit per-candidate memory")


class ZnOSlurmDockerEvaluator:
    """Prepare one Slurm array element per functional and collect JSON results."""

    def __init__(self, config: ZnOSlurmEvaluatorConfig, *, repo_root: Path | None = None) -> None:
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
            raise ValueError("ZnO evaluator labels must match candidate count")
        if not xcs:
            return []
        batch_dir, tasks = self.prepare_batch(xcs, labels=labels, stage=stage)
        script = batch_dir / "run_zno_array.sbatch"
        logger.info(
            "ZNO_BATCH_SUBMIT stage=%s candidates=%d concurrency=%d cpus_each=%d mem_each=%s timeout_s=%d script=%s",
            stage,
            len(tasks),
            min(self.config.max_concurrent, len(tasks)),
            self.config.cpus_per_candidate,
            self.config.memory,
            self.config.hard_timeout_s,
            script,
        )
        completed = subprocess.run(
            ["sbatch", "--wait", str(script)],
            cwd=self.repo_root,
            env=clean_slurm_submit_env(),
            check=False,
        )
        if completed.returncode != 0:
            logger.warning("ZnO Slurm array exited rc=%d; collecting all recoverable payloads", completed.returncode)
        results = []
        for task in tasks:
            output = self.repo_root / task["output_path"]
            if output.exists():
                try:
                    results.append(json.loads(output.read_text(encoding="utf-8")))
                    continue
                except (OSError, json.JSONDecodeError) as exc:
                    error = f"unreadable_result: {exc}"
            else:
                error = "missing_result_after_slurm_completion"
            results.append(
                {
                    "ok": False,
                    "stage": "failed",
                    "label": task["label"],
                    "source_id": task["source_id"],
                    "error": error,
                }
            )
        return results

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
        tasks: list[dict[str, Any]] = []
        for index, (xc, label) in enumerate(zip(xcs, labels)):
            label_slug = f"{index:03d}_{_safe(label)}"
            output_path = result_dir / f"{label_slug}.json"
            task = {
                "label": str(label),
                "safe_label": label_slug,
                "source_id": xc.name,
                "xc_dict": xc.to_dict(),
                "cif_path": self.config.cif_path,
                "output_dir": str(result_dir.relative_to(self.repo_root)),
                "output_path": str(output_path.relative_to(self.repo_root)),
                "basis": self.config.basis,
                "pseudo": self.config.pseudo,
                "kmesh": list(self.config.kmesh),
                "threads_per_functional": self.config.cpus_per_candidate,
                "max_memory_mb": self.config.max_memory_mb,
                "precision": self.config.precision,
                "smearing_sigma": self.config.smearing_sigma,
                "precondition_conv_tol": self.config.precondition_conv_tol,
                "final_conv_tol": self.config.final_conv_tol,
                "precondition_max_cycle": self.config.precondition_max_cycle,
                "final_max_cycle": self.config.final_max_cycle,
                "precondition_damping": self.config.precondition_damping,
                "precondition_level_shift": self.config.precondition_level_shift,
                "verbose": self.config.verbose,
                "hard_timeout_s": self.config.hard_timeout_s,
            }
            task_path = task_dir / f"task_{index}.json"
            task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
            tasks.append(task)
        (batch_dir / "batch_manifest.json").write_text(
            json.dumps({"stage": stage, "tasks": tasks}, indent=2) + "\n",
            encoding="utf-8",
        )
        script = self._render_script(batch_dir=batch_dir, task_count=len(tasks))
        script_path = batch_dir / "run_zno_array.sbatch"
        script_path.write_text(script, encoding="utf-8", newline="\n")
        return batch_dir, tasks

    def _render_script(self, *, batch_dir: Path, task_count: int) -> str:
        relative_batch = batch_dir.relative_to(self.repo_root).as_posix()
        maximum = task_count - 1
        concurrency = min(self.config.max_concurrent, task_count)
        image = shlex.quote(self.config.image)
        repo = shlex.quote(self.config.container_repo_root)
        return f"""#!/bin/bash
#SBATCH --partition={self.config.partition}
#SBATCH --job-name=zno-basin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={self.config.cpus_per_candidate}
#SBATCH --mem={self.config.memory}
#SBATCH --time={self.config.slurm_time}
#SBATCH --array=0-{maximum}%{concurrency}
#SBATCH --output={relative_batch}/slurm_logs/task_%A_%a.out
#SBATCH --error={relative_batch}/slurm_logs/task_%A_%a.err

set -u
task_rel={relative_batch}/tasks/task_${{SLURM_ARRAY_TASK_ID}}.json
task_in={self.config.container_repo_root}/$task_rel

sudo -n slurm-docker-run --pull missing --network host \\
  --workdir {repo} {image} \\
  bash -lc "export PYTHONPATH={self.config.container_repo_root}/src:\${{PYTHONPATH:-}}; \\
    export OMP_NUM_THREADS={self.config.cpus_per_candidate}; \\
    export MKL_NUM_THREADS={self.config.cpus_per_candidate}; \\
    timeout --signal=TERM --kill-after=60s {self.config.hard_timeout_s}s \\
      python3 case_study/python/zno_task_worker.py --task-json '$task_in'"
rc=$?

if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] || [ "$rc" -eq 143 ]; then
  sudo -n slurm-docker-run --pull missing --network host \\
    --workdir {repo} {image} \\
    bash -lc "export PYTHONPATH={self.config.container_repo_root}/src:\${{PYTHONPATH:-}}; \\
      python3 case_study/python/zno_task_worker.py --task-json '$task_in' --mark-timeout"
fi

# Scientific failures are represented in JSON and must not abort sibling tasks.
exit 0
"""


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)
