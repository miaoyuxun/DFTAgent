"""Crystal guard evaluator backed by a run-level persistent VC queue."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..persistent_vc_queue import PersistentVCQueue
from ..xc_functional import XCFunctional
from .crystal_slurm import CrystalGuardSlurmDockerEvaluator, CrystalGuardSlurmEvaluatorConfig


logger = logging.getLogger(__name__)


@dataclass
class CrystalGuardPersistentVCEvaluatorConfig:
    output_dir: Path
    queue_dir: Path
    cpus_per_candidate: int = 8
    hard_timeout_s: int = 7200
    poll_seconds: float = 2.0
    wait_timeout_seconds: float = 0.0
    band_mesh: tuple[int, int, int] = (2, 2, 2)
    materials: tuple[str, ...] = ("ZnS",)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.queue_dir = Path(self.queue_dir)
        self.cpus_per_candidate = max(1, int(self.cpus_per_candidate))
        self.hard_timeout_s = max(60, int(self.hard_timeout_s))


class CrystalGuardPersistentVCEvaluator:
    """Prepare crystal tasks locally, then stream them to persistent workers."""

    def __init__(
        self,
        config: CrystalGuardPersistentVCEvaluatorConfig,
        *,
        repo_root: Path,
    ) -> None:
        self.config = config
        self.repo_root = Path(repo_root).resolve()
        self.queue = PersistentVCQueue(config.queue_dir, repo_root=self.repo_root)
        prep_config = CrystalGuardSlurmEvaluatorConfig(
            output_dir=config.output_dir,
            image="unused-by-persistent-vc",
            cpus_per_candidate=config.cpus_per_candidate,
            max_concurrent=1,
            memory="1G",
            hard_timeout_s=config.hard_timeout_s,
            slurm_time="00:01:00",
            band_mesh=config.band_mesh,
            materials=config.materials,
            adaptive_straggler_cutoff=False,
        )
        self.helper = CrystalGuardSlurmDockerEvaluator(prep_config, repo_root=self.repo_root)

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
        batch_dir, tasks = self.helper.prepare_batch(xcs, labels=labels, stage=stage)
        queue_tasks = []
        for task in tasks:
            task_path = batch_dir / "tasks" / f"task_{int(task['task_index']) - 1}.json"
            queue_tasks.append(
                {
                    "kind": "crystal_guard",
                    "task_json": self.queue.repo_relative(task_path),
                    "result_path": str(task["result_path"]),
                    "hard_timeout_s": int(task.get("hard_timeout_s") or self.config.hard_timeout_s),
                }
            )
        submitted = self.queue.submit(
            queue_tasks,
            batch_id=f"crystal_{stage}_{self.helper._batch_index:05d}",
        )
        logger.info(
            "PERSISTENT_VC_CRYSTAL_ENQUEUED stage=%s tasks=%d queue=%s",
            stage,
            len(submitted),
            self.config.queue_dir,
        )
        self.queue.wait(
            submitted,
            poll_seconds=self.config.poll_seconds,
            timeout_seconds=self.config.wait_timeout_seconds,
        )
        results = self.helper.collect_results(tasks, candidate_count=len(xcs))
        logger.info(
            "PERSISTENT_VC_CRYSTAL_DONE stage=%s candidates=%d failed=%d",
            stage,
            len(results),
            sum(1 for item in results if not item.get("ok")),
        )
        return results

