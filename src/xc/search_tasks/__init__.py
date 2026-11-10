"""Task adapters used by search algorithms.

Search algorithms operate on :class:`SearchTaskEvaluation` instead of knowing
whether an evaluation came from molecular benchmarks, a periodic solid, or a
future scientific target.
"""

from .base import SearchTask, SearchTaskEvaluation
from .benchmark import BenchmarkSearchTask
from .mean_mae import MeanMAEBenchmarkTask, MeanMAETaskConfig
from .zno_gap import (
    ZnOGapSearchTask,
    ZnOGapTaskConfig,
    load_existing_zno_root_evaluation,
    make_mtask_pw92_functional,
)
from .zno_slurm import ZnOSlurmDockerEvaluator, ZnOSlurmEvaluatorConfig
from .crystal_slurm import CrystalGuardSlurmDockerEvaluator, CrystalGuardSlurmEvaluatorConfig
from .crystal_vc import CrystalGuardVCEvaluator, CrystalGuardVCEvaluatorConfig
from .crystal_persistent_vc import (
    CrystalGuardPersistentVCEvaluator,
    CrystalGuardPersistentVCEvaluatorConfig,
)
from .sie_crystal_guard import SIECrystalGuardSearchTask, SIECrystalGuardTaskConfig

__all__ = [
    "BenchmarkSearchTask",
    "MeanMAEBenchmarkTask",
    "MeanMAETaskConfig",
    "CrystalGuardSlurmDockerEvaluator",
    "CrystalGuardSlurmEvaluatorConfig",
    "CrystalGuardVCEvaluator",
    "CrystalGuardVCEvaluatorConfig",
    "CrystalGuardPersistentVCEvaluator",
    "CrystalGuardPersistentVCEvaluatorConfig",
    "SearchTask",
    "SearchTaskEvaluation",
    "SIECrystalGuardSearchTask",
    "SIECrystalGuardTaskConfig",
    "ZnOGapSearchTask",
    "ZnOGapTaskConfig",
    "load_existing_zno_root_evaluation",
    "make_mtask_pw92_functional",
    "ZnOSlurmDockerEvaluator",
    "ZnOSlurmEvaluatorConfig",
]
