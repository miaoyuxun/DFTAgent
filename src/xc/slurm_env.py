"""Environment handling for submitting independent Slurm jobs."""

from __future__ import annotations

import os
from collections.abc import Mapping


_PRESERVED_SLURM_KEYS = {
    "SLURM_CLUSTERS",
    "SLURM_CONF",
}


def clean_slurm_submit_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Remove resource settings inherited from a parent Slurm allocation.

    ``sbatch`` exports the caller environment by default.  If a controller is
    itself launched by Slurm, values such as ``SLURM_CPUS_PER_TASK`` can then
    conflict with the resource directives in the newly generated batch script.
    Keep only cluster-selection/configuration variables from the outer Slurm
    environment; the child job recreates all runtime ``SLURM_*`` values.
    """

    env = dict(os.environ if source is None else source)
    for key in list(env):
        if key.startswith("SBATCH_"):
            env.pop(key, None)
        elif key.startswith("SLURM_") and key not in _PRESERVED_SLURM_KEYS:
            env.pop(key, None)
    return env
