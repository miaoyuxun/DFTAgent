"""Worker for one Slurm+Docker benchmark shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xc.benchmark.runner import compute_molecule_batch  # noqa: E402
from xc.xc_functional import XCFunctional  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    task_path = Path(args.task)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    result_path = Path(task["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists() and task.get("skip_existing", True):
        return 0

    xc = XCFunctional.from_dict(task["xc_dict"])
    result = compute_molecule_batch(
        subset_dir=task["subset_dir"],
        mol_names=list(task["mol_names"]),
        xc=xc,
        basis=task["basis"],
        scf_conv_tol=float(task["scf_conv_tol"]),
        scf_max_cycle=int(task["scf_max_cycle"]),
        n_workers=int(task["n_workers"]),
        n_threads=int(task["n_threads"]),
        density_output_dir=task.get("density_output_dir"),
        profile=bool(task.get("profile", False)),
    )
    result.update(
        {
            "task_path": str(task_path),
            "candidate_label": task.get("candidate_label"),
            "subset_name": task.get("subset_name"),
            "task_index": task.get("task_index"),
            "candidate_index": task.get("candidate_index"),
        }
    )
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
