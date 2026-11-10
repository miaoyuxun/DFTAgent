"""Worker for one vc benchmark shard.

Each task reads a JSON payload, computes a batch of molecule energies, and writes
one result JSON.  It is meant to run inside a ``vc submit`` job with at most
8 CPU cores.
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Path to shard task JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_path = Path(args.task)
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    result_path = Path(payload["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)

    xc = XCFunctional.from_dict(payload["xc_dict"])
    result = compute_molecule_batch(
        subset_dir=payload["subset_dir"],
        mol_names=list(payload["mol_names"]),
        xc=xc,
        basis=payload["basis"],
        scf_conv_tol=float(payload["scf_conv_tol"]),
        scf_max_cycle=int(payload["scf_max_cycle"]),
        n_workers=int(payload["n_workers"]),
        n_threads=int(payload["n_threads"]),
        density_output_dir=payload.get("density_output_dir"),
    )
    result.update(
        {
            "task_path": str(task_path),
            "candidate_label": payload.get("candidate_label"),
            "subset_name": payload.get("subset_name"),
            "shard_index": payload.get("shard_index"),
            "n_shards": payload.get("n_shards"),
        }
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
