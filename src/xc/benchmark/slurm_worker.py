"""Execute one benchmark subset task inside a Slurm allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..xc_functional import XCFunctional
from .runner import compute_molecule_batch, compute_subset_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one XC benchmark subset task")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    task = json.loads(Path(args.task_file).read_text())
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xc = XCFunctional.from_dict(task["xc_dict"])
    if task.get("mode") == "molecule_batch":
        payload = compute_molecule_batch(
            subset_dir=Path(task["gmtkn55_dir"]) / task["subset_name"],
            mol_names=[str(name) for name in task.get("mol_names", [])],
            xc=xc,
            basis=task["basis"],
            scf_conv_tol=float(task["scf_conv_tol"]),
            scf_max_cycle=int(task["scf_max_cycle"]),
            n_workers=int(task["n_workers"]),
            n_threads=int(task["n_threads"]),
            profile=bool(task.get("profile", False)),
        )
        output_path.write_text(json.dumps(payload, indent=2))
    else:
        result = compute_subset_result(
            gmtkn55_dir=task["gmtkn55_dir"],
            subset_name=task["subset_name"],
            xc=xc,
            basis=task["basis"],
            scf_conv_tol=float(task["scf_conv_tol"]),
            scf_max_cycle=int(task["scf_max_cycle"]),
            n_workers=int(task["n_workers"]),
            n_threads=int(task["n_threads"]),
            log_label=str(task.get("log_label", "")),
        )
        output_path.write_text(json.dumps(result.to_dict(include_reaction_errors=True), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
