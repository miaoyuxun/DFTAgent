"""Result store — JSON file persistence for optimization results."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from .xc_functional import XCFunctional
from .benchmark.result import BenchmarkResult


class ResultStore:
    """Simple JSON-based result store."""

    def __init__(self, path: Path | str = "results/results.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"entries": [], "best": None}

    def add(
        self,
        name: str,
        result: BenchmarkResult,
        is_best: bool = False,
        signature: str = "",
    ) -> int:
        """Add a result entry. Deduplicates by name or signature."""
        # Check for existing entry with same name or signature
        for i, e in enumerate(self.data["entries"]):
            if e["name"] == name or (signature and e.get("signature") == signature):
                # Update existing entry instead of duplicating
                e["timestamp"] = datetime.now().isoformat()
                e["overall_mae"] = result.overall_mae
                e["overall_rms"] = result.overall_rms
                e["total_failures"] = result.total_failures
                e["subsets"] = {
                    k: {
                        "mae": v.mae_kcal,
                        "rms": v.rms_kcal,
                        "max_error": v.max_error_kcal,
                        "n_reactions": v.n_reactions,
                        "n_failures": v.n_failures,
                    }
                    for k, v in result.subsets.items()
                }
                e["is_best"] = is_best
                if signature:
                    e["signature"] = signature
                if is_best:
                    self.data["best"] = e
                self._save()
                return i

        entry = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "overall_mae": result.overall_mae,
            "overall_rms": result.overall_rms,
            "total_failures": result.total_failures,
            "subsets": {
                k: {
                    "mae": v.mae_kcal,
                    "rms": v.rms_kcal,
                    "max_error": v.max_error_kcal,
                    "n_reactions": v.n_reactions,
                    "n_failures": v.n_failures,
                }
                for k, v in result.subsets.items()
            },
            "is_best": is_best,
        }
        if signature:
            entry["signature"] = signature
        self.data["entries"].append(entry)
        if is_best:
            self.data["best"] = entry
        self._save()
        return len(self.data["entries"]) - 1

    @property
    def best(self) -> dict | None:
        return self.data.get("best")

    @property
    def entries(self) -> list[dict]:
        return self.data["entries"]

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=2))
