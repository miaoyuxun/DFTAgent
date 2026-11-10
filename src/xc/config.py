"""Configuration for XC optimizer."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

import yaml


@dataclass
class LLMConfig:
    model: str = "gpt-5.6-sol"
    base_url: str | None = None
    api_key: str | None = None


@dataclass
class XCOptimizerConfig:
    initial_functional: str = "PBE0"
    basis: str = "def2-TZVP"
    max_iterations: int = 20
    search_subsets: list[str] = field(default_factory=lambda: ["SIE4x4", "BH76", "G21IP", "G21EA"])
    validation_subsets: list[str] = field(default_factory=lambda: ["W4-11", "S66", "ISO34"])
    improvement_threshold: float = 0.05
    regression_threshold: float = 0.10
    gmtkn55_dir: str = "benchmark/GMTKN55"
    store_path: str = "results/xc_optimizer.db"
    llm: LLMConfig = field(default_factory=LLMConfig)
    target_problem: str = "improve SIE4x4 MAE while maintaining BH76 and G21IP accuracy"

    @classmethod
    def from_yaml(cls, path: Path | str) -> XCOptimizerConfig:
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Resolve environment variables
        def _resolve_env(val):
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                env_name = val[2:-1]
                return os.environ.get(env_name, "")
            return val

        llm_raw = raw.get("llm", {})
        llm = LLMConfig(
            model=llm_raw.get("model", "gpt-5.6-sol"),
            base_url=_resolve_env(llm_raw.get("base_url")),
            api_key=_resolve_env(llm_raw.get("api_key")),
        )

        return cls(
            initial_functional=raw.get("initial_functional", "PBE0"),
            basis=raw.get("basis", "def2-TZVP"),
            max_iterations=raw.get("max_iterations", 20),
            search_subsets=raw.get("search_subsets", ["SIE4x4", "BH76", "G21IP", "G21EA"]),
            validation_subsets=raw.get("validation_subsets", ["W4-11", "S66", "ISO34"]),
            improvement_threshold=raw.get("improvement_threshold", 0.05),
            regression_threshold=raw.get("regression_threshold", 0.10),
            gmtkn55_dir=raw.get("gmtkn55_dir", "benchmark/GMTKN55"),
            store_path=raw.get("store_path", "results/xc_optimizer.db"),
            llm=llm,
            target_problem=raw.get("target_problem", "improve SIE4x4 MAE while maintaining BH76 and G21IP accuracy"),
        )
