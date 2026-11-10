"""Small helpers for readable terminal logs."""

from __future__ import annotations

import json
import os

from .benchmark.result import BenchmarkResult


_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def _color_enabled() -> bool:
    return os.getenv("NO_COLOR") is None and os.getenv("DFTAGENT_COLOR", "1") != "0"


def colorize(text: str, color: str = "", *, bold: bool = False, dim: bool = False) -> str:
    """Wrap text in ANSI color codes unless disabled with NO_COLOR or DFTAGENT_COLOR=0."""
    if not _color_enabled():
        return text
    prefix = ""
    if bold:
        prefix += _COLORS["bold"]
    if dim:
        prefix += _COLORS["dim"]
    if color:
        prefix += _COLORS.get(color, "")
    return f"{prefix}{text}{_COLORS['reset']}" if prefix else text


def log_block(title: str, body: str, *, color: str = "cyan") -> str:
    """Format a multiline log block with a colored header."""
    line = "=" * 88
    header = f"{line}\n{title}\n{line}"
    footer = line
    return f"\n{colorize(header, color, bold=True)}\n{body.rstrip()}\n{colorize(footer, color, bold=True)}"


def pretty_json_or_text(text: str) -> str:
    """Pretty-print JSON responses, otherwise return the original text."""
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        return json.dumps(json.loads(stripped), indent=2, ensure_ascii=False)
    except (TypeError, json.JSONDecodeError):
        return text


def format_benchmark_result(result: BenchmarkResult, *, title: str = "Benchmark") -> str:
    """Render benchmark MAE/RMS/failure numbers as a compact table."""
    lines = [
        colorize(title, "magenta", bold=True),
        f"overall  MAE={result.overall_mae:8.3f}  RMS={result.overall_rms:8.3f}  "
        f"failures={result.total_failures:3d}  wall={result.total_walltime_s:7.1f}s",
        "subset          MAE       RMS       MAX     n  fail    wall(s)",
        "------------  -------  --------  --------  ----  ----  ---------",
    ]
    for name, sr in result.subsets.items():
        lines.append(
            f"{name:<12}  {sr.mae_kcal:7.3f}  {sr.rms_kcal:8.3f}  "
            f"{sr.max_error_kcal:8.3f}  {sr.n_reactions:4d}  {sr.n_failures:4d}  {sr.walltime_s:9.1f}"
        )
    return "\n".join(lines)


def format_score_table(raw_scores: dict[str, int], normalized: dict[str, float]) -> str:
    """Render reward sub-scores."""
    lines = [
        colorize("MCTS scores", "blue", bold=True),
        "metric                       raw   normalized",
        "-------------------------  -----  ----------",
    ]
    for key in sorted(raw_scores):
        lines.append(f"{key:<25}  {raw_scores[key]:5d}  {normalized.get(key, 0.0):10.3f}")
    return "\n".join(lines)
