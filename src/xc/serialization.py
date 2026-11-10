"""Serialization helpers for XCFunctional (re-exports from xc_functional.py)."""

from .xc_functional import XCFunctional

# to_dict / from_dict / to_json / from_json are methods on XCFunctional itself.
# This module provides file I/O convenience functions.

import json
from pathlib import Path


def save(xc: XCFunctional, path: Path | str) -> None:
    path = Path(path)
    path.write_text(xc.to_json(), encoding="utf-8")


def load(path: Path | str) -> XCFunctional:
    path = Path(path)
    return XCFunctional.from_json(path.read_text(encoding="utf-8"))
