"""Small JSON-safety helpers for persisted search metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def json_safe_copy(value: Any, _active: set[int] | None = None) -> Any:
    """Return a JSON-compatible copy, replacing circular references explicitly."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)

    active = _active if _active is not None else set()
    value_id = id(value)
    if value_id in active:
        return "<circular-reference>"
    if is_dataclass(value):
        active.add(value_id)
        result = json_safe_copy(asdict(value), active)
        active.remove(value_id)
        return result

    if isinstance(value, dict):
        active.add(value_id)
        result = {
            str(key): json_safe_copy(item, active)
            for key, item in value.items()
        }
        active.remove(value_id)
        return result
    if isinstance(value, (list, tuple, set)):
        active.add(value_id)
        result = [json_safe_copy(item, active) for item in value]
        active.remove(value_id)
        return result
    return str(value)


def safe_json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize arbitrary search metadata without allowing cycles to abort a run."""
    return json.dumps(json_safe_copy(value), **kwargs)
