from __future__ import annotations

import json

from xc.idea.basin_agent import _normalize_json_string_expressions


def test_normalize_json_string_replace_expression_from_basin_response() -> None:
    malformed = """
    {
      "points": [
        {"label": "gk72_zvpb esol".replace(" ", "")},
        {"label": "gk72_k tbmheavy".replace(" ", "")}
      ]
    }
    """

    payload = json.loads(_normalize_json_string_expressions(malformed))

    assert payload == {
        "points": [
            {"label": "gk72_zvpbesol"},
            {"label": "gk72_ktbmheavy"},
        ]
    }


def test_normalize_json_string_expressions_does_not_execute_other_code() -> None:
    content = '{"label": dangerous_call("x")}'

    assert _normalize_json_string_expressions(content) == content
