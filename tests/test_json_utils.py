import json

from xc.json_utils import json_safe_copy, safe_json_dumps


def test_safe_json_replaces_circular_reference():
    payload = {"alpha": 0.3}
    payload["self"] = payload
    copied = json_safe_copy(payload)
    assert copied["alpha"] == 0.3
    assert copied["self"] == "<circular-reference>"
    assert json.loads(safe_json_dumps(payload))["self"] == "<circular-reference>"
