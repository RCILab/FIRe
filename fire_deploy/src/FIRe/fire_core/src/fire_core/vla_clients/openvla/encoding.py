from __future__ import annotations

import json
from typing import Any, Dict

import json_numpy

json_numpy.patch()


def encode_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    """Encode numpy arrays using the OpenVLA server's double-encoded JSON path."""
    return {"encoded": json.dumps(payload)}


def decode_response(data: Any) -> Any:
    """Decode normal JSON responses and json_numpy double-encoded responses."""
    if isinstance(data, str):
        return json.loads(data)
    return data
