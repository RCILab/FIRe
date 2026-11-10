from __future__ import annotations

import base64
import json
from typing import Any, Dict

import numpy as np


def _encode_array(obj: Any):
    if isinstance(obj, (np.ndarray, np.generic)):
        arr = np.ascontiguousarray(obj)
        return {"__numpy__": base64.b64encode(arr.data).decode("ascii"), "dtype": arr.dtype.str, "shape": arr.shape}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _decode_array(dct: Dict[str, Any]):
    if "__numpy__" in dct:
        arr = np.frombuffer(base64.b64decode(dct["__numpy__"]), dtype=np.dtype(dct["dtype"]))
        return arr.reshape(dct["shape"])
    return dct


def encode_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    return {"encoded": json.dumps(payload, default=_encode_array)}


def decode_response(data: Any) -> Any:
    if isinstance(data, str):
        return json.loads(data, object_hook=_decode_array)
    if isinstance(data, dict):
        return json.loads(json.dumps(data), object_hook=_decode_array)
    return data
