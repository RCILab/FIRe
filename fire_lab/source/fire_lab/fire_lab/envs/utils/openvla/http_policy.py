from __future__ import annotations

from typing import Any, Dict

import requests

from .encoding import decode_response, encode_payload


class OpenVLAHTTPPolicy:
    """Synchronous HTTP client for openvla-oft/vla-scripts/deploy_batch.py."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8778,
        timeout: float | None = 120.0,
        max_batch_size: int | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self.base_url = self._make_base_url(host, port)
        self._session = requests.Session()

    @staticmethod
    def _make_base_url(host: str, port: int) -> str:
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/")
        return f"http://{host}:{port}"

    def infer(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = "/act_batch" if "observations" in observations else "/act"
        if endpoint == "/act_batch" and self._should_split_batch(observations):
            return self._infer_micro_batches(observations)

        return self._post(endpoint, observations)

    def _should_split_batch(self, payload: Dict[str, Any]) -> bool:
        return (
            self.max_batch_size is not None
            and self.max_batch_size > 0
            and len(payload["observations"]) > self.max_batch_size
        )

    def _infer_micro_batches(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        observations = payload["observations"]
        instructions = payload.get("instructions")
        all_actions = []

        for start in range(0, len(observations), self.max_batch_size):
            end = min(start + self.max_batch_size, len(observations))
            micro_payload = dict(payload)
            micro_payload["observations"] = observations[start:end]
            if instructions is not None:
                micro_payload["instructions"] = instructions[start:end]

            result = self._post("/act_batch", micro_payload)
            all_actions.extend(result["actions"])

        return {"actions": all_actions}

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._session.post(
            f"{self.base_url}{endpoint}",
            json=encode_payload(payload),
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        result = decode_response(response.json())
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        if endpoint == "/act":
            result = {"actions": [result]}
        self._validate_result(result, endpoint)
        return result

    @staticmethod
    def _validate_result(result: Any, endpoint: str):
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected OpenVLA {endpoint} response type: {type(result).__name__}")
        if "actions" not in result:
            raise RuntimeError(f"Unexpected OpenVLA {endpoint} response keys: {tuple(result.keys())}")

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
