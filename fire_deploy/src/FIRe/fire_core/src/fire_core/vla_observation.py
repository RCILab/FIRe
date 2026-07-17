from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np

_warned_no_cameras = False


class VLAObservationNotReady(RuntimeError):
    """Raised when a VLA request would be made with incomplete live inputs."""


class VLAObservationConfigError(VLAObservationNotReady):
    """Raised when VLA observation cannot become ready without changing config."""


def _validate_camera_ready(robot: Any) -> None:
    camera_manager = getattr(robot, "camera_sensor_manager", None)
    config = getattr(robot, "config", None)
    use_cameras = bool(getattr(config, "use_cameras", False))

    if not use_cameras:
        # Cameras disabled: the robot feeds zero-filled camera frames into the VLA
        # observation, so inference can still run for testing. Warn once instead of
        # blocking, since the visual inputs are not real.
        global _warned_no_cameras
        if not _warned_no_cameras:
            warnings.warn(
                "VLA observation running without cameras: zero-filled camera frames "
                "are being used. Run with --use_cameras for real visual inputs.",
                stacklevel=2,
            )
            _warned_no_cameras = True
        return
    if camera_manager is None:
        raise VLAObservationConfigError("VLA observation is not ready: camera manager is missing.")
    if not getattr(camera_manager, "is_connected", False):
        raise VLAObservationNotReady("VLA observation is not ready: cameras are not connected.")
    if not getattr(camera_manager, "is_initialized", False):
        missing = getattr(camera_manager, "missing_frames", [])
        detail = f" Missing frames: {missing}." if missing else ""
        raise VLAObservationNotReady(f"VLA observation is not ready: cameras are not initialized.{detail}")
    if hasattr(camera_manager, "is_ready") and not camera_manager.is_ready:
        missing = getattr(camera_manager, "missing_frames", [])
        raise VLAObservationNotReady(
            f"VLA observation is not ready: missing or stale camera frames: {missing}."
        )


def _validate_array(name: str, value: Any) -> None:
    if value is None:
        raise VLAObservationNotReady(f"VLA observation is not ready: {name} is None.")
    arr = np.asarray(value)
    if arr.size == 0:
        raise VLAObservationNotReady(f"VLA observation is not ready: {name} is empty.")
    if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
        raise VLAObservationNotReady(f"VLA observation is not ready: {name} contains non-finite values.")


def get_ready_vla_observation(robot: Any) -> dict[str, Any]:
    if not getattr(robot, "is_connected", False):
        raise VLAObservationNotReady("VLA observation is not ready: robot is not connected.")

    _validate_camera_ready(robot)
    obs = robot.get_vla_observation()
    if not obs:
        raise VLAObservationNotReady("VLA observation is not ready: observation is empty.")

    for name, value in obs.items():
        _validate_array(name, value)
    return obs


def wait_for_ready_vla_observation(
    robot: Any,
    *,
    timeout_s: float = 5.0,
    poll_s: float = 0.05,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: VLAObservationNotReady | None = None
    while time.monotonic() < deadline:
        try:
            return get_ready_vla_observation(robot)
        except VLAObservationConfigError:
            raise
        except VLAObservationNotReady as exc:
            last_error = exc
            time.sleep(poll_s)
    if last_error is not None:
        raise last_error
    raise VLAObservationNotReady("VLA observation is not ready.")
