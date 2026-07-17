from __future__ import annotations

import select
import sys
import termios
import time
import tty
from typing import Any, Optional, TextIO

import numpy as np
from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator

from .configuration_keyboard import FIReKeyboardTeleopConfig


class FIReKeyboardTeleop(Teleoperator):
    config_class = FIReKeyboardTeleopConfig
    name = "fire_keyboard"

    def __init__(
        self,
        config: FIReKeyboardTeleopConfig,
        stream: TextIO | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self._stream = stream or sys.stdin
        self._old_termios: Optional[list[int | bytes | list[int]]] = None
        self._is_connected = False
        self.last_key: Optional[str] = None
        self._last_action = np.zeros((config.action_dim,), dtype=np.float32)
        self._hold_until = 0.0

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (self.config.action_dim,),
            "names": [f"action_{idx}" for idx in range(self.config.action_dim)],
        }

    @property
    def feedback_features(self) -> dict[str, Any]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return
        if self._stream.isatty():
            self._old_termios = termios.tcgetattr(self._stream)
            tty.setcbreak(self._stream.fileno())
        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        action, is_intervention = self.get_intervention_action()
        return {
            "arm_actions": action,
            "is_intervention": np.array([float(is_intervention)], dtype=np.float32),
        }

    def get_intervention_action(self) -> tuple[np.ndarray, bool]:
        action = np.zeros((self.config.action_dim,), dtype=np.float32)
        if not self._is_connected or not self._stream.isatty():
            return action, False

        readable, _, _ = select.select([self._stream], [], [], 0.0)
        if not readable:
            if time.monotonic() < self._hold_until:
                return self._last_action.copy(), True
            return action, False

        key = self._stream.read(1).lower()
        self.last_key = key
        if key == "w":
            action[0] = 1.0
        elif key == "s":
            action[0] = -1.0
        elif key == "a":
            action[1] = 1.0
        elif key == "d":
            action[1] = -1.0
        else:
            return action, False

        scaled_action = float(self.config.scale) * action
        self._last_action = scaled_action.copy()
        self._hold_until = time.monotonic() + float(self.config.hold_s)
        return scaled_action, True

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self._old_termios is not None and self._stream.isatty():
            termios.tcsetattr(self._stream, termios.TCSADRAIN, self._old_termios)
        self._old_termios = None
        self._is_connected = False
