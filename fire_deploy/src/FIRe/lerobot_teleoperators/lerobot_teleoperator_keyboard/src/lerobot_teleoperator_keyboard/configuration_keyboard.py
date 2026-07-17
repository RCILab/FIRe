from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("fire_keyboard")
@dataclass(kw_only=True)
class FIReKeyboardTeleopConfig(TeleoperatorConfig):
    action_dim: int
    scale: float = 1.0
    hold_s: float = 0.25
