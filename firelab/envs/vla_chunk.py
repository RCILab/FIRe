from typing import Any, Dict

import numpy as np


def to_vla_chunk(actions: Any, chunk_size: int) -> Dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 2:
        actions = actions[None]
    h = actions.shape[1]
    if h < chunk_size:
        actions = np.concatenate([actions, np.repeat(actions[:, -1:], chunk_size - h, axis=1)], axis=1)
    actions = actions[:, :chunk_size]
    return {
        "action.eef_position_delta": actions[..., 0:3],
        "action.eef_rotation_delta": actions[..., 3:6],
    }
