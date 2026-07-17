from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.multiprocessing as mp


Features = Dict[str, Tuple[int, ...]]


def total_dim(features: Features) -> int:
    return sum(int(np.prod(s)) for s in features.values())


def flat_keys(features: Features) -> List[str]:
    keys: List[str] = []
    for name, shape in features.items():
        for i in range(int(np.prod(shape))):
            keys.append(f"{name}_{i}")
    return keys


def flatten(data: Dict[str, np.ndarray], features: Features) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, shape in features.items():
        arr = np.asarray(data[key], dtype=np.float32).ravel()
        dim = int(np.prod(shape))
        for i in range(dim):
            out[f"{key}_{i}"] = float(arr[i])
    return out


def obs_to_indexed(obs: Dict[str, np.ndarray], features: Features) -> Dict[str, float]:
    """Serialize into obs_0 ~ obs_N form following the obs_features order."""
    out: Dict[str, float] = {}
    idx = 0
    for key, shape in features.items():
        arr = np.asarray(obs[key], dtype=np.float32).ravel()
        for v in arr[: int(np.prod(shape))]:
            out[f"obs_{idx}"] = float(v)
            idx += 1
    return out


def flat_array_to_indexed(prefix: str, values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).ravel()
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(arr)}


def write_obs_shm(obs: dict, buf: np.ndarray, features: Features) -> None:
    """Slice-write obs directly into shared memory (minimizing GIL contention)."""
    offset = 0
    for key, shape in features.items():
        dim = int(np.prod(shape))
        buf[offset : offset + dim] = np.asarray(obs[key], dtype=np.float32).ravel()[:dim]
        offset += dim


def feature_slice(features: Features, name: str) -> Optional[slice]:
    offset = 0
    for key, shape in features.items():
        dim = int(np.prod(shape))
        if key == name:
            return slice(offset, offset + dim)
        offset += dim
    return None
