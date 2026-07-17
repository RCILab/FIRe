import numpy as np
from typing import Any
import cv2


def flat_feature_names(prefix: str, features: dict[str, tuple[int, ...]]) -> list[str]:
    names: list[str] = []
    for key, shape in features.items():
        dim = int(np.prod(shape))
        names.extend(f"{prefix}{key}_{i}" for i in range(dim))
    return names


def total_dim(features: dict[str, tuple[int, ...]]) -> int:
    return sum(int(np.prod(shape)) for shape in features.values())


def vector_feature(names: list[str]) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": (len(names),),
        "names": names,
    }


def flatten_feature_values(data: dict[str, Any], features: dict[str, tuple[int, ...]]) -> np.ndarray:
    values: list[np.ndarray] = []
    for key, shape in features.items():
        dim = int(np.prod(shape))
        arr = np.asarray(data[key], dtype=np.float32).reshape(-1)
        values.append(arr[:dim])
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values).astype(np.float32, copy=False)


def squeeze_image(value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims after squeeze, got shape {image.shape}.")

    height, width, _ = shape
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(image, dtype=np.uint8)


