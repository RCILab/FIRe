import numpy as np


def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return x / np.linalg.norm(x).clip(min=eps, max=None)

def copysign(a: float, b: float) -> float:
    return np.abs(a) * np.sign(b)