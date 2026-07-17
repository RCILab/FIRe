from typing import Tuple
import numpy as np
from .rotation_utils import quat_conjugate, quat_apply, quat_mul


def tf_inverse(q: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)

def tf_combine(q1, t1, q2, t2) -> Tuple[np.ndarray, np.ndarray]:
    return quat_mul(q1, q2) , quat_apply(q1, t2) + t1