from typing import Tuple
import numpy as np
from .math_utils import normalize


def xyzw2wxyz(q):
    return np.roll(q, 1, -1)

def wxyz2xyzw(q: np.ndarray) -> np.ndarray:
    return np.roll(q, -1, -1)

def quat_unit(a: np.ndarray) -> np.ndarray:
    return normalize(a)

def quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    theta = angle / 2
    xyz = normalize(axis) * np.sin(theta)
    w = np.cos(theta)
    return quat_unit(np.concatenate([[w], xyz]))

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape
    
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([w, x, y, z])

    return quat

def get_euler_xyz(q: np.ndarray, extrinsic: bool = True) -> Tuple[float, float, float]:
    if extrinsic:
        qw, qx, qy, qz = 0, 1, 2, 3
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q[qw] * q[qx] + q[qy] * q[qz])
        cosr_cosp = q[qw] * q[qw] - q[qx] * q[qx] - q[qy] * q[qy] + q[qz] * q[qz]
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (q[qw] * q[qy] - q[qz] * q[qx])
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q[qw] * q[qz] + q[qx] * q[qy])
        cosy_cosp = q[qw] * q[qw] + q[qx] * q[qx] - q[qy] * q[qy] - q[qz] * q[qz]
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
    else:
        result = matrix_to_euler_angle(quat_to_rot_matrix(q), extrinsic=False)
        return result[0], result[1], result[2]
    
def matrix_to_euler_angle(mat: np.ndarray, extrinsic: bool = True) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles (ZYX convention).
    
    Args:
        mat: (3, 3) rotation matrix
        extrinsic: if True, extrinsic (fixed-axis) convention
    
    Returns:
        (3,) array of [roll, pitch, yaw] in radians
    """
    _POLE_LIMIT = 1.0 - 1e-6
    result = np.zeros(3, dtype=np.float32)

    if extrinsic:
        if mat[2, 0] > _POLE_LIMIT:          # north pole
            result[0] = 0.0
            result[1] = -np.pi / 2
            result[2] = np.arctan2(mat[0, 1], mat[0, 2])
        elif mat[2, 0] < -_POLE_LIMIT:       # south pole
            result[0] = 0.0
            result[1] = np.pi / 2
            result[2] = np.arctan2(mat[0, 1], mat[0, 2])
        else:
            result[0] = np.arctan2(mat[2, 1], mat[2, 2])
            result[1] = -np.arcsin(mat[2, 0])
            result[2] = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        if mat[2, 0] > _POLE_LIMIT:          # north pole
            result[0] = np.arctan2(mat[1, 0], mat[1, 1])
            result[1] = np.pi / 2
            result[2] = 0.0
        elif mat[2, 0] < -_POLE_LIMIT:       # south pole
            result[0] = np.arctan2(mat[1, 0], mat[1, 1])
            result[1] = -np.pi / 2
            result[2] = 0.0
        else:
            result[0] = -np.arctan2(mat[1, 2], mat[2, 2])
            result[1] =  np.arcsin(mat[0, 2])
            result[2] = -np.arctan2(mat[0, 1], mat[0, 0])

    return result

def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quat: (4,) array [w, x, y, z]
    
    Returns:
        (3, 3) rotation matrix
    """
    nq = np.dot(quat, quat)

    if nq < 1e-10:
        return np.eye(3, dtype=np.float32)

    s = np.sqrt(2.0 / nq)
    q = quat * s
    qq = np.outer(q, q)  # einsum("i,j->ij", q, q)

    result = np.zeros((3, 3), dtype=np.float32)
    result[0, 0] = 1.0 - qq[2, 2] - qq[3, 3]
    result[0, 1] = qq[1, 2] - qq[3, 0]
    result[0, 2] = qq[1, 3] + qq[2, 0]
    result[1, 0] = qq[1, 2] + qq[3, 0]
    result[1, 1] = 1.0 - qq[1, 1] - qq[3, 3]
    result[1, 2] = qq[2, 3] - qq[1, 0]
    result[2, 0] = qq[1, 3] - qq[2, 0]
    result[2, 1] = qq[2, 3] + qq[1, 0]
    result[2, 2] = 1.0 - qq[1, 1] - qq[2, 2]

    return result

def quat_from_euler_xyz(roll: float, pitch: float, yaw: float, extrinsic: bool = True) -> np.ndarray:
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    if extrinsic:
        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp
    else:
        qw = -sr * sp * sy + cr * cp * cy
        qx = sr * cp * cy + sp * sy * cr
        qy = -sr * sy * cp + sp * cr * cy
        qz = sr * sp * cy + sy * cr * cp

    return np.stack([qw, qx, qy, qz])

def quat_conjugate(a: np.ndarray) -> np.ndarray:
    return np.concatenate([a[0:1], -a[1:]])

def quat_apply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xyz = a[1:]
    t = np.cross(xyz, b) * 2
    return b + a[0:1] * t + np.cross(xyz, t)