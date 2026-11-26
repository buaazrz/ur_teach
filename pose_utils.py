"""SO(3) / SE(3) pose utilities for UR RTDE flange recording and MiDiGaP preprocessing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert UR axis-angle rotation vector [rx, ry, rz] (rad) to 3x3 rotation matrix."""
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to UR axis-angle rotation vector [rx, ry, rz] (rad)."""
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_rotvec()


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix (or (N, 3, 3)) to unit quaternion [qw, qx, qy, qz]."""
    xyzw = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    if xyzw.ndim == 1:
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=-1)


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert [qw, qx, qy, qz] (or (..., 4)) to [qx, qy, qz, qw]."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert [qx, qy, qz, qw] (or (..., 4)) to [qw, qx, qy, qz]."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def pose6d_to_matrix(pose6d: np.ndarray) -> np.ndarray:
    """Convert UR 6D pose [x, y, z, rx, ry, rz] to 4x4 homogeneous transform matrix."""
    p = np.asarray(pose6d, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotvec_to_matrix(p[3:6])
    T[:3, 3] = p[:3]
    return T


def matrix_to_pose6d(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous transform matrix to UR 6D pose [x, y, z, rx, ry, rz]."""
    T = np.asarray(T, dtype=np.float64)
    pose = np.empty(6, dtype=np.float64)
    pose[:3] = T[:3, 3]
    pose[3:6] = matrix_to_rotvec(T[:3, :3])
    return pose


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid-body homogeneous transformation matrix."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def tcp_pose_to_flange_matrix(tcp_pose6d: np.ndarray, tcp_offset6d: np.ndarray) -> np.ndarray:
    """Recover base->flange 4x4 matrix from base->TCP pose and flange->TCP offset.

    T_base_tcp = T_base_flange @ T_flange_tcp
    => T_base_flange = T_base_tcp @ inv(T_flange_tcp)
    """
    T_base_tcp = pose6d_to_matrix(tcp_pose6d)
    if np.allclose(tcp_offset6d, 0.0):
        return T_base_tcp
    T_flange_tcp = pose6d_to_matrix(tcp_offset6d)
    return T_base_tcp @ invert_transform(T_flange_tcp)


def enforce_quaternion_continuity(
    quats: np.ndarray, reference: np.ndarray | None = None
) -> np.ndarray:
    """Ensure consecutive unit quaternions stay in the same hemisphere (q_t . q_{t-1} >= 0).

    Args:
        quats: Array of shape (N, 4), either wxyz or xyzw.
        reference: Optional reference quaternion of shape (4,) to align quats[0].
    """
    q = np.array(quats, dtype=np.float64, copy=True)
    if len(q) == 0:
        return q
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(norms, 1e-12, None)

    if reference is not None:
        if float(np.dot(q[0], reference)) < 0.0:
            q[0] = -q[0]
    elif q[0, 0] < 0.0:
        # Canonicalize first sample to positive first component when no reference is given
        q[0] = -q[0]

    for i in range(1, len(q)):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    return q


def trim_static_ends(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    pos_thresh: float = 0.002,
    rot_thresh_rad: float = 0.015,
    pad_steps: int = 2,
) -> Tuple[int, int]:
    """Find [start_idx, end_idx) removing leading/trailing stationary segments.

    A sample is considered moving once it deviates from the initial/final pose
    by more than `pos_thresh` (meters) or `rot_thresh_rad` (radians).
    """
    n = len(positions)
    if n <= 2:
        return 0, n

    def _deviates(idx_a: int, idx_b: int) -> bool:
        dp = float(np.linalg.norm(positions[idx_a] - positions[idx_b]))
        if dp > pos_thresh:
            return True
        dot = abs(float(np.dot(quats_wxyz[idx_a], quats_wxyz[idx_b])))
        dot = min(1.0, max(0.0, dot))
        angle = 2.0 * np.arccos(dot)
        return angle > rot_thresh_rad

    start = 0
    for i in range(1, n):
        if _deviates(i, 0):
            start = max(0, i - pad_steps)
            break

    end = n
    for i in range(n - 2, start, -1):
        if _deviates(i, n - 1):
            end = min(n, i + 1 + pad_steps)
            break

    if end - start < 2:
        return 0, n
    return start, end


def resample_trajectory(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    gripper: np.ndarray,
    num_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample a SE(3) + gripper trajectory to `num_steps` uniform discrete time steps.

    Uses linear interpolation on R^3 (positions, gripper) and SLERP on SO(3) (orientations).

    Returns:
        t_new: (num_steps,) normalized time in [0, 1]
        pos_new: (num_steps, 3)
        quat_wxyz_new: (num_steps, 4) with sign continuity
        matrices_new: (num_steps, 4, 4)
        gripper_new: (num_steps,)
    """
    t = np.asarray(timestamps, dtype=np.float64)
    # Ensure strictly increasing timestamps for SLERP
    keep = np.ones(len(t), dtype=bool)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            keep[i] = False
    t = t[keep]
    pos = np.asarray(positions, dtype=np.float64)[keep]
    q_wxyz = enforce_quaternion_continuity(np.asarray(quats_wxyz, dtype=np.float64)[keep])
    grip = np.asarray(gripper, dtype=np.float64)[keep]

    if len(t) < 2:
        raise ValueError("Trajectory must contain at least 2 distinct timestamps to resample.")

    t_norm = (t - t[0]) / (t[-1] - t[0])
    t_new = np.linspace(0.0, 1.0, num_steps)

    pos_new = np.column_stack([np.interp(t_new, t_norm, pos[:, d]) for d in range(3)])
    grip_new = np.interp(t_new, t_norm, grip)

    rots = Rotation.from_quat(quat_wxyz_to_xyzw(q_wxyz))
    slerp = Slerp(t_norm, rots)
    rots_new = slerp(t_new)
    q_wxyz_new = enforce_quaternion_continuity(quat_xyzw_to_wxyz(rots_new.as_quat()))

    matrices_new = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], num_steps, axis=0)
    matrices_new[:, :3, :3] = rots_new.as_matrix()
    matrices_new[:, :3, 3] = pos_new

    return t_new, pos_new, q_wxyz_new, matrices_new, grip_new


def transform_trajectory_to_frame(
    matrices: np.ndarray, frame_matrix_world: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Express world-frame trajectory T_world_flange in a local object frame T_world_obj.

    T_obj_flange = inv(T_world_obj) @ T_world_flange
    """
    T_obj_world = invert_transform(frame_matrix_world)
    T_local = np.einsum("ij,njk->nik", T_obj_world, matrices)
    pos_local = T_local[:, :3, 3]
    quat_wxyz_local = enforce_quaternion_continuity(matrix_to_quat_wxyz(T_local[:, :3, :3]))
    return pos_local, quat_wxyz_local, T_local
