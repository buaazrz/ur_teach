#!/usr/bin/env python3
"""Preprocess recorded UR flange demonstrations into a MiDiGaP-ready discrete-time dataset.

Operations performed:
  1. Load all `demo_*.npz` in the given task directory.
  2. Optionally trim leading/trailing stationary segments from human Freedrive start/stop.
  3. Optionally transform trajectories into a local object-centric frame (as used in MiDiGaP).
  4. Resample every demonstration to a uniform discrete time horizon T using:
     - Linear interpolation for 3D positions & gripper
     - SLERP on SO(3) for unit quaternions
  5. Enforce quaternion hemisphere continuity both along time AND across demonstrations
     (aligned to demo 0) to prevent antipodal jumps during Riemannian Fréchet mean /
     tangent-space covariance estimation in MiDiGaP.
  6. Export `midigap_dataset.npz` and optionally a 3D trajectory visualization PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from pose_utils import (
    enforce_quaternion_continuity,
    pose6d_to_matrix,
    quat_wxyz_to_xyzw,
    resample_trajectory,
    transform_trajectory_to_frame,
    trim_static_ends,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将采集的 UR 法兰示教轨迹预处理为 MiDiGaP 所需的等长离散时间数据集"
    )
    parser.add_argument("--task-dir", type=str, required=True, help="包含 demo_*.npz 的目录路径")
    parser.add_argument("--num-steps", type=int, default=100, help="重采样后的统一离散时间步数 T")
    parser.add_argument("--no-trim", action="store_true", help="不裁剪首尾静止段")
    parser.add_argument("--pos-thresh", type=float, default=0.002, help="静止段位置阈值 (m)")
    parser.add_argument("--rot-thresh", type=float, default=0.015, help="静止段姿态阈值 (rad)")
    parser.add_argument(
        "--object-frame",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="可选：将轨迹转换到指定物体局部坐标系 [x, y, z, rx, ry, rz] (base->object)",
    )
    parser.add_argument("--out", type=str, default=None, help="输出 .npz 路径（默认在 task-dir 下）")
    parser.add_argument("--plot", action="store_true", help="导出 3D 轨迹可视化图 (trajectories_3d.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_dir = Path(args.task_dir)
    demo_files = sorted(task_dir.glob("demo_*.npz"))
    if not demo_files:
        raise FileNotFoundError(f"在 {task_dir} 下未找到任何 demo_*.npz 文件！")

    T_obj_world = pose6d_to_matrix(np.array(args.object_frame)) if args.object_frame else None

    all_times: List[np.ndarray] = []
    all_pos: List[np.ndarray] = []
    all_quat_wxyz: List[np.ndarray] = []
    all_matrices: List[np.ndarray] = []
    all_gripper: List[np.ndarray] = []
    demo_names: List[str] = []

    ref_quat: np.ndarray | None = None

    print(f"正在处理 {len(demo_files)} 条示教轨迹 (目标时间步 T={args.num_steps})...")
    for path in demo_files:
        data = np.load(path)
        timestamps = data["timestamps"]
        pos = data["flange_pos"]
        quat_wxyz = data["flange_quat_wxyz"]
        matrices = data["flange_matrix"]
        gripper = data["gripper"] if "gripper" in data else np.zeros(len(timestamps))

        raw_len = len(timestamps)
        if not args.no_trim:
            s_idx, e_idx = trim_static_ends(
                pos, quat_wxyz, pos_thresh=args.pos_thresh, rot_thresh_rad=args.rot_thresh
            )
            timestamps = timestamps[s_idx:e_idx]
            pos = pos[s_idx:e_idx]
            quat_wxyz = quat_wxyz[s_idx:e_idx]
            matrices = matrices[s_idx:e_idx]
            gripper = gripper[s_idx:e_idx]

        if T_obj_world is not None:
            pos, quat_wxyz, matrices = transform_trajectory_to_frame(matrices, T_obj_world)

        t_new, pos_new, q_wxyz_new, mat_new, grip_new = resample_trajectory(
            timestamps=timestamps,
            positions=pos,
            quats_wxyz=quat_wxyz,
            gripper=gripper,
            num_steps=args.num_steps,
        )

        # Align quaternion hemisphere across all demos using demo 0's first frame as reference
        if ref_quat is None:
            ref_quat = q_wxyz_new[0].copy()
        q_wxyz_new = enforce_quaternion_continuity(q_wxyz_new, reference=ref_quat)

        all_times.append(t_new)
        all_pos.append(pos_new)
        all_quat_wxyz.append(q_wxyz_new)
        all_matrices.append(mat_new)
        all_gripper.append(grip_new)
        demo_names.append(path.name)

        print(f"  - {path.name}: 原始 {raw_len} 帧 -> 裁剪后 {len(timestamps)} 帧 -> 重采样 {args.num_steps} 步")

    positions = np.stack(all_pos, axis=0)  # (M, T, 3)
    quaternions_wxyz = np.stack(all_quat_wxyz, axis=0)  # (M, T, 4)
    quaternions_xyzw = quat_wxyz_to_xyzw(quaternions_wxyz)  # (M, T, 4)
    poses_wxyz = np.concatenate([positions, quaternions_wxyz], axis=-1)  # (M, T, 7)
    poses_xyzw = np.concatenate([positions, quaternions_xyzw], axis=-1)  # (M, T, 7)
    matrices_arr = np.stack(all_matrices, axis=0)  # (M, T, 4, 4)
    gripper_arr = np.stack(all_gripper, axis=0)  # (M, T)
    times_arr = np.stack(all_times, axis=0)  # (M, T)

    out_path = Path(args.out) if args.out else (task_dir / "midigap_dataset.npz")
    np.savez_compressed(
        out_path,
        positions=positions,
        quaternions_wxyz=quaternions_wxyz,
        quaternions_xyzw=quaternions_xyzw,
        poses_wxyz=poses_wxyz,
        poses_xyzw=poses_xyzw,
        matrices=matrices_arr,
        gripper=gripper_arr,
        times=times_arr,
        demo_names=np.array(demo_names),
    )

    print(f"\n[DONE] 已导出 MiDiGaP 数据集: {out_path}")
    print(f"  positions        : {positions.shape}  (M, T, 3)")
    print(f"  quaternions_wxyz : {quaternions_wxyz.shape}  (M, T, 4)")
    print(f"  poses_wxyz       : {poses_wxyz.shape}  (M, T, 7) [x,y,z,qw,qx,qy,qz]")
    print(f"  matrices         : {matrices_arr.shape}  (M, T, 4, 4)")
    print(f"  gripper          : {gripper_arr.shape}  (M, T)")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        for i in range(len(positions)):
            p = positions[i]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], label=demo_names[i], linewidth=2)
            ax.scatter(p[0, 0], p[0, 1], p[0, 2], marker="o", s=35)
            ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], marker="^", s=45)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"UR Flange Demonstrations (M={len(positions)}, T={args.num_steps})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = task_dir / "trajectories_3d.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  3D 轨迹图        : {plot_path}")


if __name__ == "__main__":
    main()
