#!/usr/bin/env python3
"""Record UR end-effector flange poses during Freedrive/teaching mode via RTDE.

Designed for collecting kinesthetic teaching demonstrations to reproduce MiDiGaP
(Mixture of Discrete-time Gaussian Processes).

Usage examples:
  # 1. Listen-only mode (safest: hold Freedrive on teach pendant, works in Local mode)
  python3 record_ur_teach.py --robot-ip 192.168.1.100 --task pick_mug --freq 50

  # 2. If your teach pendant has a non-zero TCP offset, pass it so flange pose is exact
  python3 record_ur_teach.py --robot-ip 192.168.1.100 --task pick_mug --tcp-offset 0 0 0.15 0 0 0

  # 3. Auto-Freedrive mode (requires UR in Remote Control mode; auto-reads TCP offset)
  python3 record_ur_teach.py --robot-ip 192.168.1.100 --task pick_mug --freedrive

  # 4. Dry-run without a robot (generates realistic mock trajectories)
  python3 record_ur_teach.py --mock --task demo_test --freq 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pose_utils import (
    enforce_quaternion_continuity,
    matrix_to_pose6d,
    matrix_to_quat_wxyz,
    quat_wxyz_to_xyzw,
    tcp_pose_to_flange_matrix,
)


class RobotBackend:
    """Abstracts real UR RTDE connection and offline mock generator."""

    def __init__(
        self,
        robot_ip: str,
        freq: float,
        use_freedrive: bool,
        tcp_offset: np.ndarray,
        mock: bool = False,
    ) -> None:
        self.robot_ip = robot_ip
        self.freq = freq
        self.use_freedrive = use_freedrive
        self.tcp_offset = np.asarray(tcp_offset, dtype=np.float64)
        self.mock = mock
        self.rtde_r = None
        self.rtde_c = None
        self._freedrive_active = False
        self._mock_t0 = time.perf_counter()
        self._mock_demo_seed = 0

    def connect(self) -> None:
        if self.mock:
            print("[MOCK] 已启用离线模拟模式（不连接真实 UR 机器人）。")
            return

        try:
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError(
                "未安装 ur-rtde！请先运行: pip install ur-rtde"
            ) from exc

        print(f"[RTDE] 正在连接 UR 接收接口 {self.robot_ip} (频率={self.freq} Hz)...")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip, self.freq)

        if self.use_freedrive:
            try:
                import rtde_control
            except ImportError as exc:
                raise RuntimeError("未安装 ur-rtde 的 rtde_control 模块。") from exc

            print(
                f"[RTDE] 正在连接 UR 控制接口 {self.robot_ip} (需示教器处于 Remote Control 模式)..."
            )
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            try:
                active_offset = np.asarray(self.rtde_c.getTCPOffset(), dtype=np.float64)
                print(f"[RTDE] 自动读取到当前激活的 TCP Offset: {np.round(active_offset, 5).tolist()}")
                self.tcp_offset = active_offset
            except Exception as exc:
                print(f"[WARN] 自动读取 TCP Offset 失败 ({exc})，将使用命令行参数值。")

        pose = self.rtde_r.getActualTCPPose()
        print(f"[RTDE] 连接成功！当前 TCP 位姿: {np.round(pose, 4).tolist()}")
        print(f"[RTDE] 用于换算法兰位姿的 TCP Offset: {np.round(self.tcp_offset, 5).tolist()}")

    def set_freedrive(self, enable: bool) -> None:
        if self.mock:
            self._freedrive_active = enable
            if enable:
                self._mock_t0 = time.perf_counter()
                self._mock_demo_seed += 1
            return

        if not self.use_freedrive or self.rtde_c is None:
            return

        if enable and not self._freedrive_active:
            self.rtde_c.freedriveMode()
            self._freedrive_active = True
            print("[RTDE] 已开启程序 Freedrive 自由拖动模式。")
        elif not enable and self._freedrive_active:
            self.rtde_c.endFreedriveMode()
            self._freedrive_active = False
            print("[RTDE] 已退出 Freedrive 模式。")

    def read_sample(self) -> Dict[str, np.ndarray]:
        if self.mock:
            t = time.perf_counter() - self._mock_t0
            seed = self._mock_demo_seed
            # Smooth synthetic 3D arc + gentle wrist rotation with slight per-demo variation
            phase = min(t / 3.0, 1.0)
            s = 0.5 - 0.5 * np.cos(np.pi * phase)  # smooth 0 -> 1
            x = -0.35 + 0.25 * s + 0.015 * np.sin(seed)
            y = -0.25 + 0.12 * np.sin(np.pi * s) + 0.01 * np.cos(seed)
            z = 0.25 + 0.10 * np.sin(np.pi * s) - 0.05 * s
            rx = np.pi * 0.95 + 0.15 * np.sin(np.pi * s)
            ry = 0.05 * np.cos(seed) + 0.20 * s
            rz = 0.10 * s
            tcp_pose = np.array([x, y, z, rx, ry, rz], dtype=np.float64)
            joint_q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float64) + 0.1 * s
            tcp_speed = np.zeros(6, dtype=np.float64)
        else:
            assert self.rtde_r is not None
            tcp_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float64)
            joint_q = np.asarray(self.rtde_r.getActualQ(), dtype=np.float64)
            tcp_speed = np.asarray(self.rtde_r.getActualTCPSpeed(), dtype=np.float64)

        T_flange = tcp_pose_to_flange_matrix(tcp_pose, self.tcp_offset)
        flange_pose6d = matrix_to_pose6d(T_flange)
        flange_quat_wxyz = matrix_to_quat_wxyz(T_flange[:3, :3])

        return {
            "tcp_pose_6d": tcp_pose,
            "flange_matrix": T_flange,
            "flange_pose_6d": flange_pose6d,
            "flange_pos": T_flange[:3, 3].copy(),
            "flange_rotvec": flange_pose6d[3:6].copy(),
            "flange_quat_wxyz": flange_quat_wxyz,
            "joint_q": joint_q,
            "tcp_speed": tcp_speed,
        }

    def disconnect(self) -> None:
        try:
            self.set_freedrive(False)
        except Exception:
            pass
        if self.rtde_c is not None:
            try:
                self.rtde_c.stopScript()
            except Exception:
                pass
            self.rtde_c = None
        if self.rtde_r is not None:
            try:
                self.rtde_r.disconnect()
            except Exception:
                pass
            self.rtde_r = None


class RawKeyReader:
    """Non-blocking single-key reader for Linux terminals."""

    def __init__(self) -> None:
        self._is_tty = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self._is_tty else -1
        self._old_settings = None

    def __enter__(self) -> "RawKeyReader":
        if self._is_tty:
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._is_tty and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self, timeout: float = 0.05) -> Optional[str]:
        if not self._is_tty:
            line = sys.stdin.readline()
            return line.strip() if line else "q"
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None


def next_demo_index(task_dir: Path) -> int:
    existing = sorted(task_dir.glob("demo_*.npz"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        try:
            indices.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return (max(indices) + 1) if indices else 0


def save_episode(
    task_dir: Path,
    demo_idx: int,
    records: List[Dict[str, Any]],
    tcp_offset: np.ndarray,
    freq: float,
    robot_ip: str,
    save_csv: bool = True,
) -> Path:
    timestamps = np.array([r["t_rel"] for r in records], dtype=np.float64)
    wall_timestamps = np.array([r["t_wall"] for r in records], dtype=np.float64)
    flange_pos = np.stack([r["flange_pos"] for r in records], axis=0)
    flange_quat_wxyz = enforce_quaternion_continuity(
        np.stack([r["flange_quat_wxyz"] for r in records], axis=0)
    )
    flange_quat_xyzw = quat_wxyz_to_xyzw(flange_quat_wxyz)
    flange_rotvec = np.stack([r["flange_rotvec"] for r in records], axis=0)
    flange_pose_6d = np.stack([r["flange_pose_6d"] for r in records], axis=0)
    flange_matrix = np.stack([r["flange_matrix"] for r in records], axis=0)
    tcp_pose_6d = np.stack([r["tcp_pose_6d"] for r in records], axis=0)
    joint_q = np.stack([r["joint_q"] for r in records], axis=0)
    tcp_speed = np.stack([r["tcp_speed"] for r in records], axis=0)
    gripper = np.array([r["gripper"] for r in records], dtype=np.float64)

    npz_path = task_dir / f"demo_{demo_idx:03d}.npz"
    np.savez_compressed(
        npz_path,
        timestamps=timestamps,
        wall_timestamps=wall_timestamps,
        flange_pos=flange_pos,
        flange_quat_wxyz=flange_quat_wxyz,
        flange_quat_xyzw=flange_quat_xyzw,
        flange_rotvec=flange_rotvec,
        flange_pose_6d=flange_pose_6d,
        flange_matrix=flange_matrix,
        tcp_pose_6d=tcp_pose_6d,
        tcp_offset=np.asarray(tcp_offset, dtype=np.float64),
        joint_q=joint_q,
        tcp_speed=tcp_speed,
        gripper=gripper,
    )

    if save_csv:
        csv_path = task_dir / f"demo_{demo_idx:03d}.csv"
        header = [
            "t_rel",
            "x", "y", "z",
            "qw", "qx", "qy", "qz",
            "rx", "ry", "rz",
            "q0", "q1", "q2", "q3", "q4", "q5",
            "gripper",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(timestamps)):
                writer.writerow(
                    [
                        f"{timestamps[i]:.6f}",
                        *[f"{v:.6f}" for v in flange_pos[i]],
                        *[f"{v:.6f}" for v in flange_quat_wxyz[i]],
                        *[f"{v:.6f}" for v in flange_rotvec[i]],
                        *[f"{v:.6f}" for v in joint_q[i]],
                        f"{gripper[i]:.1f}",
                    ]
                )

    meta_path = task_dir / "meta.json"
    meta = {
        "robot_ip": robot_ip,
        "target_freq_hz": freq,
        "tcp_offset_used": tcp_offset.tolist(),
        "pose_frame": "base_to_flange",
        "quaternion_convention_primary": "wxyz",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return npz_path


def run_auto_mock(
    backend: RobotBackend,
    task_dir: Path,
    num_demos: int,
    duration: float,
    freq: float,
    save_csv: bool,
) -> None:
    """Non-interactive batch recording for mock testing."""
    dt = 1.0 / freq
    for _ in range(num_demos):
        demo_idx = next_demo_index(task_dir)
        backend.set_freedrive(True)
        records: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        step = 0
        while True:
            t_now = time.perf_counter()
            t_rel = t_now - t0
            if t_rel > duration:
                break
            sample = backend.read_sample()
            sample["t_rel"] = t_rel
            sample["t_wall"] = time.time()
            sample["gripper"] = 1.0 if t_rel > duration * 0.5 else 0.0
            records.append(sample)
            step += 1
            next_t = t0 + step * dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
        backend.set_freedrive(False)
        out = save_episode(
            task_dir, demo_idx, records, backend.tcp_offset, freq, backend.robot_ip, save_csv
        )
        print(f"[MOCK] 已保存 {out.name} ({len(records)} 帧, {records[-1]['t_rel']:.2f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 UR RTDE 采集拖动示教时的末端法兰位姿（适配 MiDiGaP）"
    )
    parser.add_argument("--robot-ip", type=str, default="192.168.1.100", help="UR 机器人 IP 地址")
    parser.add_argument("--task", type=str, default="default_task", help="任务名称（用于数据子目录）")
    parser.add_argument("--out-dir", type=str, default="data", help="数据保存根目录")
    parser.add_argument("--freq", type=float, default=50.0, help="采样频率 Hz（UR CB3 <=125, e-Series <=500）")
    parser.add_argument(
        "--tcp-offset",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="当前示教器激活的 TCP 偏移 [m, rad]，用于精确反算末端法兰位姿（默认全 0）",
    )
    parser.add_argument(
        "--freedrive",
        action="store_true",
        help="由脚本自动开关 Freedrive（需要 UR 处于 Remote Control 模式；不加此项则为纯监听模式，手动按示教器 Freedrive 即可）",
    )
    parser.add_argument("--no-csv", action="store_true", help="仅保存 .npz，不额外生成 .csv")
    parser.add_argument("--mock", action="store_true", help="离线模拟模式（无需连接真实机器人）")
    parser.add_argument(
        "--auto-mock-demos",
        type=int,
        default=0,
        help="配合 --mock 自动连续录制指定条数（非交互测试用）",
    )
    parser.add_argument(
        "--auto-mock-duration",
        type=float,
        default=1.5,
        help="配合 --auto-mock-demos 每条模拟示教时长（秒）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_dir = Path(args.out_dir) / args.task
    task_dir.mkdir(parents=True, exist_ok=True)

    backend = RobotBackend(
        robot_ip=args.robot_ip,
        freq=args.freq,
        use_freedrive=args.freedrive,
        tcp_offset=np.array(args.tcp_offset, dtype=np.float64),
        mock=args.mock,
    )

    try:
        backend.connect()

        if args.mock and args.auto_mock_demos > 0:
            run_auto_mock(
                backend=backend,
                task_dir=task_dir,
                num_demos=args.auto_mock_demos,
                duration=args.auto_mock_duration,
                freq=args.freq,
                save_csv=not args.no_csv,
            )
            return

        print("\n" + "=" * 62)
        print(f"UR 示教数据采集已就绪 | 任务: {args.task} | 保存目录: {task_dir}")
        print("操作按键:")
        print("  [Space] 或 [Enter] : 开始 / 停止录制当前示教轨迹")
        print("  [g]                : 切换夹爪状态标记 (0=张开 / 1=闭合)")
        print("  [d]                : 删除刚刚保存的上一条轨迹")
        print("  [q]                : 安全退出程序")
        if not args.freedrive and not args.mock:
            print("提示: 当前为纯监听模式，请直接在 UR 示教器上按住/开启 Freedrive 拖动机械臂。")
        print("=" * 62 + "\n")

        recording = False
        gripper_state = 0.0
        records: List[Dict[str, Any]] = []
        record_lock = threading.Lock()
        stop_event = threading.Event()
        last_saved_idx: Optional[int] = None
        dt = 1.0 / args.freq

        def sampler_loop() -> None:
            nonlocal records
            t0 = time.perf_counter()
            step = 0
            while not stop_event.is_set():
                t_now = time.perf_counter()
                sample = backend.read_sample()
                sample["t_rel"] = t_now - t0
                sample["t_wall"] = time.time()
                sample["gripper"] = gripper_state
                with record_lock:
                    records.append(sample)
                    n = len(records)
                if n % max(1, int(args.freq // 5)) == 0:
                    p = sample["flange_pos"]
                    q = sample["flange_quat_wxyz"]
                    sys.stdout.write(
                        f"\r[REC] {sample['t_rel']:5.2f}s | 帧数:{n:4d} | "
                        f"Flange XYZ: [{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}] | "
                        f"Quat(wxyz): [{q[0]:+.2f}, {q[1]:+.2f}, {q[2]:+.2f}, {q[3]:+.2f}] | "
                        f"Grip:{int(gripper_state)}  "
                    )
                    sys.stdout.flush()
                step += 1
                sleep_s = (t0 + step * dt) - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        worker: Optional[threading.Thread] = None

        with RawKeyReader() as keys:
            while True:
                key = keys.read_key(timeout=0.05)
                if key is None:
                    continue

                if key in (" ", "\r", "\n"):
                    if not recording:
                        records = []
                        stop_event.clear()
                        backend.set_freedrive(True)
                        recording = True
                        demo_idx = next_demo_index(task_dir)
                        print(f"\n>>> 开始录制 demo_{demo_idx:03d} (按 Space/Enter 结束)...")
                        worker = threading.Thread(target=sampler_loop, daemon=True)
                        worker.start()
                    else:
                        stop_event.set()
                        if worker is not None:
                            worker.join(timeout=2.0)
                        backend.set_freedrive(False)
                        recording = False
                        print()
                        if len(records) < 2:
                            print("[WARN] 采集帧数过少 (<2)，已忽略本次录制。")
                        else:
                            demo_idx = next_demo_index(task_dir)
                            out_path = save_episode(
                                task_dir=task_dir,
                                demo_idx=demo_idx,
                                records=records,
                                tcp_offset=backend.tcp_offset,
                                freq=args.freq,
                                robot_ip=args.robot_ip,
                                save_csv=not args.no_csv,
                            )
                            last_saved_idx = demo_idx
                            dur = records[-1]["t_rel"]
                            eff_hz = len(records) / max(dur, 1e-6)
                            print(
                                f"<<< 已保存 {out_path} | 帧数: {len(records)} | "
                                f"时长: {dur:.2f}s | 实测频率: {eff_hz:.1f} Hz"
                            )

                elif key.lower() == "g":
                    gripper_state = 0.0 if gripper_state > 0.5 else 1.0
                    if not recording:
                        print(f"[INFO] 夹爪状态已切换为: {int(gripper_state)}")

                elif key.lower() == "d":
                    if recording:
                        print("\n[WARN] 正在录制中，请先停止录制再删除。")
                    elif last_saved_idx is None:
                        print("[INFO] 本次会话暂无可删除的上一条轨迹。")
                    else:
                        for ext in (".npz", ".csv"):
                            p = task_dir / f"demo_{last_saved_idx:03d}{ext}"
                            if p.exists():
                                p.unlink()
                        print(f"[INFO] 已删除 demo_{last_saved_idx:03d}。")
                        last_saved_idx = None

                elif key.lower() == "q":
                    if recording:
                        stop_event.set()
                        if worker is not None:
                            worker.join(timeout=2.0)
                    print("\n[INFO] 正在安全退出...")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，正在安全退出...")
    finally:
        backend.disconnect()
        print("[INFO] 连接已关闭。")


if __name__ == "__main__":
    main()
