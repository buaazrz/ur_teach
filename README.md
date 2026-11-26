# UR RTDE 拖动示教末端法兰位姿采集工具（适配 MiDiGaP 复现）

本项目用于通过 **Universal Robots RTDE (`ur_rtde`)** 在拖动示教（Freedrive）模式下实时采集机械臂**末端法兰（Flange）位姿**，并将多条示教轨迹预处理为适配 **MiDiGaP**（*Mixture of Discrete-time Gaussian Processes*）的离散时间 $\mathbb{R}^3 \times \text{SO}(3)$ 数据集。

## 1. 环境安装

```bash
pip install -r requirements.txt
```

> 说明：只需安装 `ur-rtde`、`numpy`、`scipy`（可视化可选 `matplotlib`）。

## 2. 为什么能拿到精确的“末端法兰位姿”？

UR RTDE 的 `RTDEReceiveInterface.getActualTCPPose()` 返回的是**当前激活 TCP** 相对基座的位姿 $T_{\text{base}}^{\text{tcp}}$。
- 如果示教器里 TCP 偏移设为 `[0, 0, 0, 0, 0, 0]`，则 `getActualTCPPose()` 本身就是法兰位姿。
- 如果示教器里装了夹爪并设置了非零 TCP 偏移 $T_{\text{flange}}^{\text{tcp}}$，本工具会自动通过下式精确反算末端法兰位姿：
  $$T_{\text{base}}^{\text{flange}} = T_{\text{base}}^{\text{tcp}} \cdot \left(T_{\text{flange}}^{\text{tcp}}\right)^{-1}$$
  - **纯监听模式（默认，推荐）**：通过 `--tcp-offset x y z rx ry rz` 传入当前 TCP 偏移（默认全 0）。此模式仅连接 `RTDEReceiveInterface`，**不抢占控制权，UR 处于 Local 模式也能用**，直接按住示教器背面 Freedrive 按钮拖动即可。
  - **程序自动 Freedrive 模式（`--freedrive`）**：同时连接 `RTDEControlInterface`（需将 UR 切换到 Remote Control 模式），开始录制时自动进入 Freedrive、停止时自动退出，并自动调用 `getTCPOffset()` 获取当前 TCP 偏移。

## 3. 采集示教数据

### 真机采集（纯监听模式，最安全）
```bash
python3 record_ur_teach.py --robot-ip 192.168.1.100 --task pick_mug --freq 50
```
如果示教器激活了非零 TCP（例如 Z 轴偏移 0.15 m）：
```bash
python3 record_ur_teach.py --robot-ip 192.168.1.100 --task pick_mug --freq 50 --tcp-offset 0 0 0.15 0 0 0
```

### 真机采集（程序自动开关 Freedrive）
```bash
python3 record_ur_teach.py --robot-ip 192.168.1.100 --task pick_mug --freq 50 --freedrive
```

### 离线模拟测试（不连真机即可测试全流程）
```bash
python3 record_ur_teach.py --mock --task test_mock --freq 50
```

### 交互按键说明
- `Space` / `Enter`：开始 / 停止录制当前示教轨迹
- `g`：切换夹爪状态标记（`0` 张开 / `1` 闭合）
- `d`：删除上一条误录的轨迹
- `q`：安全退出并关闭 RTDE 连接

每条示教轨迹会保存在 `data/<task>/demo_000.npz`（及便于肉眼查看的 `demo_000.csv`），包含：
- `timestamps` `(N,)`：相对时间（秒）
- `flange_pos` `(N, 3)`：法兰位置 `[x, y, z]`（米）
- `flange_quat_wxyz` `(N, 4)`：连续符号对齐的单位四元数 `[qw, qx, qy, qz]`
- `flange_quat_xyzw` `(N, 4)`：同上，`[qx, qy, qz, qw]` 格式
- `flange_rotvec` `(N, 3)`：UR 轴角表示 `[rx, ry, rz]`（弧度）
- `flange_matrix` `(N, 4, 4)`：SE(3) 齐次变换矩阵
- `joint_q` `(N, 6)`：关节角（弧度）
- `tcp_pose_6d` / `tcp_offset` / `tcp_speed` / `gripper`

## 4. 预处理为 MiDiGaP 离散时间数据集

MiDiGaP 需要在离散时间步 $t = 1 \dots T$ 上拟合 $\mathbb{R}^3 \times \text{SO}(3)$ 轨迹分布。运行以下命令即可自动完成**首尾静止段裁剪 + 位置线性插值 + 姿态 SLERP 重采样 + 跨轨迹四元数半球连续性对齐**：

```bash
python3 prepare_midigap_dataset.py --task-dir data/pick_mug --num-steps 100 --plot
```

如果复现 MiDiGaP 的 **object-centric（物体局部坐标系）** 设定，可传入物体在基座系下的位姿：
```bash
python3 prepare_midigap_dataset.py --task-dir data/pick_mug --num-steps 100 --object-frame 0.4 -0.1 0.05 0 0 0 --plot
```

输出的 `midigap_dataset.npz` 包含：
- `positions`: `(M, T, 3)`
- `quaternions_wxyz`: `(M, T, 4)`
- `quaternions_xyzw`: `(M, T, 4)`
- `poses_wxyz`: `(M, T, 7)` (`[x, y, z, qw, qx, qy, qz]`)
- `matrices`: `(M, T, 4, 4)`
- `gripper`: `(M, T)`
- `times`: `(M, T)`（归一化到 `[0, 1]`）
