# Mini3 手柄命令与三点遥操：接口参考

本分支基于 cuboi，分别提供 command / teleop encoder，复用原 Mini3 mimic encoder 和 decoder。两个接口目前各保存一份模型；不是 NVIDIA SONIC 的 FSQ token，也不能直接加载 G1 SONIC encoder。zita checkpoint 的适配不在此实现中。

已完成的工作、环境准备、数据子集、蒸馏、SuperMimic 微调、PPO 对照和仿真检查步骤统一见 [README](../README.md)。本文仅保留接口、标定和验证范围说明。

## 输入约定

所有长度为米、角度为弧度。坐标轴 X 前、Y 左、Z 上，四元数 WXYZ。

- `command` 为 4 维：`vx, vy, yaw_rate, pelvis_height`。速度在机器人 heading frame，左转角速度为正。
- `teleop` 为 31 维：上述 4 维，加 head / left_hand / right_hand 各 9 维。每个目标包含相对骨盆原点、在 heading frame 表达的位置 3 维，以及相对 heading 的旋转矩阵前两列 6 维，按行展开：`r00,r01,r10,r11,r20,r21`。
- student encoder 接收当前目标和与原 checkpoint 一致的 term-major 本体历史。没有参考相位、未来关节轨迹或未来人体姿态输入。
- teacher 在蒸馏时仍可使用原有未来参考。两者监督在仿真 step 前一起采集，防止 autoreset 后错配。

VR 数据必须先完成操作者到 Mini3 的尺度、坐标和零位标定。不能直接把人体米制手部坐标当作 Mini3 的手部目标。这里包含目标接收和训练接口，不包含 PICO SDK、人体追踪或标定求解器。Mini3 没有灵巧手自由度，此接口也不控制手指。

## 三点目标配置

teleop 必须提供 `--sites /绝对路径/mini3-sites.json`。JSON 需包含 `head`、`left_hand`、`right_hand`，每项包含 `body`、`pos`（该 body 局部坐标中的物理跟踪点）和可选的 `quat_wxyz`（局部跟踪坐标系，默认单位旋转）。三个 body 名称会在当前训练 MJCF 中校验。

Mini3 没有 G1 的 wrist link，必须根据你们的手部安装位置填写前臂上的实际偏移。不要使用肘关节原点代替手掌。`check_control_cpu.py` 中的零偏移仅为测试坐标变换，不是部署标定值。标定信息会存入 checkpoint，推理不依赖原 JSON 的路径。

## 实时输入协议

Linux 手柄轴顺序和方向由 `--axes`、`--signs` 配置，目标生成与训练使用相同的速度／高度单位。

遥操可用 `--mode udp --port 5558`。接收器仅监听本机，发送端发布 JSON：

```json
{"interface":"command","sequence":1,"timestamp":1793865600.0,"target":[0.1,0.0,0.0,0.43]}
```

`timestamp` 必须使用发送时的 Unix 秒数；上面的值仅说明格式。`teleop` 包将接口改为 `teleop`，目标填入符合前述约定的 31 个浮点数。序号严格递增；默认目标超过 250 ms、断连、跌倒或回合超时就停止仿真，不自动生成“中立姿态”继续控制。发送端重启时也需重启接收器。

## 验证范围

已通过 21 项 CPU 单元测试和真实 Mini3 checkpoint 兼容性检查。CPU 单元测试使用合成小模型的数值更新检查梯度，包括穿过世界模型的 BPTT、冻结参数不变、动力学训练/验证分组及多步误差门限；不构成正式训练。真实资产检查只做 FK、policy/world-model 推理、原生观测布局对比和环境配置构造，优化器步数和仿真步数均为零。手柄响应、启停和转向、三点误差、跌倒率、延迟及丢包表现需在后续有界仿真测试中验收。
