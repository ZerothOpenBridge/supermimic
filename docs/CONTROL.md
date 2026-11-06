# Mini3 手柄命令与三点遥操

本开发分支基于 cuboi。新增接口复用已有 Mini3 mimic checkpoint，保留原 motion encoder 和 decoder，分别训练 command / teleop encoder。两个接口目前各保存一份模型；不是 NVIDIA SONIC 的 FSQ token，也不能直接加载 G1 SONIC encoder。zita checkpoint 的适配不在此实现中。

推荐路线为两个阶段：`train_control.py` 用已有 motion teacher 做在线蒸馏和 DAgger；`finetune_control_wm.py` 继续用 SuperMimic 世界模型和 BPTT 做闭环微调。`finetune_control.py` 也提供 PPO 入口，供后续对照。代码和 CPU 检查通过不表示已经获得可用的手柄或遥操权重。此次没有启动正式训练，也没有验证 GPU 仿真收敛、设备延迟或真机能力。

## 输入约定

所有长度为米、角度为弧度。坐标轴 X 前、Y 左、Z 上，四元数 WXYZ。

- `command` 为 4 维：`vx, vy, yaw_rate, pelvis_height`。速度在机器人 heading frame，左转角速度为正。
- `teleop` 为 31 维：上述 4 维，加 head / left_hand / right_hand 各 9 维。每个目标包含相对骨盆原点、在 heading frame 表达的位置 3 维，以及相对 heading 的旋转矩阵前两列 6 维，按行展开：`r00,r01,r10,r11,r20,r21`。
- student encoder 接收当前目标和与原 checkpoint 一致的 term-major 本体历史。没有参考相位、未来关节轨迹或未来人体姿态输入。
- teacher 在蒸馏时仍可使用原有未来参考。两者监督在仿真 step 前一起采集，防止 autoreset 后错配。

VR 数据必须先完成操作者到 Mini3 的尺度、坐标和零位标定。不能直接把人体米制手部坐标当作 Mini3 的手部目标。这里包含目标接收和训练接口，不包含 PICO SDK、人体追踪或标定求解器。Mini3 没有灵巧手自由度，此接口也不控制手指。

## 动作子集

优先从已有 BONES-SEED Mini3 **train** 库挑选动作，不需要先下载新的 G1 库。command 第一轮使用站立、走路、后退、侧移、转向、启停等动作，排除依赖家具、接触物体和腾空的动作；teleop 再加入可行的伸手、弯腰等动作。

`clips.json` 是人工审查后的精确 clip ID 数组。以下命令生成独立子库，不修改原数据：

```bash
export PYTHONPATH="$PWD/src"
python scripts/prepare_control_library.py \
  --input /mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/packed/train.pt \
  --clips /绝对路径/clips.json \
  --output /新数据目录/command-train.pt
```

输出旁有 `.manifest.json`，训练入口会核对其中的精确 clip 顺序。文件名检查不能证明划分；需要依据原 `split-proof.json` 和训练 manifest 确认来源。同一动作的镜像/变体应留在同一划分。dev316 用于开发验证，Full-6000 不参与筛选、蒸馏或 PPO。

## 三点目标配置

teleop 必须提供 `--sites /绝对路径/mini3-sites.json`。JSON 需包含 `head`、`left_hand`、`right_hand`，每项包含 `body`、`pos`（该 body 局部坐标中的物理跟踪点）和可选的 `quat_wxyz`（局部跟踪坐标系，默认单位旋转）。三个 body 名称会在当前训练 MJCF 中校验。

Mini3 没有 G1 的 wrist link，必须根据你们的手部安装位置填写前臂上的实际偏移。不要使用肘关节原点代替手掌。`check_control_cpu.py` 中的零偏移仅为测试坐标变换，不是部署标定值。标定信息会存入 checkpoint，推理不依赖原 JSON 的路径。

## 服务器训练入口

在 cuboi 已配置的 mjlab 环境中执行。先运行下面的 `--dry-run`；它只打印计划，不导入 torch、不创建输出、不申请 GPU。删除该选项才会启动训练。每次选择新的输出目录，并自行选定空闲 GPU。

```bash
export PYTHONPATH="$PWD/src"
python scripts/train_control.py \
  --checkpoint /绝对路径/candidate_0320000.pt \
  --library /新数据目录/command-train.pt \
  --interface command \
  --output /新run目录/command-distill \
  --num-envs 256 --steps 10000 --max-seconds 3600 \
  --dry-run
```

teleop 将 `--interface` 改为 `teleop`，使用审查后的对应训练子库，并增加 `--sites`。默认逐渐减少 teacher 执行动作的概率，最后由学生执行；所有学生状态上的监督均由原 motion teacher 提供。原 teacher、decoder 和原 checkpoint 不更新，新 encoder 单独优化。

蒸馏完成后，推荐在新的 run 中继续用 SuperMimic：

```bash
python scripts/finetune_control_wm.py \
  --checkpoint /新run目录/command-distill/last.pt \
  --world-model-checkpoint /绝对路径/candidate_0320000.pt \
  --library /新数据目录/command-train.pt \
  --output /新run目录/command-supermimic \
  --num-envs 256 --batch-size 512 --horizon 16 \
  --steps 10000 --max-seconds 3600 \
  --dry-run
```

此路线复用原 Mini3 checkpoint 中的 `wm` 动力学参数。循环为：真实仿真采样 → replay → 多步动力学拟合 → 未参与拟合的环境数据验证 → 冻结动力学、通过想象 rollout 对 encoder 做 BPTT → 更新 control critic。梯度可穿过固定 decoder、动力学与 FK 奖励；原 mimic 权重不更新。命令由独立采样生成，teleop 当前目标来自可行参考位姿，actor 只在对应时刻接收当前目标，没有未来参考输入。

原 termination head 学的是相对参考动作的偏差，这里不使用。真实仿真采用绝对高度/直立性跌倒检查；想象中的 survival 使用相同物理边界的平滑近似。控制奖励复用同一个可微函数，并与原任务的动作变化和关节限位惩罚一致。此实现更新原动力学，不重新训练旧参考终止头。

环境 ID 0、8、16……的 replay 窗口保留给动力学验证，不参与动力学拟合。验证使用与想象 horizon 等长的多步误差；任一步 normalized MSE 超过 `--wm-max-nse`（默认 0.1，尚未校准的起始值）或非有限时，跳过 actor 更新并继续采样和校正动力学。没有足够长的同回合验证窗口时也不做想象更新。这个检查用于暴露模型误差，不能保证策略不会利用模型缺陷。

默认 `--compile default` 对想象中的动力学/奖励/观测单步启用 `torch.compile`；先做 GPU 接口检查时可用 `--compile none` 排除编译影响。CPU 单元测试使用未编译的小模型。时间预算是循环的软上限，首次 CUDA/编译初始化及单次更新可能越过检查点，正式服务器作业仍应放在既有资源 guard 中。样本效率和墙钟加速比均需后续同预算测量，不能直接套用原 mimic 实验结果。

如需 PPO 对照，入口保持为：

```bash
python scripts/finetune_control.py \
  --checkpoint /新run目录/command-distill/last.pt \
  --library /新数据目录/command-train.pt \
  --output /新run目录/command-ppo \
  --num-envs 256 --iterations 10000 --max-seconds 3600 \
  --dry-run
```

PPO 使用独立采样的前后/侧向速度、转向和高度命令，包含 20% 零速度停止命令。默认范围是 ±0.3 m/s、±0.15 m/s、±0.5 rad/s、高度 0.38–0.48 m；它们只是起始配置，需按 Mini3 可行范围调整。teleop 的头/手目标来自参考 FK，同时速度命令独立采样，以训练上肢与移动的组合。任意新组合的物理可行性仍需评估。

奖励在物理 step 后、autoreset 前读取实际速度、高度、姿态，以及 teleop 跟踪点误差，另保留关节限位和动作变化惩罚。取消相对于原动作的终止条件和 clip-end 终止，改为绝对跌倒检查及有界回合超时。actor/critic 不消费参考 goal；全部 reset（包括超时）作为本实现有界回合目标的终态，GAE 不跨 reset bootstrap。

各阶段都有时间预算、信号退出、独立日志和原子 `last.pt`。SuperMimic 输出会同时保存适配后的动力学、encoder 和 control critic；`--world-model-checkpoint` 也支持读取这样的控制 checkpoint。保存的 optimizer 用于诊断；目前没有完整 simulator/RNG 恢复入口，不应当作精确 resume。再次启动是新实验。冻结 decoder 能保留底座，但其表达能力也可能限制新任务，需要根据闭环结果决定是否解冻，不能预先保证收敛。

## 仿真接口测试

参考目标回放只用于接口诊断：

```bash
python scripts/run_control.py \
  --checkpoint /新run目录/command-ppo/last.pt \
  --library /开发数据目录/small-dev.pt \
  --mode reference --steps 500
```

Linux 手柄模式：增加 `--mode gamepad --joystick /dev/input/js0 --height 实际骨盆高度`。`--axes` 和 `--signs` 分别配置 forward/left/yaw 对应轴及方向，默认轴 1/0/3、方向均为 -1。手柄限幅采用 PPO checkpoint 中保存的命令范围。此入口是有界无可视化仿真，不发送真机电机命令。

遥操可用 `--mode udp --port 5558`。接收器仅监听本机，发送端发布 JSON：

```json
{"interface":"command","sequence":1,"timestamp":1793865600.0,"target":[0.1,0.0,0.0,0.43]}
```

`timestamp` 必须使用发送时的 Unix 秒数；上面的值仅说明格式。`teleop` 包将接口改为 `teleop`，目标填入符合前述约定的 31 个浮点数。序号严格递增；默认目标超过 250 ms、断连、跌倒或回合超时就停止仿真，不自动生成“中立姿态”继续控制。发送端重启时也需重启接收器。

## 验证范围

无需训练即可运行 CPU 单元测试：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m unittest discover -s tests -p test_control.py -v
```

真实资产/checkpoint 只读兼容性测试：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/check_control_cpu.py \
    --checkpoint /绝对路径/candidate_0320000.pt \
    --library /mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/packed/pilot.pt
```

CPU 单元测试使用合成小模型的数值更新检查梯度，包括穿过世界模型的 BPTT、冻结参数不变、动力学训练/验证分组及多步误差门限；不构成正式训练。真实资产检查只做 FK、policy/world-model 推理、原生观测布局对比和环境配置构造，优化器步数和仿真步数均为零。手柄响应、启停和转向、三点误差、跌倒率、延迟及丢包表现需在后续有界仿真测试中验收。
