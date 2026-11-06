# SuperMimic cuboi · Mini3 v0.3.1

本分支在已训练的 Mini3 v0.3.1 mimic 底座上，完善手柄命令和头部＋双手三点遥操的训练与仿真接口。基于 [cuboi 分支](https://github.com/ZerothRobot/supermimic/tree/cuboi)；另一套 mimic 实现见 [zita 分支](https://github.com/ZerothRobot/supermimic/tree/zita)，共同导航见 [main](https://github.com/ZerothRobot/supermimic/tree/main)。本分支不是上游 G1 默认示例。

我们在本分支完成的工作：

- 新增 `command` / `teleop` encoder：分别接收速度、转向和高度命令，以及头部、双手目标；复用并冻结原 mimic encoder 和 decoder。
- 新增在线蒸馏与 DAgger：用现有 mimic checkpoint 监督新接口，学生只接收当前目标和本体历史。
- 新增 SuperMimic 闭环微调：复用已有世界模型，通过可微想象 rollout 和 BPTT 优化新 encoder，并用真实仿真 replay 校正动力学；另保留 PPO 对照入口。
- 新增与自由控制匹配的奖励和跌倒条件，以及独立环境上的多步动力学误差检查；预测误差超限时暂停想象策略更新。
- 新增训练子集整理、Linux 手柄与 UDP 仿真输入、断连／数据过期处理，以及 checkpoint 保存与加载。

已通过 21 项 CPU 单元测试和真实 Mini3 policy / world-model checkpoint 兼容性检查，包括动力学到遥操奖励的动作梯度检查。**目前完成的是代码和 CPU 验证；没有启动新接口正式训练，没有发布已训练的手柄／遥操权重，GPU 仿真收敛、设备延迟和加速比仍待验证。**

**先读：[先导](docs/INTRODUCTION.md) · [环境](docs/ENVIRONMENT.md) · [训练](docs/TRAINING.md) · [日志](docs/LOGGING.md) · [CPFS数据与权重](docs/CPFS_ASSETS.md) · [权重与复现](docs/WEIGHTS.md) · [Contract链路](docs/CONTRACT_PIPELINE.md) · [本次验收范围](docs/VALIDATION.md)**

## 快速开始：复用已部署环境

### 手柄与遥操：获取本分支和复用环境

建议在新的 checkout 中操作，复用服务器上已有的 cuboi 环境。先选定空闲 GPU；下列 `--dry-run` 只打印计划，不启动训练。已克隆仓库时可先 `git fetch origin`，再切换到 `feat/mini3-command-teleop`；已在该分支时使用 `git pull --ff-only`。

```bash
git clone --branch feat/mini3-command-teleop --single-branch \
  https://github.com/ZerothRobot/supermimic.git \
  /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/checkouts/mini3-command-teleop
cd /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/checkouts/mini3-command-teleop
source /mnt/cpfs/workspace/yxq/supermimic/cuboi/.venv/bin/activate
export PYTHONPATH="$PWD/src"
```

已有环境路径仅适用于上述已部署服务器；其他机器按[环境说明](docs/ENVIRONMENT.md)配置。先做 CPU 检查：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_control.py -v
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python scripts/check_control_cpu.py \
  --checkpoint /mnt/cpfs/workspace/yxq/supermimic/experiments/mini3_v1/cuboi/formal-seed0-r1/candidate_0320000.pt \
  --library /mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/packed/pilot.pt
```

这些检查不创建仿真环境，也不启动正式训练；单元测试只对合成小模型做数值更新。

### 手柄与遥操：准备动作子集和目标配置

优先从已有 BONES-SEED Mini3 **train** 库挑选动作，不需要先下载新的 G1 库。command 第一轮使用站立、走路、后退、侧移、转向、启停等动作，排除依赖家具、接触物体和腾空的动作；teleop 再加入可行的伸手、弯腰等动作。

`clips.json` 是人工审查后的精确 clip ID 数组。以下命令生成独立子库，不修改原数据：

```bash
export PYTHONPATH="$PWD/src"
python scripts/prepare_control_library.py \
  --input /mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/packed/train.pt \
  --clips /绝对路径/clips.json \
  --output /新数据目录/command-train.pt
```

输出旁有 `.manifest.json`，训练入口会核对其中的精确 clip 顺序。文件名检查不能证明划分；需要依据原 `split-proof.json` 和训练 manifest 确认来源。同一动作的镜像/变体应留在同一划分。dev316 用于开发验证，Full-6000 不参与筛选、蒸馏或微调。

训练示例中的 `/绝对路径/…`、`/新数据目录/…` 和 `/新run目录/…` 需要换成实际路径。`candidate_0320000.pt` 可使用上面已验证的服务器绝对路径。每次训练选择新的输出目录。

遥操还需要 `mini3-sites.json`：配置 `head`、`left_hand`、`right_hand` 对应的 MJCF body、实际局部跟踪点 `pos` 和可选的 `quat_wxyz`。必须按 Mini3 的头部与手部安装位置标定，不能把肘关节原点直接当作手部目标。输入维度、坐标约定和配置字段见[接口参考](docs/CONTROL.md)。

### 手柄与遥操：蒸馏后用 SuperMimic 微调

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

SuperMimic 和 PPO 微调均使用独立采样的前后/侧向速度、转向和高度命令，包含 20% 零速度停止命令。默认范围是 ±0.3 m/s、±0.15 m/s、±0.5 rad/s、高度 0.38–0.48 m；它们只是起始配置，需按 Mini3 可行范围调整。teleop 的头/手目标来自参考 FK，同时速度命令独立采样，以训练上肢与移动的组合。任意新组合的物理可行性仍需评估。

奖励在物理 step 后、autoreset 前读取实际速度、高度、姿态，以及 teleop 跟踪点误差，另保留关节限位和动作变化惩罚。取消相对于原动作的终止条件和 clip-end 终止，改为绝对跌倒检查及有界回合超时。actor/critic 不消费参考 goal。PPO 将全部 reset（包括超时）作为有界回合目标的终态，GAE 不跨 reset bootstrap；SuperMimic 想象 rollout 使用前述平滑跌倒概率计算 survival 和 critic 目标。

各阶段都有时间预算、信号退出、独立日志和原子 `last.pt`。SuperMimic 输出会同时保存适配后的动力学、encoder 和 control critic；`--world-model-checkpoint` 也支持读取这样的控制 checkpoint。保存的 optimizer 用于诊断；目前没有完整 simulator/RNG 恢复入口，不应当作精确 resume。再次启动是新实验。冻结 decoder 能保留底座，但其表达能力也可能限制新任务，需要根据闭环结果决定是否解冻，不能预先保证收敛。

### 原 mimic 底座：保留复现流程

历史主机为海豚 zero-2；先确认所选卡空闲。zita的现有容器挂载覆盖shared/data，所以checkout推荐如下位置。不会自动创建/重启容器，也不会修改历史run。

```bash
git clone --branch cuboi --single-branch https://github.com/ZerothRobot/supermimic.git /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/checkouts/cuboi
cd /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/checkouts/cuboi
python3 scripts/run.py preflight
python3 scripts/run.py check-fk
python3 scripts/run.py train --run-id colleague-seed0-r1 --gpu 2 --dry-run
# 确认路径、空闲卡和预算后，启动一个有界训练段：
python3 scripts/run.py train --run-id colleague-seed0-r1 --gpu 2 --max-seconds 7800
```

`preflight`和`check-fk`隐藏CUDA，不做GPU仿真。`train`不是一次性自动跑满48小时：默认单次上限7800秒；zita约每7200秒保存退出，Cuboi由有界guard请求正常保存退出。每个新run有独立48 GPU小时账本，native-eval也计费；不会使用已耗尽的历史账本。

```bash
python3 scripts/run.py native-eval --run-id colleague-seed0-r1 --gpu 2 --max-seconds 900
python3 scripts/run.py resume --run-id colleague-seed0-r1 --gpu 2 --max-seconds 7800
```

在每段结束后评dev316、记录候选，再resume，直到原生320000 iteration或该新run的预算先到。Cuboi内部定时候选已有dev评估，native-eval可用于显式复查；zita需显式native-eval以更新selection。不要根据Full-6000追加训练或选模。

新的输出根为`/mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/runs/cuboi/<run-id>/`；原生输出在`training/`，完整checkpoint、CSV/TensorBoard和stdout都留CPFS。

## 导出与统一评测

以下导出与 contract 评测命令沿用原 mimic 流程，适用于原 mimic checkpoint。新增控制 checkpoint 使用后面的手柄／遥操仿真入口，尚未接入这套 ONNX contract 导出与统一评分。

```bash
python3 scripts/run.py export --run-id colleague-seed0-r1
python3 scripts/run.py verify-package --run-id colleague-seed0-r1 \
  --package /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/runs/cuboi/colleague-seed0-r1/policy-package
python3 scripts/run.py evaluate --run-id colleague-contract-smoke-r1 \
  --package /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/runs/cuboi/colleague-seed0-r1/policy-package --workers 2 \
  --clip baby_full_diaper_walk_ff_180_start_R_002__A460_M \
  --clip idle_crawl_stop_001__A214_M
```

导出默认读取本新run的selection.json；未选模时必须显式给`--checkpoint`。CPU导出执行64组source parity和完整contract检查。上面先做2条动作的小集合验收，保存逐帧轨迹；通过后用独立run ID、不传--clip，再做Full-6000。全量默认32 CPU workers、ORT单线程，使用已部署冻结evaluator。完整协议与实跑回执见Contract链路文档。不要先跑Full-6000再按结果改配方。

### 手柄与遥操：仿真接口检查

参考目标回放只用于接口诊断：

```bash
python scripts/run_control.py \
  --checkpoint /新run目录/command-supermimic/last.pt \
  --library /开发数据目录/small-dev.pt \
  --mode reference --steps 500
```

Linux 手柄模式：增加 `--mode gamepad --joystick /dev/input/js0 --height 实际骨盆高度`。`--axes` 和 `--signs` 分别配置 forward/left/yaw 对应轴及方向，默认轴 1/0/3、方向均为 -1。手柄限幅采用微调 checkpoint 中保存的命令范围。此入口是有界无可视化仿真，不发送真机电机命令。

遥操可用 `--mode udp --port 5558`。接收器仅监听本机，发送端发布 JSON：

```json
{"interface":"command","sequence":1,"timestamp":1793865600.0,"target":[0.1,0.0,0.0,0.43]}
```

`timestamp` 必须使用发送时的 Unix 秒数；上面的值仅说明格式。`teleop` 包将接口改为 `teleop`，目标填入符合前述约定的 31 个浮点数。序号严格递增；默认目标超过 250 ms、断连、跌倒或回合超时就停止仿真，不自动生成“中立姿态”继续控制。发送端重启时也需重启接收器。

## 源码、环境与外部依赖

`configs/recorded-formal.json`是实际run保存配置（可能含历史resume路径），仅用于审阅；`configs/formal.json`用于新启动，resume已清空。`provenance/native-source-sha256.json`记录2026-11-03历史模块源码的身份；本分支的控制接口及共享观测布局修改属于后续开发，不在该历史源码一致性声明内。原上游README见docs/UPSTREAM_README.md，许可证保留。Cuboi原导出tree含AppleDouble元数据，仓库移除这些非代码文件，另外保留原导出tree与干净代码tree两个哈希。

Git历史不包含二进制大文件；两份选中的checkpoint和完整ONNX包位于本仓库[Release](https://github.com/ZerothRobot/supermimic/releases/tag/mini3-seed0-v1)，下载脚本、SHA256和复现范围见[权重与复现](docs/WEIGHTS.md)。数据、机器人资产、完整恢复文件、完整日志、容器和缓存仍在CPFS。当前native Mini3适配固定从`/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/control/control.json`读取本体；更换资产根/机器人必须显式适配并重做FK和接口验收。

`configs/selected-checkpoint.json`另外保留交付checkpoint自己的配置；`recorded-formal.json`来自训练终段。zita在保存间隔和部分launcher字段上有差异，核心训练超参一致，详见权重复核回执。

仓库与Release已于2026-11-04设为公开，可直接阅读、克隆和匿名下载。协作推送代码仍需仓库Write权限。下载脚本默认不读取凭据；仅在需要私有访问或提高API额度时使用--authenticate。
