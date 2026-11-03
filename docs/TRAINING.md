# cuboi 训练配方与恢复

目标是Mini3 v0.3.1的21关节机器人参考跟踪，seed0从零训练，50Hz，物理2ms。默认4096环境、WM batch2048×10、8步自回归；actor batch4096、H32；history10、future10；replay每环境1000步。Adam初始2e-4，actor beta=(0.9,0.95)，γ=.99、λ=.95；critic5epoch×4minibatch；norm clip1、actor逐元素clip.1；动作噪声std.1，观测/reset噪声和物理随机化开启。原生250k iteration余弦降至0.03倍，此后保持；两种iteration不同。

WM hidden[2048,2048,2048]，encoder/decoder[2048,1024,512,256]、latent32，critic[256,128]。Cuboi固定尺度、obs690/goal520、采样预热40；zita RunningNorm、obs750/goal570、bf16 MLP/FP32积分FKreward、采样预热64与500轮WM-only。以本分支JSON和源码为准。

实际数据入口是CPFS packed/train.pt。Cuboi记录配置中的默认npz_dir=data/npz/lafan1不是本次实际数据源。训练episode上限10秒，原生及统一评测覆盖完整动作。

每轮collect→replay→WM监督→冻结WM参数后BPTT actor→critic。梯度不穿过真实物理仿真。部署只保留actor ONNX，动作目标q_default+.25*action；不是q_ref+residual。世界模型32步想象为0.64秒，不是actor实际前视0.64秒。

## 新实验

先新run ID和--dry-run，再preflight/CPU FK，核对资源后启动。启动器以PYTHONPATH加载当前checkout；不会自动用CPFS旧源码目录代替。GPU动作会检查占用、使用独立账本，并仅向本次启动的进程/容器PID发终止信号。其他作业占用时拒绝启动；这不是集群资源预约，需要团队协调卡位。

每次train/resume单独有界。正式配方总iterations仍为320000，不能把它改为单段轮数，否则会改变学习率日程。--max-seconds限制本次占卡，run累计账本上限48h。两分钟正常checkpoint宽限包含在该上限中；如果CPFS保存未能在时限内完成，guard会终止本次作业，须先核对checkpoint有效性后恢复。不要更改历史预算文件来续用。

resume默认读取新run/training下的完整恢复文件（Cuboi ckpt_latest.pt；zita last.pt）。轻量candidate/ckpt_<iteration>.pt用于推理、原生评测和导出，不能当完整恢复文件。完整恢复包括optimizer/replay/RNG/history/sampler/计数和物理状态；修改后要重新验证恢复后optimizer继续更新与状态有限性。

每段后用dev316选模，固定完成率→progress→局部MAE→更早候选；不按最高return选。两个新run可以并行，但需分别使用空闲GPU、独立run ID，遵循团队同时卡数上限。当前仓库发布不会启动任何训练。

## 历史已完成实验

Cuboi run终点320000，交付同一候选；zita在221158预算结束，交付161203。实际计数和资源见evidence与LOGGING；配置语义不同，不能用nativeiteration直接比效率。完整训练日志仍在CPFS。单seed结果不证明多seed稳定性；Mini3平地成绩不直接证明G1或感知WBC能力。

每次启动记录当前Git提交、dirty状态、模块逐文件SHA及配置SHA。感知变体请另开feature分支和独立run；baseline的preflight严格核对冻结源码，修改后应明确记录并评审新的源身份，不静默冒充原baseline。

感知变体需先在feature分支评审源码差异，并有意更新 `provenance/native-source-sha256.json`（相对仓库根的模块文件名→SHA256），保留原baseline分支不变。新run的launch回执会记录实际源码与配置SHA；更新清单只表示变体身份，不能继承原baseline验收结论。

## 配置来源补充

recorded-formal.json记录终段启动参数；selected-checkpoint.json保存本次交付权重的内嵌配置。zita选中点的checkpoint_seconds为7200秒，终段因预算收口为6909.611607827246秒，launcher可视化字段也有差异；核心训练超参相同。新formal配方保持7200秒常规分段，不能把末段参数冒充选中点原始配置。
