# ONNX 与统一 Contract 评测链路

2026-11-03 实际检查。两套 Mini3 复现都沿用我们内部的 `robot-policy-onnx` v1.0 协议，通过 `adapter=supermimic` 交给冻结的统一 evaluator。机器可读回执见 [链路检查结果](../evidence/contract-pipeline-check-20261103.json)。

## 流程与交付边界

选中的 checkpoint → 原生环境中的 FP32 actor 导出 → `policy.onnx`＋`onnx_contract.yaml`＋`tests/` → 独立 consumer 验包 → CPU MuJoCo 闭环评测。

交付时复制整个 package 目录，不只复制 ONNX。contract 声明图、特征生产器与布局、时间窗口、关节顺序、动作变换、机器人兼容性、文件哈希、interface ID、package ID 和 golden tests。源端 parity JSON 与 raw-to-feature 测试向量在包的同级目录，供额外审计；运行期只需要完整 package。它们在CPFS保留，同时在本仓库Release提供可校验副本，见[权重与复现](WEIGHTS.md)。

世界模型、critic、optimizer 和 replay 不进入部署图。encoder/decoder、actor 及推理所需的归一化保留在单个 FP32、静态 batch=1 的图中。zita 在导出前关闭 bf16 autocast，保留训练得到的 RunningNorm 状态。

|项目|Cuboi|zita|
|---|---|---|
|交付 checkpoint iteration|320000|161203（训练终点221158）|
|`obs_hist`|`[1,690]`，按特征分组堆10帧|`[1,750]`，按10个时间步堆叠，含参考相对朝向|
|`goal`|`[1,520]`，10×52|`[1,570]`，10×57|
|观测历史／参考未来|10帧／未来第1–10帧|10帧／未来第1–10帧|
|控制频率|50Hz|50Hz|
|ONNX 输出|`raw_action[1,21]`|`raw_action[1,21]`|
|逻辑输出|`q_default + 0.25 * raw_action`，rad|相同|
|原生导出 MuJoCo|3.16.0|3.15.0|
|统一评测 MuJoCo／ORT|3.12.0／1.29.0，CPU|相同|

两套图的 interface ID 不同，这是输入语义与维数不同的正常结果；它们使用同一个 contract 标准和 evaluator。动作目标不是 `q_reference + residual`。参考末端处理也不同：Cuboi clamp 最后一帧及速度；zita 对超出末端的请求保持位姿并将速度置零，均写进对应 extension。

## 本次发现与修复

新整理的 Cuboi 启动器曾把评测用 MuJoCo 3.12 放到导出进程的 PYTHONPATH 前面，导入原生 mjlab／MuJoCo-Warp 3.16 时报 `mjINT_DISCRETE` 缺失。这是新启动入口的环境冲突，历史正式包与成绩没有变更。

已修正环境优先级：导出沿用各自原生环境，只在 site-packages 之后补充缺少的 ONNX／ORT／contract 工具；`verify-package` 与 `evaluate` 使用冻结评测环境，不加载训练框架。没有修改 native 模块、训练配置或权重。

## 已完成的检查

- 从两个分支源码重新加载历史选中 checkpoint，严格核对 state_dict，执行各64步 FP32 source／ONNX parity。包含 reset、连续动作历史、转向和参考末端窗口。
- 输入 producer 最大绝对误差：Cuboi `1.19e-7`，zita `2.38e-7`；raw action 误差：`1.61e-6`／`3.52e-6`；关节目标误差：`4.02e-7`／`8.79e-7` rad。满足既有 atol=rtol=`1e-5`。
- 两份重新导出的 ONNX SHA256 **分别与历史正式包完全相同**。schema、semantic、artifact／metadata hash、canonical interface/package ID、ONNX 和 golden tests 通过。
- 在干净 consumer 中再次核验64步 raw-to-feature 向量；未导入 torch 或 native supermimic。
- 每套使用相同4条完整动作做 CPU 闭环：行走、舞蹈、4187帧长动作、已知失败 `idle_crawl_stop_001__A214_M`。每套均3条完成、1条真实失败；与历史逐动作终态、失败帧和指标一致。失败分别为247／251个控制步，保留其真实结果。
- 逐帧轨迹保存 qpos、qvel、参考 qpos、raw action、joint target、时间；检查有限性、50Hz 时钟、动作变换、有效帧数及哈希，使用独立 MjData 从保存轨迹重新计算 Fresh FK 指标。

历史 Full-6000 每套6000个唯一终态已只读核对：Cuboi 5964完成／36失败，zita 5965／35。本次复查8条模型×动作任务，不重跑12000条全量评分、不用结果重新选模，也不启动训练或占GPU。局部指标沿用冻结 Mini3 的 pelvis XYZ／关键点方向 yaw 对齐，不能混用后续 G1 协议的定义。

## 运行方法

先按实现分支 README 在服务器 checkout；zita 的导出需在章鱼已有容器挂载范围内运行。以下以 Cuboi 的历史选中权重为例，使用新的 run ID：

```bash
python3 scripts/run.py export --run-id contract-check-r1 \
  --checkpoint /mnt/cpfs/workspace/yxq/supermimic/experiments/mini3_v1/cuboi/formal-seed0-r1/candidate_0320000.pt

package_dir=/mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/runs/cuboi/contract-check-r1/policy-package
python3 scripts/run.py verify-package --run-id contract-check-r1 --package "$package_dir"

# 小集合接口验证：完整动作，不截短时间，保存选定动作的逐帧轨迹。
python3 scripts/run.py evaluate --run-id contract-smoke-r1 --package "$package_dir" --workers 2 \
  --clip baby_full_diaper_walk_ff_180_start_R_002__A460_M \
  --clip idle_crawl_stop_001__A214_M

# 配方/模型冻结后再做全量正式评分；使用独立输出目录。
python3 scripts/run.py evaluate --run-id contract-full-r1 --package "$package_dir" --workers 32
```

zita 分支使用 `.../zita/formal-seed0-r1/ckpt_161203.pt`，包路径中的实现名相应改为 zita。未显式指定 checkpoint 时，export 仍从该新 run 的 `training/selection.json` 取选模结果，不能以 `last.pt` 默认替代交付候选。

`verify-package` 只读验包；若同级有 source-parity JSON，会额外验证 producer 向量，否则明确标记该项未提供。`evaluate --clip` 只运行精确指定的动作，回执 `full6000=false`；不传 `--clip` 才检查6000个唯一终态。重复 clip ID、技术错误、任务缺失都不能验收为通过。每次评测使用新 run ID，已有 evaluation 目录会拒绝覆盖。

## CPFS 依赖与本次证据

固定根：`/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/`。

- 原交付包：`packages/final-cuboi-mini3/`、`packages/final-zita-mini3/`。
- evaluator：`evaluator-frozen-v1/evaluator/src/`，本次身份 `c4bb87cb57c99b5906393ea30b2f7a9c2ecaa2bbf99bbc5cca63bd20cdd1df72`。
- 通用 contract 校验器／schema：`contract-tooling/`；统一评测依赖：`runtime/export-runtime/`、`runtime/contracts/`。
- 本次新导出包与轨迹：`/mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/contract-check-20261103T172535/runs/<cuboi|zita>/contract-audit-r2/`。
- 独立轨迹复算脚本与回执：上述 audit 根的 `audit_saved_traces.py`、`trace-audit.json`。

本仓库引用现有 CPFS evaluator 和资产，并未把完整 evaluator 发布成另一个安装包；离开这些挂载路径要显式重建依赖和资产。历史完整日志／权重不进入 Git。

## 后续增加感知输入

当前 contract 的严格布局只支持这两套 Mini3 动作跟踪 actor。新增相机、视觉历史、场景参考或网络结构时，要声明相应输入、时间戳／坐标／单位、producer 与兼容版本，并同步 consumer；不能补零假装原接口兼容。应重新做 source parity、golden tests、物理闭环与任务输入干预检查，再与 blind baseline 比较。

## 发布前进一步加固

导出／验包／评测现在逐文件验证冻结evaluator的61个文件及contract工具/schema的5个文件。evaluate入口自身强制执行完整验包，不能跳过canonical ID、文件哈希或golden tests。GitHub附件下载回读后已重新导出并回放；结果和配置差异见[发布复核回执](../evidence/github-release-audit-20261103.json)。
