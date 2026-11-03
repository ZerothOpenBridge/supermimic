# CPFS路径与文件身份

固定根：`/mnt/cpfs/workspace/yxq/supermimic`。机器可读清单见configs/cpfs-assets.json；机器人资产SHA见provenance/robot-asset-sha256.json。

|内容|路径（相对固定根）|
|---|---|
|共享训练数据|shared/data/mini3_v1/packed/train.pt（119925条，约4.89GB）|
|开发集与试训|同目录dev316.pt、pilot.pt|
|原划分与逐动作来源|同目录split-proof.json、train.manifest.json、dev316.manifest.json|
|原Mini3 NPZ|mini3_preparation/README.md记录的源；已有完整共享副本见shared/data/mini3_v1/raw|
|控制、MJCF、网格|shared/data/mini3_v1/control/control.json；control/assets/mini3_v031/；control/mini3.xml|
|zita转换资产|shared/data/mini3_v1/control/usd/mini3/mini3.usda及相邻USD依赖|
|Cuboi交付权重|experiments/mini3_v1/cuboi/formal-seed0-r1/candidate_0320000.pt|
|Cuboi完整恢复|同目录ckpt_latest.pt|
|zita交付权重|experiments/mini3_v1/zita/formal-seed0-r1/ckpt_161203.pt|
|zita完整恢复|同目录last.pt（终点221158，非交付候选）|
|现成ONNX包|shared/data/mini3_v1/packages/final-cuboi-mini3/和final-zita-mini3/|
|统一执行层|shared/data/mini3_v1/evaluator-frozen-v1/evaluator/src/|
|导出contract工具|shared/data/mini3_v1/contract-tooling/|
|冻结benchmark|shared/data/mini3_v1/benchmark/|
|正式结果与评分trace|experiments/mini3_v1/cohort-full-v1/<model>/|
|本机完整汇总的服务器来源|experiments/mini3_v1/及上述cohort目录|

大型文件不使用Git LFS指针；选中的轻量checkpoint和完整ONNX包另在本仓库Release分发，可用scripts/fetch_weights.py按清单下载校验，见WEIGHTS.md。训练数据和完整恢复文件仍只引用CPFS。preflight检查存在性、关键大小、控制文件与源码哈希，不把预检变成每次重读5GB训练数据；完整训练数据SHA已记录，可按需sha256sum校验。不要对资产或数据原地转换；需要变体时另建目录并重新记录身份。

数据格式是FP16压缩的55维状态：p3、wxyz四元数4、joint21、世界线速度3、世界角速度3、joint速度21，50Hz；训练FK/奖励按各实现恢复到其原生精度。不得直接把G1数据裁剪成Mini3，也不得把full6000用作训练。

仓库内所有provenance/historical-tools脚本仅供审计，包含历史绝对路径；不要拿它们对旧run执行。日常入口只用scripts/run.py，新产物写collaboration/runs。
