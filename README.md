# SuperMimic zita · Mini3 v0.3.1

这是我们实际训练使用的Mini3适配源码快照；对应另一实现见[cuboi分支](https://github.com/ZerothOpenBridge/supermimic/tree/cuboi)，共同导航见[main](https://github.com/ZerothOpenBridge/supermimic/tree/main)。本分支不是上游G1默认示例。

**先读：[先导](docs/INTRODUCTION.md) · [环境](docs/ENVIRONMENT.md) · [训练](docs/TRAINING.md) · [日志](docs/LOGGING.md) · [CPFS数据与权重](docs/CPFS_ASSETS.md) · [权重与复现](docs/WEIGHTS.md) · [Contract链路](docs/CONTRACT_PIPELINE.md) · [本次验收范围](docs/VALIDATION.md)**

## 快速开始：复用已部署环境

历史主机为章鱼 zhangyu；先确认所选卡空闲。zita的现有容器挂载覆盖shared/data，所以checkout推荐如下位置。不会自动创建/重启容器，也不会修改历史run。

```bash
git clone --branch zita --single-branch https://github.com/ZerothOpenBridge/supermimic.git /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/checkouts/zita
cd /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/checkouts/zita
python3 scripts/run.py preflight
python3 scripts/run.py check-fk
python3 scripts/run.py train --run-id colleague-seed0-r1 --gpu 0 --dry-run
# 确认路径、空闲卡和预算后，启动一个有界训练段：
python3 scripts/run.py train --run-id colleague-seed0-r1 --gpu 0 --max-seconds 7800
```

`preflight`和`check-fk`隐藏CUDA，不做GPU仿真。`train`不是一次性自动跑满48小时：默认单次上限7800秒；zita约每7200秒保存退出，Cuboi由有界guard请求正常保存退出。每个新run有独立48 GPU小时账本，native-eval也计费；不会使用已耗尽的历史账本。

```bash
python3 scripts/run.py native-eval --run-id colleague-seed0-r1 --gpu 0 --max-seconds 900
python3 scripts/run.py resume --run-id colleague-seed0-r1 --gpu 0 --max-seconds 7800
```

在每段结束后评dev316、记录候选，再resume，直到原生320000 iteration或该新run的预算先到。Cuboi内部定时候选已有dev评估，native-eval可用于显式复查；zita需显式native-eval以更新selection。不要根据Full-6000追加训练或选模。

新的输出根为`/mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/runs/zita/<run-id>/`；原生输出在`training/`，完整checkpoint、CSV/TensorBoard和stdout都留CPFS。

## 导出与统一评测

```bash
python3 scripts/run.py export --run-id colleague-seed0-r1
python3 scripts/run.py verify-package --run-id colleague-seed0-r1 \
  --package /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/runs/zita/colleague-seed0-r1/policy-package
python3 scripts/run.py evaluate --run-id colleague-contract-smoke-r1 \
  --package /mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/runs/zita/colleague-seed0-r1/policy-package --workers 2 \
  --clip baby_full_diaper_walk_ff_180_start_R_002__A460_M \
  --clip idle_crawl_stop_001__A214_M
```

导出默认读取本新run的selection.json；未选模时必须显式给`--checkpoint`。CPU导出执行64组source parity和完整contract检查。上面先做2条动作的小集合验收，保存逐帧轨迹；通过后用独立run ID、不传--clip，再做Full-6000。全量默认32 CPU workers、ORT单线程，使用已部署冻结evaluator。完整协议与实跑回执见Contract链路文档。不要先跑Full-6000再按结果改配方。

## 源码、环境与外部依赖

`configs/recorded-formal.json`是实际run保存配置（可能含历史resume路径），仅用于审阅；`configs/formal.json`用于新启动，resume已清空。`provenance/native-source-sha256.json`记录与2026-11-03服务器逐文件一致的模块源码。原上游README见docs/UPSTREAM_README.md，许可证保留。Cuboi原导出tree含AppleDouble元数据，仓库移除这些非代码文件，另外保留原导出tree与干净代码tree两个哈希。

Git历史不包含二进制大文件；两份选中的checkpoint和完整ONNX包位于本仓库[Release](https://github.com/ZerothOpenBridge/supermimic/releases/tag/mini3-seed0-v1)，下载脚本、SHA256和复现范围见[权重与复现](docs/WEIGHTS.md)。数据、机器人资产、完整恢复文件、完整日志、容器和缓存仍在CPFS。当前native Mini3适配固定从`/mnt/cpfs/workspace/yxq/supermimic/shared/data/mini3_v1/control/control.json`读取本体；更换资产根/机器人必须显式适配并重做FK和接口验收。

`configs/selected-checkpoint.json`另外保留交付checkpoint自己的配置；`recorded-formal.json`来自训练终段。zita在保存间隔和部分launcher字段上有差异，核心训练超参一致，详见权重复核回执。

仓库与Release已于2026-11-04设为公开，可直接阅读、克隆和匿名下载。协作推送代码仍需仓库Write权限。下载脚本默认不读取凭据；仅在需要私有访问或提高API额度时使用--authenticate。
