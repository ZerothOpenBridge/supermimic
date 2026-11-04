# Mini3 权重与复现

模型发布在本公开仓库的 [mini3-seed0-v1 Release](https://github.com/ZerothOpenBridge/supermimic/releases/tag/mini3-seed0-v1)。公开阅读和下载无需邀请或登录。Git 分支保存源码与清单，二进制附件保存于同一仓库的 Release，普通 `git clone` 不会自动下载权重。

## 发布内容

|实现|选中 checkpoint|实际训练终点|checkpoint 大小|ONNX 包大小|
|---|---:|---:|---:|---:|
|Cuboi|320000|320000|67,925,715 B|30,113,601 B|
|zita|161203|221158|68,102,341 B|31,016,487 B|

每套发布两个附件：PyTorch 轻量 checkpoint、完整 ONNX contract 压缩包。另附 `model-manifest.json` 和 `SHA256SUMS`。机器可读清单位于 [artifacts/weights-manifest.json](../artifacts/weights-manifest.json)，声明文件名、大小、SHA256、源码 tag、checkpoint 迭代、ONNX SHA、interface ID 和 package ID。

这两份 checkpoint 对应已报告的 dev316 选模结果；zita 不是训练终点221158权重。Cuboi 文件包含 policy／WM／critic，zita 包含 pi／WM 等轻量状态；均没有 optimizer 或 replay，不能用作完整恢复。完整恢复文件仍在 CPFS：`experiments/mini3_v1/cuboi/formal-seed0-r1/ckpt_latest.pt`、`experiments/mini3_v1/zita/formal-seed0-r1/last.pt`。从零训练不加载这些已训权重。

## 下载与校验

下载脚本只使用 Python 标准库，支持 Python≥3.9。默认匿名下载，不读取本机凭据。仅在私有访问、草稿验收或提高GitHub API额度时添加 `--authenticate`，此时读取已有 `gh` 登录或 Git HTTPS credential helper；无需把token写入命令、配置或仓库。

在对应分支／tag checkout 根目录，以 Cuboi 为例：

```bash
weights_dir=/mnt/cpfs/workspace/yxq/supermimic/shared/data/collaboration/downloads/mini3-seed0-v1
python3 scripts/fetch_weights.py --method cuboi --output "$weights_dir"
python3 scripts/fetch_weights.py --method cuboi --output "$weights_dir" --verify-only

tar -xzf "$weights_dir/cuboi-mini3-seed0-onnx.tar.gz" -C "$weights_dir"
package_dir="$weights_dir/cuboi-mini3-seed0/policy-package"
python3 scripts/run.py verify-package --run-id release-package-check --package "$package_dir"
python3 scripts/run.py evaluate --run-id release-package-smoke --package "$package_dir" --workers 1 \
  --clip baby_full_diaper_walk_ff_180_start_R_002__A460_M \
  --clip idle_crawl_stop_001__A214_M
```

zita 将 `--method`、附件名和目录名的 cuboi 改为 zita；其验包／统一评测也用同一 CPU runtime。`--method all` 下载两套，`--kind checkpoint|onnx` 可仅取一种，`--dry-run` 查看计划。现有文件若哈希不符会拒绝覆盖。`--verify-only` 完全离线，不请求 GitHub。

从下载的 checkpoint 重新导出（Cuboi）：

```bash
python3 scripts/run.py export --run-id release-export-r1 \
  --checkpoint "$weights_dir/cuboi-mini3-seed0-iter320000.pt"
```

zita 在其分支及章鱼容器挂载范围运行，checkpoint 改为 `zita-mini3-seed0-iter161203.pt`。输出遵循独立新run目录，不覆盖原模型。每次运行请使用新的 run ID；路径含相应实现名。

## 源码固定与复现范围

源码版本固定为 `mini3-seed0-v1-cuboi`、`mini3-seed0-v1-zita`，也可用相应实现分支继续开发。contract 的 provenance.source_commit 指上游提交；Mini3 适配身份以 source_variant_tree_sha256、本仓库实现 tag 和配置清单共同确定。原始v1 tag保留发布时的下载脚本认证行为；需要匿名下载请使用当前main／cuboi／zita分支的 `scripts/fetch_weights.py`，或直接在公开Release页下载附件。模型字节、SHA256和原始源码tag不变。Release 自动生成的默认 Source code ZIP 基于 main 导航分支；复现实验请克隆实现 tag：

```bash
git clone --branch mini3-seed0-v1-cuboi --single-branch https://github.com/ZerothOpenBridge/supermimic.git cuboi-mini3
git clone --branch mini3-seed0-v1-zita --single-branch https://github.com/ZerothOpenBridge/supermimic.git zita-mini3
```

完整 ONNX 包含 `policy.onnx`、`onnx_contract.yaml`、`tests/`；其同级保留 source parity 报告和 raw-to-feature 向量。消费端只需 ONNX runtime、冻结 evaluator 和机器人／动作资产，不导入训练框架。

运行前会逐文件核对冻结 evaluator 的61个文件及 contract 工具／schema 的5个文件。统一评测入口自身强制执行 schema、canonical IDs、文件／模型元数据哈希、ONNX 与 golden tests，不依赖调用者事先手动验包。它们是我们内部 `robot-policy-onnx` v1.0 协议及 SuperMimic 扩展，不意味着任意 ONNX 工具都能直接消费。

已验证范围是 Linux 服务器复用 CPFS 环境：锁定源码、选中 checkpoint 加载、FP32 导出、contract 验包和同动作 CPU 闭环。独立环境重建的锁文件和配方已提供，但本次没有重建全套环境或重跑从零训练；Ubuntu基础镜像标签及系统包不是逐比特固定。数据、Mini3资产、冻结 evaluator、原生环境／容器和完整日志仍需访问 CPFS。不能将这些检查描述为任意新机器的一键完整复现。

## 模型适用范围与实验口径

两套均为 Mini3 v0.3.1、21关节、50Hz机器人参考跟踪策略，seed0从随机初始化训练。原生源码保留了G1Kinematics等上游命名及部分历史注释，实际本体以Mini3 spec、control.json、21维输出和绑定资产为准；不能据旧名称将本权重当作G1权重。使用119925条训练动作、316条开发动作选模；Full-6000不参与梯度更新或选模。Cuboi曾做过一次中途Full-6000诊断，因此不称全程密封盲测。单seed、平地仿真结果不直接证明感知控制、G1或真机能力。

冻结统一Full-6000结果：Cuboi完成5964／6000、Fresh局部欧氏MPJPE约30.97mm；zita完成5965／6000、约28.03mm。指标按实际有效帧统计，失败保留；完整设置见 [Contract链路](CONTRACT_PIPELINE.md)。两套输入、表示和训练预算不同，不能把这一差异解释成严格的单变量算法提升。

checkpoint及部署包属于本项目内部实验产物，公开分发不改变上游源码许可证或机器人／数据原来源的使用范围。许可证及来源见仓库根的 LICENSE／LICENSES 和 THIRD_PARTY_NOTICES.md。

## 发布复核记录

[GitHub发布复核回执](../evidence/github-release-audit-20261103.json)包含远端源码版本、文档链接、配置差异、依赖指纹、上传后下载校验、原生重新导出、独立consumer、CPU闭环、FK与负例结果。
