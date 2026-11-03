# SuperMimic · Zeroth Mini3 协作入口

两份实际Mini3复现分别在独立分支；本main分支只提供导航与共同交接。整理日期2026-11-03。

|分支|内容|历史正式run|
|---|---|---|
|[cuboi](https://github.com/ZerothOpenBridge/supermimic/tree/cuboi)|mjlab / MuJoCo-Warp，21关节Mini3适配源码、锁文件、配置和启动脚本|320000 iteration；34.0054 GPUh|
|[zita](https://github.com/ZerothOpenBridge/supermimic/tree/zita)|Isaac Lab 3无Kit / Newton-MJWarp，独立Mini3适配|221158 iteration；47.9526 GPUh；交付161203|

先读[同事对接说明](docs/COLLEAGUE_HANDOFF.md)与[日志口径](docs/LOGGING.md)，再切实现分支按README操作。两份策略从随机初始化训练seed0，各使用单张48GB RTX4090。不同iteration、输入与精度不能直接横比。

数据、权重、完整日志、视频、replay和镜像留在`/mnt/cpfs/workspace/yxq/supermimic/`。仓库只保存代码、配置、路径/哈希、结果摘要和说明。这里尚未完成Mini3真机或感知WBC性能验收。

```bash
git clone --branch cuboi --single-branch https://github.com/ZerothOpenBridge/supermimic.git cuboi-mini3
git clone --branch zita --single-branch https://github.com/ZerothOpenBridge/supermimic.git zita-mini3
```

在服务器使用已有容器时请按分支README把checkout放在其CPFS共享挂载内。每个实现分支均包含环境复用/重建说明、preflight、CPU FK、fresh/resume、native dev316、导出和统一评测命令。新启动器经过的验收范围见对应分支`docs/VALIDATION.md`；未为归档而启动新训练。

上游代码版权与MIT许可分别保留于LICENSES和实现分支LICENSE；机器人资产、数据和模型的使用范围另循原来源。
