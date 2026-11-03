# 环境：复用与重建

## 复用已验证环境（推荐先走此路径）

Cuboi：CPFS根目录cuboi/.venv/bin/python；Python3.12.13、Torch2.14.1+cu130、MuJoCo3.16.0、Warp1.18.0。zita：章鱼现有专用容器yxq-supermimic-zita-zhangyu-gpu0，私有Docker入口deploy/docker-zhangyu.sh；Python3.12.3、Torch2.11.0+cu130、MuJoCo3.15.0、Warp1.17.0、固定IsaacLab46d6cc3。此处版本是原生训练环境。

脚本可用SM_PYTHON、SM_DOCKER_CLI、SM_CONTAINER覆盖解释器/容器。zita所选容器必须已经运行、挂载当前checkout、只暴露请求的那张物理GPU；CPU预检隐藏CUDA。不会启动、迁移、重建或重启现有Docker服务。需要其他GPU时新建独立容器，不能改正在使用的容器。

CPU导出/统一评测另使用shared/data/mini3_v1/runtime/export-runtime与冻结evaluator；统一MuJoCo为3.12.0、ORT1.29.0。不要将训练后端版本与统一evaluator版本混为一谈；启动器只在export/evaluate加载这些额外依赖，原生train/preflight不加载它们。

## 在新位置重建Cuboi

在CPFS上的cuboi分支checkout中，使用已部署deploy/bin/uv（0.12.9）：

```bash
export UV_CACHE_DIR=/mnt/cpfs/workspace/yxq/supermimic/shared/cache/colleague-uv
export UV_PYTHON_INSTALL_DIR=/mnt/cpfs/workspace/yxq/supermimic/shared/python
export TMPDIR=/mnt/cpfs/workspace/yxq/supermimic/shared/tmp
/mnt/cpfs/workspace/yxq/supermimic/deploy/bin/uv sync --frozen --group dev
export SM_PYTHON="$PWD/.venv/bin/python"
python3 scripts/run.py preflight
```

uv.lock保留实际锁定版本、wheel哈希和网络地址调整；environment/upstream.uv.lock保留原锁，environment/recorded-runtime.json与recorded-requirements.txt记录现场环境。网络可能需配置可用下载通道；不要uv lock或升级依赖来绕过失败。本次归档未重新下载全环境，重建命令不是本次全新环境验收。

## zita重建

IsaacLab固定46d6cc3d7b115906fad12efd372fbf69f4e3d3ed；环境锁在environment/isaaclab.uv.lock，Docker配方在environment/Dockerfile.kitless-cpfs，不能把该IsaacLab锁当作本仓库根目录的uv.lock。克隆该IsaacLab提交作为构建上下文，复制配方和记录锁到上下文后构建。Docker镜像/层缓存必须使用现有CPFS隔离存储，不用空间有限的系统Docker根盘。

历史镜像及sha见environment/recorded-container.json；16.3GB tar位于shared/data/mini3_v1/runtime/images/zita-4f29ffba-20261030.tar。可由有权限的维护者在自己的隔离Docker中load并新建容器；不会由脚本自动load或修改服务。单GPU新容器至少挂载整个shared/data、experiments与deploy原绝对路径、当前checkout和独立CPFS cache，workdir设为checkout；--gpus device=<物理卡>，CPU内逻辑设备固定cuda:0。沿用已验收镜像可避免重复网络构建。

新环境须先preflight、CPU FK，再按团队预算另做GPU仿真/更新/恢复工程门。本次只验证已有运行环境上的CPU入口，不把语法检查称为新镜像构建或GPU训练验证。
