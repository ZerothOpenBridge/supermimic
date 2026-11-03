# 日志、计数与选模

历史run根目录：`/mnt/cpfs/workspace/yxq/supermimic/experiments/mini3_v1/`。

* Cuboi：`cuboi/formal-seed0-r1/tb/` 两份TensorBoard event；终态`training-result.json`；`selection.json`与`candidate_*-dev316.json`。
* zita：`zita/formal-seed0-r1/train_log.csv`；`args.json`、`training-result.json`、`dev316_*.json`、`resume-verification.json`；分段stdout在该实验根目录的zita-formal-seed0-r1-chunk*.log。

目前未接SwanLab，没有这两条实验的在线SwanLab run。仓库提供说明和轻量回执，完整event/CSV/stdout留CPFS。本地8766工作台是该Mac的入口，不能把localhost链接当成同事的共享网址。

## 计数应如何比较

Cuboi一轮收集一个4096环境向量步；zita一轮收集2或3个向量步。原生iteration不能横比。真实采样量、Actor/WM更新和想象步分别记账：想象步=Actor更新×4096×32，不是真实物理采样。

GPU小时是专卡分配墙钟，包含调试、编译、试训和相应开发评测；不等于平均SM忙碌时长。没有完整利用率序列。Cuboi恢复后TensorBoard time/hours末值约19.17h，不是累计34.01h；zita CSV最后标量221150，实际终态221158来自训练回执，日志间隔为50轮。

## 先看固定dev，再看代理指标

Cuboi 8步online NSE约0.604→0.00146；zita wm_err8约0.862→0.0250。两者归一化不同，不是毫米误差，不能直接比较大小。想象survival约0.999不是全动作实际成功率。

zita ep_ret_norm末点约-2.565，sim_rew约0.915。前者按结束episode汇总，后者按所有环境采样步汇总；奖励存在负罚项、随机化与难例reset。完整逐episode奖励分解尚缺，不能断言负值由哪项主导，也不能单凭负值判断崩溃。求解器iterations limit reached警告已归档，尚未证明与具体失败存在因果联系。

dev局部MAE与统一evaluator Fresh欧氏MPJPE不是同一指标；例如6mm原生MAE不能与30.97mm Fresh MPJPE直接相比。两者原生return也不作为跨实现统一排名。

## 冻结选模规则

dev316的完成率 → progress → 较低局部MAE → 更早候选，按未四舍五入数值进行字典序比较。Full-6000不参与选模。

Cuboi选择320000；zita选择161203而非终点221158。zita在161203完成315/316、progress=0.9968879819、MAE=5.014mm；211474同样315/316，MAE更低4.838mm，但progress略低0.9968793392；终点221158为314/316。规则产生的微小排名差别不是统计显著性结论。

## 新run的日志

新启动器在`shared/data/collaboration/runs/<backend>/<run-id>/`下记录launch JSON、stdout、cache及独立resource-ledger；原生训练输出在其`training/`子目录。Cuboi会在原生定时checkpoint时做dev316候选验证；zita每约2小时保存退出后运行native-eval，再resume。native-eval将更新该新run的selection.json。数据与完整恢复文件不提交Git。

原有恢复已核验网络、optimizer、replay、采样、RNG、历史、计数和物理状态；不把模型加载本身当成恢复成功。后续修改必须重新检查真实optimizer步和有限性。
