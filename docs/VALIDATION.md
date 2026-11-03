# 本次归档验收

2026-11-03：native模块逐文件SHA与CPFS实际训练源码复核，Cuboi47个、zita16个有效Python文件；AppleDouble非代码元数据不纳入Git。原上游许可证、实际配置、环境锁、资源/选模/导出回执保留。新run输出与历史实验隔离。

本次进行语法检查、dry-run、命令/配置与大文件排除检查，以及已有环境上的CPU预检和128状态FK检查。实际通过的CPU回执见evidence/handoff-validation.json；如该文件尚不存在则不应声称相关CPU检查完成。

本次不启动GPU训练、不重跑36000条评测、不重建完整容器环境，也不修改历史权重或分数。GPU仿真/真实更新/恢复的已通过证据属于历史实验，不能冒充新启动器已完成一轮端到端训练。新GPU作业仍需按团队资源约定运行。

## 统一 Contract 链路复查

同日后续已完成从选中checkpoint重新导出、每套64步source parity、schema/semantic/hash/golden tests、独立consumer及各4条完整动作的CPU闭环与轨迹复算。修复Cuboi导出环境覆盖冲突；两份ONNX与历史正式包字节一致。详见[Contract链路](CONTRACT_PIPELINE.md)与evidence/contract-pipeline-check-20261103.json。未重跑Full-6000、未启动GPU训练。
