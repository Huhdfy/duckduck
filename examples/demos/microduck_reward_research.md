# MicroDuck 奖励机制研究

从仓库根目录运行：

```bash
uv run python examples/demos/microduck_reward_research.py \
  --output logs/experiments/microduck_reward_research_2026-10-28
```

固定使用 4096 个环境、每组新增 4000 次更新。复用旧四组实验中的 Baseline、双 std
终点；先重新评估并保存逐步状态和奖励，再依次训练 V-std、W-std。全部正式训练从同一
6000 轮检查点出发。默认沿用 Newton MJWarp；不指定可视化器。

`--prepare-only` 在已有模型评估及小规模恢复验证成功后停止，尚未启动正式训练。
使用相同输出目录加 `--continue-prepared` 可继续成功准备的队列；失败队列不会自动恢复。
两个选项互斥，输出目录不得覆盖旧结果。全局研究锁阻止两个研究队列并行。

## 冻结规则

`rules.json` 保存筛选阈值和最多 8 次正式训练的预算。首先执行两个 std 对照，随后根据
试验级 bootstrap 与奖励分解决定是否启动 H1/H2 干预。短测不占正式训练预算。
H2 权重干预同时缩放对应课程，避免重置撤销修改。没有足够样本、候选竞争项并列或
可比结果冲突时停止相应分支；不以跨模型低速轨迹代替同模型内的比较。

通过物理指标及稳定性门槛的候选，按 H1、H2、V-std、W-std 顺序选择一个，补做种子
456、789 的候选与匹配参照，共最多 4 次训练。多种子结果逐个报告，不能当作充分的
因果证明。未用预算不补跑。

## 结果和恢复记录

- `study.json`：阶段、进程、预算消耗、训练命令、模型哈希和自动决策。
- `existing_audit.json`、`preflight/`：历史配置和源码核对、原优化器学习率及成熟奖励清单。
- `input_hashes.json`、`source_snapshot/`：运行输入和源码冻结，变化时停止队列。
- `evaluation_*/step_traces.npz`：每档 `(控制步, 试验, 列)`，列定义在 `trace_metadata.json`。
  数组索引对应 `t=(step+1)*step_dt`，试验编号对应第二轴索引。
- `evaluation_*/analysis.json`：逐试验物理指标、训练/参照两套回报、H1 分布、H2 分解。
- `early_training_*.json`：最初 500 次更新的训练和优化曲线；完整曲线保留在 TensorBoard。
- `comparison.csv`、`report.md`、`figures/`：统一物理指标、证据边界及图表。

采集时刻为物理推进和奖励计算之后、重置和指令重采样之前。固定 20 秒回报含前 3 秒
预热，运动指标排除预热。终止步保留，重置后的行不纳入；未完成部分的累计回报不填补
也不按存活时间放大。原始稀疏足部事件可为 NaN，状态有效性和奖励有限性分别检查。

参照评分使用真实 Baseline 奖励管理器，保留空中时间、摆动峰值、头部误差等状态。
离线只重算两个无状态速度核及明确选中的权重，不能从单帧状态重建有记忆的奖励。
改变角速度函数形状同时改变整条奖励分布；误差敏感度不等于 PPO 策略梯度。

## 实验记录规范

新实验的命名、默认指标与表格、绘图选数、数据目录及展示板接入方式，见
[MicroDuck 实验记录规范 v1.1](microduck_experiment_records.md)。
