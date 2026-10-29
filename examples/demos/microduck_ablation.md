# MicroDuck 四组独立续训

> 历史实验说明：本文的四组设计基于合并前的奖励配方。`study1` 默认采用上游的新奖励和课程，
> 权重及 std 干预可能已与默认值相同；重新运行前必须重设实验对照并通过配置核对。
> 原配方与工具保存在 `backup/microduck-before-study1-20261029` 分支。

从仓库根目录运行：

```bash
uv run python examples/demos/microduck_ablation.py
```

默认从 `2026-10-27_10-24-47_rerun_baseline/model_6000.pt` 开始，依次运行基线、
速度奖励权重 4.0、速度奖励 std 0.15/0.35、关节跟踪惩罚四组实验，每组新增 4000 次更新。
全部使用 4096 个环境、种子 123、原优化器和成熟课程；不恢复原仿真状态或随机流。

流程为配置核对 → 七档指令标定 → 四组小规模续训验证 → 四组正式训练 → 五个模型的统一评估。
任一步失败立即停止，并在 `batch.json` 记录失败阶段；不会自动跳过或重跑。

可以先完成标定和短测：

```bash
uv run python examples/demos/microduck_ablation.py --prepare-only --output logs/experiments/my_batch
uv run python examples/demos/microduck_ablation.py --continue-prepared --output logs/experiments/my_batch
```

查看 `batch.json` 获取状态、命令和正式训练目录，查看 `train_<实验名>.log` 获取中文训练输出。
对应训练目录的 TensorBoard 标签也是中文；普通训练默认仍使用英文。

```bash
uv run python -m tensorboard.main --logdir logs/rsl_rl/microduck_velocity_flat
```

`calibration/calibration.json` 固定保存关节顺序、容差与惩罚权重。
标定给七档指令相同权重；每关节容差为绝对跟踪误差 P75，最低 0.015 rad。
代价为所有动作关节超容差误差除以 0.1 rad 后的平方均值，初始平均惩罚为正奖励总贡献的 5%。
比较的是当前物理目标与推进后的真实关节角，包括 15–30 ms 指令延迟，不做目标时间平移。

原生 RSL-RL 从迭代标签 6000 开始追加训练，最终标签为 9999；实际新增更新数为 4000。
新增更新 1000/2000/3000/4000 次时分别保存 `model_6999/7999/8999/9999.pt`。
环境课程控制步从 144024 恢复，不乘并行环境数量。

最终结果在 `comparison.csv`、`comparison.json` 和 `comparison.md`。
每模型每档指令 16 次试验、20 秒、排除前 3 秒；失败后样本不再作为新试验。
原模型与四组终点使用同一评估环境、确定性策略和种子 42。
训练总奖励的定义不同，不能用于直接排名；比较速度、偏航、完成率与关节跟踪指标。
单训练种子的结果属于探索性对照，需要多种子实验才能判断提升是否稳定。

## 实验记录规范

新实验的命名、默认指标与表格、绘图选数、数据目录及展示板接入方式，见
[MicroDuck 实验记录规范 v1.1](microduck_experiment_records.md)。
