# MicroDuck 多模型键盘对照

从仓库根目录启动，把任意数量的兼容 MicroDuck RSL-RL 检查点依次放在参数后面：

```bash
uv run python examples/demos/microduck_keyboard.py \
  --checkpoints /path/to/model_A.pt /path/to/model_B.pt /path/to/model_C.pt \
  --labels A B C
```

编号严格按输入顺序排列，支持重复检查点，数量不限于 9 个。`--labels` 可省略；提供时必须
与模型数量一致。原来的 `--checkpoint MODEL.pt`、不带参数默认加载 `latest` 的用法仍可用。
各检查点需要与当前 MicroDuck 任务的策略网络结构兼容，启动时会逐个检查。

## 操作

先点击仿真窗口获得键盘焦点。相机仍用 WASD。

|按键|作用|
|---|---|
|数字编号，然后 Enter|选择模型，例如 `1`、`0`、Enter 选择第 10 个|
|Backspace|修改正在输入的编号|
|`[` / `]`|上一个 / 下一个模型，首尾循环|
|F1 / F2 / F3 / F4|前后速度档位 0.1 / 0.2 / 0.3 / 0.4 m/s|
|`-` / `=`|前后线速度幅值减小 / 增大，每次 0.05 m/s|
|`;` / `'`|转向角速度幅值减小 / 增大，每次 0.1 rad/s|
|按住 I / K|前进 / 后退；松开停止|
|按住 J / L|左转 / 右转，使用当前角速度幅值，默认为 ±0.5 rad/s|
|按住 U / O|横移 +0.15 / −0.15 m/s|
|B|停止，取消正在进行的标准测试|
|R|用相同种子重置机器人，取消正在进行的测试|
|C|开关跟随镜头|
|P|开始当前模型的标准测试；再次按 P 取消并保存部分结果|

侧栏显示模型编号、名称、正在输入的编号、目标速度、实际速度、横滚/俯仰角速度大小、
标准测试进度。终端输出中文提示。若侧栏字体不支持中文名称，可将 `--labels` 设为简短英文。

侧栏 `Linear speed (m/s)`、`Turn speed (rad/s)` 输入框可直接输入精确幅值；幅值最小为 0。
切换模型与重置保留所选幅值，方便对照。启动时也可追加 `--linear-speed 0.2 --angular-speed 0.35`。
横移保持原来的 ±0.15 m/s；速度调节只作用于手动控制，P 标准测试继续使用固定的七档指令。

切换模型会同时载入该模型的观测归一化参数，并用相同种子重置环境；切换前按住的运动键
必须松开再按才生效。窗口失焦或 UI 接管输入时，手动运动键清零。仅保留当前策略在 GPU
中，增加模型数量不会按数量复制 GPU 环境。

## 标准测试

P 依次执行站立、前进 0.1/0.2/0.3/0.4 m/s、原地负向/正向转弯。每档 20 秒，前 3 秒
预热，按仿真时间计算。测试中忽略手动运动键；数字切换、R、B 或再次 P 可中断。
各档重置种子为 `--seed + 档位序号`，默认基准种子 42。

重置位置和朝向、成熟课程、关闭观测噪声与推力、零头部/身体目标等条件与研究评估保持一致。
这里只运行 1 个环境、每档 1 次，适合观察和快速验证，不替代原来的每档 16 次统计评估。

结果默认写入 `logs/keyboard/microduck/`，可用 `--output` 修改。JSON 保存模型身份、种子、
完成/中断状态、逐档速度误差和终止原因；CSV 保存包含预热的逐步状态、目标和奖励。
终止步在重置前采集，提前终止后直接进入下一档，不把自动重置后的片段混进原试验。

## 本轮五个模型

```bash
uv run python examples/demos/microduck_keyboard.py --checkpoints \
  logs/rsl_rl/microduck_velocity_flat/microduck_four_way_2026-10-27_v2_train_01_baseline/model_9999.pt \
  logs/rsl_rl/microduck_velocity_flat/microduck_reward_research_2026-10-28_v2_train_04_v_std/model_9999.pt \
  logs/rsl_rl/microduck_velocity_flat/microduck_reward_research_2026-10-28_v2_train_05_w_std/model_9999.pt \
  logs/rsl_rl/microduck_velocity_flat/microduck_four_way_2026-10-27_v2_train_03_velocity_std/model_9999.pt \
  logs/rsl_rl/microduck_velocity_flat/microduck_reward_research_2026-10-28_v2_train_06_w_rational/model_9999.pt \
  --labels Baseline V-std W-std Both-std Rational
```
