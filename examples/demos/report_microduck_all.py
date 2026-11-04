# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build offline HTML and Markdown tables from a matched MicroDuck evaluation suite."""

import argparse
import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

SPEEDS = (0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.5)
RATES = (0.1, -0.1, 0.5, -0.5)
DESCRIPTIONS = {
    "FM": "fast_mixed 原基线；无累计朝向观测",
    "YO": "从零训练；仅加累计朝向观测",
    "YR": "从零训练；朝向观测＋朝向奖励",
    "PHa": "从零训练；固定尺度位置奖励 0.75 m，死区 0.05 m",
    "PHr": "从零训练；相对位置奖励，速度权重减半",
    "POr": "从零训练；相对位置奖励替换两项速度奖励",
    "PAr": "从零训练；保留速度奖励＋相对位置奖励",
    "S4-9k": "pos_add新 续训；扩大速度范围；中间 checkpoint",
    "S4": "pos_add新 续训 4000 轮；扩大速度范围",
    "S6-9k": "pos_add新 续训；扩大范围＋线速度权重 6；中间 checkpoint",
    "S6": "pos_add新 续训 4000 轮；扩大范围＋线速度权重 6",
    "P0": "yaw_reward 续训 4000 轮；不加位置奖励",
    "P2": "yaw_reward 续训 4000 轮；位置权重 2",
    "P4": "yaw_reward 续训 4000 轮；位置权重 4",
}
METHOD = (
    "评估 seed=137；每类 16 个实例、20 秒、dt=0.02 秒；确定性均值策略，保留观测噪声和物理随机化，"
    "关闭随机推力，头部/身体姿态指令为零。每 5 秒重置指令参考，物理速度保留。"
    "线速度为当前机体朝向坐标系下的前进速度，角速度为世界竖直轴角速度。"
    "速度均值及 RMSE 排除开始 2 秒；偏航统计覆盖全部有效时间。"
    "全局偏航=实际朝向减去从初始时刻积分指令角速度得到的理想朝向，取最短角差的绝对值（0～180°），"
    "不会每 5 秒归零。均值先在各实例内按时间平均，再对实例等权平均；"
    "最大值是所有有效实例、所有时刻的最大误差，不是各实例最大值的均值。"
    "提前终止后的重置轨迹排除，终止数单独报告；失败多时存活偏差会使误差看起来偏小。"
    "全部训练种子为 43，评估种子相同不等于多训练种子验证。"
    "1.2/1.5 m/s 仅在 speed4/6 的训练范围内，其他组均为范围外测试。"
    "fast_mixed 的历史奖励源码未保留，统一评估环境使用已说明的重建版本。"
    "本次未施加扰动，因此不能据此判断扰动后的回到原轨迹能力。"
)


def fmt(value, digits=3):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def speed_cell(metrics):
    value = fmt(metrics["actual_vx"])
    failures = metrics["failures"]
    return f"{value}（终止 {failures}/16）" if failures else value


def table(headers, rows, detail=False):
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        attributes = f' data-search="{html.escape(" ".join(str(x) for x in row), quote=True)}"' if detail else ""
        body.append(f"<tr{attributes}>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def markdown_table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in rows),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    models, details = [], []
    for m in manifest:
        directory = args.directory / m["id"]
        status = json.loads((directory / "status.json").read_text())
        assert status["exit_code"] == 0, status
        result = json.loads((directory / f"{m['variant']}.json").read_text())
        trace = np.load(directory / f"{m['variant']}_traces.npz")["trace"]
        assert result["seed"] == 137 and result["trials_per_case"] == 16
        assert trace.shape == (1000, 176, 18) and np.isfinite(trace).all()
        assert Path(result["checkpoint"]) == Path(m["checkpoint"])
        m["sha256"] = hashlib.sha256(Path(m["checkpoint"]).read_bytes()).hexdigest()
        m["evaluation_revision"] = result["source_revision"]
        m["description"] = DESCRIPTIONS[m["id"]]
        m["range"] = m["training"].get("forward_range", [-1, 1])
        m["weights"] = m["training"]["weights"]
        m["configuration"] = "/".join(f"{x:g}" for x in m["weights"])
        m["metrics"] = {}
        for index, (case, metrics) in enumerate(result["cases"].items()):
            data = trace[:, index * 16 : (index + 1) * 16]
            alive = data[:, :, 8] > 0.5
            initial_yaw = data[0, :, 16] - data[0, :, 7] - data[0, :, 2] * result["dt"]
            ideal_yaw = initial_yaw + np.cumsum(data[:, :, 2], axis=0) * result["dt"]
            difference = data[:, :, 16] - ideal_yaw
            yaw_deg = np.rad2deg(np.abs(np.arctan2(np.sin(difference), np.cos(difference))))
            per_trial = [
                float(yaw_deg[:, trial][alive[:, trial]].mean()) for trial in range(16) if alive[:, trial].any()
            ]
            per_trial_max = [
                float(yaw_deg[:, trial][alive[:, trial]].max()) for trial in range(16) if alive[:, trial].any()
            ]
            row = dict(metrics)
            row.update(
                experiment=m["id"],
                case=case,
                global_yaw_mean_deg=float(np.mean(per_trial)) if per_trial else None,
                global_yaw_max_deg=float(yaw_deg[alive].max()) if alive.any() else None,
                trial_max_yaw_mean_deg=float(np.mean(per_trial_max)) if per_trial_max else None,
                cycle_yaw_max_deg=float(np.rad2deg(np.abs(data[:, :, 7][alive])).max()) if alive.any() else None,
                failures=int((~alive[-1]).sum()),
            )
            # Independently verify the velocity summary against the raw per-trial valid samples.
            mask = alive.copy()
            mask[:100] = False
            for col, key in [(3, "actual_vx"), (5, "actual_wz")]:
                trial_means = [data[100:, j, col][mask[100:, j]].mean() for j in range(16) if mask[100:, j].any()]
                assert np.isclose(np.mean(trial_means), row[key], atol=1e-6)
            assert row["global_yaw_max_deg"] + 1e-5 >= row["global_yaw_mean_deg"]
            m["metrics"][case] = row
            details.append(row)
        models.append(m)
    config_headers = ["ID", "实验", "配置说明", "线/角/朝向/位置权重", "vx训练范围", "累计更新数", "父模型"]
    config_rows = []
    linear_rows, angular_rows, yaw_rows, detail_rows = [], [], [], []
    for m in models:
        parent = m["training"].get("parent_checkpoint")
        parent_name = Path(parent).parent.name if parent else "从零训练"
        config_rows.append(
            [
                m["id"],
                m["label"],
                m["description"],
                m["configuration"],
                str(m["range"]),
                m["iteration"] + 1,
                parent_name,
            ]
        )
        linear_rows.append(
            [m["id"], m["label"], m["configuration"], *(speed_cell(m["metrics"][f"vx_{s:g}"]) for s in SPEEDS)]
        )
        angular_rows.append([m["id"], m["label"], *(fmt(m["metrics"][f"wz_{w:g}"]["actual_wz"]) for w in RATES)])
        row = m["metrics"]["vx_1"]
        yaw_rows.append(
            [
                m["id"],
                fmt(row["actual_vx"]),
                fmt(row["global_yaw_mean_deg"], 2),
                fmt(row["global_yaw_max_deg"], 2),
                fmt(row["heading_mean_deg"], 2),
                f"{row['failures']}/16",
            ]
        )
        for case, r in m["metrics"].items():
            linear = case.startswith("vx")
            detail_rows.append(
                [
                    m["id"],
                    case,
                    fmt(r["actual_vx"] if linear else r["actual_wz"]),
                    fmt(r["linear_rmse"]),
                    fmt(r["yaw_rate_rmse"]),
                    fmt(r["global_yaw_mean_deg"], 2),
                    fmt(r["global_yaw_max_deg"], 2),
                    fmt(r["heading_mean_deg"], 2),
                    fmt(r["cycle_yaw_max_deg"], 2),
                    fmt(r["position_mean"]),
                    f"{r['failures']}/16",
                    fmt(r["mean_survival_s"], 2),
                ]
            )
    linear_headers = ["ID", "实验", "线/角/朝向/位置权重", *[f"目标 {s:g}" for s in SPEEDS]]
    angular_headers = ["ID", "实验", *[f"目标 {w:+g}" for w in RATES]]
    yaw_headers = ["ID", "实际vx", "全局平均偏航°", "全局最大偏航°", "周期内平均偏航°", "终止"]
    detail_headers = [
        "ID",
        "指令",
        "实际均值",
        "线速度RMSE",
        "角速度RMSE",
        "全局平均偏航°",
        "全局最大偏航°",
        "周期平均偏航°",
        "周期最大偏航°",
        "周期平均位置误差m",
        "终止",
        "存活秒",
    ]
    sections = [
        (
            "config",
            "实验配置",
            "除 fast_mixed 外均有累计朝向误差观测。相对位置奖励为 exp[−(e/(0.12(L+0.05)))²]。",
            config_headers,
            config_rows,
        ),
        (
            "linear",
            "线速度跟随 · m/s",
            "显示实际平均前进速度；括号表示提前终止实例数，无括号即 0/16。目标 1.2/1.5 对非 speed 系列属于训练范围外。"
            "均值接近目标不等于瞬时误差小，详表提供 RMSE。",
            linear_headers,
            linear_rows,
        ),
        (
            "angular",
            "原地旋转 · rad/s",
            "正负方向单独列出；平均角速度可能掩盖摆动，详表提供角速度 RMSE。",
            angular_headers,
            angular_rows,
        ),
        (
            "yaw",
            "1.0 m/s 直行偏航",
            "全局偏航不随 5 秒周期重置。应结合实际速度看，几乎不走的模型可能有较小偏航。"
            "其余全部指令的平均/最大偏航见详表。",
            yaw_headers,
            yaw_rows,
        ),
        (
            "details",
            "全部指令详情",
            "vx 为线速度指令，wz 为原地角速度指令。偏航含从启动至终止前的所有有效时刻。",
            detail_headers,
            detail_rows,
        ),
    ]
    training_rows = []
    for name in ("P0", "P2", "P4"):
        log = (args.directory.parent / "position_ablation" / f"{name}.log").read_text()
        values = []
        for key in (
            "Metrics/base_velocity/error_vel_xy",
            "Metrics/base_velocity/error_vel_yaw",
            "Episode_Termination/fell_over",
            "Episode_Termination/nan_state",
        ):
            samples = [float(x) for x in re.findall(re.escape(key) + r":\s*([-+\d.eE]+)", log)]
            values.append(fmt(float(np.mean(samples[-100:]))))
        training_rows.append([name, "10000 / 10000", *values])
    sections.insert(
        0,
        (
            "training",
            "P0 / P2 / P4 训练完成情况",
            "全部正常完成并保存 model_9999；下列为训练日志最后 100 轮均值，不是固定指令回放指标。",
            ["实验", "累计训练轮数", "线速度误差 m/s", "角速度误差 rad/s", "跌倒终止占比", "NaN终止占比"],
            training_rows,
        ),
    )
    generated = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M 北京时间")
    markdown = ["# MicroDuck 统一实验对照", "", generated, "", METHOD]
    for _, title, note, headers, rows in sections:
        markdown.extend(["", "## " + title, "", note, "", markdown_table(headers, rows)])
    (args.directory / "comparison.md").write_text("\n".join(markdown) + "\n")
    data = {"generated": generated, "method": METHOD, "models": models, "details": details}
    (args.directory / "comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    pieces = []
    for id_, title, note, headers, rows in sections:
        control = (
            '<label>筛选详表 <input id="filter" placeholder="例如 P4 或 vx_0.4" aria-label="筛选详表"></label>'
            '<span id="count"></span>'
            if id_ == "details"
            else ""
        )
        pieces.append(
            f'<section id="{id_}"><h2>{title}</h2><p>{note}</p>{control}'
            f"{table(headers, rows, id_ == 'details')}</section>"
        )
    sources = "".join(
        f"<li><b>{m['id']}</b> {html.escape(m['checkpoint'])}<br>"
        f"<small>SHA256 {m['sha256']} · 训练提交 "
        f"{html.escape(m['training'].get('source_revision', '历史源码未保留'))}</small></li>"
        for m in models
    )
    document = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MicroDuck 实验对照</title>
<style>
:root{font-family:system-ui,"Noto Sans CJK SC",sans-serif;
color:#182338;
background:#f3f6fa;
font-size:15px}
body{margin:0}
header{background:#172c46;
color:white;
padding:32px max(24px,5vw)}
header p{color:#d5e3f4;
max-width:1100px;
line-height:1.7}
h1{margin:0 0 10px;
font-size:30px}
nav{display:flex;
gap:20px;
flex-wrap:wrap}
nav a{color:#b9ddff}
main{max-width:1700px;
margin:auto;
padding:22px}
section{background:white;
border:1px solid #dae3ee;
border-radius:10px;
margin:0 0 26px;
padding:22px}
h2{font-size:22px;
margin:0 0 10px}
p{line-height:1.75;
color:#43526a}
.scroll{overflow:auto;
max-height:650px}
table{border-collapse:separate;
border-spacing:0;
width:100%;
font-variant-numeric:tabular-nums;
font-size:14px}
th,td{text-align:right;
padding:10px 12px;
border-bottom:1px solid #e4eaf1;
white-space:nowrap}
th{background:#eaf0f7;
position:sticky;
top:0;
z-index:2}
th:first-child,td:first-child{text-align:left;
position:sticky;
left:0;
background:#f5f8fc;
font-weight:650}
td:nth-child(2){text-align:left}
tr:hover td{background:#edf5ff}
#config td:nth-child(3){white-space:normal;
min-width:220px;
text-align:left}
input{padding:9px 12px;
margin:0 15px 16px;
border:1px solid #a8b9ca;
border-radius:6px;
width:260px}
small{color:#596b7d}
li{margin:12px 0;
overflow-wrap:anywhere}
.method{font-size:14px}
footer{padding:10px 24px 30px;
color:#596b7d}
a{color:#1560a3}
@media print{nav,input,label,#count{display:none}
.scroll{max-height:none;
overflow:visible}
section{break-inside:auto}
th{position:static}
body{background:white}
main{padding:0}
table{font-size:9px}
th,td{padding:5px}
}

</style>
<header>
<h1>MicroDuck · 统一实验对照</h1>
<p>14 个模型快照 · 11 类指令 · 每类 16 个实例 · 相同评估种子 137</p>
<nav>
<a href="#config">实验配置</a>
<a href="#linear">线速度</a>
<a href="#angular">角速度</a>
<a href="#yaw">偏航</a>
<a href="#details">完整详表</a>
<a href="#method">口径与来源</a>
</nav>
</header>
<main>"""
    document += "".join(pieces)
    document += f"""<section id="method">
<h2>统计口径与来源</h2>
<p class="method">{html.escape(METHOD)}</p>
<p>生成于 {generated}。<a href="comparison.json">全部数值与来源 JSON</a> · <a href="comparison.md">Markdown 表格</a>
</p>
<details>
<summary>展开 14 个 checkpoint 的路径和校验值</summary>
<ul>{sources}</ul>
</details>
</section>
</main>
<footer>研究性单种子实验；不以总训练奖励排列优劣。未完成的旧版 pos_only / pos_add 运行不作为有效对照。</footer>"""
    document += """<script>const input=document.getElementById('filter');
const rows=[...document.querySelectorAll('#details tbody tr')];
function filter(){let n=0;
const q=input.value.toLowerCase().trim();
for(const r of rows){const keep=r.dataset.search.toLowerCase().includes(q);
r.hidden=!keep;
if(keep)n++;
}document.getElementById('count').textContent=`显示 ${n} / ${rows.length} 行`;
}
input.addEventListener('input',filter);
filter();
</script></html>"""
    (args.directory / "comparison.html").write_text(document)
    print(
        json.dumps(
            {"models": len(models), "rows": len(details), "output": str(args.directory / "comparison.html")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
