# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trial-level reward diagnostics and frozen screening rules for MicroDuck continuations.

This module needs no simulator. Reference rewards come from the live, stateful reward manager;
only the stateless velocity kernels and a selected weight are rescored offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RULES = {
    "version": 1,
    "max_training_runs": 8,
    "num_envs": 4096,
    "updates": 4000,
    "training_seeds": [123, 456, 789],
    "evaluation_seed": 42,
    "evaluation_trials": 16,
    "duration_s": 20.0,
    "warmup_s": 3.0,
    "bootstrap_seed": 20261028,
    "bootstrap_draws": 10000,
    "saturation_x": 3.0,
    "saturation_fraction": 0.25,
    "error_reduction_fraction": 0.1,
    "saturated_reward_gain_max": 0.02,
    "primary_change_fraction": 0.1,
    "planar_guard_fraction": 0.05,
    "angular_guard_fraction": 0.1,
    "failure_guard_absolute": 0.05,
    "low_motion_ratio": [0.0, 0.2],
    "tracking_ratio": [0.8, 1.2],
    "h2_min_trials_per_group": 4,
    "h2_min_positive_fraction": 0.02,
    "priority": ["w_rational", "competitor_half", "v_std", "w_std"],
}
FORWARD = tuple(f"forward_{speed:g}" for speed in (0.1, 0.2, 0.3, 0.4))
TURNS = ("turn_negative", "turn_positive")


def bootstrap_difference(left, right, paired: bool = True) -> dict:
    """Bootstrap equal-trial mean differences, preserving paired trial indices when requested."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.size == 0 or right.size == 0 or not (np.isfinite(left).all() and np.isfinite(right).all()):
        return {"mean": None, "ci95": [None, None], "n": [left.size, right.size]}
    if paired and left.shape != right.shape:
        raise ValueError("Paired comparisons require aligned trial arrays.")
    rng = np.random.default_rng(RULES["bootstrap_seed"])
    indices = rng.integers(0, len(left), (RULES["bootstrap_draws"], len(left)))
    other = indices if paired else rng.integers(0, len(right), (RULES["bootstrap_draws"], len(right)))
    samples = left[indices].mean(1) - right[other].mean(1)
    return {
        "mean": float(left.mean() - right.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "n": [len(left), len(right)],
    }


def analyze_evaluation(folder: Path, spec: dict) -> dict:
    """Rescore training rewards and retain equal-trial physical and reward summaries."""
    metadata = json.loads((folder / "trace_metadata.json").read_text())
    old_trials = json.loads((folder / "trials.json").read_text())
    index = {name: i for i, name in enumerate(metadata["columns"])}
    terms = [name.split("/", 1)[1] for name in index if name.startswith("reward_step/")]
    variant = spec["variant"]
    v_std = 0.15 if variant in ("v_std", "velocity_std") else 0.1**0.5
    w_std = 0.35 if variant in ("w_std", "w_rational", "velocity_std") else 0.5**0.5
    dt = metadata["step_dt"]
    warmup = round(metadata["warmup_s"] / dt)
    summaries, distributions = [], {}
    with np.load(folder / "step_traces.npz") as archive:
        for case in archive.files:
            samples = archive[case].astype(np.float64)

            def column(name):
                return samples[:, :, index[name]]

            alive = column("alive_before_step").astype(bool)
            omega_error = column("yaw_error_radps") ** 2 + column("roll_pitch_sq_rad2ps2")
            linear_error = sum(
                (column(f"link_v{axis}_b_mps") - (column(f"command_{axis}") if axis != "z" else 0)) ** 2
                for axis in "xyz"
            )
            finite = np.isfinite(omega_error) & np.isfinite(linear_error)
            stable = alive & finite
            stable[:warmup] = False
            reference = np.stack([column(f"reward_step/{name}") for name in terms], axis=-1)
            np.testing.assert_allclose(
                reference.sum(-1)[alive], column("reward_total_step")[alive], atol=2e-6, rtol=2e-5
            )
            training = reference.copy()
            weight = 4.0 if variant == "velocity_weight" else 2.0
            x = omega_error / w_std**2
            angular = 1 / (1 + x) if variant == "w_rational" else np.exp(-x)
            # Match the live kernels' zero_non_finite contract on a terminal, diverged state.
            angular = np.nan_to_num(angular, nan=0.0, posinf=0.0, neginf=0.0)
            linear = np.nan_to_num(np.exp(-linear_error / v_std**2), nan=0.0, posinf=0.0, neginf=0.0)
            training[:, :, terms.index("track_ang_vel")] = weight * angular * dt
            training[:, :, terms.index("track_lin_vel")] = weight * linear * dt
            if variant == "competitor_half":
                training[:, :, terms.index(spec["competitor"])] *= 0.5
            reference_return = np.where(alive[:, :, None], reference, 0).sum(0)
            training_return = np.where(alive[:, :, None], training, 0).sum(0)
            saturated = stable & (x >= RULES["saturation_x"])
            gain = np.exp(-0.9 * x) - np.exp(-x)
            distributions[case] = {
                "x": x[stable],
                "E_omega": omega_error[stable],
                "training_angular_normalized": angular[stable],
                "exp_reward_gain_for_10pct_error_reduction": gain[stable],
            }
            for trial in sorted((t for t in old_trials if t["case"] == case), key=lambda t: t["trial"]):
                i = trial["trial"]
                mask = stable[:, i]
                sat = saturated[:, i]
                row = dict(trial)
                row["angular_rms_radps"] = float(np.sqrt(omega_error[mask, i].mean())) if mask.any() else None
                row["roll_pitch_rms_radps"] = (
                    float(np.sqrt(column("roll_pitch_sq_rad2ps2")[mask, i].mean())) if mask.any() else None
                )
                row["saturation_fraction"] = float(sat.sum() / mask.sum()) if mask.any() else None
                row["saturated_gain_median"] = float(np.median(gain[sat, i])) if sat.any() else None
                row["reference_rewards"] = dict(zip(terms, reference_return[i].tolist()))
                row["training_rewards"] = dict(zip(terms, training_return[i].tolist()))
                row["reference_return"] = float(reference_return[i].sum())
                row["training_return"] = float(training_return[i].sum())
                row["reference_positive_return"] = float(reference_return[i][reference_return[i] > 0].sum())
                ratio = row["vx_mps"] / row["target_vx_mps"] if row["target_vx_mps"] and mask.any() else None
                row["motion_ratio"] = ratio
                row["motion_class"] = (
                    "unclassified"
                    if ratio is None
                    else "reverse"
                    if ratio < 0
                    else "low"
                    if ratio < 0.2
                    else "tracking"
                    if 0.8 <= ratio <= 1.2
                    else "other"
                )
                summaries.append(row)
    distribution_summary = {}
    for case, arrays in distributions.items():
        distribution_summary[case] = {
            name: {"n": len(values), "quantiles": np.quantile(values, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist()}
            if len(values)
            else {"n": 0, "quantiles": []}
            for name, values in arrays.items()
        }
    np.savez_compressed(
        folder / "reward_distributions.npz",
        **{f"{case}/{name}": values for case, arrays in distributions.items() for name, values in arrays.items()},
    )
    result = {"spec": spec, "trials": summaries, "distributions": distribution_summary, "h2": diagnose_h2(summaries)}
    (folder / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def diagnose_h2(trials: list[dict]) -> dict:
    """Require within-model, within-command support and consistent competitor attribution."""
    evidence = []
    for case in FORWARD[:2]:
        complete = [t for t in trials if t["case"] == case and t["completed"]]
        low = [t for t in complete if t["motion_class"] == "low"]
        tracking = [t for t in complete if t["motion_class"] == "tracking"]
        row = {"case": case, "low_n": len(low), "tracking_n": len(tracking), "eligible": False}
        evidence.append(row)
        if min(len(low), len(tracking)) < RULES["h2_min_trials_per_group"]:
            row["reason"] = "同模型同指令的完整试验样本不足"
            continue
        difference = bootstrap_difference(
            [t["reference_return"] for t in low], [t["reference_return"] for t in tracking], paired=False
        )
        contributions = {
            name: float(
                np.mean([t["reference_rewards"][name] for t in low])
                - np.mean([t["reference_rewards"][name] for t in tracking])
            )
            for name in low[0]["reference_rewards"]
        }
        candidates = sorted(
            ((value, name) for name, value in contributions.items() if name not in ("track_lin_vel", "track_ang_vel")),
            reverse=True,
        )
        positive = np.mean([t["reference_positive_return"] for t in complete])
        best, name = candidates[0]
        tied = len(candidates) > 1 and np.isclose(best, candidates[1][0], rtol=1e-6, atol=1e-9)
        other_loss = sum(value for key, value in contributions.items() if key != "track_lin_vel")
        linear_gain = -contributions["track_lin_vel"]
        row.update(
            difference=difference,
            contributions=contributions,
            competitor=name,
            positive_return_mean=float(positive),
            competitor_fraction=float(best / positive) if positive else 0,
        )
        row["eligible"] = bool(
            difference["ci95"][0] > 0
            and linear_gain > 0
            and other_loss > linear_gain
            and best >= RULES["h2_min_positive_fraction"] * positive
            and not tied
        )
        row["reason"] = "奖励分解支持" if row["eligible"] else "回报差、竞争贡献或唯一性未通过门槛"
    supported = [r for r in evidence if r["eligible"]]
    measured = [r for r in evidence if "difference" in r]
    conflict = any(not r["eligible"] for r in measured) or len({r["competitor"] for r in supported}) > 1
    return {
        "eligible": bool(supported) and not conflict,
        "competitor": supported[0]["competitor"] if supported and not conflict else None,
        "evidence": evidence,
        "limitation": "轨迹分组是观察性比较；不能证明未执行的动作会产生何种回报。",
    }


def paired_metric(analysis: dict, cases: tuple[str, ...], key: str) -> np.ndarray:
    """Average cases within aligned trial indices, requiring every case's metric to be finite."""
    arrays = [
        [t[key] for t in sorted(analysis["trials"], key=lambda t: t["trial"]) if t["case"] == case] for case in cases
    ]
    return np.asarray(arrays, dtype=float).mean(0)


def h1_gate(baseline: dict, w_std: dict) -> dict:
    """Gate the rational W-std intervention on measured degradation and saturation."""
    base = paired_metric(baseline, TURNS, "angular_rms_radps")
    changed = paired_metric(w_std, TURNS, "angular_rms_radps")
    difference = bootstrap_difference(changed, base)
    saturation = {}
    for case in TURNS:
        rows = [t for t in w_std["trials"] if t["case"] == case]
        fraction = np.mean([t["saturation_fraction"] or 0 for t in rows])
        with np.load(Path(w_std["folder"]) / "reward_distributions.npz") as values:
            x = values[f"{case}/x"]
            gain = values[f"{case}/exp_reward_gain_for_10pct_error_reduction"]
            gains = gain[x >= RULES["saturation_x"]]
        median = float(np.median(gains)) if len(gains) else None
        saturation[case] = {
            "fraction": float(fraction),
            "gain_median": median,
            "eligible": bool(fraction >= 0.25 and median is not None and median <= 0.02),
        }
    eligible = (
        difference["mean"] is not None
        and base.mean() > 0
        and difference["mean"] >= 0.1 * base.mean()
        and difference["ci95"][0] > 0
        and any(v["eligible"] for v in saturation.values())
    )
    return {"eligible": bool(eligible), "turning_difference": difference, "saturation": saturation}


def screen_candidate(candidate: dict, control: dict, variant: str) -> dict:
    """Apply effect and per-case guardrails to physical measurements, never training return."""
    cases = TURNS if variant in ("w_std", "w_rational") else FORWARD[:2] if variant == "competitor_half" else FORWARD
    key = "angular_rms_radps" if cases == TURNS else "planar_error_mps"
    base = paired_metric(control, cases, key)
    treatment = paired_metric(candidate, cases, key)
    difference = bootstrap_difference(treatment, base)
    improvement = (
        float(1 - treatment.mean() / base.mean())
        if np.isfinite(base).all() and np.isfinite(treatment).all() and base.mean() > 0
        else None
    )
    guards = []
    for case in FORWARD:
        for metric, limit in (("planar_error_mps", 0.05), ("yaw_rms_radps", 0.1), ("roll_pitch_rms_radps", 0.1)):
            before = paired_metric(control, (case,), metric).mean()
            after = paired_metric(candidate, (case,), metric).mean()
            guards.append(
                {
                    "case": case,
                    "metric": metric,
                    "passed": bool(np.isfinite(after) and after <= before * (1 + limit) + 1e-12),
                }
            )
    for case in ("standing",) + FORWARD + TURNS:
        before = np.mean([not t["completed"] for t in control["trials"] if t["case"] == case])
        after = np.mean([not t["completed"] for t in candidate["trials"] if t["case"] == case])
        guards.append({"case": case, "metric": "failure_rate", "passed": bool(after - before <= 0.05)})
    return {
        "eligible": bool(improvement is not None and improvement >= 0.1 and all(g["passed"] for g in guards)),
        "primary_metric": key,
        "primary_cases": cases,
        "improvement_fraction": improvement,
        "difference": difference,
        "guards": guards,
    }


def write_report(output: Path, models: dict, decisions: dict) -> None:
    """Write inspectable per-case tables and scientific figures, with explicit causal limits."""
    import csv

    rows = []
    for name, analysis in models.items():
        for case in ("standing",) + FORWARD + TURNS:
            trials = [t for t in analysis["trials"] if t["case"] == case]
            row = {"model": name, "case": case, "completion": float(np.mean([t["completed"] for t in trials]))}
            for metric in (
                "vx_mps",
                "planar_error_mps",
                "yaw_rms_radps",
                "angular_rms_radps",
                "roll_pitch_rms_radps",
                "saturation_fraction",
                "reference_return",
                "training_return",
            ):
                values = [t[metric] for t in trials if t[metric] is not None]
                row[metric] = float(np.mean(values)) if values else None
            rows.append(row)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    lines = [
        "# MicroDuck 奖励机制研究",
        "",
        "## 发生了什么",
        "",
        "|模型|指令|完成率|实际 vx m/s|平面误差 m/s|角速度综合 RMS rad/s|低奖励比例|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            "—" if row[k] is None else f"{row[k]:.4f}"
            for k in ("completion", "vx_mps", "planar_error_mps", "angular_rms_radps", "saturation_fraction")
        ]
        lines.append(f"|{row['model']}|{row['case']}|" + "|".join(values) + "|")
    lines += [
        "",
        "## 证据与自动决策",
        "",
        "```json",
        json.dumps(decisions, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 解释边界",
        "",
        "奖励数值敏感度不是 PPO 策略梯度。低运动轨迹的回报分解属于观察性证据。",
        "替代解释包括奖励整体尺度、Critic 适应、运动学耦合与训练随机性。失败试验保留，稳定指标只使用有效样本。",
        "固定窗口回报包含预热；训练奖励与统一 Baseline 参照奖励分开报告。",
        "",
        "![物理指标](figures/physical_metrics.png)",
        "",
        "![奖励饱和分布](figures/saturation.png)",
        "",
        "![低速奖励分解](figures/low_speed_rewards.png)",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    _plot_report(output, models, rows)


def _plot_report(output: Path, models: dict, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for name in models:
        forward = [r for r in rows if r["model"] == name and r["case"] in FORWARD]
        for axis, metric in zip(axes, ("vx_mps", "yaw_rms_radps", "roll_pitch_rms_radps")):
            axis.plot([0.1, 0.2, 0.3, 0.4], [r[metric] for r in forward], marker="o", label=name)
    axes[0].plot([0.1, 0.4], [0.1, 0.4], "k--", label="command")
    for axis, title, unit in zip(
        axes, ("Forward tracking", "Straight-line yaw error", "Roll/pitch rate"), ("m/s", "rad/s RMS", "rad/s RMS")
    ):
        axis.set(title=title, xlabel="Command vx (m/s)", ylabel=unit)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7)
    figure.savefig(folder / "physical_metrics.png", dpi=180)
    figure.savefig(folder / "physical_metrics.svg")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, analysis in models.items():
        with np.load(Path(analysis["folder"]) / "reward_distributions.npz") as archive:
            for axis, case in zip(axes, TURNS):
                values = np.sort(archive[f"{case}/x"])
                if len(values):
                    axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=name)
    for axis, case in zip(axes, TURNS):
        axis.axvline(3, color="k", linestyle="--", label="x=3")
        axis.set(
            title=case,
            xlabel="Normalized angular error x (training scale)",
            ylabel="Empirical CDF",
            xscale="symlog",
            xlim=(0, None),
        )
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7)
    figure.savefig(folder / "saturation.png", dpi=180)
    figure.savefig(folder / "saturation.svg")
    plt.close(figure)
    primary = {name: data for name, data in models.items() if "_seed" not in name}
    figure, axes = plt.subplots(len(primary), 2, figsize=(13, 3 * len(primary)), squeeze=False, constrained_layout=True)
    for axes_row, (name, data) in zip(axes, primary.items()):
        for axis, case in zip(axes_row, FORWARD[:2]):
            rows = [t for t in data["trials"] if t["case"] == case and t["completed"]]
            low = [r for r in rows if r["motion_class"] == "low"]
            tracking = [r for r in rows if r["motion_class"] == "tracking"]
            axis.set_title(f"{name}, {case}: low n={len(low)}, tracking n={len(tracking)}", fontsize=9)
            if not low or not tracking:
                axis.text(0.5, 0.5, "Insufficient group overlap", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            differences = {
                term: np.mean([r["reference_rewards"][term] for r in low])
                - np.mean([r["reference_rewards"][term] for r in tracking])
                for term in low[0]["reference_rewards"]
            }
            terms = sorted(differences, key=lambda k: abs(differences[k]), reverse=True)[:6]
            axis.barh(
                terms,
                [differences[k] for k in terms],
                color=["#d55e00" if differences[k] > 0 else "#0072b2" for k in terms],
            )
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_xlabel("Reference return: low motion minus tracking (20 s)")
            axis.tick_params(labelsize=8)
    figure.savefig(folder / "low_speed_rewards.png", dpi=160)
    figure.savefig(folder / "low_speed_rewards.svg")
    plt.close(figure)
