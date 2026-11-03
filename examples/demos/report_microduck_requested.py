# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report matched final-checkpoint speed and yaw evaluations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "fast_mixed": "yaw_obs",
    "yaw_obs": "yaw_obs",
    "yaw_reward": "yaw_reward",
    "pos_half_old": "pos_half",
    "pos_half_rel": "pos_half",
    "pos_only_rel": "pos_only",
    "pos_add_rel": "pos_add",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows, yaw_rows, manifest = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for label, variant in RUNS.items():
        root = args.directory / label
        result = json.loads((root / f"{variant}.json").read_text())
        trace = np.load(root / f"{variant}_traces.npz")["trace"]
        trials = result["trials_per_case"]
        assert trace.shape == (1000, 8 * trials, 18)
        manifest.append(
            {"label": label, "checkpoint": result["checkpoint"], "evaluation_revision": result["source_revision"]}
        )
        for index, (case, metrics) in enumerate(result["cases"].items()):
            data = trace[:, index * trials : (index + 1) * trials]
            valid = data[100:, :, 8] > 0.5
            column = 3 if case.startswith("v") else 5
            samples = np.where(valid, data[100:, :, column], np.nan)
            trial_means = np.nanmean(samples, axis=0)
            rows.append(
                {"experiment": label, "case": case, **metrics, "actual_trial_sd": float(np.nanstd(trial_means, ddof=1))}
            )
        data = trace[:, 3 * trials : 4 * trials]
        alive = data[:, :, 8] > 0.5
        initial_yaw = data[0, :, 16] - data[0, :, 7]
        yaw = np.unwrap(data[:, :, 16], axis=0) - initial_yaw
        initial_xy = data[0, :, 14:16] - result["dt"] * np.stack((np.cos(initial_yaw), np.sin(initial_yaw)), axis=-1)
        delta = data[:, :, 12:14] - initial_xy
        side = -np.sin(initial_yaw) * delta[:, :, 0] + np.cos(initial_yaw) * delta[:, :, 1]
        forward = np.cos(initial_yaw) * delta[:, :, 0] + np.sin(initial_yaw) * delta[:, :, 1]
        survivors = alive[-1]
        yaw_rows.append(
            {
                "experiment": label,
                "survivors": int(survivors.sum()),
                "period_heading_mean_deg": result["cases"]["v10"]["heading_mean_deg"],
                "final_abs_yaw_deg": float(np.rad2deg(np.abs(yaw[-1, survivors])).mean()) if survivors.any() else None,
                "final_abs_side_m": float(np.abs(side[-1, survivors]).mean()) if survivors.any() else None,
                "final_forward_m": float(forward[-1, survivors].mean()) if survivors.any() else None,
            }
        )
        cases = result["cases"]
        axes[0, 0].plot(
            [0.1, 0.2, 0.4, 1.0], [cases[c]["actual_vx"] for c in ["v01", "v02", "v04", "v10"]], "o-", label=label
        )
        axes[0, 1].plot(
            [-0.5, -0.1, 0.1, 0.5],
            [cases[c]["actual_wz"] for c in ["w05_neg", "w01_neg", "w01_pos", "w05_pos"]],
            "o-",
            label=label,
        )
        time = (np.arange(len(trace)) + 1) * result["dt"]
        axes[1, 0].plot(time, np.nanmean(np.where(alive, np.rad2deg(np.abs(yaw)), np.nan), axis=1), label=label)
        axes[1, 1].plot(time, np.nanmean(np.where(alive, np.abs(side), np.nan), axis=1), label=label)
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 1].plot([-0.5, 0.5], [-0.5, 0.5], "k--", alpha=0.5)
    titles = [
        "Forward speed (m/s)",
        "Yaw rate (rad/s)",
        "1 m/s: heading drift from initial heading",
        "1 m/s: lateral displacement from initial line",
    ]
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].set(xlabel="Command", ylabel="Actual mean")
    axes[0, 1].set(xlabel="Command", ylabel="Actual mean")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Mean absolute degrees")
    axes[1, 1].set(xlabel="Time (s)", ylabel="Mean absolute meters")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.directory / "comparison.png", dpi=160)
    for name, data in [("metrics", rows), ("yaw_drift", yaw_rows)]:
        keys = list(dict.fromkeys(key for row in data for key in row))
        with (args.directory / f"{name}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    (args.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Final checkpoint comparison",
        "",
        "All checkpoints: model_5999 (6000 training iterations). Evaluation seed 137, 16 trials per "
        "case, 20 s; mean policy; noisy observations; no pushes; head/body targets zero. Velocity "
        "statistics exclude the first 2 s. Command reference reanchors every 5 s; global "
        "yaw/lateral drift does not reanchor. Failures are excluded after termination and reported "
        "explicitly. Trial SD measures evaluation variability, not uncertainty across training "
        "seeds. Fast-mixed uses a reconstructed common evaluation environment; exact historical "
        "source parity is unavailable.",
        "",
        "## Speed and rotation",
        "",
        "| Experiment | Case | Actual mean | Trial SD | Linear RMSE | Yaw-rate RMSE | Failure fraction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        actual = row["actual_vx"] if row["case"].startswith("v") else row["actual_wz"]
        lines.append(
            f"| {row['experiment']} | {row['case']} | {actual:.4f} | {row['actual_trial_sd']:.4f}"
            f" | {row['linear_rmse']:.4f} | {row['yaw_rate_rmse']:.4f} | {row['failure_fraction']:.3f} |"
        )
    lines += [
        "",
        "## 1 m/s global drift",
        "",
        "| Experiment | Survivors | Period mean abs heading (deg) | Final abs heading (deg) | Final "
        "abs lateral (m) | Final forward (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in yaw_rows:
        lines.append(
            f"| {r['experiment']} | {r['survivors']} | {r['period_heading_mean_deg']:.3f}"
            f" | {r['final_abs_yaw_deg']:.3f} | {r['final_abs_side_m']:.3f}"
            f" | {r['final_forward_m']:.3f} |"
        )
    (args.directory / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(yaw_rows, indent=2))


if __name__ == "__main__":
    main()
