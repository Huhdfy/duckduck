# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write checkpoint evaluation tables and physical tracking curves from saved traces."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = ("yaw_obs", "yaw_reward", "pos_half")
MOVING = ("v01", "v03", "v06", "v10", "back", "side", "arc_left", "arc_right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    documents = {name: json.loads((args.folder / f"{name}.json").read_text()) for name in NAMES}
    rows = []
    for name, document in documents.items():
        for case, values in document["cases"].items():
            rows.append({"variant": name, "case": case, **values})
    with (args.folder / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# MicroDuck checkpoint 5000 evaluation",
        "",
        "Three deterministic policies, evaluation seed 137, 15 cases × 16 trials × 20 seconds per policy.",
        "Observation noise enabled; random interval pushes disabled; specified paired disturbances retained.",
        "Reference trajectories re-anchor every 5 seconds. Head/body pose commands are zero.",
        "Velocity averages exclude the first 2 seconds only. "
        "Fallen trials remain failed; post-reset samples are excluded.",
        "This is one training seed and one evaluation seed. It does not establish multi-seed significance.",
        "",
        "Moving-case aggregates below give equal weight to eight fixed translation commands.",
        "",
        "| Variant | Mean position error (m) | Mean absolute heading error (deg) "
        "| Linear RMSE (m/s) | Failed trials / 240 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, document in documents.items():
        cases = document["cases"]
        pos = mean(cases[c]["position_mean"] for c in MOVING)
        heading = mean(cases[c]["heading_mean_deg"] for c in MOVING)
        velocity = mean(cases[c]["linear_rmse"] for c in MOVING)
        failed = sum(round(c["failure_fraction"] * c["trials"]) for c in cases.values())
        lines.append(f"| {name} | {pos:.3f} | {heading:.2f} | {velocity:.3f} | {failed} |")
    lines.extend(
        [
            "",
            "| Case / target (vx, vy, wz) | Variant | Actual vx | Actual vy "
            "| Actual wz | Position error (m) | Heading error (deg) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in documents[NAMES[0]]["cases"]:
        for name in NAMES:
            r = documents[name]["cases"][case]
            lines.append(
                f"| {case} / {r['target']} | {name} | {r['actual_vx']:.3f} | {r['actual_vy']:.3f} "
                f"| {r['actual_wz']:.3f} | {r['position_mean']:.3f} | {r['heading_mean_deg']:.2f} |"
            )
    lines.extend(
        [
            "",
            "The speed_change target sequence is 0.1 → 0.6 → 0.1 → 0 m/s, each held for 5 seconds.",
            "push_mid/push_late add ±0.25 m/s world-Y velocity at t=7/9 seconds; "
            "yaw_push adds ±0.4 rad/s at t=7 seconds.",
            "The reported speed_change target tuple in the per-case table is the initial command only.",
            "",
            "![Tracking curves](comparison.png)",
            "",
        ]
    )
    (args.folder / "comparison.md").write_text("\n".join(lines))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, document in documents.items():
        trace = np.load(args.folder / f"{name}_traces.npz")["trace"]
        cases = list(document["cases"])
        n = document["trials_per_case"]
        time = (np.arange(len(trace)) + 1) * document["dt"]

        def data(case):
            index = cases.index(case) * n
            selected = trace[:, index : index + n].copy()
            selected[selected[:, :, 8] < 0.5] = np.nan
            return selected

        axes[0, 0].plot(time, np.nanmean(data("speed_change")[:, :, 3], axis=1), label=name)
        axes[0, 1].plot(time, np.rad2deg(np.nanmean(np.abs(data("v03")[:, :, 7]), axis=1)), label=name)
        axes[1, 0].plot(time, np.nanmean(data("v10")[:, :, 6], axis=1), label=name)
        axes[1, 1].plot(time, np.nanmean(data("spin_left")[:, :, 5], axis=1), label=name)
        if name == NAMES[0]:
            axes[0, 0].plot(time, data("speed_change")[:, 0, 0], "k--", label="command")
    axes[0, 0].set(title="Speed changes", ylabel="Forward velocity (m/s)")
    axes[0, 1].set(title="0.3 m/s straight: heading error", ylabel="Absolute heading error (deg)")
    axes[1, 0].set(title="1.0 m/s straight: position error", ylabel="Position error (m)")
    axes[1, 1].set(title="0.8 rad/s spin: actual yaw rate", ylabel="Yaw rate (rad/s)")
    axes[1, 1].axhline(0.8, color="black", linestyle="--", label="command")
    for axis in axes.flat:
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(args.folder / "comparison.png", dpi=160)
    plt.close(figure)
    print(args.folder / "comparison.md")


if __name__ == "__main__":
    main()
