# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare MicroDuck checkpoints on fixed straight-line velocity commands.

Run from the repository root::

    uv run python examples/demos/microduck_evaluate.py --run_dir logs/rsl_rl/microduck_velocity_flat/RUN

Outputs per-trial CSV, per-step compressed CSV, summary CSV, configuration, and
standalone PNG/SVG figures. Failed trials stop contributing at their first failure;
the environment's internal autoresets never create additional evaluation trials.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import to_dict

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--run_dir", type=Path, required=True)
parser.add_argument("--iterations", type=int, nargs="+", default=[0, 2000, 8000, 16000, 23999])
parser.add_argument("--speeds", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
parser.add_argument("--num_envs", type=int, default=16, help="Independent trials per speed and checkpoint.")
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--warmup", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=None)
add_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or not 0 <= args.warmup < args.duration:
    parser.error("Require num_envs >= 1 and 0 <= warmup < duration.")
if any(not math.isfinite(s) or not 0 <= s <= 0.4 for s in args.speeds):
    parser.error("Speeds must be finite and inside the trained forward range [0, 0.4] m/s.")
if len(set(args.iterations)) != len(args.iterations) or len(set(args.speeds)) != len(args.speeds):
    parser.error("Iterations and speeds must be unique.")

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab_rl.entrypoints.common import dump_train_configs
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.contrib.microduck.mdp.diagnostics import DiagnosticAccumulator
from isaaclab_tasks.utils import resolve_task_config

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty result table with stable column ordering."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values) -> float:
    """Average valid per-trial values without giving longer trials extra weight."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def summarize(trials: list[dict]) -> list[dict]:
    """Keep completion denominators and available steady-state sample counts explicit."""
    rows = []
    for iteration in args.iterations:
        for speed in args.speeds:
            group = [r for r in trials if r["iteration"] == iteration and r["target_mps"] == speed]
            valid = [r for r in group if r["stable_samples"] > 0]
            vx = [r["mean_vx_mps"] for r in valid]
            rows.append(
                {
                    "iteration": iteration,
                    "target_mps": speed,
                    "trials": len(group),
                    "completed": sum(r["completed"] for r in group),
                    "completion_pct": 100 * sum(r["completed"] for r in group) / len(group),
                    "stable_trials": len(valid),
                    "mean_vx_mps": finite_mean(vx),
                    "between_trial_vx_sd_mps": float(np.std(vx, ddof=1)) if len(vx) > 1 else math.nan,
                    "planar_error_mps": finite_mean([r["planar_error_mps"] for r in valid]),
                    "within_trial_vx_sd_mps": finite_mean([r["vx_sd_mps"] for r in valid]),
                    "mean_heading_error_deg": finite_mean([r["heading_error_deg"] for r in valid]),
                    "mean_survival_s": finite_mean([r["survival_s"] for r in group]),
                    "falls": sum(r["fell_over"] for r in group),
                    "nan_failures": sum(r["nan_state"] for r in group),
                }
            )
    return rows


def plot_results(summary: list[dict], output: Path) -> None:
    """Export comparison figures; spread is trial SD, never a confidence interval."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    panels = [
        (axes[0, 0], "mean_vx_mps", "Actual forward speed (m/s)", "Speed tracking"),
        (axes[0, 1], "completion_pct", "Completed trials (%)", f"{args.duration:g}-second completion"),
        (axes[1, 0], "planar_error_mps", "Mean planar error (m/s)", "Tracking error: lower is better"),
        (axes[1, 1], "within_trial_vx_sd_mps", "Within-trial speed SD (m/s)", "Speed fluctuation: lower is better"),
    ]
    for ci, iteration in enumerate(args.iterations):
        rows = [r for r in summary if r["iteration"] == iteration]
        x = [r["target_mps"] for r in rows]
        color, marker = colors[ci % len(colors)], markers[ci % len(markers)]
        for ax, field, ylabel, title in panels:
            y = [r[field] for r in rows]
            ax.plot(x, y, marker=marker, color=color, label=f"{iteration:,}", linewidth=1.8)
            ax.set(xlabel="Target forward speed (m/s)", ylabel=ylabel, title=title)
            ax.set_xticks(args.speeds)
            ax.grid(alpha=0.22)
        # No stable samples => missing point. Bands show variability, not estimation certainty.
        means = np.asarray([r["mean_vx_mps"] for r in rows])
        spread = np.asarray([r["between_trial_vx_sd_mps"] for r in rows])
        axes[0, 0].fill_between(x, means - spread, means + spread, color=color, alpha=0.10)
    axes[0, 0].plot(args.speeds, args.speeds, "--", color="#555555", label="Ideal")
    axes[0, 1].set_ylim(-3, 103)
    axes[0, 0].legend(title="Checkpoint iteration", fontsize=9)
    fig.suptitle("MicroDuck | fixed-speed checkpoint comparison", fontsize=17)
    fig.supxlabel(
        f"{args.num_envs} trials / speed; {args.duration:g} s; first {args.warmup:g} s excluded from speed metrics. "
        "Shading: between-trial SD.\nSpeed metrics use available pre-failure samples; "
        "completion includes every trial. Missing points mean no steady-state samples.",
        fontsize=10,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for si, speed in enumerate(args.speeds):
        rows = [r for r in summary if r["target_mps"] == speed]
        for ax, field in zip(axes, ("planar_error_mps", "completion_pct")):
            ax.plot(
                [r["iteration"] for r in rows],
                [r[field] for r in rows],
                marker=markers[si % 5],
                color=colors[si % 5],
                label=f"{speed:g} m/s",
            )
            ax.set_xlabel("Checkpoint iteration")
            ax.grid(alpha=0.22)
    axes[0].set(title="Tracking error over training", ylabel="Mean planar error (m/s)")
    axes[1].set(title="Completion over training", ylabel="Completed trials (%)", ylim=(-3, 103))
    axes[1].legend(title="Target speed")
    fig.suptitle("Fixed evaluation conditions across training stages", fontsize=15)
    fig.savefig(output / "training_stages.png", dpi=180)
    fig.savefig(output / "training_stages.svg")
    plt.close(fig)


def main() -> None:
    """Evaluate paired checkpoints with pre-reset state, gait and actual reward contributions."""
    checkpoints = [args.run_dir / f"model_{i}.pt" for i in args.iterations]
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output or Path("logs/evaluations/microduck_velocity") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", play_mode=True, overrides=())
    # Replay only reward-weight schedules. Keep the previous evaluation's physical randomization protocol.
    weight_schedules = [
        term.params
        for term in vars(env_cfg.curriculum).values()
        if term is not None and term.params["address"].startswith("rewards.")
    ]
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.duration + 2 * env_cfg.sim.dt * env_cfg.decimation
    env_cfg.curriculum = None
    env_cfg.seed = args.seed
    env_cfg.recorders.diagnostics.capture_samples = True
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_forward_envs = 0.0
    command_cfg.rel_turn_in_place_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    for pose_cfg in (env_cfg.commands.head_pose, env_cfg.commands.body_pose):
        pose_cfg.ranges = tuple((0.0, 0.0) for _ in pose_cfg.ranges)
    if args.device is not None:
        env_cfg.sim.device = args.device
    manifest = {
        "task": TASK,
        "seed": args.seed,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "speeds_mps": args.speeds,
        "trials_per_condition": args.num_envs,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoints": [
            {"iteration": i, "path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for i, p in zip(args.iterations, checkpoints)
        ],
        "protocol": "Deterministic actor; zero head/body commands; no pushes; retained startup/reset randomization; "
        "paired reset seeds and initial physical properties. Reward weights use checkpoint-stage schedules. "
        "Samples are post-physics/post-reward, before reset/resampling; first failure ends scoring. "
        "Warmup excludes t <= warmup_s. Primary velocity is root-link origin in body coordinates; COM velocity "
        "is separately recorded. Means and RMS are computed per trial, then trials are equally weighted.",
        "reward_units": "reward_weighted = weight * raw; reward_step = reward_weighted * step_dt; "
        "sum(reward_step) equals returned environment reward. Zero-weight terms have zero contribution.",
        "foot_definitions": "Contact force norm > 1 N at control frequency. Landing is ankle origin in world/body "
        "frames. Sole-center horizontal speed during contact is a slip proxy. First/last incomplete stances "
        "are censored; stance summary includes only complete stances within the scored interval.",
        "joint_definitions": "joint_target is articulation position target after encoder-bias correction, "
        "before BAM internal delay; joint_pos/vel are actual servo coordinates. Action order is servo order.",
    }
    (output / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--", "source/", "examples/"]))
    (output / "evaluation_script.py").write_text(Path(__file__).read_text())
    dump_train_configs(str(output), env_cfg, agent_cfg)
    trials, diagnostic_trials, joint_trials, training_logs = [], [], [], []
    with launch_simulation(env_cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv

        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        try:
            raw, device = env.unwrapped, env.unwrapped.device
            recorder = raw.recorder_manager._terms["diagnostics"]
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=device)
            steps = math.ceil(args.duration / raw.step_dt)
            warmup_steps = math.ceil(args.warmup / raw.step_dt)
            manifest["step_dt_s"] = raw.step_dt
            manifest["joint_names"] = recorder.joint_names
            manifest["reward_weights"] = {}
            baseline_initial = {}
            with gzip.open(output / "samples.csv.gz", "wt", newline="") as samples_file:
                writer = csv.writer(samples_file)
                for iteration, checkpoint in zip(args.iterations, checkpoints):
                    runner.load(str(checkpoint))
                    policy = runner.get_inference_policy(device=device)
                    for schedule in weight_schedules:
                        _, name, _ = schedule["address"].split(".")
                        cfg = raw.reward_manager.get_term_cfg(name)
                        for start, value in schedule["modify_params"]["stages"]:
                            if iteration * agent_cfg.num_steps_per_env >= start:
                                cfg.weight = value
                        raw.reward_manager.set_term_cfg(name, cfg)
                    manifest["reward_weights"][str(iteration)] = {
                        name: raw.reward_manager.get_term_cfg(name).weight for name in raw.reward_manager.active_terms
                    }
                    for si, speed in enumerate(args.speeds):
                        raw.reset(seed=args.seed + si)
                        policy.reset(torch.ones(args.num_envs, dtype=torch.bool, device=device))
                        robot = raw.scene["robot"]
                        initial = torch.cat(
                            (
                                robot.data.root_pos_w.torch,
                                robot.data.root_quat_w.torch,
                                robot.data.joint_pos.torch,
                                robot.data.body_com_pose_b.torch.flatten(start_dim=1),
                                robot.data.body_mass.torch,
                            ),
                            dim=1,
                        ).clone()
                        if si in baseline_initial:
                            torch.testing.assert_close(initial, baseline_initial[si], atol=1e-6, rtol=0)
                        else:
                            baseline_initial[si] = initial
                        initial_heading = robot.data.heading_w.torch.clone()
                        active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
                        survival = torch.full((args.num_envs,), args.duration, device=device)
                        falls = torch.zeros_like(active)
                        broken = torch.zeros_like(active)
                        trace = None
                        accumulator = None
                        with torch.no_grad():
                            for step in range(steps):
                                command = raw.command_manager.get_command("base_velocity")
                                command.zero_()
                                command[:, 0] = speed
                                obs = TensorDict(raw.observation_manager.compute(), batch_size=[args.num_envs])
                                _, returned_reward, dones, _ = env.step(policy(obs))
                                policy.reset(dones)
                                if trace is None:
                                    names = recorder.sample_names
                                    columns = {name: i for i, name in enumerate(names)}
                                    trace = torch.full((steps, args.num_envs, len(names)), torch.nan, device=device)
                                    accumulator = DiagnosticAccumulator(args.num_envs, recorder.metric_names, device)
                                    if not manifest.get("sample_columns"):
                                        manifest["sample_columns"] = names
                                        writer.writerow(
                                            ["iteration", "target_mps", "env_id", "time_s", "stable_phase", *names]
                                        )
                                trace[step] = torch.where(active[:, None], recorder.snapshot, torch.nan)
                                torch.testing.assert_close(
                                    recorder.snapshot[:, columns["reward_total_step"]], returned_reward, equal_nan=True
                                )
                                if step >= warmup_steps:
                                    metrics = recorder.metric_values.clone()
                                    for side in ("left", "right"):
                                        index = recorder.metric_names.index(f"{side}_stance_duration_mean_s")
                                        metrics[:, index] = torch.where(
                                            metrics[:, index] <= (step + 1) * raw.step_dt - args.warmup,
                                            metrics[:, index],
                                            torch.nan,
                                        )
                                    accumulator.update(metrics, active)
                                failed = active & raw.reset_terminated
                                survival[failed] = (step + 1) * raw.step_dt
                                falls |= failed & raw.termination_manager.get_term("fell_over")
                                broken |= failed & raw.termination_manager.get_term("nan_state")
                                active &= ~failed
                                if not active.any():
                                    break
                        data = trace.cpu().numpy()
                        diagnostics = accumulator.values().cpu().numpy()
                        counts = accumulator.counts.cpu().numpy()
                        survival_np, falls_np, broken_np = (x.cpu().numpy() for x in (survival, falls, broken))
                        for eid in range(args.num_envs):
                            stable = data[warmup_steps:, eid]
                            valid = np.isfinite(stable[:, columns["link_vx_b_mps"]])
                            stable = stable[valid]
                            vx, vy = stable[:, columns["link_vx_b_mps"]], stable[:, columns["link_vy_b_mps"]]
                            heading = stable[:, columns["heading_w_rad"]] - float(initial_heading[eid])
                            heading = np.abs(np.arctan2(np.sin(heading), np.cos(heading))) * 180 / math.pi
                            identity = {"iteration": iteration, "target_mps": speed, "env_id": eid}
                            trials.append(
                                {
                                    **identity,
                                    "seed": args.seed + si,
                                    "completed": int(not (falls_np[eid] or broken_np[eid])),
                                    "survival_s": float(survival_np[eid]),
                                    "fell_over": int(falls_np[eid]),
                                    "nan_state": int(broken_np[eid]),
                                    "stable_samples": len(stable),
                                    "mean_vx_mps": float(vx.mean()) if len(stable) else math.nan,
                                    "planar_error_mps": float(np.hypot(vx - speed, vy).mean())
                                    if len(stable)
                                    else math.nan,
                                    "vx_sd_mps": float(vx.std()) if len(stable) else math.nan,
                                    "heading_error_deg": float(heading.mean()) if len(stable) else math.nan,
                                }
                            )
                            diagnostic_trials.append(
                                {
                                    **identity,
                                    **dict(zip(recorder.metric_names, diagnostics[eid].tolist())),
                                    **{
                                        f"count/{name}": int(count)
                                        for name, count in zip(recorder.metric_names, counts[eid])
                                    },
                                }
                            )
                            for joint in recorder.joint_names:
                                target = stable[:, columns[f"joint_target/{joint}"]]
                                actual = stable[:, columns[f"joint_pos/{joint}"]]
                                joint_trials.append(
                                    {
                                        **identity,
                                        "joint": joint,
                                        "target_sd_rad": float(target.std()) if len(stable) else math.nan,
                                        "actual_sd_rad": float(actual.std()) if len(stable) else math.nan,
                                        "target_range_rad": float(np.ptp(target)) if len(stable) else math.nan,
                                        "actual_range_rad": float(np.ptp(actual)) if len(stable) else math.nan,
                                        "error_mean_rad": float((target - actual).mean()) if len(stable) else math.nan,
                                        "error_rms_rad": float(np.sqrt(np.mean((target - actual) ** 2)))
                                        if len(stable)
                                        else math.nan,
                                    }
                                )
                            for step in range(steps):
                                if np.isfinite(data[step, eid, columns["command_x"]]):
                                    writer.writerow(
                                        [
                                            iteration,
                                            speed,
                                            eid,
                                            round((step + 1) * raw.step_dt, 6),
                                            int(step >= warmup_steps),
                                            *data[step, eid].tolist(),
                                        ]
                                    )
                        # Exercise the same episode-log path used by training, including surviving episodes.
                        raw.reset(seed=args.seed + si)
                        training_logs.append(
                            {
                                "iteration": iteration,
                                "target_mps": speed,
                                "values": {
                                    key: float(value)
                                    for key, value in raw.extras["log"].items()
                                    if key.startswith("Diagnostics/")
                                },
                            }
                        )
                        print(
                            f"Checkpoint {iteration}, target {speed:.1f}: "
                            f"{sum(r['completed'] for r in trials[-args.num_envs :])}/{args.num_envs} completed",
                            flush=True,
                        )
                        write_csv(output / "trials.csv", trials)
                        write_csv(output / "diagnostic_trials.csv", diagnostic_trials)
                        write_csv(output / "joint_trials.csv", joint_trials)
                        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
                        (output / "training_log_samples.json").write_text(json.dumps(training_logs, indent=2))
        finally:
            env.close()
    summary = summarize(trials)
    write_csv(output / "summary.csv", summary)
    diagnostic_summary = []
    for iteration in args.iterations:
        for speed in args.speeds:
            group = [r for r in diagnostic_trials if r["iteration"] == iteration and r["target_mps"] == speed]
            diagnostic_summary.append(
                {
                    "iteration": iteration,
                    "target_mps": speed,
                    **{name: finite_mean([r[name] for r in group]) for name in recorder.metric_names},
                }
            )
    write_csv(output / "diagnostic_summary.csv", diagnostic_summary)
    plot_results(summary, output)
    print(f"Evaluation complete: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
