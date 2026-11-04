# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paired command-suite evaluation of MicroDuck tracking checkpoints without training."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from microduck_tracking_experiment import TASK, VARIANTS, make_config
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

CASES = {
    "stand": (0.0, 0.0, 0.0),
    "v01": (0.1, 0.0, 0.0),
    "v03": (0.3, 0.0, 0.0),
    "v06": (0.6, 0.0, 0.0),
    "v10": (1.0, 0.0, 0.0),
    "back": (-0.3, 0.0, 0.0),
    "side": (0.0, 0.2, 0.0),
    "arc_left": (0.4, 0.0, 0.5),
    "arc_right": (0.4, 0.0, -0.5),
    "spin_left": (0.0, 0.0, 0.8),
    "spin_right": (0.0, 0.0, -0.8),
    "speed_change": (0.1, 0.0, 0.0),
    "push_mid": (0.4, 0.0, 0.0),
    "push_late": (0.4, 0.0, 0.0),
    "yaw_push": (0.4, 0.0, 0.0),
}
COLUMNS = [
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "actual_vx",
    "actual_vy",
    "actual_wz",
    "position_error",
    "heading_error",
    "alive",
    "ref_forward_error",
    "ref_side_error",
    "estimator_residual",
    "actual_x",
    "actual_y",
    "reference_x",
    "reference_y",
    "actual_yaw",
    "action_delta_sq",
]


def summarize(trace, dt, trials):
    rows = {}
    # Velocity statistics exclude initial startup only; all later command transitions are included.
    velocity_start = round(2.0 / dt)
    cycle_ends = np.arange(round(5.0 / dt) - 1, len(trace), round(5.0 / dt))
    for case_index, case in enumerate(CASES):
        data = trace[:, case_index * trials : (case_index + 1) * trials]
        valid = data[:, :, 8] > 0.5
        velocity_valid = valid.copy()
        velocity_valid[:velocity_start] = False
        per_trial = []
        for trial in range(trials):
            v = velocity_valid[:, trial]
            p = valid[:, trial]
            if not v.any() or not p.any():
                continue
            sample = data[:, trial]
            velocity = sample[v]
            planar_sq = ((velocity[:, 3:5] - velocity[:, :2]) ** 2).sum(axis=-1)
            per_trial.append(
                {
                    "actual_vx": float(velocity[:, 3].mean()),
                    "actual_vy": float(velocity[:, 4].mean()),
                    "actual_wz": float(velocity[:, 5].mean()),
                    "linear_rmse": float(np.sqrt(planar_sq.mean())),
                    "yaw_rate_rmse": float(np.sqrt(((velocity[:, 5] - velocity[:, 2]) ** 2).mean())),
                    "position_mean": float(sample[p, 6].mean()),
                    "heading_mean_deg": float(np.rad2deg(np.abs(sample[p, 7])).mean()),
                    "side_error_mean": float(np.abs(sample[p, 10]).mean()),
                    "estimator_rmse_deg": float(np.rad2deg(np.sqrt((sample[p, 11] ** 2).mean()))),
                    "action_delta_sq": float(velocity[:, 17].mean()),
                }
            )
        row = {key: float(np.mean([r[key] for r in per_trial])) for key in per_trial[0]} if per_trial else {}
        row["target"] = list(CASES[case])
        row["trials"] = trials
        row["trials_with_velocity_samples"] = len(per_trial)
        row["failure_fraction"] = float(1.0 - valid[-1].mean())
        row["mean_survival_s"] = float(valid.sum(axis=0).mean() * dt)
        row["position_p90"] = float(np.quantile(data[:, :, 6][valid], 0.9)) if valid.any() else None
        endpoint_valid = valid[cycle_ends]
        row["cycle_end_position_mean"] = (
            float(data[cycle_ends, :, 6][endpoint_valid].mean()) if endpoint_valid.any() else None
        )
        moving = velocity_valid & (np.linalg.norm(data[:, :, :2], axis=-1) >= 0.05)
        if moving.any():
            targets = data[:, :, :2][moving]
            actual = data[:, :, 3:5][moving]
            row["speed_completion_ratio"] = float(((targets * actual).sum(-1) / (targets**2).sum(-1)).mean())
        rows[case] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--suite", choices=("original", "requested", "speedup", "comprehensive"), default="original")
    parser.add_argument("--baseline", action="store_true", help="Evaluate the original 61/76-observation policy.")
    args = parser.parse_args()
    if args.suite == "requested":
        CASES.clear()
        CASES.update(
            {
                "v01": (0.1, 0.0, 0.0),
                "v02": (0.2, 0.0, 0.0),
                "v04": (0.4, 0.0, 0.0),
                "v10": (1.0, 0.0, 0.0),
                "w01_pos": (0.0, 0.0, 0.1),
                "w01_neg": (0.0, 0.0, -0.1),
                "w05_pos": (0.0, 0.0, 0.5),
                "w05_neg": (0.0, 0.0, -0.5),
            }
        )
    if args.suite == "speedup":
        CASES.clear()
        CASES.update({f"vx_{speed:g}": (speed, 0.0, 0.0) for speed in (-1.0, -0.5, 0.1, 0.4, 1.0, 1.2, 1.5)})
    if args.suite == "comprehensive":
        CASES.clear()
        CASES.update({f"vx_{speed:g}": (speed, 0.0, 0.0) for speed in (0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.5)})
        CASES.update({f"wz_{rate:g}": (0.0, 0.0, rate) for rate in (0.1, -0.1, 0.5, -0.5)})
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    cfg = make_config(args.variant, len(CASES) * args.trials, args.seed)
    if args.suite == "speedup":
        cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
    if args.baseline:
        cfg.observations.policy.heading_error = None
        cfg.observations.critic.heading_error = None
    cfg.episode_length_s = 30.0
    cfg.events.push_robot = None
    for key in list(vars(cfg.curriculum)):
        if not key.startswith("_"):
            setattr(cfg.curriculum, key, None)
    cfg.log_dir = str(args.output)
    sim_utils.create_new_stage()
    env = gym.make(TASK, cfg=cfg).unwrapped
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True, "critic": True})
    policy = runner.get_inference_policy(device=env.device)
    torch.manual_seed(args.seed)
    env.reset()
    term = env.command_manager.get_term("base_velocity")
    robot = env.scene["robot"]
    for name in ("head_pose", "body_pose"):
        other = env.command_manager.get_term(name)
        other.command[:] = 0
        other.time_left[:] = 1e9
    device = env.device
    commands = torch.tensor(list(CASES.values()), device=device).repeat_interleave(args.trials, dim=0)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    ref_p = robot.data.root_link_pos_w.torch[:, :2].clone()
    ref_yaw = euler_xyz_from_quat(robot.data.root_link_quat_w.torch)[2].clone()
    steps = round(20.0 / env.step_dt)
    period = round(5.0 / env.step_dt)
    previous_action = torch.zeros(env.num_envs, 14, device=device)
    traces = []
    with torch.inference_mode():
        for step in range(steps):
            if step % period == 0:
                term._resample(torch.arange(env.num_envs, device=device))
                sequence_speed = (0.1, 0.6, 0.1, 0.0)[step // period]
                if "speed_change" in CASES:
                    index = list(CASES).index("speed_change") * args.trials
                    commands[index : index + args.trials, 0] = sequence_speed
                term.vel_command_b[:] = commands
                term.is_standing_env[:] = commands.abs().sum(-1) == 0
                term.time_left[:] = 1e9
                ref_p = robot.data.root_link_pos_w.torch[:, :2].clone()
                ref_yaw = euler_xyz_from_quat(robot.data.root_link_quat_w.torch)[2].clone()
            for case, push_time, axis, magnitude in (
                ("push_mid", 7, 1, 0.25),
                ("push_late", 9, 1, 0.25),
                ("yaw_push", 7, 5, 0.4),
            ):
                if case in CASES and step == round(push_time / env.step_dt):
                    index = list(CASES).index(case) * args.trials
                    ids = torch.arange(index, index + args.trials, device=device)
                    velocity = robot.data.root_vel_w.torch[ids].clone()
                    velocity[:, axis] += torch.where(
                        torch.arange(args.trials, device=device) % 2 == 0, magnitude, -magnitude
                    )
                    robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
            # Reset commands for failed, auto-reset environments too; their subsequent samples remain excluded.
            term.vel_command_b[:] = commands
            term.is_standing_env[:] = commands.abs().sum(-1) == 0
            term.time_left[:] = 1e9
            obs = TensorDict(env.observation_manager.compute(), batch_size=[env.num_envs])
            action = policy(obs)
            midpoint = ref_yaw + 0.5 * commands[:, 2] * env.step_dt
            c, s = torch.cos(midpoint), torch.sin(midpoint)
            ref_p += (
                torch.stack((c * commands[:, 0] - s * commands[:, 1], s * commands[:, 0] + c * commands[:, 1]), dim=-1)
                * env.step_dt
            )
            ref_yaw = wrap_to_pi(ref_yaw + commands[:, 2] * env.step_dt)
            _, _, dones, _ = wrapped.step(action)
            policy.reset(dones)
            alive &= ~dones.bool()
            pos = robot.data.root_link_pos_w.torch[:, :2]
            actual_yaw = euler_xyz_from_quat(robot.data.root_link_quat_w.torch)[2]
            actual_v = quat_apply_inverse(
                yaw_quat(robot.data.root_link_quat_w.torch), robot.data.root_link_lin_vel_w.torch
            )[:, :2]
            actual_w = robot.data.root_link_ang_vel_w.torch[:, 2:3]
            error = pos - ref_p
            heading_error = wrap_to_pi(actual_yaw - ref_yaw)
            c, s = torch.cos(ref_yaw), torch.sin(ref_yaw)
            local_error = torch.stack((c * error[:, 0] + s * error[:, 1], -s * error[:, 0] + c * error[:, 1]), dim=-1)
            estimator_residual = wrap_to_pi(term.yaw_error_estimate - heading_error)
            sample = torch.cat(
                (
                    commands,
                    actual_v,
                    actual_w,
                    error.norm(dim=-1, keepdim=True),
                    heading_error[:, None],
                    alive[:, None],
                    local_error,
                    estimator_residual[:, None],
                    pos,
                    ref_p,
                    actual_yaw[:, None],
                    (action - previous_action).square().mean(-1, keepdim=True),
                ),
                dim=-1,
            )
            traces.append(sample.cpu().numpy())
            previous_action = action.clone()
    trace = np.stack(traces)
    np.savez_compressed(args.output / f"{args.variant}_traces.npz", trace=trace)
    result = {
        "variant": args.variant,
        "suite": args.suite,
        "baseline": args.baseline,
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dt": env.step_dt,
        "steps": steps,
        "trials_per_case": args.trials,
        "cycle_s": 5.0,
        "velocity_startup_exclusion_s": 2.0,
        "observation_noise": True,
        "random_pushes": False,
        "columns": COLUMNS,
        "cases": summarize(trace, env.step_dt, args.trials),
    }
    (args.output / f"{args.variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("EVALUATION_COMPLETE", args.variant, flush=True)
    wrapped.close()


if __name__ == "__main__":
    main()
