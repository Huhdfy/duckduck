# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent continuation variants and paired rollout measurement for MicroDuck.

The queue passes one immutable JSON specification through ISAACLAB_MICRODUCK_EXPERIMENT_SPEC.
Only the child process registers the config factory; normal task registration remains unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
VARIANTS = (
    "baseline",
    "velocity_weight",
    "velocity_std",
    "joint_tracking",
    "v_std",
    "w_std",
    "w_rational",
    "competitor_half",
)
SPEC_ENV = "ISAACLAB_MICRODUCK_EXPERIMENT_SPEC"


def read_spec() -> dict:
    """Read the explicit per-process experiment specification."""
    return json.loads(Path(os.environ[SPEC_ENV]).read_text())


def register_experiment() -> None:
    """External-task callback for the standard training entrypoint, isolated to this process."""
    import gymnasium as gym

    gym.spec(TASK).kwargs["env_cfg_entry_point"] = f"{__name__}:experiment_env_cfg"


def apply_variant(cfg, variant: str, calibration: dict | None = None, competitor: str | None = None):
    """Apply exactly one experiment, leaving the supplied baseline's other settings intact."""
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg

    from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

    from .mdp.rewards import joint_tracking_error_l2

    if variant not in VARIANTS:
        raise ValueError(f"Unknown MicroDuck experiment: {variant}")
    if variant == "velocity_weight":
        cfg.rewards.track_lin_vel.weight = 4.0
        cfg.rewards.track_ang_vel.weight = 4.0
    elif variant in ("velocity_std", "v_std", "w_std", "w_rational"):
        if variant in ("velocity_std", "v_std"):
            cfg.rewards.track_lin_vel.params["std"] = 0.15
        if variant in ("velocity_std", "w_std", "w_rational"):
            cfg.rewards.track_ang_vel.params["std"] = 0.35
        if variant == "w_rational":
            cfg.rewards.track_ang_vel.params["shape"] = "rational"
    elif variant == "competitor_half":
        if competitor in (None, "track_lin_vel", "track_ang_vel"):
            raise ValueError("A non-velocity competitor reward must be selected explicitly.")
        term = getattr(cfg.rewards, competitor)
        if term is None:
            raise ValueError(f"Disabled competitor reward: {competitor}")
        term.weight *= 0.5
        # Scale the schedule too: reset-time curricula must not undo the intervention.
        for curriculum in vars(cfg.curriculum).values():
            if curriculum is not None and curriculum.params.get("address") == f"rewards.{competitor}.weight":
                stages = curriculum.params["modify_params"]["stages"]
                curriculum.params["modify_params"]["stages"] = [(step, value * 0.5) for step, value in stages]
    if calibration is not None:
        if calibration["joint_names"] != MICRODUCK_JOINT_NAMES:
            raise ValueError("Calibration joint order does not match the policy action order.")
        cfg.recorders.diagnostics.tracking_tolerance = calibration["tolerance_rad"]
    if variant == "joint_tracking":
        if calibration is None:
            raise ValueError("Joint tracking experiment requires frozen calibration.")
        cfg.rewards.joint_tracking_error = RewardTermCfg(
            func=joint_tracking_error_l2,
            weight=-calibration["weight_magnitude"],
            params={
                "tolerance": calibration["tolerance_rad"],
                "scale": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True),
            },
        )
    return cfg


def experiment_env_cfg():
    """Construct a variant without modifying the source baseline or the saved checkpoint."""
    from .flat_env_cfg import MicroDuckVelocityFlatEnvCfg

    spec = read_spec()
    calibration = json.loads(Path(spec["calibration"]).read_text()) if spec.get("calibration") else None
    return apply_variant(MicroDuckVelocityFlatEnvCfg(), spec["variant"], calibration, spec.get("competitor"))


def config_differences(saved, current, prefix: str = "") -> list[dict]:
    """Report changed leaves, including additions/removals; used to gate experiment preparation."""
    if isinstance(saved, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(saved.keys() | current.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in saved or key not in current:
                differences.append({"path": path, "saved": saved.get(key), "current": current.get(key)})
            else:
                differences.extend(config_differences(saved[key], current[key], path))
        return differences
    return [] if saved == current else [{"path": prefix, "saved": saved, "current": current}]


def check_baseline(checkpoint: Path, output: Path) -> dict:
    """Check saved/current behavior before allowing any rollout or continuation."""
    from isaaclab.utils import to_dict

    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=())
    env_cfg.seed = agent_cfg.seed = 123
    env_cfg.scene.num_envs = 4096
    differences = []

    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value.replace("{ENV_REGEX_NS}", "/World/envs/env_[^/]+") if isinstance(value, str) else value

    # BaseLoader compares YAML representations, including tuples/slices, without constructing objects.
    for name, cfg in (("env", env_cfg), ("agent", agent_cfg)):
        saved = yaml.load((checkpoint.parent / "params" / f"{name}.yaml").read_text(), Loader=yaml.BaseLoader)
        current = yaml.load(yaml.dump(to_dict(cfg)), Loader=yaml.BaseLoader)
        if name == "env":
            current["scene"]["terrain"]["num_envs"] = current["scene"]["num_envs"]
            current["scene"]["terrain"]["env_spacing"] = current["scene"]["env_spacing"]
        differences.extend(config_differences(canonical(saved), canonical(current), name))
    allowed = {
        "env.log_dir",
        "env.sim.log_dir",
        "env.sim.visualizer_cfgs",
        "env.sim.default_visualizer_cfg",
        "env.rewards.joint_tracking_error",
        "env.recorders.diagnostics.tracking_tolerance",
        "agent.max_iterations",
        "agent.save_interval",
        "agent.run_name",
        "agent.load_run",
        "agent.load_checkpoint",
        "env.scene.robot.spawn.spawn_path",
        "env.scene.sky_light.spawn.spawn_path",
    }
    for difference in differences:
        path = difference["path"]
        difference["allowed"] = path in allowed or path.endswith((".joint_ids", ".body_ids"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["iter"] != 6000:
        raise ValueError("Expected the original iter=6000 checkpoint.")
    result = {
        "differences": differences,
        "checkpoint_iteration": state["iter"],
        "resume_env_steps": 6001 * 24,
        "learning_rate": state["optimizer_state_dict"]["param_groups"][0]["lr"],
    }
    (output / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    unexpected = [d for d in differences if not d["allowed"]]
    if unexpected:
        raise ValueError(f"Baseline configuration changed; inspect {output / 'preflight.json'}: {unexpected}")
    mature = env_cfg.copy()
    freeze_curriculum(mature, result["resume_env_steps"])
    audit = {
        "step_dt": mature.sim.dt * mature.decimation,
        "linear_error": "(vx-cmd_vx)^2 + (vy-cmd_vy)^2 + vz^2; root_link body frame",
        "angular_error": "(wz-cmd_wz)^2 + wx^2 + wy^2; root_link body frame",
        "kernel": "exp(-E/std**2); no Gaussian 1/2 factor; non-finite outputs replaced by zero",
        "aggregation": "sum(weight * raw_reward) * step_dt; no total reward clipping or normalization",
        "rewards": {
            name: {
                "weight": term.weight,
                "function": term.func.__name__,
                "params": yaml.load(yaml.dump(to_dict(term.params)), Loader=yaml.BaseLoader),
            }
            for name, term in vars(mature.rewards).items()
            if term is not None
        },
        "training_commands": yaml.load(yaml.dump(to_dict(mature.commands)), Loader=yaml.BaseLoader),
    }
    (output / "reward_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return result


def command_cases(cfg) -> list[tuple[str, tuple[float, float, float]]]:
    """Fixed commands shared by calibration and evaluation, with turns inside saved limits."""
    low, high = cfg.commands.base_velocity.ranges.ang_vel_z
    return [
        ("standing", (0.0, 0.0, 0.0)),
        *((f"forward_{speed:g}", (speed, 0.0, 0.0)) for speed in (0.1, 0.2, 0.3, 0.4)),
        ("turn_negative", (0.0, 0.0, low / 2)),
        ("turn_positive", (0.0, 0.0, high / 2)),
    ]


def freeze_curriculum(cfg, steps: int) -> None:
    """Resolve this task's staged curriculum before fixed-command measurement, then freeze it.

    Training continues to use the normal manager. Measurement freezes schedules so a reset cannot
    overwrite a fixed turn/forward command or the evaluation's zero head commands.
    """
    for term in vars(cfg.curriculum).values():
        if term is None:
            continue
        params = term.params
        address = params["address"].split(".")
        owner = cfg
        for key in address[:-1]:
            owner = owner[key] if isinstance(owner, dict) else getattr(owner, key)
        for start, value in params["modify_params"]["stages"]:
            if steps >= start:
                if isinstance(owner, dict):
                    owner[address[-1]] = value
                else:
                    setattr(owner, address[-1], value)
    cfg.curriculum = None


def freeze_calibration(groups: list[dict], joint_names: list[str]) -> dict:
    """Equal-command empirical quantiles and a 5%-of-positive-reward tracking weight."""
    if len(groups) != 7:
        raise ValueError("Calibration requires all seven command cases.")
    for group in groups:
        if len(group["errors"]) < 1000 or group["valid_trials"] < 8:
            raise ValueError(f"Insufficient calibration samples: {group['case']}")
        if not np.isfinite(group["errors"]).all() or not np.isfinite(group["positive_rewards"]).all():
            raise ValueError("Non-finite calibration data.")
    # Exact weighted CDF avoids periodic gait aliasing when command cases have different sample counts.
    errors = np.concatenate([g["errors"] for g in groups])
    weights = np.concatenate([np.full(len(g["errors"]), 1 / (len(groups) * len(g["errors"]))) for g in groups])
    quantiles = np.empty((3, errors.shape[1]))
    for joint in range(errors.shape[1]):
        order = np.argsort(errors[:, joint])
        cdf = np.cumsum(weights[order])
        indices = np.searchsorted(cdf, np.array([0.5, 0.75, 0.95]) - 1e-12)
        quantiles[:, joint] = errors[order[indices], joint]
    tolerance = np.maximum(quantiles[1], 0.015)
    costs = [np.mean(np.maximum(g["errors"] - tolerance, 0) ** 2 / 0.1**2) for g in groups]
    mean_cost = float(np.mean(costs))
    positive = float(np.mean([np.mean(g["positive_rewards"]) for g in groups]))
    if not np.isfinite(mean_cost + positive) or mean_cost < 1e-6 or positive <= 0:
        raise ValueError("Calibration cost is near zero or positive reward is invalid.")
    return {
        "joint_names": joint_names,
        "tolerance_rad": tolerance.tolist(),
        "scale_rad": 0.1,
        "weight_magnitude": 0.05 * positive / mean_cost,
        "positive_reward_mean": positive,
        "cost_mean": mean_cost,
        "target_penalty_fraction": 0.05,
        "absolute_error_p50_rad": quantiles[0].tolist(),
        "absolute_error_p95_rad": quantiles[2].tolist(),
        "cases": [{"case": g["case"], "samples": len(g["errors"]), "valid_trials": g["valid_trials"]} for g in groups],
    }


def rollout_measurements(checkpoint: Path, output: Path, args, calibration: dict | None = None) -> list[dict]:
    """Measure pre-reset state under training stochasticity or fixed deterministic evaluation.

    Reuses the task's recorder, so terminal states are captured before autoresets. The alive mask
    never admits a new trial after a failure. Calibration retains training noise/events; evaluation
    shares one inference protocol across every checkpoint and reward variant.
    """
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import to_dict

    from isaaclab_rl.entrypoints.common import dump_train_configs
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils import resolve_task_config

    evaluation = args.phase == "evaluate"
    cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restore_steps = (state["iter"] + 1) * agent.num_steps_per_env
    freeze_curriculum(cfg, restore_steps)
    if evaluation:
        cfg.play_mode()
        cfg.events.push_robot = None
        cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        for pose in (cfg.commands.head_pose, cfg.commands.body_pose):
            pose.ranges = tuple((0.0, 0.0) for _ in pose.ranges)
    cases = command_cases(cfg)
    cfg.scene.num_envs = args.rollout_envs
    cfg.sim.device = agent.device = args.device or "cuda:0"
    cfg.seed = agent.seed = 42 if evaluation else 123
    cfg.episode_length_s = args.duration + 1.0  # keep the measurement horizon below automatic timeout
    cfg.recorders.diagnostics.capture_samples = True
    if calibration is not None:
        cfg.recorders.diagnostics.tracking_tolerance = calibration["tolerance_rad"]
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_forward_envs = 0.0
    cfg.commands.base_velocity.rel_turn_in_place_envs = 0.0
    dump_train_configs(str(output), cfg, agent)
    results = []
    capture = evaluation and getattr(args, "capture_traces", False)
    traces = {}
    reward_sum_max_error = 0.0
    with torch.inference_mode(), launch_simulation(cfg, args):
        raw = ManagerBasedRLEnv(cfg=cfg)
        try:
            raw.common_step_counter = restore_steps
            env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
            runner = OnPolicyRunner(env, to_dict(agent), log_dir=None, device=agent.device)
            runner.load(str(checkpoint), map_location=agent.device)
            policy = runner.get_inference_policy(device=agent.device)
            recorder = raw.recorder_manager._terms["diagnostics"]
            names = recorder.joint_names
            if names != raw.action_manager.get_term("joint_pos").IO_descriptor.joint_names:
                raise ValueError("Recorder/action joint order differs.")
            cmd = raw.command_manager.get_term("base_velocity")
            for case_index, (case, velocity) in enumerate(cases):
                for axis, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), velocity):
                    setattr(cmd.cfg.ranges, axis, (value, value))
                raw.reset(seed=cfg.seed + case_index)
                raw.episode_length_buf.zero_()
                obs = env.get_observations()
                alive = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
                finite_counts = torch.zeros(raw.num_envs, device=raw.device)
                sums = torch.zeros((raw.num_envs, 7), device=raw.device)
                joint_sq = torch.zeros((raw.num_envs, len(names)), device=raw.device)
                exceed = torch.zeros(raw.num_envs, device=raw.device)
                falls = torch.zeros_like(alive)
                nan_failures = torch.zeros_like(alive)
                survival = torch.zeros(raw.num_envs, device=raw.device)
                errors, positive_rewards = [], []
                case_samples = []
                for step in range(round(args.duration / raw.step_dt)):
                    with torch.inference_mode():
                        action = policy(obs.to(agent.device), stochastic_output=not evaluation)
                        obs, _, dones, _ = env.step(action)
                    values = dict(zip(recorder.sample_names, recorder.snapshot.unbind(dim=1)))
                    if capture:
                        contributions = torch.stack(
                            [values[f"reward_step/{name}"] for name in raw.reward_manager.active_terms], dim=1
                        )
                        torch.testing.assert_close(
                            contributions.sum(1), values["reward_total_step"], atol=2e-6, rtol=2e-5
                        )
                        if not torch.isfinite(contributions[alive]).all():
                            raise ValueError("Non-finite reward in research trajectory.")
                        reward_sum_max_error = max(
                            reward_sum_max_error,
                            (contributions.sum(1) - values["reward_total_step"]).abs().max().item(),
                        )
                        actual_command = torch.stack([values[f"command_{axis}"] for axis in "xyz"], 1)
                        expected_command = actual_command.new_tensor(velocity).expand_as(actual_command)
                        torch.testing.assert_close(actual_command[alive], expected_command[alive], atol=1e-6, rtol=0)
                        # Retain the terminal step but never treat an autoreset as a new trial.
                        sample = torch.cat((recorder.snapshot, alive[:, None].float()), dim=1).cpu().numpy()
                        sample[~alive.cpu().numpy(), :-1] = np.nan
                        case_samples.append(sample)
                    error = torch.stack(
                        [values[f"joint_target/{name}"] - values[f"joint_pos/{name}"] for name in names], 1
                    )
                    measurements = torch.stack(
                        [
                            values["link_vx_b_mps"],
                            (
                                (values["link_vx_b_mps"] - velocity[0]) ** 2
                                + (values["link_vy_b_mps"] - velocity[1]) ** 2
                            ).sqrt(),
                            values["yaw_error_radps"],
                            values["yaw_error_radps"].square(),
                            values["roll_pitch_sq_rad2ps2"],
                            values["link_vz_b_mps"].square(),
                            error.square().mean(1),
                        ],
                        1,
                    )
                    finite = torch.isfinite(error).all(1) & torch.isfinite(measurements).all(1)
                    valid = alive & finite
                    if step >= round(args.warmup / raw.step_dt):
                        finite_counts += valid
                        sums += torch.where(valid[:, None], measurements, 0)
                        joint_sq += torch.where(valid[:, None], error.square(), 0)
                        if calibration is not None:
                            tau = torch.tensor(calibration["tolerance_rad"], device=raw.device)
                            exceed += torch.where(valid, (error.abs() > tau).float().mean(1), 0)
                        if not evaluation:
                            positive = sum(
                                values[f"reward_weighted/{name}"].clamp_min(0)
                                for name in raw.reward_manager.active_terms
                                if raw.reward_manager.get_term_cfg(name).weight > 0
                            )
                            errors.append(error[valid].abs().cpu().numpy())
                            positive_rewards.append(positive[valid].cpu().numpy())
                    survival += alive * raw.step_dt
                    falls |= alive & raw.termination_manager.get_term("fell_over")
                    nan_failures |= alive & raw.termination_manager.get_term("nan_state")
                    alive &= ~dones.bool()
                means = sums / finite_counts.clamp_min(1)[:, None]
                if capture:
                    traces[case] = np.stack(case_samples)
                if not evaluation:
                    results.append(
                        {
                            "case": case,
                            "errors": np.concatenate(errors),
                            "positive_rewards": np.concatenate(positive_rewards),
                            "valid_trials": int((finite_counts > 0).sum().item()),
                        }
                    )
                else:
                    for i in range(raw.num_envs):
                        v = means[i].tolist()
                        valid = finite_counts[i].item() > 0
                        results.append(
                            {
                                "case": case,
                                "trial": i,
                                "target_vx_mps": velocity[0],
                                "target_yaw_radps": velocity[2],
                                "completed": bool(alive[i].item()),
                                "fell_over": bool(falls[i].item()),
                                "nan_failure": bool(nan_failures[i].item()),
                                "survival_s": survival[i].item(),
                                "stable_samples": int(finite_counts[i].item()),
                                "vx_mps": v[0] if valid else None,
                                "planar_error_mps": v[1] if valid else None,
                                "yaw_bias_radps": v[2] if valid else None,
                                "yaw_rms_radps": v[3] ** 0.5 if valid else None,
                                "roll_pitch_sq_rad2ps2": v[4] if valid else None,
                                "vertical_rms_mps": v[5] ** 0.5 if valid else None,
                                "joint_rms_rad": v[6] ** 0.5 if valid else None,
                                "head_joint_rms_rad": (joint_sq[i, 5:9].mean() / finite_counts[i].clamp_min(1))
                                .sqrt()
                                .item()
                                if valid
                                else None,
                                "leg_joint_rms_rad": (
                                    torch.cat((joint_sq[i, :5], joint_sq[i, 9:])).mean() / finite_counts[i].clamp_min(1)
                                )
                                .sqrt()
                                .item()
                                if valid
                                else None,
                                "tracking_exceedance_fraction": (exceed[i] / finite_counts[i].clamp_min(1)).item()
                                if valid
                                else None,
                                "joint_rms_by_name_rad": dict(
                                    zip(names, (joint_sq[i] / finite_counts[i].clamp_min(1)).sqrt().tolist())
                                ),
                            }
                        )
                print(
                    f"{'评估' if evaluation else '标定'}完成：{case}；"
                    f"有效试验：{int((finite_counts > 0).sum())}/{raw.num_envs}",
                    flush=True,
                )
            if not evaluation:
                calibrated = freeze_calibration(results, names)
                (output / "calibration.json").write_text(json.dumps(calibrated, ensure_ascii=False, indent=2))
                np.savez_compressed(
                    output / "calibration_samples.npz",
                    **{f"{g['case']}_{key}": g[key] for g in results for key in ("errors", "positive_rewards")},
                )
            else:
                (output / "trials.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
                if capture:
                    np.savez_compressed(output / "step_traces.npz", **traces)
                    metadata = {
                        "columns": recorder.sample_names + ["alive_before_step"],
                        "step_dt": raw.step_dt,
                        "duration_s": args.duration,
                        "warmup_s": args.warmup,
                        "seed": cfg.seed,
                        "shape": "step, trial, column",
                        "state_timing": "post_physics_post_reward_pre_reset_pre_command_resample",
                        "reward_scoring": "mature_baseline_reference",
                        "reward_sum_max_abs_error": reward_sum_max_error,
                        "reward_weights": {
                            name: raw.reward_manager.get_term_cfg(name).weight
                            for name in raw.reward_manager.active_terms
                        },
                    }
                    (output / "trace_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        finally:
            raw.close()
    return results
