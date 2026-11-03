# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck cumulative-heading and trajectory reward experiments.

The original fast-mixed reward module was not preserved. Its fixed/relative mixture below
is reconstructed from saved parameters and the agreed formula, not verified source parity.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg, RewardTermCfg
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.mdp.commands import MicroDuckVelocityCommand
from isaaclab_tasks.contrib.microduck.mdp.observations import ImuNoiseCfg, imu_noise, misaligned_imu
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
VARIANTS = {
    # linear velocity, angular velocity, heading, position
    "yaw_obs": (4.0, 4.0, 0.0, 0.0),
    "yaw_reward": (4.0, 4.0, 1.0, 0.0),
    "pos_half": (2.0, 2.0, 1.0, 4.0),
    "pos_only": (0.0, 0.0, 1.0, 8.0),
    "pos_add": (4.0, 4.0, 1.0, 4.0),
}
IMU_PARAMS = {"max_angle_deg": 6.0, "min_lag": 0, "max_lag": 1, "update_period": 64}


class TrackingVelocityCommand(MicroDuckVelocityCommand):
    """Own the command-period reference and the IMU-based heading error estimate."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reference_position = torch.zeros(self.num_envs, 2, device=self.device)
        self.reference_yaw = torch.zeros(self.num_envs, device=self.device)
        self.yaw_error_estimate = torch.zeros(self.num_envs, device=self.device)
        self._generation = torch.zeros(self.num_envs, device=self.device)
        self._resampled_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self._last_update_step = -1
        self._imu_step = -1
        self._imu_needs_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._imu_output = torch.zeros(self.num_envs, 6, device=self.device)
        self._imu_noise_cfg = ImuNoiseCfg(ang_vel=0.03, gravity=0.01)
        self._imu = misaligned_imu(ObservationTermCfg(func=misaligned_imu, params=IMU_PARAMS), env)
        self._command_history = CircularBuffer(max_len=2, batch_size=self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.reference_position[env_ids] = self.robot.data.root_link_pos_w.torch[env_ids, :2]
        self.reference_yaw[env_ids] = euler_xyz_from_quat(self.robot.data.root_link_quat_w.torch[env_ids])[2]
        self.yaw_error_estimate[env_ids] = 0.0
        self._generation[env_ids] += 1
        self._resampled_step[env_ids] = self._env.common_step_counter

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._imu.reset(ids)
        self._imu_needs_reset[ids] = True
        return super().reset(env_ids)

    def imu_sample(self):
        """Cache the exact noisy actor sample once per control step."""
        step = self._env.common_step_counter
        if step != self._imu_step or self._imu_needs_reset.any():
            raw = self._imu(self._env, **IMU_PARAMS)
            noisy = imu_noise(raw, self._imu_noise_cfg)
            if step != self._imu_step:
                self._imu_output = noisy
                self._imu_step = step
            else:
                self._imu_output = torch.where(self._imu_needs_reset[:, None], noisy, self._imu_output)
            self._imu_needs_reset.fill_(False)
        return self._imu_output

    def update_reference(self):
        """Advance once with the command that governed the just-completed physics step."""
        step = self._env.common_step_counter
        if self._last_update_step == step:
            return
        self._last_update_step = step
        dt = self._env.step_dt
        command = self.command
        # The first manager observation precedes the first physics step.
        if step == 0:
            self.imu_sample()
            self._command_history.append(torch.stack((command[:, 2], self._generation), dim=-1))
            return
        middle = self.reference_yaw + 0.5 * command[:, 2] * dt
        c, s = torch.cos(middle), torch.sin(middle)
        active = self._resampled_step != step
        delta = torch.stack((c * command[:, 0] - s * command[:, 1], s * command[:, 0] + c * command[:, 1]), dim=-1) * dt
        self.reference_position += delta * active[:, None]
        self.reference_yaw = wrap_to_pi(self.reference_yaw + command[:, 2] * dt * active)
        imu = self.imu_sample()
        self._command_history.append(torch.stack((command[:, 2], self._generation), dim=-1))
        gyro_lag = self._imu._lags[:, 0]
        aligned = self._imu._buffer[gyro_lag]
        # Gravity and gyro use the same underlying time sample; add gravity noise at the current sample.
        gravity = aligned[:, 3:6] + (torch.rand_like(aligned[:, 3:6]) * 2.0 - 1.0) * 0.01
        up = -torch.nn.functional.normalize(gravity, dim=-1)
        measured_yaw_rate = (imu[:, :3] * up).sum(dim=-1)
        delayed = self._command_history[gyro_lag]
        valid = (delayed[:, 1] == self._generation) & active
        self.yaw_error_estimate += torch.where(valid, measured_yaw_rate - delayed[:, 0], 0.0) * dt

    def _update_metrics(self):
        self.update_reference()
        super()._update_metrics()


def observed_imu(env):
    return env.command_manager.get_term("base_velocity").imu_sample().clone()


def observed_yaw_error(env):
    term = env.command_manager.get_term("base_velocity")
    return (wrap_to_pi(term.yaw_error_estimate) / math.pi).unsqueeze(-1)


def _mixture(error_sq, magnitude, std, relative_std=0.5, command_floor=0.1, absolute_fraction=0.5):
    tolerance = relative_std * magnitude.clamp_min(command_floor)
    absolute = torch.exp(-error_sq / std**2)
    relative = torch.exp(-error_sq / tolerance.square())
    return torch.nan_to_num(absolute_fraction * absolute + (1.0 - absolute_fraction) * relative)


def linear_mixed(env, command_name, std):
    command = env.command_manager.get_command(command_name)
    robot = env.scene["robot"]
    actual = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w.torch), robot.data.root_link_lin_vel_w.torch)
    error = (command[:, :2] - actual[:, :2]).square().sum(dim=-1)
    return _mixture(error, command[:, :2].norm(dim=-1), std)


def angular_mixed(env, command_name, std):
    command = env.command_manager.get_command(command_name)
    actual = env.scene["robot"].data.root_link_ang_vel_w.torch[:, 2]
    return _mixture((command[:, 2] - actual).square(), command[:, 2].abs(), std)


def heading_reward(env):
    term = env.command_manager.get_term("base_velocity")
    term.update_reference()
    yaw = euler_xyz_from_quat(term.robot.data.root_link_quat_w.torch)[2]
    error = wrap_to_pi(yaw - term.reference_yaw).abs()
    return torch.nan_to_num(torch.exp(-(error.sub(math.radians(3)).clamp_min(0) / math.radians(15)).square()))


def position_reward(env):
    term = env.command_manager.get_term("base_velocity")
    term.update_reference()
    error = (term.robot.data.root_link_pos_w.torch[:, :2] - term.reference_position).norm(dim=-1)
    elapsed = (env.common_step_counter - term._resampled_step).clamp_min(0) * env.step_dt
    path_length = term.command[:, :2].norm(dim=-1) * elapsed
    # Commands are constant within each period; normalize by its ideal traveled distance.
    scale = 0.12 * (path_length + 0.05)
    return torch.nan_to_num(torch.exp(-(error / scale).square()))


def make_config(variant, num_envs, seed):
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=num_envs)
    cfg.seed = seed
    cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
    cfg.commands.base_velocity.class_type = TrackingVelocityCommand
    cfg.observations.policy.imu.func = observed_imu
    cfg.observations.policy.imu.params = {}
    cfg.observations.policy.imu.noise = None
    cfg.observations.policy.heading_error = ObservationTermCfg(func=observed_yaw_error)
    cfg.observations.critic.heading_error = ObservationTermCfg(func=observed_yaw_error)
    v, w, heading, position = VARIANTS[variant]
    cfg.rewards.track_lin_vel.func = linear_mixed
    cfg.rewards.track_lin_vel.weight = v
    cfg.rewards.track_ang_vel.func = angular_mixed
    cfg.rewards.track_ang_vel.weight = w
    cfg.rewards.heading_tracking = RewardTermCfg(func=heading_reward, weight=heading)
    cfg.rewards.position_tracking = RewardTermCfg(func=position_reward, weight=position)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-root", default="logs/rsl_rl/microduck_velocity_flat")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--forward-max", type=float, default=1.0)
    parser.add_argument("--linear-weight", type=float)
    parser.add_argument("--run-name")
    parser.add_argument("--save-interval", type=int, default=1000)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cfg = make_config(args.variant, args.num_envs, args.seed)
    cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, args.forward_max)
    if args.linear_weight is not None:
        cfg.rewards.track_lin_vel.weight = args.linear_weight
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = args.save_interval
    log_dir = Path(args.output_root).resolve() / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name or args.variant}"
    cfg.log_dir = str(log_dir)
    log_dir.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "task": TASK,
        "variant": args.variant,
        "seed": args.seed,
        "source_revision": revision,
        "fast_mixed": "reconstructed from saved parameters; original implementation unavailable",
        "weights": [cfg.rewards.track_lin_vel.weight, *VARIANTS[args.variant][1:]],
        "parent_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "forward_range": [-1.0, args.forward_max],
        "additional_iterations": args.iterations,
        "save_interval": args.save_interval,
    }
    (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    sim_utils.create_new_stage()
    env = gym.make(TASK, cfg=cfg).unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device="cuda:0")
    if args.checkpoint:
        runner.load(str(args.checkpoint))
        # Saved iteration is the last completed update; continue with the next update.
        runner.current_learning_iteration += 1
        metadata["start_iteration"] = runner.current_learning_iteration
        metadata["loaded_learning_rate"] = runner.alg.learning_rate
        (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"RESUMED_FROM={args.checkpoint} START_ITERATION={runner.current_learning_iteration}", flush=True)
    runner.add_git_repo_to_log(__file__)
    dump_yaml(str(log_dir / "params" / "env.yaml"), cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    print(f"EXPERIMENT_LOG_DIR={log_dir}", flush=True)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=agent_cfg.init_at_random_ep_len)
    wrapped.close()


if __name__ == "__main__":
    main()
