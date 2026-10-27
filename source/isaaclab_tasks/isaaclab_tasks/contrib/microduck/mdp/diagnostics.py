# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only, pre-reset MicroDuck diagnostics shared by training and fixed-command evaluation."""

from __future__ import annotations

import torch

from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES


class DiagnosticAccumulator:
    """Per-environment means/RMS with independent finite-sample counts, including sparse gait events.

    Non-events are NaN rather than zero. RMS is computed from squared samples, never from a signed mean.
    Summaries retain an environment axis so callers can average trials/episodes equally.
    """

    def __init__(self, num_envs: int, names: list[str], device: str):
        self.names = names
        self.sums = torch.zeros((num_envs, len(names)), device=device)
        self.counts = torch.zeros_like(self.sums)
        self.rms = torch.tensor(["_rms_" in name for name in names], device=device)

    def update(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        valid = torch.isfinite(values) & mask[:, None]
        transformed = torch.where(self.rms, values.square(), values)
        self.sums += torch.where(valid, transformed, 0.0)
        self.counts += valid

    def values(self, env_ids=slice(None)) -> torch.Tensor:
        means = self.sums[env_ids] / self.counts[env_ids].clamp_min(1)
        result = torch.where(self.rms, means.clamp_min(0).sqrt(), means)
        return torch.where(self.counts[env_ids] > 0, result, torch.nan)

    def reset(self, env_ids=slice(None)) -> None:
        self.sums[env_ids] = 0
        self.counts[env_ids] = 0


class MicroDuckDiagnostics(RecorderTerm):
    """Capture state after reward computation, before reset or command resampling.

    Training logs are equal-episode averages of time means or per-episode RMS, with separate all-command
    and forward-only groups. Reward means use the weights actually applied at each step, without ``dt``;
    step contributions include ``dt``. They differ from Episode_Reward's maximum-duration normalization.
    Stance events and slip use a 1 N force-norm threshold sampled at control frequency. Landing positions
    are ankle-frame origins, not contact centroids. Sole-center speed is a slip proxy, not exact contact slip.
    """

    def __init__(self, cfg: MicroDuckDiagnosticsCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.sensor = env.scene["contact_forces"]
        self.joint_ids, self.joint_names = self.robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)
        self.foot_ids, _ = self.robot.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)
        self.sensor_ids, _ = self.sensor.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)
        self.previous_contact = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stance_time = torch.zeros((env.num_envs, 2), device=env.device)
        self.complete_stance = torch.zeros_like(self.previous_contact)
        self.snapshot = None
        self.sample_names: list[str] = []
        self.metric_names: list[str] = []
        self.metric_values = None
        self.accumulators: dict[str, DiagnosticAccumulator] = {}

    def record_post_step(self):
        env, data = self._env, self.robot.data
        command = env.command_manager.get_command("base_velocity")
        link = data.root_link_lin_vel_b.torch
        com = data.root_com_lin_vel_b.torch
        angular = data.root_link_ang_vel_b.torch
        yaw_error = angular[:, 2] - command[:, 2]
        roll_pitch_sq = angular[:, :2].square().sum(dim=1)
        std = env.reward_manager.get_term_cfg("track_ang_vel").params["std"]
        yaw_factor = torch.exp(-yaw_error.square() / std**2)
        roll_pitch_factor = torch.exp(-roll_pitch_sq / std**2)
        root_pos = data.root_link_pos_w.torch
        root_quat = data.root_link_quat_w.torch
        foot_pos = data.body_link_pos_w.torch[:, self.foot_ids]
        foot_quat = data.body_link_quat_w.torch[:, self.foot_ids]
        foot_vel = data.body_link_lin_vel_w.torch[:, self.foot_ids]
        foot_ang = data.body_link_ang_vel_w.torch[:, self.foot_ids]
        offset = torch.zeros_like(foot_pos)
        offset[:, :, 2] = -self.cfg.sole_offset_m
        sole_offset = quat_apply(foot_quat, offset)
        sole_vel = foot_vel + torch.cross(foot_ang, sole_offset, dim=-1)
        foot_rel = quat_apply_inverse(root_quat[:, None].expand(-1, 2, -1), foot_pos - root_pos[:, None])
        force = self.sensor.data.net_forces_w.torch[:, self.sensor_ids]
        contact = force.norm(dim=-1) > self.cfg.contact_threshold_n
        touchdown = contact & ~self.previous_contact & self.initialized[:, None]
        liftoff = ~contact & self.previous_contact & self.initialized[:, None]
        # A stance already in progress on the first sample is left-censored; do not report it as complete.
        duration = torch.where(liftoff & self.complete_stance, self.stance_time, torch.nan)
        self.complete_stance = (self.complete_stance | touchdown) & contact
        self.stance_time = torch.where(contact, self.stance_time + env.step_dt, 0.0)
        self.previous_contact.copy_(contact)
        self.initialized[:] = True

        samples = {}
        metrics = {}
        for i, axis in enumerate("xyz"):
            samples[f"command_{axis}"] = command[:, i]
            samples[f"link_v{axis}_b_mps"] = link[:, i]
            samples[f"com_v{axis}_b_mps"] = com[:, i]
            samples[f"omega_{axis}_b_radps"] = angular[:, i]
            samples[f"root_{axis}_w_m"] = root_pos[:, i]
        samples["heading_w_rad"] = data.heading_w.torch
        for i, axis in enumerate("xyzw"):
            samples[f"root_q{axis}"] = root_quat[:, i]
        metrics.update(
            link_vx_mean_mps=link[:, 0],
            com_vx_mean_mps=com[:, 0],
            link_vx_error_mean_mps=link[:, 0] - command[:, 0],
            link_vx_error_rms_mps=link[:, 0] - command[:, 0],
            link_vy_rms_mps=link[:, 1],
            link_vz_rms_mps=link[:, 2],
            reference_vx_delta_mean_mps=link[:, 0] - com[:, 0],
            reference_velocity_delta_rms_mps=(link - com).norm(dim=1),
            yaw_error_mean_radps=yaw_error,
            yaw_error_rms_radps=yaw_error,
            roll_pitch_sq_mean_rad2ps2=roll_pitch_sq,
            yaw_factor_mean=yaw_factor,
            roll_pitch_factor_mean=roll_pitch_factor,
            angular_reward_raw_mean=yaw_factor * roll_pitch_factor,
        )
        samples["yaw_error_radps"] = yaw_error
        samples["roll_pitch_sq_rad2ps2"] = roll_pitch_sq
        samples["yaw_factor"] = yaw_factor
        samples["roll_pitch_factor"] = roll_pitch_factor
        samples["angular_reward_raw"] = yaw_factor * roll_pitch_factor
        for i, name in enumerate(env.reward_manager.active_terms):
            weighted = env.reward_manager.weighted_term_values[:, i]
            metrics[f"reward_weighted/{name}"] = weighted
            samples[f"reward_weighted/{name}"] = weighted
            samples[f"reward_step/{name}"] = weighted * env.step_dt
        samples["reward_total_step"] = env.reward_buf
        metrics["reward_weighted/total"] = env.reward_manager.weighted_term_values.sum(dim=1)

        targets = self.robot.actuators.target_command.position.torch[:, self.joint_ids]
        positions = data.joint_pos.torch[:, self.joint_ids]
        velocities = data.joint_vel.torch[:, self.joint_ids]
        actions = env.action_manager.action
        for i, name in enumerate(self.joint_names):
            samples[f"joint_target/{name}"] = targets[:, i]
            samples[f"joint_pos/{name}"] = positions[:, i]
            samples[f"joint_vel/{name}"] = velocities[:, i]
            samples[f"action/{name}"] = actions[:, i]
            metrics[f"joint_error_rms_rad/{name}"] = targets[:, i] - positions[:, i]
        for indices, side in ((slice(0, 5), "left"), (slice(9, 14), "right")):
            metrics[f"{side}_joint_error_rms_rad"] = (
                (targets[:, indices] - positions[:, indices]).square().mean(1).sqrt()
            )
            default = data.default_joint_pos.torch[:, self.joint_ids][:, indices]
            metrics[f"{side}_target_offset_rms_rad"] = (targets[:, indices] - default).square().mean(1).sqrt()
            metrics[f"{side}_actual_offset_rms_rad"] = (positions[:, indices] - default).square().mean(1).sqrt()

        for i, side in enumerate(("left", "right")):
            ankle_speed = foot_vel[:, i, :2].norm(dim=1)
            sole_speed = sole_vel[:, i, :2].norm(dim=1)
            samples[f"{side}_contact"] = contact[:, i].float()
            samples[f"{side}_touchdown"] = touchdown[:, i].float()
            samples[f"{side}_liftoff"] = liftoff[:, i].float()
            samples[f"{side}_stance_elapsed_s"] = self.stance_time[:, i]
            samples[f"{side}_completed_stance_s"] = duration[:, i]
            samples[f"{side}_ankle_xy_speed_mps"] = ankle_speed
            samples[f"{side}_sole_xy_speed_mps"] = sole_speed
            for j, axis in enumerate("xyz"):
                samples[f"{side}_ankle_{axis}_w_m"] = foot_pos[:, i, j]
                samples[f"{side}_ankle_{axis}_b_m"] = foot_rel[:, i, j]
                samples[f"{side}_force_{axis}_w_n"] = force[:, i, j]
                metrics[f"{side}_landing_{axis}_b_mean_m"] = torch.where(touchdown[:, i], foot_rel[:, i, j], torch.nan)
            metrics[f"{side}_contact_fraction"] = contact[:, i].float()
            metrics[f"{side}_stance_duration_mean_s"] = duration[:, i]
            metrics[f"{side}_touchdowns_per_s"] = touchdown[:, i].float() / env.step_dt
            metrics[f"{side}_contact_ankle_speed_mean_mps"] = torch.where(contact[:, i], ankle_speed, torch.nan)
            metrics[f"{side}_contact_sole_speed_mean_mps"] = torch.where(contact[:, i], sole_speed, torch.nan)

        self.metric_names = list(metrics)
        self.metric_values = torch.stack(list(metrics.values()), dim=1)
        if not self.accumulators:
            self.accumulators = {
                group: DiagnosticAccumulator(env.num_envs, self.metric_names, env.device)
                for group in ("all", "forward")
            }
        self.accumulators["all"].update(self.metric_values, torch.ones_like(self.initialized))
        forward = (command[:, 0] > 0.01) & (command[:, 1:].abs().amax(dim=1) < 1e-6)
        self.accumulators["forward"].update(self.metric_values, forward)
        if self.cfg.capture_samples:
            self.sample_names = list(samples)
            self.snapshot = torch.stack(list(samples.values()), dim=1)
        return None, None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for group, accumulator in self.accumulators.items():
            values = accumulator.values(ids)
            available = torch.isfinite(values).any(dim=0).tolist()
            means = torch.nanmean(values, dim=0)
            for name, value, present in zip(self.metric_names, means, available):
                if present:
                    self._env.extras["log"][f"Diagnostics/{group}/{name}"] = value
            accumulator.reset(ids)
        self.previous_contact[ids] = False
        self.initialized[ids] = False
        self.stance_time[ids] = 0
        self.complete_stance[ids] = False


@configclass
class MicroDuckDiagnosticsCfg(RecorderTermCfg):
    """Control diagnostics capture without changing policy inputs or rewards."""

    class_type: type[RecorderTerm] = MicroDuckDiagnostics
    capture_samples: bool = False
    contact_threshold_n: float = 1.0
    sole_offset_m: float = 0.0225


@configclass
class MicroDuckRecordersCfg(RecorderManagerBaseCfg):
    """Streaming scalar diagnostics; no per-environment HDF5 trajectory accumulation."""

    diagnostics: MicroDuckDiagnosticsCfg = MicroDuckDiagnosticsCfg()
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    record_post_step_observations = False
