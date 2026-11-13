# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Commanded posture targets following pollen-robotics/microduck_rl's SitStand recipe.

Reference: upstream revision 0daab160d3fb1c2cbaba16a4a19052f36709dd9e.
The one-dimensional command is separate from velocity observations. Quaternion operations
use Isaac Lab's xyzw convention; joints are resolved by name, never USD traversal order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, RewardTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_assets.robots.microduck import MICRODUCK_HEAD_JOINT_NAMES, MICRODUCK_LEG_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

SITTING_JOINT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.4079,
    "left_knee": 1.35,
    "left_ankle": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.4079,
    "right_knee": -1.35,
    "right_ankle": 0.0,
}


class SitStandCommand(CommandTerm):
    """Binary sit request with a two-second, height-initialized reward target ramp."""

    cfg: SitStandCommandCfg

    def __init__(self, cfg: SitStandCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.leg_ids = self.robot.find_joints(MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)[0]
        self.head_ids = self.robot.find_joints(MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)[0]
        self.play_ids = None
        if "passive_left_knee_backlash" in self.robot.joint_names:
            self.play_ids = self.robot.find_joints(
                [f"passive_{name}_backlash" for name in MICRODUCK_LEG_JOINT_NAMES], preserve_order=True
            )[0]
        self.sit_pose = torch.tensor(
            [SITTING_JOINT_POS[name] for name in MICRODUCK_LEG_JOINT_NAMES], device=self.device
        )
        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        self.alpha = torch.zeros(self.num_envs, device=self.device)
        for name in ("height_error", "leg_error", "tilt_deg"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def height(self) -> torch.Tensor:
        return self.robot.data.root_link_pos_w.torch[:, 2] - self._env.scene.env_origins[:, 2]

    @property
    def target_height(self) -> torch.Tensor:
        return self.cfg.stand_height + self.alpha * (self.cfg.sit_height - self.cfg.stand_height)

    @property
    def leg_error(self) -> torch.Tensor:
        stand = self.robot.data.default_joint_pos.torch[:, self.leg_ids]
        target = stand + self.alpha[:, None] * (self.sit_pose - stand)
        actual = self.robot.data.joint_pos.torch[:, self.leg_ids]
        if self.play_ids is not None:
            actual = actual + self.robot.data.joint_pos.torch[:, self.play_ids]
        return actual - target

    @property
    def upright(self) -> torch.Tensor:
        # projected gravity distinguishes upright and upside down without quaternion indexing.
        return -self.robot.data.projected_gravity_b.torch[:, 2]

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
        ids = slice(None) if env_ids is None else env_ids
        result = super().reset(env_ids)
        # Isaac Lab applies reset events before resetting commands.
        self.alpha[ids] = (
            (self.cfg.stand_height - self.height[ids]) / (self.cfg.stand_height - self.cfg.sit_height)
        ).clamp(0.0, 1.0)
        return result

    def _resample_command(self, env_ids: Sequence[int] | slice):
        self._command[env_ids] = (torch.rand_like(self._command[env_ids]) < self.cfg.sit_probability).float()

    def _update_command(self):
        delta = self._command[:, 0] - self.alpha
        step = self._env.step_dt / self.cfg.ramp_time
        self.alpha += delta.clamp(-step, step)

    def _update_metrics(self):
        self.metrics["height_error"] = (self.height - self.target_height).abs()
        self.metrics["leg_error"] = self.leg_error.abs().mean(-1)
        self.metrics["tilt_deg"] = torch.rad2deg(self.upright.clamp(-1.0, 1.0).acos())


@configclass
class SitStandCommandCfg(CommandTermCfg):
    """Posture sampling and measured rest-height targets [m]."""

    class_type: type = SitStandCommand
    resampling_time_range: tuple[float, float] = (3.5, 6.5)
    sit_probability: float = 0.5
    ramp_time: float = 2.0
    sit_height: float = 0.060
    stand_height: float = 0.115


def reset_posture(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None, sitting_probability: float = 0.5
) -> None:
    """Mix seated and standing starts independently of the command sampled after reset."""
    robot = env.scene["robot"]
    command: SitStandCommand = env.command_manager.get_term("posture")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    count = len(env_ids)
    seated = torch.rand(count, device=env.device) < sitting_probability
    joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    joint_pos[:, command.leg_ids] = torch.where(seated[:, None], command.sit_pose, joint_pos[:, command.leg_ids])
    noise = torch.randn(count, len(command.leg_ids), device=env.device) * 0.10
    joint_pos[:, command.leg_ids] += noise * seated[:, None]
    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
    root = robot.data.default_root_state.torch[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    height = torch.where(seated, command.cfg.sit_height, command.cfg.stand_height)
    root[:, 2] = env.scene.env_origins[env_ids, 2] + height + torch.rand(count, device=env.device) * 0.015
    roll = torch.empty(count, device=env.device).uniform_(-math.radians(8), math.radians(8)) * seated
    pitch = torch.empty_like(roll).uniform_(-math.radians(8), math.radians(8)) * seated
    yaw = torch.empty_like(roll).uniform_(-math.pi, math.pi)
    root[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    root[:, 7:] = 0.0
    robot.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)


def posture_pose(env: ManagerBasedRLEnv, std: float = 0.5) -> torch.Tensor:
    """Average per-joint Gaussian tracking of the ramped leg pose."""
    return torch.exp(-(env.command_manager.get_term("posture").leg_error / std).square()).mean(-1)


def posture_pose_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mean absolute leg error [rad], used with a negative weight."""
    return env.command_manager.get_term("posture").leg_error.abs().mean(-1)


def posture_height(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Gaussian tracking of the ramped trunk height."""
    term = env.command_manager.get_term("posture")
    return torch.exp(-((term.height - term.target_height) / std).square())


def posture_height_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Absolute trunk-height error [m], used with a negative weight."""
    term = env.command_manager.get_term("posture")
    return (term.height - term.target_height).abs()


def posture_upright(env: ManagerBasedRLEnv, height_gated: bool = False) -> torch.Tensor:
    """Linear upright score, optionally fading between 7.5 and 10 cm."""
    term = env.command_manager.get_term("posture")
    if not height_gated:
        return term.upright
    blend = ((term.height - 0.075) / 0.025).clamp(0.0, 1.0)
    return term.upright * blend.square() * (3.0 - 2.0 * blend)


def posture_composite(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Require height, uprightness, leg pose and raised head together."""
    term = env.command_manager.get_term("posture")
    robot = term.robot
    head = robot.data.joint_pos.torch[:, term.head_ids] - robot.data.default_joint_pos.torch[:, term.head_ids]
    head_error = head - env.command_manager.get_command("head_pose")
    return (
        posture_height(env, 0.03)
        * torch.exp(-(1.0 - term.upright) / 0.40**2)
        * torch.exp(-term.leg_error.square().mean(-1) / 0.40**2)
        * torch.exp(-head_error.square().mean(-1) / 0.40**2)
    )


def posture_stillness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward quiet rest only once the ramp is complete and the body is upright."""
    term = env.command_manager.get_term("posture")
    height_gate = ((0.03 - (term.height - term.target_height).abs()) / (0.03 - 0.012)).clamp(0.0, 1.0)
    tilt_gate = (
        (term.upright - math.cos(math.radians(60))) / (math.cos(math.radians(25)) - math.cos(math.radians(60)))
    ).clamp(0.0, 1.0)
    speed_sq = term.robot.data.root_link_lin_vel_w.torch.square().sum(-1)
    ramp_done = (term.command[:, 0] - term.alpha).abs() < 0.02
    return (
        torch.exp(-speed_sq / 0.05**2)
        * height_gate.square()
        * (3 - 2 * height_gate)
        * tilt_gate.square()
        * (3 - 2 * tilt_gate)
        * ramp_done
    )


def rise_bootstrap(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward upward progress below the stand target, capped at 8 cm/s."""
    term = env.command_manager.get_term("posture")
    vz = term.robot.data.root_link_lin_vel_w.torch[:, 2]
    return vz.clamp(0.0, 0.08) * (term.height < term.cfg.stand_height + 0.01) * (1 - term.command[:, 0])


def vertical_speed_excess(env: ManagerBasedRLEnv, direction: float, limit: float) -> torch.Tensor:
    """Positive excess speed [m/s] in the given direction, used with a negative weight."""
    vz = env.scene["robot"].data.root_link_lin_vel_w.torch[:, 2]
    return (direction * vz - limit).clamp_min(0.0)


class vertical_acceleration(ManagerTermBase):
    """Positive control-rate vertical acceleration magnitude, excluding episode resets."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_vz = torch.zeros(env.num_envs, device=env.device)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        vz = env.scene["robot"].data.root_link_lin_vel_w.torch[:, 2]
        result = torch.where(self.fresh, 0.0, (vz - self.previous_vz).abs() / env.step_dt)
        self.previous_vz.copy_(vz)
        self.fresh.fill_(False)
        return result
