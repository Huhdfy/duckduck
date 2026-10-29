# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck walking rewards.

Rewards run before resets, so a state that ``robot_state_is_nan`` terminates still reaches them once;
terms that read the robot state return zero, or ``upright`` its minimum, instead of a non-finite value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import body_projected_gravity_b
from isaaclab.envs.mdp import joint_pos_limits as _joint_pos_limits
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.string import resolve_matching_names_values

from .observations import foot_height, joint_pos_rel_biased, zero_non_finite

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _command_magnitude(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Upstream's scalar command magnitude ``|v_xy| + |w_z|``."""
    command = env.command_manager.get_command(command_name)
    # 这是沿用上游的运动指令判据，混合线速度与角速度；并非具有统一物理单位的速度大小。
    return torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])


class joint_tracking_error_l2(ManagerTermBase):
    """Dead-zone tracking cost after physics, including command delay, in physical joint coordinates.

    ``asset_cfg`` must select the action joints in action order. The tolerance is per joint [rad];
    ``scale`` [rad] makes the mean squared excess dimensionless. RewardManager supplies dt and weight.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        action = env.action_manager.get_term(cfg.params.get("action_name", "joint_pos"))
        self.tolerance = torch.tensor(cfg.params["tolerance"], device=env.device)
        if (
            self.tolerance.shape != (action.action_dim,)
            or not torch.isfinite(self.tolerance).all()
            or (self.tolerance < 0).any()
        ):
            raise ValueError("Tracking tolerance must contain one finite non-negative angle per action joint.")
        if cfg.params.get("scale", 0.1) <= 0:
            raise ValueError("Tracking error scale must be positive.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tolerance: list[float],
        asset_cfg: SceneEntityCfg,
        action_name: str = "joint_pos",
        scale: float = 0.1,
    ) -> torch.Tensor:
        action = env.action_manager.get_term(action_name)
        actual = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
        excess = (action.physical_targets - actual).abs().sub(self.tolerance).clamp_min(0)
        return zero_non_finite((excess / scale).square().mean(dim=1))


"""
Velocity-tracking rewards.
"""


def track_linear_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track linear velocity, including vertical drift, with tolerance ``std`` [m/s]."""
    # 获取机器人对象
    asset: Articulation = env.scene[asset_cfg.name]
    # [vx.vy.vz]*N
    command = env.command_manager.get_command(command_name)
    # 后缀 b 表示机体坐标系；同时惩罚竖直速度，因此平面速度准确但上下跳动仍会降低奖励。
    actual = asset.data.root_link_lin_vel_b.torch
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    # why 指数？变正数？比较自然
    return zero_non_finite(torch.exp(-(xy_error + z_error) / std**2))


def track_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    shape: str = "exponential",
) -> torch.Tensor:
    """Track yaw, including roll/pitch drift, using an exponential or rational kernel.

    ``std`` is in rad/s. Both kernels have unit value and derivative ``-1/std**2``
    with respect to squared error at zero; the rational kernel has a heavier tail.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_ang_vel_b.torch
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    # 指数核最大为 1；误差平方和等于 std² 时奖励为 e⁻¹，这里没有高斯密度中的 1/2 系数。
    if shape == "exponential":
        return zero_non_finite(torch.exp(-(z_error + xy_error) / std**2))
    if shape == "rational":
        return zero_non_finite(1.0 / (1.0 + (z_error + xy_error) / std**2))
    raise ValueError(f"Unknown angular tracking reward shape: {shape}")


"""
Posture rewards.
"""


def upright(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward keeping one body's frame aligned with gravity, using a Gaussian kernel."""
    projected_gravity_b = body_projected_gravity_b(env, asset_cfg)
    # 只看重力在机体 xy 平面的分量，不限制航向；完全倒立时该项也无法辨别，需配合姿态终止。
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    xy_squared = torch.nan_to_num(xy_squared, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.exp(-xy_squared / std**2)


class joint_deviation_exp(ManagerTermBase):
    """Reward staying near the default joint positions, with a per-joint tolerance that widens once the robot moves.

    The tolerance is ``std_standing`` [rad] while the command magnitude is below ``walking_threshold``, and
    ``std_walking`` [rad] otherwise.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Resolve the standing and walking per-joint standard deviations."""
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        joint_names = [env.scene[asset_cfg.name].joint_names[index] for index in asset_cfg.joint_ids]
        self._std = {}
        for key in ("std_standing", "std_walking"):
            _, _, values = resolve_matching_names_values(cfg.params[key], joint_names)
            self._std[key] = torch.tensor(values, dtype=torch.float32, device=env.device).unsqueeze(0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        walking_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the posture reward."""
        asset: Articulation = env.scene[asset_cfg.name]
        # 站立/行走由目标指令判断，而非实际速度；收到行走指令但尚未移动时也放宽关节容差。
        standing = (_command_magnitude(env, command_name) < walking_threshold).unsqueeze(1)
        std = torch.where(standing, self._std["std_standing"], self._std["std_walking"])
        joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        default_joint_pos = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        error_squared = torch.square(joint_pos - default_joint_pos)
        return zero_non_finite(torch.exp(-torch.mean(error_squared / std**2, dim=1)))


"""
Head-pose rewards.
"""


def head_pose_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    backlash_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Track head encoder offsets with tolerance ``std`` [rad], including play when configured."""
    command = env.command_manager.get_command(command_name)
    error = joint_pos_rel_biased(env, asset_cfg, biased=False, backlash_cfg=backlash_cfg) - command
    # 先对各头部关节计算指数奖励再平均；与先平均误差再取指数的姿态奖励不同。
    return zero_non_finite(torch.exp(-((error / std) ** 2)).mean(dim=-1))


class head_pose_bias_penalty(ManagerTermBase):
    """Penalize a head-pose error that persists, as opposed to one that averages out."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-environment moving average."""
        super().__init__(cfg, env)
        self._error_ema = torch.zeros(env.num_envs, len(cfg.params["asset_cfg"].joint_ids), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the average for the environments that restarted."""
        if env_ids is None:
            self._error_ema[:] = 0.0
        else:
            self._error_ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        tau_s: float,
        asset_cfg: SceneEntityCfg,
        backlash_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        """Return the absolute error averaged with time constant ``tau_s`` [s]."""
        command = env.command_manager.get_command(command_name)
        error = zero_non_finite(joint_pos_rel_biased(env, asset_cfg, biased=False, backlash_cfg=backlash_cfg) - command)
        # 先对有符号误差做滑动平均，再取绝对值：持续偏向一侧受罚，正负交替误差可相互抵消。
        alpha = min(1.0, env.step_dt / tau_s)
        self._error_ema = (1.0 - alpha) * self._error_ema + alpha * error
        return zero_non_finite(self._error_ema.abs().mean(dim=-1))


"""
Gait and foot rewards.
"""


def feet_air_time_windowed(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold_min: float,
    threshold_max: float,
    command_threshold: float,
) -> torch.Tensor:
    """Reward air time strictly between ``threshold_min`` and ``threshold_max`` [s] while moving."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    # 使用当前腾空时长，每个控制步都判断；不是落地时才结算，且上下边界均不计入。
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    reward = reward * (_command_magnitude(env, command_name) > command_threshold).float()
    return reward


def foot_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    height_sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Penalize moving feet that deviate from ``target_height`` [m] above the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    heights = foot_height(env, asset_cfg, height_sensor_names)
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    # 高度误差由足部水平速度加权；静止足不会因低于抬脚目标而在此项受罚。
    cost = torch.sum(torch.abs(heights - target_height) * torch.norm(foot_vel_xy, dim=-1), dim=1)
    return zero_non_finite(cost * (_command_magnitude(env, command_name) > command_threshold).float())


class foot_swing_height(ManagerTermBase):
    """Penalize, on touchdown, how far the peak of the swing was from a target height."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-foot peak heights."""
        super().__init__(cfg, env)
        num_feet = len(cfg.params["sensor_cfg"].body_ids)
        self._peak_heights = torch.zeros(env.num_envs, num_feet, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the tracked peaks for the environments that restarted."""
        if env_ids is None:
            self._peak_heights[:] = 0.0
        else:
            self._peak_heights[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float,
        height_sensor_names: tuple[str, ...] = (),
    ) -> torch.Tensor:
        """Penalize touchdown error relative to ``target_height`` [m]."""
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_air = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] == 0.0
        heights = zero_non_finite(foot_height(env, asset_cfg, height_sensor_names))
        self._peak_heights = torch.where(in_air, torch.maximum(self._peak_heights, heights), self._peak_heights)
        # 摆动中累计最高点，首次接触时结算一次；结算后清零，防止上一摆动周期污染下一周期。
        first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids] > 0.5
        # 使用相对高度误差，结果无量纲；目标高度改变时，同样的绝对误差会得到不同惩罚。
        error = self._peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1)
        cost = cost * (_command_magnitude(env, command_name) > command_threshold).float()
        self._peak_heights = torch.where(first_contact, torch.zeros_like(self._peak_heights), self._peak_heights)
        return zero_non_finite(cost)


def foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize the squared planar speed of feet that are on the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = (contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0).float()
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact, dim=1)
    return zero_non_finite(cost * (_command_magnitude(env, command_name) > command_threshold).float())


"""
Regularization penalties.
"""


def body_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the roll and pitch rate of selected bodies, in the world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    return zero_non_finite(torch.sum(torch.square(ang_vel_xy), dim=(1, 2)))


def angular_momentum_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the squared magnitude of the articulation's angular momentum about its own centre of mass."""
    asset: Articulation = env.scene[asset_cfg.name]
    mass = asset.data.body_mass.torch[:, asset_cfg.body_ids]
    pos_w = asset.data.body_com_pos_w.torch[:, asset_cfg.body_ids]
    lin_vel_w = asset.data.body_com_lin_vel_w.torch[:, asset_cfg.body_ids]
    ang_vel_w = asset.data.body_com_ang_vel_w.torch[:, asset_cfg.body_ids]
    quat_w = asset.data.body_com_quat_w.torch[:, asset_cfg.body_ids]
    inertia_b = asset.data.body_inertia.torch[:, asset_cfg.body_ids].view(*mass.shape, 3, 3)
    total_mass = mass.sum(dim=1, keepdim=True).unsqueeze(-1)
    com_pos_w = (mass.unsqueeze(-1) * pos_w).sum(dim=1, keepdim=True) / total_mass
    com_lin_vel_w = (mass.unsqueeze(-1) * lin_vel_w).sum(dim=1, keepdim=True) / total_mass
    # 惯量先从局部质心坐标系旋转到世界系；总角动量包含各刚体自转和绕整机质心的轨道项。
    rot = math_utils.matrix_from_quat(quat_w)
    inertia_w = torch.matmul(rot, torch.matmul(inertia_b, rot.transpose(-2, -1)))
    spin = torch.matmul(inertia_w, ang_vel_w.unsqueeze(-1)).squeeze(-1)
    orbit = mass.unsqueeze(-1) * torch.cross(pos_w - com_pos_w, lin_vel_w - com_lin_vel_w, dim=-1)
    # 先对各刚体角动量求和再平方，方向相反的贡献可抵消；不是各刚体角动量平方之和。
    momentum = (spin + orbit).sum(dim=1)
    return zero_non_finite(torch.sum(torch.square(momentum), dim=-1))


def joint_pos_limits_finite(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """:func:`~isaaclab.envs.mdp.joint_pos_limits`, zero for environments whose state is not finite."""
    return zero_non_finite(_joint_pos_limits(env, asset_cfg))


"""
Contact penalties.
"""


def self_collision_cost(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0
) -> torch.Tensor:
    """Penalize parts of the robot that are touching the robot itself."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = contact_sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(f"{sensor_cfg.name} needs a filtered contact sensor to measure self-contacts.")
    forces = force_matrix_w.torch[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.norm(forces, dim=-1) > force_threshold
    # 对每个被监测刚体判断是否接触任一过滤对象，再计数；不是累加接触力或接触点数量。
    touching = in_contact.any(dim=-1)
    return touching.sum(dim=-1).float()
