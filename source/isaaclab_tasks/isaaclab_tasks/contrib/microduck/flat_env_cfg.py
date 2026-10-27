# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat walking recipe ported from pollen-robotics/microduck_rl and the original Lab task."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.microduck import (
    MICRODUCK_BACKLASH_CFG,
    MICRODUCK_CFG,
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
)

from . import mdp
from .mdp.diagnostics import MicroDuckRecordersCfg

MICRODUCK_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
# 所有关节向量都按舵机顺序解析；足部列表也固定为左、右，须与奖励和传感器顺序一致。
# 这里一个cfg是一类关节的指针 ，实际的函数介绍cfg作为参数，计算对应的关节。
# 不同函数用到不同的关节， 故设置多个cfg。
_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=["trunk_base"])
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_PLAY_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_JOINT_NAMES], preserve_order=True
)
_HEAD_PLAY_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=[f"passive_{name}_backlash" for name in MICRODUCK_HEAD_JOINT_NAMES], preserve_order=True
)
MICRODUCK_SOLE_TO_ANKLE_OFFSET = 0.0225
"""Standing ankle-frame height above the sole [m]."""
# 奖励读取踝部坐标系高度：脚底离地 2 cm 还需加上脚底到踝部原点的偏移。
MICRODUCK_FOOT_TARGET_HEIGHT = 0.02 + MICRODUCK_SOLE_TO_ANKLE_OFFSET
MICRODUCK_STEPS_PER_ITERATION = 24


# 选择是否考虑齿轮间隙
def _backlash_preset(term, func=None, **params):
    """Use ``term`` as is, or with ``func`` and ``params`` for the backlash robot under ``presets=backlash``."""
    # 仅在 backlash 预设中替换函数或参数；编码器相关项需考虑舵机角与被动齿隙角之和。
    backlash = term.replace(params={**term.params, **params})
    if func is not None:
        backlash.func = func
    return preset(default=term, backlash=backlash)


@configclass
class MicroDuckSceneCfg(InteractiveSceneCfg):
    """Scene with the MicroDuck robot on a ground plane."""

    # 地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),  # 摩擦系数
        debug_vis=False,
    )
    robot = preset(
        default=MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        backlash=MICRODUCK_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )
    # The walking USD enables sole colliders only, so self-contact is foot against foot.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_left",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_right"],
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9))
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # 速度指令为 vx、vy（m/s）和绕 z 轴角速度（rad/s）；重采样时间以秒计。
    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,  # 原地不动
        rel_forward_envs=0.2,  # 纯前进
        rel_turn_in_place_envs=0.15,  # 原地转弯
        debug_vis=True,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
    )
    head_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0), ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
    )
    # Retained in the deployed observation vector, although body-pose reward is disabled.
    body_pose = mdp.UniformPoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005), (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 14 维动作表示相对默认姿态的位置偏移（rad），不是力矩。
    # 目标位置 = 默认角度 + scale × 动作 - 编码器偏置；执行器再将目标转换为驱动力。
    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True, scale=1.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    观测信息
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract."""

        # 策略输入顺序属于部署接口：IMU 6 + 关节位置 14 + 关节速度 14 + 上次动作 14
        # + 速度指令 3 + 头部指令 4 + 身体指令 6 = 61 维；改变顺序会改变已有策略的输入含义。
        # Angular velocity [rad/s], then projected gravity.
        # 陀螺仪+加速度计= 3+3
        imu = ObsTerm(
            func=mdp.misaligned_imu,
            params={"max_angle_deg": 6.0, "min_lag": 0, "max_lag": 1, "update_period": 64},
            noise=mdp.ImuNoiseCfg(ang_vel=0.03, gravity=0.01),
        )
        # With backlash, encoders measure servo plus play angle.
        # 关节14
        joint_pos = _backlash_preset(
            ObsTerm(
                func=mdp.joint_pos_rel_biased,
                params={"asset_cfg": _SERVO_JOINT_CFG, "biased": True},
                noise=Unoise(n_min=-0.001, n_max=0.001),
            ),
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        # 速度14
        joint_vel = _backlash_preset(
            ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": _SERVO_JOINT_CFG},
                noise=Unoise(n_min=-0.25, n_max=0.25),
                # 此处延迟按观测更新计；每个控制步更新时为 20 ms，与舵机的物理步延迟不同。
                delay_min_lag=1,
                delay_max_lag=1,
            ),
            func=mdp.joint_vel_rel_backlash,
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    # 价值网络训练时可使用真实速度、足部接触等特权信息；部署 actor 只使用 policy 组。
    # 非对称的AC架构？
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function (76 values)."""

        """多了 身体速度+左右脚高度+腾空时间+接触+接触力
        """
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 身体线速度 3
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # 角 速度 3  两者都有
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 重力方向
        joint_pos = _backlash_preset(
            ObsTerm(func=mdp.joint_pos_rel_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False}),
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        joint_vel = _backlash_preset(
            ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG}),
            func=mdp.joint_vel_rel_backlash,
            backlash_cfg=_PLAY_JOINT_CFG,
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_height = ObsTerm(func=mdp.foot_height_safe, params={"asset_cfg": _FOOT_BODY_CFG})  #
        foot_air_time = ObsTerm(func=mdp.foot_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # startup 在环境初始化时随机化，reset 在对应环境重置时执行，interval 按时间间隔执行。
    # 例如编码器偏置在初始化后保持，而质心位置会在每次重置时重新采样。
    # 随机摩擦力
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # 随机化误差，模拟不同机器人的差异
    encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="startup", params={"bias_range": (-0.015, 0.015)})
    mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    # 质量和惯性随机化，模拟不同机器人的差异
    # 恢复位置
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            # 额外的位置偏差
            "velocity_range": {},
        },
    )
    # 恢复默认
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    # 随机质心
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )
    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["neck", "neck_pitch(_[0-9]+)?", "yaw_roll_motion", "jaw_soft"]
            ),
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )
    # 随机关节摩擦
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction, mode="reset", params={"scale_range": (0.9, 1.1)}
    )
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    # 随机模拟撞击 给予速度
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 奖励函数返回每个环境的原始值，管理器再乘权重并以控制周期 dt 缩放总和。
    # 非负代价项通过负权重成为惩罚；实际权重还可能被下方 curriculum 修改。
    # TensorBoard 的 Episode_Reward/<名称> 是结束回合的加权累计贡献均值 / 最大回合时长 20 s，
    # 不是原始函数值，也不是除以实际存活时间的均值；回合变长会同时放大正奖励和负代价的累计量。
    # 下方“趋近 0/上升更好”均需在相近回合长度、任务分布和权重下比较，不能只看单条曲线。
    # 线速度跟随：奖励 exp(-(平面跟踪误差² + 竖直速度²)/std²)，原始值在 [0, 1]。
    # 同等回合长度下，日志上升通常表示跟踪改善、竖直跳动减少；也应对照 error_vel_xy。
    track_lin_vel = RewTerm(
        func=mdp.track_linear_velocity, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.1)}
    )
    # 角速度跟随：奖励目标偏航角速度，同时抑制机体横滚、俯仰角速度；原始值在 [0, 1]。
    # 日志越高通常越好；与 error_vel_yaw 不完全等价，后者只衡量偏航跟踪误差。
    track_ang_vel = RewTerm(
        func=mdp.track_angular_velocity, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    # 直立奖励：躯干坐标系中重力的水平分量越小，奖励越高；不约束航向。
    # 它不能单独区分正立与倒立，需配合 fell_over；日志上升也可能来自回合变长。
    upright = RewTerm(func=mdp.upright, weight=2.0, params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG})
    # 腿部姿态奖励：按各关节容差归一化偏离默认站姿的误差，再计算指数奖励。
    # 目标运动指令超过阈值时使用 walking 容差；容差不是硬限位，过度追求此项可能限制迈步。
    pose = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std_standing": {".*hip_yaw": 0.1, ".*hip_roll": 0.05, ".*hip_pitch": 0.15, ".*knee": 0.15, ".*ankle": 0.1},
            "std_walking": {".*hip_yaw": 0.3, ".*hip_roll": 0.05, ".*hip_pitch": 0.4, ".*knee": 0.4, ".*ankle": 0.25},
            "walking_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True),
        },
    )
    # 平方和 抑制晃动
    # 躯干晃动代价：世界坐标系 xy 角速度平方和；负权重使其成为非正贡献。
    # 相同权重与回合长度下，曲线趋近 0 通常表示晃动减少。
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    # 角动量
    # 整机角动量代价：绕整机质心的总角动量平方，包含各刚体自转和轨道贡献。
    # 先求和再平方，部件间可相互抵消；同等条件下负值趋近 0 表示净角动量减小。
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    # Play hinges rest against their stops, so only servos incur the soft-limit penalty with backlash.
    # 软限位越界代价：超出关节软位置范围越多，负贡献越大；这不是直接截断关节角度。
    # 日志接近 0 表示较少越界；齿隙版仅检查舵机关节，排除被动齿隙关节。
    dof_pos_limits = _backlash_preset(
        RewTerm(func=mdp.joint_pos_limits_finite, weight=-1.0), asset_cfg=_SERVO_JOINT_CFG
    )
    # 动作突变代价：sum((本次动作 - 上次动作)²)，没有除以控制周期。
    # 课程会在 500/750/1000/1250/1500 轮加重负权重；曲线骤降未必是动作变差。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # 腾空窗口奖励：有运动指令时，每只脚当前腾空时间严格处于 0.125–0.3 s 才计 1 分。
    # 逐控制步累计而非落地结算；较高不必然代表更好步态，需排查双脚跳跃并对照速度误差。
    air_time = RewTerm(
        func=mdp.feet_air_time_windowed,
        weight=3.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold_min": 0.125,
            "threshold_max": 0.3,
            "command_threshold": 0.01,
        },
    )
    # 移动足高度代价：sum(|踝部高度 - 目标高度| × 足部水平速度)，仅有运动指令时启用。
    # 目标 0.0425 m 对应脚底约离地 2 cm；负值趋近 0 也可能只是脚不动，需结合速度跟随。
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=-2.0,
        params={
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    # 摆动峰值代价：摆动期间记录最高点，首次落地时结算 (峰值/目标高度 - 1)²。
    # 仅有运动指令时启用；曲线有脉冲或波动正常，接近 0 也可能因未形成有效摆动和落地。
    foot_swing_height = RewTerm(
        func=mdp.foot_swing_height,
        weight=-0.25 * (MICRODUCK_FOOT_TARGET_HEIGHT / 0.02) ** 2,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "asset_cfg": _FOOT_BODY_CFG,
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "command_threshold": 0.01,
        },
    )
    # 滑动代价：仅对接触中的脚累加世界系水平速度平方，且需有运动指令。
    # 相同条件下负值趋近 0 表示滑动减少，但站着不动也会很小，应同时确认确实在行走。
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.1,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    # 自碰撞代价：过滤传感器中接触力超过默认 1 N 阈值的被监测刚体数，不是接触力总和。
    # 当前传感器监测左脚与右脚；负值趋近 0 表示触发这种接触的情况减少。
    self_collisions = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("self_collision")}
    )
    # 头部跟随：4 个头颈关节分别计算 exp(-(角度误差/std)²) 后平均，std=0.5 rad。
    # 课程逐步扩大目标范围，曲线暂降可能因任务变难；齿隙版考虑相应被动角度。
    head_pose_tracking = _backlash_preset(
        RewTerm(
            func=mdp.head_pose_tracking,
            weight=2.0,
            params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
        ),
        backlash_cfg=_HEAD_PLAY_JOINT_CFG,
    )
    # 头部持续误差代价：对有符号误差作时间常数 1 s 的滑动平均，再求绝对值均值。
    # 初始权重为 0，600/1000/1500 轮改为 -1/-2/-3；由 0 变负是启用惩罚，未必发生退步。
    head_pose_bias = _backlash_preset(
        RewTerm(
            func=mdp.head_pose_bias_penalty,
            weight=0.0,
            params={"command_name": "head_pose", "tau_s": 1.0, "asset_cfg": _HEAD_JOINT_CFG},
        ),
        backlash_cfg=_HEAD_PLAY_JOINT_CFG,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 超时标记为时间截断，区别于摔倒等任务终止，供训练器正确处理价值估计。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 区分
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    nan_state = DoneTerm(func=mdp.robot_state_is_nan, time_out=False, params={"sensor_names": ("contact_forces",)})


# 课程学习
def _schedule(address: str, *stages: tuple[int, object]) -> CurrTerm:
    """Step ``address`` through ``(upstream_iteration, value)`` stages after its configured value."""
    return CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": address,
            "modify_fn": mdp.staged_value,
            "modify_params": {
                # 课程按环境控制步计数；每轮 PPO 采样 24 步，因此用迭代数 × 24 转换，勿乘并行环境数。
                "stages": [(iteration * MICRODUCK_STEPS_PER_ITERATION, value) for iteration, value in stages]
            },
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # 第 xxx 轮
    action_rate_weight = _schedule(
        "rewards.action_rate_l2.weight", (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0)
    )
    head_pose_bias_weight = _schedule("rewards.head_pose_bias.weight", (600, -1.0), (1000, -2.0), (1500, -3.0))
    standing_envs = _schedule(
        "commands.base_velocity.rel_standing_envs", (500, 0.05), (750, 0.1), (1000, 0.15), (1500, 0.2), (2000, 0.25)
    )
    head_pose_range = _schedule(
        "commands.head_pose.ranges",
        (500, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
        (1000, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
        (1500, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.2, 0.2))),
        (2000, ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-0.31, 0.31))),
    )
    com_range = _schedule(
        "events.randomize_com.params.com_range",
        (500, {axis: (-0.005, 0.005) for axis in "xyz"}),
        (1000, {axis: (-0.01, 0.01) for axis in "xyz"}),
        (1500, {axis: (-0.015, 0.015) for axis in "xyz"}),
    )
    head_com_range = _schedule(
        "events.randomize_head_com.params.com_range",
        (500, {axis: (-0.005, 0.005) for axis in "xyz"}),
        (1000, {axis: (-0.01, 0.01) for axis in "xyz"}),
    )


@configclass
class MicroDuckVelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck flat velocity walking with Newton MJWarp and BAM servos."""

    sim: SimulationCfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=96,
                nconmax=16,
                # microduck_rl's 10 iterations stop some contact solves short; MJWarp exits once every world converges.
                iterations=100,
                # The original 20-step budget exhausts line search with the current MJWarp solver.
                ls_iterations=50,
                cone="pyramidal",
                impratio=1.0,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2500000),
            default_shape_cfg=NewtonShapeCfg(margin=0.0),
            num_substeps=1,
        )
    )
    scene: MicroDuckSceneCfg = MicroDuckSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    recorders: MicroDuckRecordersCfg = MicroDuckRecordersCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 物理仿真 200 Hz；每 4 个物理步更新一次策略，控制频率 50 Hz，每回合最多 1000 个控制步。
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.self_collision.update_period = self.sim.dt
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))

    def play_mode(self):
        """Use the shared inference defaults and disable interval pushes."""
        super().play_mode()
        self.events.push_robot = None
