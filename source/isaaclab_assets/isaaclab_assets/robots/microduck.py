# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Pollen Robotics' MicroDuck with Dynamixel XL330 BAM servos.

The following configurations are available:

* :data:`MICRODUCK_CFG`: the 14-joint walking robot with its standing pose.
* :data:`MICRODUCK_ALLCOLLISIONS_CFG`: the robot with additional body and head collisions.
* :data:`MICRODUCK_BACKLASH_CFG` and :data:`MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG`: the corresponding variants
  with gearbox play.

Reference: https://github.com/pollen-robotics/microduck_rl
"""

from isaaclab_newton.sim.schemas import NewtonArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuatorCfg, BamMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import clone, replace
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# 这些数值是 XL330 的辨识参数，描述电气响应和摩擦；不能直接当成普通 PD 增益调节。
# 舵机 封装，接受指令 执行具体转动  电压->原始电机->关节转动 
_XL330_MOTOR_CFG = BamMotorCfg(
    model="m6",
    kt=0.36601349688984386,
    resistance=2.8113923539223227,
    error_gain=0.0028773775022263564,
    max_pwm=1.0,
    max_current=1.75,
    friction_base=0.004771183165566,
    friction_viscous=0.005359668274599504,
    friction_stribeck=0.004676345799486616,
    dtheta_stribeck=2.890372094130307,
    alpha=8.683259907618984,
    load_friction_motor=0.2667860954283698,
    load_friction_external=8.515871897059342e-06,
    load_friction_motor_stribeck=1.0722918395099123e-05,
    load_friction_external_stribeck=0.08077928978935671,
    load_friction_motor_quad=0.009972471242139415,
    load_friction_external_quad=0.004902565732332559,
)
"""XL330 m6 fit from Rhoban BAM revision ``62bd8ce12154340be97e06f7f41a0ca8f116d967``."""

# 动作和关节观测共用此顺序：左腿 5 维、颈头 4 维、右腿 5 维。
# USD 内部先列右腿，使用列表选择关节时需保留顺序，避免策略输出驱动错关节。
# 相对默认姿态的偏移角度
MICRODUCK_JOINT_NAMES: list[str] = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""The 14 servos in the robot's servo order: left leg, neck and head, right leg (the mouth servo is not modeled).

Every MicroDuck USD lists the right leg first, so select these with ``preserve_order=True`` where order matters.
"""

MICRODUCK_HEAD_JOINT_NAMES: list[str] = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
"""The 4 neck and head servos, in servo order."""

MICRODUCK_LEG_JOINT_NAMES: list[str] = [n for n in MICRODUCK_JOINT_NAMES if n not in MICRODUCK_HEAD_JOINT_NAMES]
"""The 10 leg servos, in servo order (left leg first)."""

# usd描述鸭子参数
MICRODUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/PollenRobotics/MicroDuck/microduck_walk.usd",
        activate_contact_sensors=True,
        articulation_props=NewtonArticulationCfg(self_collision_enabled=True),
    ),
    # 根位置单位为米，关节角单位为弧度；左右关节轴方向不同，因此对称站姿可能使用相反符号。
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.125),
        joint_pos={
            ".*hip_yaw": 0.0,
            "left_hip_roll": -0.0873,
            "right_hip_roll": 0.0873,
            "left_hip_pitch": -0.4579,
            "right_hip_pitch": 0.4579,
            "left_knee": -0.0049,
            "right_knee": 0.0049,
            "left_ankle": 0.4530,
            "right_ankle": -0.4530,
            "neck_pitch": 0.3491,
            "head_pitch": 0.3491,
            "head_yaw": 0.0,
            "head_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "servos": BamActuatorCfg(
            joint_names_expr=["^(?!passive_).*"],
            motor=_XL330_MOTOR_CFG,
            kp_fw=200.0,
            vin=7.4,
            vin_range=(6.5, 8.2),
            vin_drop_gain_range=(0.0, 0.2),
            vin_min=6.0,
            # 延迟以物理步计；平地任务 dt=0.005 s，对应 15–30 ms，不能按策略周期换算。
            min_delay=3,
            max_delay=6,
        ),
    },
)
"""MicroDuck with BAM voltage control, battery sag, and a 3--6-step command delay.

Requires Newton's MJWarp solver and ``SimulationCfg.use_newton_actuators=True``.
"""
# 对鸭子的全部碰撞体进行建模，增加躯干、臀部和头部的碰撞几何体，便于全身接触。
MICRODUCK_ALLCOLLISIONS_CFG = clone(MICRODUCK_CFG)
MICRODUCK_ALLCOLLISIONS_CFG.spawn.usd_path = (
    f"{ISAACLAB_NUCLEUS_DIR}/Robots/PollenRobotics/MicroDuck/microduck_allcollisions.usd"
)
"""MicroDuck with additional trunk, hip, and head collision geometry for whole-body contact."""

# 对齿轮间隙进行建模
def _with_backlash(cfg: ArticulationCfg, usd_name: str) -> ArticulationCfg:
    """Use the USD with a passive gearbox-play hinge in series with each servo."""
    cfg = clone(cfg)
    cfg.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/PollenRobotics/MicroDuck/{usd_name}"
    # 被动齿隙关节串联在舵机之后，不增加策略动作维数；模板将每个舵机映射到对应被动关节。
    cfg.actuators["servos"] = replace(cfg.actuators["servos"], backlash_joint_template="passive_{}_backlash")
    cfg.init_state.joint_pos["passive_.*"] = 0.0
    return cfg


MICRODUCK_BACKLASH_CFG = _with_backlash(MICRODUCK_CFG, "microduck_walk_backlash.usd")
"""Walking MicroDuck with 14 servos and 14 passive gearbox-play hinges spanning ±1°."""

MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG = _with_backlash(
    MICRODUCK_ALLCOLLISIONS_CFG, "microduck_allcollisions_backlash.usd"
)
"""MicroDuck with additional body collisions and passive gearbox-play hinges."""
