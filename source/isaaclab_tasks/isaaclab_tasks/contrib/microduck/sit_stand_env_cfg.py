# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sit/stand baseline using the official MicroDuck posture recipe on Newton MJWarp.

The first baseline retains the official pose, target ramp, reward weights and mixed
resets. Pushes, wide head commands and the later seated-tip curriculum are deferred
until the two transitions are validated. Actor/critic observations append one posture
flag (62/77 values); the 14 joint-position actions and PPO network are unchanged.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG, MICRODUCK_ALLCOLLISIONS_CFG

from . import mdp
from .flat_env_cfg import (
    _HEAD_JOINT_CFG,
    _HEAD_PLAY_JOINT_CFG,
    MicroDuckVelocityFlatEnvCfg,
    ObservationsCfg,
    _backlash_preset,
    _schedule,
)
from .mdp import sit_stand


@configclass
class SitStandObservationsCfg(ObservationsCfg):
    """Append the independent posture command to both policy and critic."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        posture_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "posture"})

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        posture_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "posture"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class SitStandCurriculumCfg:
    """Consolidate transitions before increasing motion regularization."""

    action_rate_weight = _schedule(
        "rewards.action_rate_l2.weight", (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0)
    )
    descent_speed_weight = _schedule("rewards.descent_speed.weight", (500, -20.0))
    rise_speed_weight = _schedule("rewards.rise_speed.weight", (1500, -5.0), (2500, -10.0))


@configclass
class MicroDuckSitStandEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """Commanded sitting, seated rest, rising and standing rest on a flat floor."""

    observations: SitStandObservationsCfg = SitStandObservationsCfg()
    curriculum: SitStandCurriculumCfg = SitStandCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.scene.robot = preset(
            default=MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
            backlash=MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        )
        self.sim.physics.solver_cfg.nconmax = 200
        self.sim.physics.solver_cfg.njmax = 1024
        # Ground and self contact of the trunk/hips/head must remain physical.
        self.commands.posture = sit_stand.SitStandCommandCfg()
        velocity = self.commands.base_velocity
        velocity.ranges.lin_vel_x = (0.0, 0.0)
        velocity.ranges.lin_vel_y = (0.0, 0.0)
        velocity.ranges.ang_vel_z = (0.0, 0.0)
        velocity.rel_standing_envs = 1.0
        velocity.rel_forward_envs = 0.0
        velocity.rel_turn_in_place_envs = 0.0
        velocity.debug_vis = False
        self.commands.body_pose.ranges = ((0.0, 0.0),) * 6
        self.commands.head_pose.ranges = ((0.0, 0.0),) * 4
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.events.reset_posture = EventTerm(func=sit_stand.reset_posture, mode="reset")
        self.events.push_robot = None
        # Allow recoverable intermediate tilts, as in the official SitStand task.
        self.terminations.fell_over = None

        for name in (
            "track_lin_vel",
            "track_ang_vel",
            "pose",
            "upright",
            "air_time",
            "foot_clearance",
            "foot_swing_height",
            "foot_slip",
            "head_pose_bias",
        ):
            setattr(self.rewards, name, None)
        # Keep existing head tracking, including its backlash-aware implementation.
        self.rewards.head_pose_tracking = _backlash_preset(
            RewTerm(
                func=mdp.head_pose_tracking,
                weight=0.75,
                params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
            ),
            backlash_cfg=_HEAD_PLAY_JOINT_CFG,
        )
        self.rewards.posture_pose = RewTerm(func=sit_stand.posture_pose, weight=4.0)
        self.rewards.posture_pose_l1 = RewTerm(func=sit_stand.posture_pose_error, weight=-1.0)
        self.rewards.posture_height = RewTerm(func=sit_stand.posture_height, weight=1.0, params={"std": 0.04})
        self.rewards.posture_height_sharp = RewTerm(func=sit_stand.posture_height, weight=1.0, params={"std": 0.015})
        self.rewards.posture_height_l1 = RewTerm(func=sit_stand.posture_height_error, weight=-6.0)
        self.rewards.posture_composite = RewTerm(func=sit_stand.posture_composite, weight=3.0)
        self.rewards.posture_stillness = RewTerm(func=sit_stand.posture_stillness, weight=2.0)
        self.rewards.upright_linear = RewTerm(func=sit_stand.posture_upright, weight=2.5)
        self.rewards.upright_while_tall = RewTerm(
            func=sit_stand.posture_upright, weight=1.5, params={"height_gated": True}
        )
        self.rewards.rise_bootstrap = RewTerm(func=sit_stand.rise_bootstrap, weight=0.75)
        self.rewards.descent_speed = RewTerm(
            func=sit_stand.vertical_speed_excess, weight=-10.0, params={"direction": -1.0, "limit": 0.05}
        )
        self.rewards.rise_speed = RewTerm(
            func=sit_stand.vertical_speed_excess, weight=0.0, params={"direction": 1.0, "limit": 0.08}
        )
        self.rewards.gentle_motion = RewTerm(func=sit_stand.vertical_acceleration, weight=-0.05)
