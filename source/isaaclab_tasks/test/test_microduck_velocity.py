# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck walking: deployed policy interface, rough-terrain foot sensing, and the backlash preset.

``test/contrib/test_contrib_environments_kitless.py`` steps one task per family, the rough task here, so the flat
task steps its default robot below and the backlash test steps the play-hinge robot.
"""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.mdp.diagnostics import DiagnosticAccumulator
from isaaclab_tasks.contrib.microduck.mdp.observations import foot_height_safe
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
ROUGH_TASK = "IsaacContrib-Velocity-Rough-MicroDuck"

POLICY_TERMS = [
    "imu",
    "joint_pos",
    "joint_vel",
    "actions",
    "velocity_commands",
    "head_pose_commands",
    "body_pose_commands",
]
"""Actor input order of the deployed ONNX policy."""


def test_tracking_cost_uses_physical_targets_and_ordered_post_step_positions():
    """Bias and default pose are coordinate transforms, not tracking error; only dead-zone excess costs."""
    from isaaclab.managers import RewardTermCfg

    from isaaclab_tasks.contrib.microduck.mdp.actions import BiasedJointPositionAction
    from isaaclab_tasks.contrib.microduck.mdp.rewards import joint_tracking_error_l2

    action = object.__new__(BiasedJointPositionAction)
    action._num_joints = 2
    action._processed_actions = torch.tensor([[0.3, 0.6]]).expand(4, -1)
    action.encoder_bias = torch.tensor([[0.1, 0.2]]).expand(4, -1)
    positions = torch.tensor([[0.4, 9, 0.2], [0.44, 9, 0.24], [0.6, 9, 0.4], [float("nan"), 9, 0.2]])
    env = SimpleNamespace(
        device="cpu",
        action_manager=SimpleNamespace(get_term=lambda _: action),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=positions)))},
    )
    cfg = RewardTermCfg(
        func=joint_tracking_error_l2,
        weight=-1,
        params={
            "tolerance": [0.05, 0.05],
            "asset_cfg": SceneEntityCfg("robot", joint_ids=[2, 0]),
            "scale": 0.1,
        },
    )
    term = joint_tracking_error_l2(cfg, env)
    torch.testing.assert_close(term(env, **cfg.params), torch.tensor([0.0, 0.0, 2.25, 0.0]))


def test_continuation_variants_are_independent_and_calibration_is_balanced():
    """Reward ablations preserve the baseline and calibration targets exactly 5% positive contribution."""
    import numpy as np

    from isaaclab_tasks.contrib.microduck.experiments import apply_variant, freeze_calibration
    from isaaclab_tasks.utils import resolve_task_config

    baseline, _ = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=())
    groups = []
    for i in range(7):
        values = np.tile(np.array([0.01, 0.03, 0.05, 0.07]), 300 * (i + 1))[:, None]
        groups.append(
            {
                "case": str(i),
                "errors": np.repeat(values, 14, 1),
                "positive_rewards": np.full(len(values), 4.0),
                "valid_trials": 16,
            }
        )
    calibration = freeze_calibration(groups, MICRODUCK_JOINT_NAMES)
    assert calibration["weight_magnitude"] * calibration["cost_mean"] == pytest.approx(0.2)
    for variant in ("baseline", "velocity_weight", "velocity_std", "joint_tracking", "v_std", "w_std", "w_rational"):
        cfg = apply_variant(baseline.copy(), variant, calibration)
        assert cfg.rewards.track_lin_vel.weight == (4 if variant == "velocity_weight" else 2)
        assert cfg.rewards.track_ang_vel.weight == (4 if variant == "velocity_weight" else 2)
        assert cfg.rewards.track_lin_vel.params["std"] == pytest.approx(
            0.15 if variant in ("velocity_std", "v_std") else 0.1**0.5
        )
        assert cfg.rewards.track_ang_vel.params["std"] == pytest.approx(
            0.35 if variant in ("velocity_std", "w_std", "w_rational") else 0.5**0.5
        )
        assert (cfg.rewards.joint_tracking_error is not None) == (variant == "joint_tracking")
    assert baseline.rewards.joint_tracking_error is None
    assert baseline.recorders.diagnostics.tracking_tolerance is None
    halved = apply_variant(baseline.copy(), "competitor_half", competitor="action_rate_l2")
    from isaaclab_tasks.contrib.microduck.experiments import freeze_curriculum

    freeze_curriculum(halved, 144024)
    assert halved.rewards.action_rate_l2.weight == -0.5, "the mature curriculum must preserve the intervention"
    assert halved.rewards.head_pose_bias.weight == -3
    groups[0]["errors"] = groups[0]["errors"][:2]
    with pytest.raises(ValueError, match="Insufficient"):
        freeze_calibration(groups, MICRODUCK_JOINT_NAMES)


def test_angular_reward_shapes_preserve_units_and_local_scale():
    """Both kernels include roll/pitch and yaw, share local scale, and differ in their high-error tail."""
    import math

    from isaaclab_tasks.contrib.microduck.mdp.rewards import track_angular_velocity

    angular = torch.tensor([[0.0, 0.0, 0.0], [0.35, 0.0, 0.0], [0.0, 0.0, 1.05]], dtype=torch.float64)
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=SimpleNamespace(torch=angular)))},
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros_like(angular)),
    )
    for shape, expected in (("exponential", [1.0, math.exp(-1), math.exp(-9)]), ("rational", [1.0, 0.5, 0.1])):
        result = track_angular_velocity(env, 0.35, "base_velocity", shape=shape)
        torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
        angular[0, 2] = 1e-4
        perturbed = track_angular_velocity(env, 0.35, "base_velocity", shape=shape)[0]
        assert (1 - perturbed.item()) / 1e-8 == pytest.approx(1 / 0.35**2, rel=1e-6)
        angular[0, 2] = 0
    with pytest.raises(ValueError, match="Unknown"):
        track_angular_velocity(env, 0.35, "base_velocity", shape="unknown")


def _make_env(task: str, device: str, num_envs: int, overrides: tuple[str, ...] = ()):
    cfg = parse_env_cfg(task, device=device, num_envs=num_envs, overrides=overrides)
    sim_utils.create_new_stage()
    return gym.make(task, cfg=cfg).unwrapped


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_deploy_interface_and_friction_reset(device, monkeypatch):
    """The deployed policy's 61/76 inputs and joint order hold, resets redraw BAM friction, and steps stay finite.

    A diverged state terminates only its environment, whose final reward and reset observations stay finite.
    """
    from isaaclab.managers import RewardTermCfg

    from isaaclab_tasks.contrib.microduck.mdp.rewards import joint_tracking_error_l2

    cfg = parse_env_cfg(TASK, device=device, num_envs=2)
    cfg.recorders.diagnostics.capture_samples = True
    cfg.recorders.diagnostics.tracking_tolerance = [0.05] * 14
    cfg.rewards.joint_tracking_error = RewardTermCfg(
        func=joint_tracking_error_l2,
        weight=-0.2,
        params={
            "tolerance": [0.05] * 14,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True),
        },
    )
    sim_utils.create_new_stage()
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 61)
        assert obs["critic"].shape == (2, 76)
        assert env.observation_manager.active_terms["policy"] == POLICY_TERMS
        # The IMU reads angular velocity, at rest after reset, then unit gravity.
        assert ((obs["policy"][:, 3:6].norm(dim=-1) - 1.0).abs() < 0.05).all()
        assert env.action_manager.get_term("joint_pos").IO_descriptor.joint_names == MICRODUCK_JOINT_NAMES

        robot = env.scene["robot"]
        before = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale").clone()
        env.reset()
        after = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale")
        assert not (before == after).all()
        assert (after == after[:, :1]).all(), "one friction scale per environment"
        assert ((after >= 0.9) & (after <= 1.1)).all()

        observation_calls = []
        compute_observations = env.observation_manager.compute

        def counted_observations(*args, **kwargs):
            observation_calls.append(1)
            return compute_observations(*args, **kwargs)

        monkeypatch.setattr(env.observation_manager, "compute", counted_observations)
        yaw_errors = []
        with torch.inference_mode():
            for _ in range(20):
                command = env.command_manager.get_command("base_velocity")[:, 2].clone()
                obs, reward, *_ = env.step(torch.randn(2, 14, device=env.device) * 0.1)
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
                recorder = env.recorder_manager._terms["diagnostics"]
                samples = dict(zip(recorder.sample_names, recorder.snapshot.unbind(dim=1)))
                errors = torch.stack(
                    [samples[f"joint_target/{name}"] - samples[f"joint_pos/{name}"] for name in MICRODUCK_JOINT_NAMES],
                    1,
                )
                expected_cost = ((errors.abs() - 0.05).clamp_min(0) / 0.1).square().mean(1)
                reward_index = env.reward_manager.active_terms.index("joint_tracking_error")
                torch.testing.assert_close(
                    env.reward_manager.weighted_term_values[:, reward_index], -0.2 * expected_cost
                )
                yaw_errors.append(robot.data.root_link_ang_vel_b.torch[:, 2].clone() - command)
            assert len(observation_calls) == 20, "state-only recording must not draw extra observation noise"
            _, info = env.reset()
            expected_rms = torch.stack(yaw_errors).square().mean(dim=0).sqrt().mean()
            torch.testing.assert_close(info["log"]["Diagnostics/all/yaw_error_rms_radps"], expected_rms)
            # A diverged state terminates, and its final reward stays finite although rewards precede resets.
            nan = torch.full((1, 1), float("nan"), device=env.device)
            robot.write_joint_velocity_to_sim_index(velocity=nan, joint_ids=[0], env_ids=[0])
            obs, reward, *_ = env.step(torch.zeros(2, 14, device=env.device))
            assert env.termination_manager.get_term("nan_state").tolist() == [True, False]
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
    finally:
        env.close()
        SimulationContext.clear_instance()


def test_microduck_diagnostic_reductions_and_partial_reset():
    """Opposite errors retain nonzero RMS; sparse events and partial resets preserve their own denominators."""
    stats = DiagnosticAccumulator(2, ["yaw_mean_radps", "yaw_rms_radps", "stance_mean_s"], "cpu")
    stats.update(torch.tensor([[1.0, 1.0, torch.nan], [2.0, 2.0, 0.2]]), torch.tensor([True, True]))
    stats.update(torch.tensor([[-1.0, -1.0, 0.4], [9.0, 9.0, 0.8]]), torch.tensor([True, False]))
    torch.testing.assert_close(stats.values(), torch.tensor([[0.0, 1.0, 0.4], [2.0, 2.0, 0.2]]))
    stats.reset([0])
    assert torch.isnan(stats.values()[0]).all()
    torch.testing.assert_close(stats.values()[1], torch.tensor([2.0, 2.0, 0.2]))


def test_microduck_foot_height_on_steps():
    """Use the closest terrain sample per foot, ignoring misses, step undersides, and world elevation.

    Missed rays and rays starting inside a step cannot be placed reliably in a live scene.
    """
    foot_positions = torch.tensor([[[0.0, 0.0, 0.05], [0.0, 0.0, 0.07]]])
    left_hits = torch.tensor([[[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]]])
    right_hits = torch.tensor([[[0.0, 0.0, 0.03], [float("inf"), float("inf"), float("inf")]]])
    left_normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

    def ray_sensor(hits, normals):
        data = SimpleNamespace(ray_hits_w=SimpleNamespace(torch=hits), ray_normals_w=SimpleNamespace(torch=normals))
        return SimpleNamespace(data=data, cfg=SimpleNamespace(max_distance=1.0))

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene(
        robot=SimpleNamespace(data=SimpleNamespace(body_link_pos_w=SimpleNamespace(torch=foot_positions))),
        left=ray_sensor(left_hits, left_normals),
        right=ray_sensor(right_hits, left_normals.clone()),
    )
    env = SimpleNamespace(scene=scene)
    params = {"asset_cfg": SceneEntityCfg("robot", body_ids=[0, 1]), "height_sensor_names": ("left", "right")}
    expected = torch.tensor([[0.03, 0.04]])
    torch.testing.assert_close(foot_height_safe(env, **params), expected)
    for tensor in (foot_positions, left_hits, right_hits, scene.env_origins):
        tensor[..., 2] += 4.0
    torch.testing.assert_close(foot_height_safe(env, **params), expected)
    right_hits.fill_(float("inf"))
    torch.testing.assert_close(foot_height_safe(env, **params), torch.tensor([[0.03, 0.07]]))
    left_normals[0, 0, 2] = -1.0
    torch.testing.assert_close(foot_height_safe(env, **params), torch.tensor([[0.0, 0.07]]))


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_rough_terrain_contact_and_foot_rays(device):
    """The terrain carries upstream's contact response, and foot rays see the ground under each foot."""
    cfg = parse_env_cfg(ROUGH_TASK, device=device, num_envs=8)
    # A two-row map with one column per sub-terrain.
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows, generator.num_cols, generator.seed = 2, 4, 42
    for sub_terrain in generator.sub_terrains.values():
        sub_terrain.proportion = 0.25
    sim_utils.create_new_stage()
    env = gym.make(ROUGH_TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        mesh = sim_utils.get_current_stage().GetPrimAtPath("/World/ground/terrain/mesh")
        assert tuple(mesh.GetAttribute("mjc:solref").Get()) == pytest.approx((0.04, 1.0))
        assert tuple(mesh.GetAttribute("mjc:solimp").Get()) == pytest.approx((0.85, 0.95, 0.001, 0.5, 2.0))
        assert env.scene.env_origins[:, 2].max() > 0.05, "some environments must start on raised tiles"
        for name in ("left_foot_height", "right_foot_height"):
            hits = env.scene[name].data.ray_hits_w.torch
            assert torch.isfinite(hits).all()
            torch.testing.assert_close(hits[..., 2], env.scene.env_origins[:, 2, None].expand(-1, 2), atol=2e-3, rtol=0)
        heights = foot_height_safe(env, **env.observation_manager.cfg.critic.foot_height.params)
        assert ((heights > 0.01) & (heights < 0.04)).all()
    finally:
        env.close()
        SimulationContext.clear_instance()


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_backlash_preset_reads_output_side_encoders(device):
    """With ``presets=backlash``, encoders, servo feedback and head rewards see servo plus play; play stays undriven.

    Distinct play values per joint and environment catch pairing or ordering mistakes. The kitless contributed-task
    test only steps default presets, so this test also steps the backlash robot.
    """
    env = _make_env(TASK, device, num_envs=8, overrides=("presets=backlash",))
    try:
        env.reset()
        robot = env.scene["robot"]
        assert robot.num_joints == 28
        play_ids = [robot.joint_names.index(f"passive_{name}_backlash") for name in MICRODUCK_JOINT_NAMES]
        torch.testing.assert_close(robot.data.joint_pos.torch[:, play_ids], torch.zeros(8, 14, device=env.device))

        play = torch.linspace(-0.014, 0.014, 8 * 14, device=env.device).reshape(8, 14)
        q = robot.data.default_joint_pos.torch.clone()
        qd = torch.zeros_like(q)
        q[:, play_ids] = play
        qd[:, play_ids] = play * 10.0
        robot.write_joint_state_to_sim_index(position=q, velocity=qd)
        bias = env.action_manager.get_term("joint_pos").encoder_bias
        for group in (env.observation_manager.cfg.policy, env.observation_manager.cfg.critic):
            term = group.joint_pos
            expected = play + (bias if term.params["biased"] else 0.0)
            torch.testing.assert_close(term.func(env, **term.params), expected)
            term = group.joint_vel
            torch.testing.assert_close(term.func(env, **term.params), play * 10.0)
        env.command_manager.get_command("head_pose").zero_()
        head = env.reward_manager.get_term_cfg("head_pose_tracking")
        expected = torch.exp(-((play[:, 5:9] / head.params["std"]) ** 2)).mean(dim=-1)
        torch.testing.assert_close(head.func(env, **head.params), expected)
        head_bias = env.reward_manager.get_term_cfg("head_pose_bias")
        head_bias.func.reset()
        expected = play[:, 5:9].abs().mean(dim=-1) * env.step_dt / head_bias.params["tau_s"]
        torch.testing.assert_close(head_bias.func(env, **head_bias.params), expected)
        # Each servo feeds back its own play angle, the only error at rest; play-hinge velocity adds no back-EMF.
        # One physics step, not a decimated env step, keeps the first-step effort analytic.
        robot.actuators.target_command.set_position_index(value=robot.data.default_joint_pos.torch)
        robot.write_data_to_sim()
        env.sim.backend_manager.set_decimation(1)
        env.sim.step(render=False)
        env.sim.backend_manager.set_decimation(env.cfg.decimation)
        robot.update(env.physics_dt)
        servos = robot.cfg.actuators["servos"]
        vin = read_group_parameter(robot.actuators, "servos", "drive", "vin")[:, :1]
        expected = -play * servos.kp_fw * servos.motor.error_gain * vin * servos.motor.kt / servos.motor.resistance
        servo_ids = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)[0]
        effort = robot.actuators.applied_effort.torch[:, servo_ids]
        torch.testing.assert_close(effort, expected, atol=1e-6, rtol=1e-5)
        # Riding the play stops must not incur a servo soft-limit penalty.
        q[:, play_ids] = robot.data.soft_joint_pos_limits.torch[:, play_ids, 1] + 0.001
        robot.write_joint_state_to_sim_index(position=q, velocity=qd)
        limits = env.reward_manager.get_term_cfg("dof_pos_limits")
        torch.testing.assert_close(limits.func(env, **limits.params), torch.zeros(8, device=env.device))

        env.reset()
        with torch.inference_mode():
            for _ in range(20):
                obs, reward, *_ = env.step(torch.randn(8, 14, device=env.device) * 0.1)
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
        effort = robot.actuators.applied_effort.torch[:, play_ids]
        torch.testing.assert_close(effort, torch.zeros_like(effort))
    finally:
        env.close()
        SimulationContext.clear_instance()
