# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check experiment reference timing and partial command resets in a live scene."""

import gymnasium as gym
import pytest
import torch
from microduck_tracking_experiment import TASK, make_config, observed_imu, observed_yaw_error

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


@pytest.mark.integration
@pytest.mark.kitless
def test_tracking_reference_and_partial_resample():
    cfg = make_config("pos_half", 4, 43)
    cfg.events.push_robot = None
    sim_utils.create_new_stage()
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (4, 62)
        assert obs["critic"].shape == (4, 77)
        term = env.command_manager.get_term("base_velocity")
        term.is_standing_env[:] = False
        term.vel_command_b[:] = torch.tensor([0.4, 0.1, 0.3], device=env.device)
        term.time_left[:] = 100
        start_p = term.reference_position.clone()
        start_yaw = term.reference_yaw.clone()
        for _ in range(12):
            _, _, terminated, truncated, _ = env.step(torch.zeros((4, 14), device=env.device))
            assert not (terminated | truncated).any()
            current = term.yaw_error_estimate.clone()
            term.update_reference()
            torch.testing.assert_close(term.yaw_error_estimate, current, rtol=0, atol=0)
            torch.testing.assert_close(observed_imu(env), observed_imu(env), rtol=0, atol=0)
        final_yaw = start_yaw + 0.3 * 12 * env.step_dt
        dx = (
            0.4 * (torch.sin(final_yaw) - torch.sin(start_yaw)) + 0.1 * (torch.cos(final_yaw) - torch.cos(start_yaw))
        ) / 0.3
        dy = (
            0.4 * (torch.cos(start_yaw) - torch.cos(final_yaw)) + 0.1 * (torch.sin(final_yaw) - torch.sin(start_yaw))
        ) / 0.3
        torch.testing.assert_close(
            term.reference_position - start_p, torch.stack((dx, dy), dim=-1), rtol=1e-4, atol=1e-6
        )
        old_other = term.reference_position[1:].clone()
        term.time_left[0] = 0
        env.command_manager.compute(dt=0)
        torch.testing.assert_close(term.reference_position[1:], old_other, rtol=0, atol=0)
        assert term.yaw_error_estimate[0].item() == 0
        assert torch.all(observed_yaw_error(env).abs() <= 1)
    finally:
        env.close()
        SimulationContext.clear_instance()
