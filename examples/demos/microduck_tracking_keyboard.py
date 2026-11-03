# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt the local MicroDuck keyboard viewer to cumulative-heading checkpoints."""

import importlib.util
from pathlib import Path

import torch
from microduck_tracking_experiment import TrackingVelocityCommand, observed_imu, observed_yaw_error

from isaaclab.managers import ObservationTermCfg

KEYBOARD_PATH = Path("/mnt/linux-data/www/IsaacLab-MicroDuck-main/examples/demos/microduck_keyboard.py")
spec = importlib.util.spec_from_file_location("microduck_keyboard_local", KEYBOARD_PATH)
keyboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keyboard)
original_config = keyboard.comparison_config
original_step = keyboard.KeyboardComparison.step
original_reset = keyboard.KeyboardComparison.reset


def tracking_config(device, seed):
    cfg, agent = original_config(device, seed)
    cfg.commands.base_velocity.class_type = TrackingVelocityCommand
    cfg.observations.policy.imu.func = observed_imu
    cfg.observations.policy.imu.params = {}
    cfg.observations.policy.imu.noise = None
    cfg.observations.policy.heading_error = ObservationTermCfg(func=observed_yaw_error)
    cfg.observations.critic.heading_error = ObservationTermCfg(func=observed_yaw_error)
    return cfg, agent


def tracking_reset(self, seed=None):
    original_reset(self, seed)
    self._tracking_last_command = None


def tracking_step(self):
    term = self.raw.command_manager.get_term("base_velocity")
    if self.command != self._tracking_last_command:
        term._resample(torch.arange(self.env.num_envs, device=self.raw.device))
        self._tracking_last_command = self.command
    original_step(self)


keyboard.comparison_config = tracking_config
keyboard.KeyboardComparison.reset = tracking_reset
keyboard.KeyboardComparison.step = tracking_step

if __name__ == "__main__":
    keyboard.main()
