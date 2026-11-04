# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt the local MicroDuck keyboard viewer to cumulative-heading checkpoints."""

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch
from microduck_tracking_experiment import TrackingVelocityCommand, observed_imu, observed_yaw_error

from isaaclab.managers import ObservationTermCfg

KEYBOARD_PATH = Path(__file__).with_name("microduck_keyboard.py")
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


def adapt_legacy_checkpoint(path, output):
    """Append an ignored heading input to a legacy actor without changing its outputs."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    actor = data["actor_state_dict"]
    if actor["mlp.0.weight"].shape[1] != 61:
        return path
    actor["mlp.0.weight"] = torch.nn.functional.pad(actor["mlp.0.weight"], (0, 1), value=0)
    for key, value in (("_mean", 0), ("_var", 1), ("_std", 1)):
        name = f"obs_normalizer.{key}"
        actor[name] = torch.nn.functional.pad(actor[name], (0, 1), value=value)
    torch.save(data, output)
    return output


def main():
    with tempfile.TemporaryDirectory(prefix="microduck_keyboard_") as temporary:
        in_checkpoints = False
        for index, value in enumerate(sys.argv[1:], 1):
            if value in ("--checkpoint", "--checkpoints"):
                in_checkpoints = True
            elif value.startswith("--"):
                in_checkpoints = False
            elif in_checkpoints and value != "latest":
                output = Path(temporary) / f"checkpoint_{index}.pt"
                sys.argv[index] = str(adapt_legacy_checkpoint(Path(value).expanduser(), output))
        keyboard.main()


if __name__ == "__main__":
    main()
