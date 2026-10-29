# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ordered keyboard selection restores policy normalization; fixed tests preserve trial boundaries."""

import csv
import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import pytest
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

spec = importlib.util.spec_from_file_location(
    "microduck_keyboard", Path(__file__).resolve().parents[3] / "examples/demos/microduck_keyboard.py"
)
keyboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keyboard)


def test_checkpoint_cli_preserves_order_and_single_model_alias():
    args = keyboard.parse_args(["--checkpoints", "b.pt", "a.pt", "b.pt", "--labels", "B", "A", "B again"])
    assert args.checkpoints == ["b.pt", "a.pt", "b.pt"]
    assert args.labels == ["B", "A", "B again"]
    assert keyboard.parse_args(["--checkpoint", "single.pt"]).checkpoints == ["single.pt"]
    with pytest.raises(SystemExit):
        keyboard.parse_args(["--checkpoints", "a.pt", "b.pt", "--labels", "only one"])


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_live_switching_restores_normalization_and_records_fixed_trials(device, tmp_path):
    """The real runner handles model 10, matching resets, focus loss and pre-reset test termination."""
    cfg, agent = keyboard.comparison_config(device, 42)
    sim_utils.create_new_stage()
    raw = gym.make(keyboard.TASK, cfg=cfg).unwrapped
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    try:
        with torch.inference_mode():
            runner = OnPolicyRunner(env, to_dict(agent), log_dir=None, device=device)
            policy = runner.alg.get_policy()
            for parameter in policy.mlp.parameters():
                parameter.zero_()
            data = torch.cat((torch.ones(32, 61, device=device), -torch.ones(32, 61, device=device)))
            policy.obs_normalizer.update(data)
            first = tmp_path / "first.pt"
            second = tmp_path / "second.pt"
            runner.save(str(first))
            last_linear = [module for module in policy.mlp.modules() if isinstance(module, torch.nn.Linear)][-1]
            last_linear.bias.fill_(0.05)
            policy.obs_normalizer.update(data + 2)
            probe = torch.zeros(1, 61, device=device)
            normalized_second = policy.obs_normalizer(probe).clone()
            runner.save(str(second))
            checkpoints = [first] * 12
            checkpoints[9] = second
            session = keyboard.KeyboardComparison(env, agent, checkpoints, [str(i) for i in range(1, 13)], 42, tmp_path)
            friction = read_group_parameter(raw.scene["robot"].actuators, "servos", "drive", "friction_scale").clone()

            def tap(code):
                session.on_key(code, True)
                session.on_key(code, False)

            for code in ("Digit1", "Digit0", "Enter"):
                tap(code)
            session.process_input()
            assert session.selected == 9
            torch.testing.assert_close(session.policy.obs_normalizer(probe), normalized_second)
            torch.testing.assert_close(session.policy(env.get_observations()), torch.full((1, 14), 0.05, device=device))
            torch.testing.assert_close(
                read_group_parameter(raw.scene["robot"].actuators, "servos", "drive", "friction_scale"), friction
            )
            tap("F1")
            session.on_key("KeyI", True)
            session.process_input()
            assert session.command == (0.1, 0.0, 0.0)
            session.on_focus_lost()
            session.process_input()
            assert session.command == (0.0, 0.0, 0.0)
            for code in ("Digit1", "Digit2", "Enter"):
                tap(code)
            session.process_input()
            assert session.selected == 11
            torch.testing.assert_close(session.policy(env.get_observations()), torch.zeros(1, 14, device=device))
            for code in ("Digit1", "Digit3", "Enter"):
                tap(code)
            session.process_input()
            assert session.selected == 11, "an invalid number must not change the active policy"
            tap("BracketRight")
            session.process_input()
            assert session.selected == 0
            tap("KeyP")
            session.process_input()
            for _ in range(7001):
                session.process_input()
                session.step()
                if session.benchmark is None:
                    break
            assert session.benchmark is None
            result = json.loads(session.last_results.read_text())
            assert result["status"] == "complete"
            assert [r["case"] for r in result["results"]] == [case for case, _ in keyboard.CASES]
            complete = [r for r in result["results"] if r["completed"]]
            assert complete, "at least the standing trial should survive the fixed horizon"
            assert all(r["stable_samples"] == 850 and r["survival_s"] == 20 for r in complete)
            with session.last_results.with_suffix(".csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            for case in complete:
                assert sum(row["case"] == case["case"] for row in rows) == 1000
            tap("KeyP")
            session.process_input()
            raw.scene["robot"].write_joint_velocity_to_sim_index(
                velocity=torch.full((1, 1), float("nan"), device=device), joint_ids=[0], env_ids=[0]
            )
            session.step()
            # The first failed trial advances to the next case, never into its autoreset trajectory.
            assert session.benchmark["case_index"] == 1
            tap("BracketRight")
            session.process_input()
            partial = json.loads(session.last_results.read_text())
            assert partial["status"] == "model_switch" and partial["model_number"] == 1
            assert not partial["results"][0]["completed"]
            assert partial["results"][0]["termination"]["nan_state"]
    finally:
        env.close()
        SimulationContext.clear_instance()
