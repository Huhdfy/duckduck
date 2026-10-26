# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive a trained MicroDuck policy with the Newton viewer keyboard.

Run from the repository root::

    uv run python examples/demos/microduck_keyboard.py --checkpoint latest

Focus the viewer. Hold I/K for forward/backward, U/O for left/right side steps,
and J/L for left/right turns. Release the keys to stand. B stops while held,
R resets the robot, and C toggles the follow camera. WASD remains available for
camera control. Commands stay inside the flat task's training velocity ranges.
"""

import argparse
import contextlib
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import to_dict

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", default="latest", help="Latest compatible checkpoint or a checkpoint file path.")
parser.add_argument("--max_steps", type=int, default=-1, help="Control steps; -1 runs until the viewer closes.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab_rl.entrypoints.common import resolve_checkpoint_selector
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
AGENT = "rsl_rl_cfg_entry_point"


def main() -> None:
    """Load a flat-walking checkpoint and replace random commands with keyboard input."""
    # argparse 已消费脚本参数；不要让 Hydra 再读取 --checkpoint、--max_steps 等选项。
    env_cfg, agent_cfg = resolve_task_config(TASK, AGENT, play_mode=True, overrides=())
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1000000
    env_cfg.curriculum = None
    # 键盘是唯一的速度来源；禁用站立掩码与特殊采样，避免随机指令覆盖人工输入。
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_forward_envs = 0.0
    command_cfg.rel_turn_in_place_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    # 保留输入维数，但让头部和身体指令保持默认姿态。
    for pose_cfg in (env_cfg.commands.head_pose, env_cfg.commands.body_pose):
        pose_cfg.ranges = tuple((0.0, 0.0) for _ in pose_cfg.ranges)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.checkpoint == "latest":
        checkpoint = resolve_checkpoint_selector(
            str(Path("logs/rsl_rl") / agent_cfg.experiment_name),
            "latest",
            library="rsl_rl",
            task=TASK,
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": AGENT},
        )
    else:
        checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve(strict=True))
    print(f"Loading model checkpoint from: {checkpoint}")

    with launch_simulation(env_cfg, args_cli):
        from isaaclab.envs import ManagerBasedRLEnv

        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        try:
            sim = env.unwrapped.sim
            viewers = [v for v in sim.visualizers if v.cfg.backend.visualizer_type == "newton_gl"]
            if not viewers:
                raise ValueError("Keyboard control requires --viz newton_gl.")
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=env.unwrapped.device)
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            env.reset()
            keys_down: set[str] = set()
            follow_camera = True
            step_count = 0
            print("Focus viewer: I/K forward/back; U/O side step; J/L turn; B stop; R reset; C camera.")
            with torch.inference_mode(), contextlib.suppress(KeyboardInterrupt):
                while sim.is_running() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
                    start = time.perf_counter()
                    held = {
                        key
                        for key in ("i", "k", "u", "o", "j", "l", "b", "r", "c")
                        if any(viewer.is_key_down(key) for viewer in viewers)
                    }
                    pressed = held - keys_down
                    keys_down = held
                    if "r" in pressed:
                        env.reset()
                        policy.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device))
                    if "c" in pressed:
                        follow_camera = not follow_camera
                    # 速度是机体坐标系中的任务指令，不是直接控制电机；松键后回到零速度。
                    command = env.unwrapped.command_manager.get_command("base_velocity")
                    command.zero_()
                    if "b" not in held:
                        command[:, 0] = 0.3 * (("i" in held) - ("k" in held))
                        command[:, 1] = 0.15 * (("u" in held) - ("o" in held))
                        command[:, 2] = 0.6 * (("j" in held) - ("l" in held))
                    # 重算观测，确保本次推理立即看到键盘指令；不额外推进观测历史。
                    obs = TensorDict(env.unwrapped.observation_manager.compute(), batch_size=[env.num_envs])
                    _, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)
                    if follow_camera:
                        pos = env.unwrapped.scene["robot"].data.root_pos_w.torch[0].tolist()
                        sim.set_camera_view(
                            eye=(pos[0] + 0.7, pos[1] + 0.7, pos[2] + 0.35),
                            target=(pos[0], pos[1], pos[2]),
                        )
                    step_count += 1
                    # 以真实时间交互，避免一秒按键在仿真中变成数十秒。
                    time.sleep(max(0.0, env.unwrapped.step_dt - (time.perf_counter() - start)))
        finally:
            env.close()


if __name__ == "__main__":
    main()
