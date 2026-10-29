# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare any number of compatible MicroDuck checkpoints in one Newton keyboard window.

    uv run python examples/demos/microduck_keyboard.py --checkpoints MODEL_A.pt MODEL_B.pt

Type a model number and Enter to select it; [/] select previous/next. F1-F4 choose
0.1/0.2/0.3/0.4 m/s. Minus/Equal adjust linear speed; Semicolon/Quote adjust turn speed.
Hold I/K forward/back, U/O sideways, J/L turn. The sidebar also accepts exact speed values.
Release movement keys to stand. B stops, R resets, C toggles the follow camera.
P runs seven fixed 20-second cases on the current model and saves measurements.
WASD remains available for the camera. --checkpoint remains an alias.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from tensordict import TensorDict

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
AGENT = "rsl_rl_cfg_entry_point"
CASES = (
    ("standing", (0.0, 0.0, 0.0)),
    *((f"forward_{speed:g}", (speed, 0.0, 0.0)) for speed in (0.1, 0.2, 0.3, 0.4)),
    ("turn_negative", (0.0, 0.0, -0.5)),
    ("turn_positive", (0.0, 0.0, 0.5)),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ordered checkpoints while retaining the original single-checkpoint invocation."""
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", "--checkpoint", nargs="+", default=["latest"])
    parser.add_argument("--labels", nargs="+", help="Optional model names in checkpoint order")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--linear-speed", type=float, default=0.3, help="Initial forward/backward speed magnitude (m/s)"
    )
    parser.add_argument("--angular-speed", type=float, default=0.5, help="Initial turning speed magnitude (rad/s)")
    parser.add_argument("--max_steps", type=int, default=-1, help="Control steps; -1 runs until the viewer closes")
    parser.add_argument("--output", type=Path, default=Path("logs/keyboard/microduck"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args(argv)
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max_steps must be positive or -1")
    if args.labels is not None and len(args.labels) != len(args.checkpoints):
        parser.error("--labels must contain one name per checkpoint")
    if any(not math.isfinite(value) or value < 0 for value in (args.linear_speed, args.angular_speed)):
        parser.error("--linear-speed and --angular-speed must be finite and non-negative")
    return args


def comparison_config(device: str | None, seed: int):
    """Use one mature, fixed-heading, noise-free inference environment for every checkpoint."""
    from isaaclab_tasks.contrib.microduck.experiments import freeze_curriculum
    from isaaclab_tasks.utils import resolve_task_config

    cfg, agent = resolve_task_config(TASK, AGENT, play_mode=True, overrides=())
    freeze_curriculum(cfg, 144024)
    cfg.scene.num_envs = 1
    cfg.seed = agent.seed = seed
    cfg.episode_length_s = 1000000
    cfg.recorders.diagnostics.capture_samples = True
    cfg.events.reset_base.params["pose_range"].update(x=(0.0, 0.0), y=(0.0, 0.0), yaw=(0.0, 0.0))
    command = cfg.commands.base_velocity
    command.rel_standing_envs = command.rel_forward_envs = command.rel_turn_in_place_envs = 0.0
    command.heading_command = False
    command.ranges.lin_vel_x = command.ranges.lin_vel_y = command.ranges.ang_vel_z = (0.0, 0.0)
    for pose in (cfg.commands.head_pose, cfg.commands.body_pose):
        pose.ranges = tuple((0.0, 0.0) for _ in pose.ranges)
    if device is not None:
        cfg.sim.device = agent.device = device
    return cfg, agent


class KeyboardComparison:
    """Own policy switching, focused input and fixed-duration measurements in one live environment."""

    def __init__(
        self,
        env,
        agent_cfg,
        checkpoints: list[Path],
        labels: list[str],
        seed: int,
        output: Path,
        linear_speed: float = 0.3,
        angular_speed: float = 0.5,
    ):
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.utils import to_dict

        self.env = env
        self.raw = env.unwrapped
        self.checkpoints = checkpoints
        self.labels = labels
        self.seed = seed
        self.output = output
        self.runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=self.raw.device)
        expected = self.runner.alg.get_policy().state_dict()
        # Validate every file before the first live switch; keep only one policy in GPU memory.
        for path in checkpoints:
            actor = torch.load(path, map_location="cpu", weights_only=True)["actor_state_dict"]
            if actor.keys() != expected.keys() or any(actor[k].shape != expected[k].shape for k in expected):
                raise ValueError(f"检查点与当前 MicroDuck 策略结构不兼容：{path}")
            if not all(torch.isfinite(value).all() for value in actor.values()):
                raise ValueError(f"检查点包含非有限策略参数：{path}")
        self.selected = 0
        self.held: set[str] = set()
        self.blocked: set[str] = set()
        self.presses: deque[str] = deque()
        self.number = ""
        self.speed = linear_speed
        self.angular_speed = angular_speed
        self.follow_camera = True
        self.message = "Type model number + Enter; [/] previous/next"
        self.command = (0.0, 0.0, 0.0)
        self.actual = (0.0, 0.0, 0.0, 0.0)
        self.benchmark: dict | None = None
        self.last_results: Path | None = None
        self.last_termination: dict[str, bool] = {}
        self.switch(0)

    def on_key(self, code: str, down: bool) -> None:
        """Consume the visualizer's W3C keyboard events without repeating toggle actions."""
        if not down:
            self.held.discard(code)
            self.blocked.discard(code)
        elif code not in self.blocked and code not in self.held:
            self.held.add(code)
            self.presses.append(code)

    def on_focus_lost(self) -> None:
        """Treat all manual movement as released when the viewer or UI loses keyboard focus."""
        self.held.clear()
        self.blocked.clear()
        self.presses.clear()
        self.number = ""

    def reset(self, seed: int | None = None) -> None:
        """Restore matched reset randomization and clear recurrent and input state."""
        self.blocked.update(self.held)
        self.held.clear()
        self.raw.reset(seed=self.seed if seed is None else seed)
        self.raw.episode_length_buf.zero_()
        self.policy.reset(torch.ones(self.env.num_envs, dtype=torch.bool, device=self.raw.device))
        self.command = (0.0, 0.0, 0.0)
        self.raw.command_manager.get_command("base_velocity").zero_()
        self.actual = (0.0, 0.0, 0.0, 0.0)

    def switch(self, index: int) -> None:
        """Load the selected actor including its normalization buffers, then reset fairly."""
        self.cancel_benchmark("model_switch")
        self.runner.load(str(self.checkpoints[index]), load_cfg={"actor": True}, map_location=self.raw.device)
        self.policy = self.runner.get_inference_policy(device=self.raw.device)
        self.selected = index
        self.reset()
        self.message = f"Model {index + 1}/{len(self.checkpoints)}: {self.labels[index]}"
        print(
            f"已选择模型 {index + 1}/{len(self.checkpoints)}：{self.labels[index]}\n{self.checkpoints[index]}",
            flush=True,
        )

    def process_input(self) -> None:
        """Apply queued selection/toggle events and derive this control step's command."""
        while self.presses:
            code = self.presses.popleft()
            if code.startswith(("Digit", "Numpad")) and code[-1:].isdigit():
                self.number += code[-1]
            elif code == "Backspace":
                self.number = self.number[:-1]
            elif code in ("Enter", "NumpadEnter") and self.number:
                index = int(self.number) - 1 if len(self.number) <= len(str(len(self.checkpoints))) else -1
                self.number = ""
                if 0 <= index < len(self.checkpoints):
                    self.switch(index)
                else:
                    self.message = f"Model number must be 1..{len(self.checkpoints)}"
            elif code in ("BracketLeft", "BracketRight"):
                self.number = ""
                self.switch((self.selected + (1 if code == "BracketRight" else -1)) % len(self.checkpoints))
            elif code in ("F1", "F2", "F3", "F4"):
                self.speed = int(code[-1]) / 10
            elif code in ("Minus", "Equal"):
                self.speed = max(0.0, round(self.speed + (0.05 if code == "Equal" else -0.05), 10))
            elif code in ("Semicolon", "Quote"):
                self.angular_speed = max(0.0, round(self.angular_speed + (0.1 if code == "Quote" else -0.1), 10))
            elif code == "KeyC":
                self.follow_camera = not self.follow_camera
            elif code in ("KeyR", "KeyB"):
                self.cancel_benchmark("reset" if code == "KeyR" else "stopped")
                if code == "KeyR":
                    self.reset()
            elif code == "KeyP":
                if self.benchmark is None:
                    self.benchmark = {"case_index": 0, "step": 0, "rows": [], "samples": [], "results": []}
                    self.reset()
                    self.message = "Standard test: standing (1/7)"
                    print("开始当前模型标准测试：7 档，每档 20 秒，前 3 秒预热。P/B 可取消。", flush=True)
                else:
                    self.cancel_benchmark("cancelled")
        if self.benchmark is not None:
            self.command = CASES[self.benchmark["case_index"]][1]
        elif "KeyB" in self.held:
            self.command = (0.0, 0.0, 0.0)
        else:
            self.command = (
                self.speed * (("KeyI" in self.held) - ("KeyK" in self.held)),
                0.15 * (("KeyU" in self.held) - ("KeyO" in self.held)),
                self.angular_speed * (("KeyJ" in self.held) - ("KeyL" in self.held)),
            )

    def step(self) -> None:
        """Infer using the current keyboard command, recording terminal measurements before reset."""
        command = self.raw.command_manager.get_command("base_velocity")
        command[:] = command.new_tensor(self.command)
        obs = TensorDict(self.raw.observation_manager.compute(update_history=False), batch_size=[self.env.num_envs])
        _, _, dones, _ = self.env.step(self.policy(obs))
        self.policy.reset(dones)
        recorder = self.raw.recorder_manager._terms["diagnostics"]
        sample = dict(zip(recorder.sample_names, recorder.snapshot[0].tolist()))
        self.actual = (
            sample["link_vx_b_mps"],
            sample["link_vy_b_mps"],
            sample["omega_z_b_radps"],
            math.sqrt(sample["roll_pitch_sq_rad2ps2"]),
        )
        self.last_termination = {k.split("/", 1)[1]: bool(v) for k, v in sample.items() if k.startswith("termination/")}
        if self.benchmark is not None:
            self._record_benchmark(sample, bool(dones[0]))

    def _record_benchmark(self, sample: dict, done: bool) -> None:
        test = self.benchmark
        case, target = CASES[test["case_index"]]
        step = test["step"]
        row = {
            "case": case,
            "step": step,
            "time_s": (step + 1) * self.raw.step_dt,
            "target_vx_mps": target[0],
            "target_yaw_radps": target[2],
            "done": done,
            **sample,
        }
        test["rows"].append(row)
        if step >= round(3 / self.raw.step_dt):
            test["samples"].append(
                (
                    self.actual[0],
                    math.hypot(self.actual[0] - target[0], self.actual[1]),
                    (self.actual[2] - target[2]) ** 2,
                    self.actual[3] ** 2,
                )
            )
        test["step"] += 1
        if not done and test["step"] < round(20 / self.raw.step_dt):
            return
        count = len(test["samples"])
        finite = [values for values in test["samples"] if all(math.isfinite(v) for v in values)]
        means = [sum(v[i] for v in finite) / len(finite) for i in range(4)] if finite else [None] * 4
        test["results"].append(
            {
                "case": case,
                "completed": not done,
                "survival_s": test["step"] * self.raw.step_dt,
                "stable_samples": len(finite),
                "invalid_samples": count - len(finite),
                "vx_mps": means[0],
                "planar_error_mps": means[1],
                "yaw_rms_radps": math.sqrt(means[2]) if finite else None,
                "roll_pitch_rms_radps": math.sqrt(means[3]) if finite else None,
                "termination": dict(self.last_termination) if done else {},
            }
        )
        print(f"标准测试：{case}；{'提前终止' if done else '完成'}；有效样本 {len(finite)}", flush=True)
        test["case_index"] += 1
        if test["case_index"] == len(CASES):
            self._save_benchmark("complete")
            self.benchmark = None
            self.reset()
            self.message = "Standard test complete; results saved"
        else:
            test["step"] = 0
            test["samples"] = []
            self.reset(self.seed + test["case_index"])
            self.message = f"Standard test: {CASES[test['case_index']][0]} ({test['case_index'] + 1}/7)"

    def _save_benchmark(self, status: str) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S-%f")
        stem = self.output / f"{stamp}_model{self.selected + 1}"
        metadata = {
            "status": status,
            "model_number": self.selected + 1,
            "label": self.labels[self.selected],
            "checkpoint": str(self.checkpoints[self.selected]),
            "seed": self.seed,
            "duration_s": 20.0,
            "warmup_s": 3.0,
            "trials_per_case": 1,
            "results": self.benchmark["results"],
            "partial_case": CASES[self.benchmark["case_index"]][0]
            if self.benchmark["case_index"] < len(CASES)
            else None,
        }
        stem.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
        rows = self.benchmark["rows"]
        if rows:
            with stem.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.last_results = stem.with_suffix(".json")
        print(f"标准测试记录已保存（{status}）：{self.last_results}", flush=True)

    def cancel_benchmark(self, reason: str) -> None:
        """Preserve partial tests explicitly instead of silently mixing models or dropping failures."""
        if self.benchmark is not None:
            self._save_benchmark(reason)
            self.benchmark = None
            self.message = "Standard test cancelled; partial results saved"

    def draw_ui(self, imgui) -> None:
        """Expose model selection, live measurements and test progress in the viewer sidebar."""
        if not imgui.collapsing_header("MicroDuck comparison", imgui.TreeNodeFlags_.default_open):
            return
        imgui.text(f"Model {self.selected + 1}/{len(self.checkpoints)}")
        imgui.text_wrapped(self.labels[self.selected])
        imgui.text(f"Select number: {self.number or '-'} (Enter)")
        imgui.text("[ / ] previous / next; F1-F4 speed")
        imgui.text("- / = linear speed; ; / ' turn speed")
        imgui.text("I/K forward/back; J/L turn; U/O side")
        imgui.text("R reset; B stop; C camera; P test")
        imgui.separator()
        changed, value = imgui.input_float("Linear speed (m/s)", self.speed, step=0.05, step_fast=0.1, format="%.3f")
        if changed and math.isfinite(value):
            self.speed = max(0.0, value)
        changed, value = imgui.input_float(
            "Turn speed (rad/s)", self.angular_speed, step=0.1, step_fast=0.5, format="%.3f"
        )
        if changed and math.isfinite(value):
            self.angular_speed = max(0.0, value)
        imgui.text(f"Target vx/vy/wz: {self.command[0]:+.2f} / {self.command[1]:+.2f} / {self.command[2]:+.2f}")
        imgui.text(f"Actual vx/vy: {self.actual[0]:+.3f} / {self.actual[1]:+.3f} m/s")
        imgui.text(f"Actual yaw: {self.actual[2]:+.3f} rad/s; roll/pitch: {self.actual[3]:.3f}")
        if self.benchmark is not None:
            imgui.text(f"Test time: {self.benchmark['step'] * self.raw.step_dt:.1f} / 20 s")
        imgui.text_wrapped(self.message)


def main() -> None:
    args = parse_args()
    from isaaclab.app import launch_simulation
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_rl.entrypoints.common import resolve_checkpoint_selector
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    cfg, agent = comparison_config(args.device, args.seed)
    checkpoints = []
    for path in args.checkpoints:
        if path == "latest":
            path = resolve_checkpoint_selector(
                str(Path("logs/rsl_rl") / agent.experiment_name),
                "latest",
                library="rsl_rl",
                task=TASK,
                checkpoint_pattern=r"model_.*\.pt",
                metadata={"agent": AGENT},
            )
        checkpoints.append(Path(path).expanduser().resolve(strict=True))
    labels = args.labels or [f"{path.parent.name}/{path.name}" for path in checkpoints]
    for number, (label, path) in enumerate(zip(labels, checkpoints), 1):
        print(f"模型 {number}：{label}\n  {path}")
    with torch.inference_mode(), launch_simulation(cfg, args):
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg), clip_actions=agent.clip_actions)
        session = None
        with contextlib.ExitStack() as stack:
            stack.callback(env.close)
            try:
                sim = env.unwrapped.sim
                viewers = [v for v in sim.visualizers if v.cfg.backend.visualizer_type == "newton_gl"]
                if not viewers:
                    raise ValueError("Keyboard comparison requires --viz newton_gl")
                session = KeyboardComparison(
                    env, agent, checkpoints, labels, args.seed, args.output, args.linear_speed, args.angular_speed
                )
                for viewer in viewers:
                    viewer.register_ui_callback(session.draw_ui, position="side")
                    source = viewer.key_event_source
                    if source is None:
                        raise RuntimeError("Keyboard input is unavailable; launch from a graphical desktop session")
                    stack.enter_context(source.add_key_listener(session.on_key, session.on_focus_lost))
                print(
                    "点击窗口后操作：数字+回车选模型，[ / ] 切换，F1-F4 选速度，- / = 调线速度，; / ' 调角速度，"
                    "I/K 前后，J/L 转向，P 标准测试。",
                    flush=True,
                )
                step = 0
                with contextlib.suppress(KeyboardInterrupt):
                    while sim.is_running() and (args.max_steps < 0 or step < args.max_steps):
                        start = time.perf_counter()
                        session.process_input()
                        session.step()
                        if session.follow_camera:
                            pos = env.unwrapped.scene["robot"].data.root_pos_w.torch[0].tolist()
                            sim.set_camera_view(
                                eye=(pos[0] + 0.7, pos[1] + 0.7, pos[2] + 0.35), target=(pos[0], pos[1], pos[2])
                            )
                        step += 1
                        time.sleep(max(0.0, env.unwrapped.step_dt - (time.perf_counter() - start)))
            finally:
                if session is not None:
                    session.cancel_benchmark("window_closed")


if __name__ == "__main__":
    main()
