# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate, validate, sequentially train and evaluate four independent MicroDuck continuations.

Run from the checkout with ``uv run python examples/demos/microduck_ablation.py``.
``--prepare-only`` stops after calibration and four short resume tests. A prepared batch can be
started with ``--continue-prepared --output PATH``. Any failed child stops the queue.
Each simulation/training child gets its own process and immutable experiment specification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "logs/rsl_rl/microduck_velocity_flat/2026-10-27_10-24-47_rerun_baseline/model_6000.pt"
VARIANTS = ("baseline", "velocity_weight", "velocity_std", "joint_tracking")
TITLES = ("基线", "提高速度奖励权重", "收紧速度奖励宽度", "增加关节跟踪惩罚")
SPEC_ENV = "ISAACLAB_MICRODUCK_EXPERIMENT_SPEC"


def write_json(path: Path, value) -> None:
    """Atomically update a batch status/artifact, retaining UTF-8 labels."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def digest(path: Path) -> str:
    """Hash a checkpoint without loading tensors into GPU memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_child(command: list[str], log: Path, env: dict | None = None) -> None:
    """Stream one child to its own log and stop the caller on any non-zero exit."""
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        metadata = {"pid": process.pid, "command": command, "started_at": datetime.now().astimezone().isoformat()}
        write_json(log.with_suffix(".process.json"), metadata)
        print(f"启动进程 {process.pid}；日志：{log}", flush=True)
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        metadata["returncode"] = code
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(log.with_suffix(".process.json"), metadata)
    if code:
        raise RuntimeError(f"子进程失败（退出码 {code}），队列已停止：{log}")


def verify_training(run_dir: Path, updates: int, checkpoint: Path, competitor: str | None = None) -> None:
    """Validate actual update count, restored optimizer LR, Chinese events and endpoint readability."""
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    state = torch.load(run_dir / f"model_{6000 + updates - 1}.pt", map_location="cpu", weights_only=True)
    if state["iter"] != 6000 + updates - 1 or "optimizer_state_dict" not in state:
        raise ValueError(f"Invalid continuation endpoint: {run_dir}")
    events = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    events.Reload()
    steps = [event.step for event in events.Scalars("优化/学习率")]
    if steps != list(range(6000, 6000 + updates)):
        raise ValueError(f"Expected exactly {updates} new updates, got {len(steps)} in {run_dir}")
    expected = torch.load(checkpoint, map_location="cpu", weights_only=True)["optimizer_state_dict"]["param_groups"][0][
        "lr"
    ]
    # The first PPO update may adapt LR; the entrypoint separately records the pre-update restored value.
    restored = json.loads((run_dir / "resume_state.json").read_text())
    if restored["learning_rate"] != expected or restored["common_step_counter"] != 144024:
        raise ValueError(f"Optimizer/curriculum restoration mismatch: {restored}")
    for name, weight in (("action_rate_l2", -1), ("head_pose_bias", -3)):
        if restored["reward_weights"][name] != weight * (0.5 if competitor == name else 1):
            raise ValueError("Reward curriculum did not resume at its mature stage.")
    commands = restored["commands"]
    if commands["base_velocity"]["rel_standing_envs"] != 0.25:
        raise ValueError("Standing curriculum did not resume at its mature stage.")
    if commands["head_pose"]["ranges"] != [[-1.1, 1.1], [-1.1, 1.1], [-1.4, 1.4], [-0.31, 0.31]]:
        raise ValueError("Head command curriculum did not resume at its mature stage.")
    for milestone in range(1000, updates + 1, 1000):
        if not (run_dir / f"model_{6000 + milestone - 1}.pt").is_file():
            raise ValueError(f"Missing relative checkpoint after {milestone} updates")


def training_command(args, output: Path, variant: str, index: int, smoke: bool) -> tuple[list[str], Path, dict]:
    """Build independent commands with only the declared reward variant and shared controls."""
    spec_path = output / f"{variant}.json"
    timestamp = f"{output.name}_{'smoke' if smoke else 'train'}_{index + 1:02d}"
    run_dir = ROOT / "logs/rsl_rl/microduck_velocity_flat" / f"{timestamp}_{variant}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {run_dir}")
    command = [
        sys.executable,
        "-m",
        "isaaclab",
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        "IsaacContrib-Velocity-Flat-MicroDuck",
        "--checkpoint",
        str(args.checkpoint),
        "--num_envs",
        str(16 if smoke else 4096),
        "--seed",
        str(getattr(args, "seed", 123)),
        "--max_iterations",
        str(2 if smoke else args.iterations),
        "--resume_env_steps",
        "144024",
        "--run_timestamp",
        timestamp,
        "--run_name",
        variant,
        "--device",
        args.device,
        "--log_language",
        "zh",
        "--save_additional_interval",
        "1000",
        "--external_callback",
        "isaaclab_tasks.contrib.microduck.experiments.register_experiment",
        "agent.save_interval=100000000",
    ]
    environment = dict(os.environ, **{SPEC_ENV: str(spec_path), "PYTHONUNBUFFERED": "1"})
    return command, run_dir, environment


def summarize(output: Path, evaluations: list[tuple[str, Path]]) -> None:
    """Retain trials and report equal-trial physical metrics, never rank by changed training rewards."""
    import numpy as np

    rows = []
    metrics = {
        "vx_mps": "实际前向速度米每秒",
        "planar_error_mps": "平面速度误差米每秒",
        "yaw_bias_radps": "偏航速度偏差弧度每秒",
        "yaw_rms_radps": "偏航速度误差均方根弧度每秒",
        "roll_pitch_sq_rad2ps2": "横滚俯仰角速度平方和",
        "vertical_rms_mps": "竖向速度均方根米每秒",
        "joint_rms_rad": "关节误差均方根弧度",
        "head_joint_rms_rad": "头部关节误差均方根弧度",
        "leg_joint_rms_rad": "腿部关节误差均方根弧度",
        "tracking_exceedance_fraction": "关节超容差比例",
    }
    for title, folder in evaluations:
        trials = json.loads((folder / "trials.json").read_text())
        for case in dict.fromkeys(t["case"] for t in trials):
            group = [t for t in trials if t["case"] == case]
            valid = [t for t in group if t["stable_samples"] > 0]
            row = {
                "模型": title,
                "指令": case,
                "目标前向速度": group[0]["target_vx_mps"],
                "目标偏航速度": group[0]["target_yaw_radps"],
                "试验数": len(group),
                "有效稳定试验数": len(valid),
                "完成率": np.mean([t["completed"] for t in group]).item(),
                "摔倒率": np.mean([t["fell_over"] for t in group]).item(),
                "非有限状态次数": sum(t["nan_failure"] for t in group),
                "平均存活秒": np.mean([t["survival_s"] for t in group]).item(),
            }
            row.update(
                {label: float(np.mean([t[key] for t in valid])) if valid else None for key, label in metrics.items()}
            )
            rows.append(row)
    write_json(output / "comparison.json", rows)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# MicroDuck 四组独立续训对照",
        "",
        "单训练种子探索；比较统一物理指标，训练总奖励不可直接横比。",
        "",
        "|模型|指令|完成率|平面误差 m/s|偏航误差 RMS rad/s|关节误差 RMS rad|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        numbers = [
            "—" if row[key] is None else f"{row[key]:.4f}"
            for key in (
                "完成率",
                "平面速度误差米每秒",
                "偏航速度误差均方根弧度每秒",
                "关节误差均方根弧度",
            )
        ]
        lines.append(f"|{row['模型']}|{row['指令']}|" + "|".join(numbers) + "|")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def run_batch(args) -> None:
    """Prepare once, then run four jobs serially and evaluate them under identical conditions."""
    output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.continue_prepared:
        manifest = json.loads((output / "batch.json").read_text())
        if manifest["status"] != "prepared" or manifest["checkpoint_sha256"] != digest(args.checkpoint):
            raise ValueError("Batch is not prepared, or the source checkpoint changed.")
        if manifest["iterations"] != args.iterations:
            raise ValueError("Prepared and requested iteration counts differ.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "preparing",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": digest(args.checkpoint),
            "iterations": args.iterations,
            "num_envs": 4096,
            "seed": 123,
            "resume_env_steps": 144024,
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "runs": [],
        }
        (output / "source.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True))
        # Include new/untracked implementation sources, which git diff alone cannot capture.
        sources = [
            Path(__file__),
            ROOT / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/experiments.py",
            ROOT / "source/isaaclab_rl/isaaclab_rl/utils/chinese_logging.py",
        ]
        for source in sources:
            destination = output / "source" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    def save_status(status):
        manifest["status"] = status
        manifest["controller_pid"] = os.getpid()
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(output / "batch.json", manifest)

    def worker(phase, checkpoint, folder, envs):
        folder.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(folder),
            "--device",
            args.device,
            "--rollout_envs",
            str(envs),
        ]
        if phase == "evaluate":
            command += ["--calibration", str(output / "calibration/calibration.json")]
        run_child(command, output / f"{folder.name}.log")

    try:
        if not args.continue_prepared:
            save_status("preflight")
            worker("preflight", args.checkpoint, output / "preflight", 128)
            save_status("calibrating")
            worker("calibrate", args.checkpoint, output / "calibration", 128)
            for index, variant in enumerate(VARIANTS):
                write_json(
                    output / f"{variant}.json",
                    {
                        "variant": variant,
                        "calibration": str(output / "calibration/calibration.json"),
                        "checkpoint_sha256": manifest["checkpoint_sha256"],
                    },
                )
                save_status(f"smoke_{variant}")
                command, run_dir, environment = training_command(args, output, variant, index, smoke=True)
                run_child(command, output / f"smoke_{variant}.log", environment)
                verify_training(run_dir, 2, args.checkpoint)
            save_status("prepared")
            if args.prepare_only:
                print(f"准备完成；正式训练尚未启动：{output}", flush=True)
                return
        else:
            save_status("training_preflight")
            worker("preflight", args.checkpoint, output / "training_preflight", 128)
        # Archive final implementation as well as the preparation snapshot (including new source files).
        paths = subprocess.check_output(
            ["git", "ls-files", "--modified", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
        ).splitlines()
        manifest["source_sha256"] = {}
        for relative in paths:
            source = ROOT / relative
            if source.is_file() and source.suffix in (".py", ".pyi", ".rst", ".md"):
                destination = output / "training_source" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
                manifest["source_sha256"][relative] = digest(source)
        manifest["calibration_sha256"] = digest(output / "calibration/calibration.json")
        manifest["spec_sha256"] = {variant: digest(output / f"{variant}.json") for variant in VARIANTS}

        def verify_inputs():
            inputs = {ROOT / relative: checksum for relative, checksum in manifest["source_sha256"].items()}
            inputs[args.checkpoint] = manifest["checkpoint_sha256"]
            inputs[output / "calibration/calibration.json"] = manifest["calibration_sha256"]
            inputs.update(
                {output / f"{variant}.json": checksum for variant, checksum in manifest["spec_sha256"].items()}
            )
            if any(not path.is_file() or digest(path) != checksum for path, checksum in inputs.items()):
                raise ValueError(
                    "Experiment source/configuration changed during the queue; refusing to mix conditions."
                )

        for index, variant in enumerate(VARIANTS):
            verify_inputs()
            save_status(f"training_{variant}")
            command, run_dir, environment = training_command(args, output, variant, index, smoke=False)
            record = {"variant": variant, "title": TITLES[index], "command": command, "log_dir": str(run_dir)}
            manifest["runs"].append(record)
            save_status(f"training_{variant}")
            run_child(command, output / f"train_{variant}.log", environment)
            verify_training(run_dir, args.iterations, args.checkpoint)
            record["status"] = "complete"
        save_status("evaluating")
        evaluations = []
        checkpoints = [("原6000轮模型", args.checkpoint)] + [
            (r["title"], Path(r["log_dir"]) / f"model_{6000 + args.iterations - 1}.pt") for r in manifest["runs"]
        ]
        for index, (title, checkpoint) in enumerate(checkpoints):
            verify_inputs()
            folder = output / f"evaluation_{index}"
            worker("evaluate", checkpoint, folder, 16)
            evaluations.append((title, folder))
        summarize(output, evaluations)
        save_status("complete")
    except BaseException as error:
        manifest["failed_phase"] = manifest["status"]
        manifest["error"] = str(error)
        save_status("failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "preflight", "calibrate", "evaluate"), default="run")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/experiments" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-prepared", action="store_true")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--rollout_envs", type=int, default=128)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--capture_traces", action="store_true", help="Persist pre-reset research trajectories")
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args()
    if args.iterations < 1 or args.rollout_envs < 1 or not 0 <= args.warmup < args.duration:
        parser.error("Require positive iterations/environments and 0 <= warmup < duration")
    args.device = args.device or "cuda:0"
    if args.phase == "run":
        import fcntl

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_batch(args)
    else:
        from isaaclab_tasks.contrib.microduck.experiments import check_baseline, rollout_measurements

        if args.phase == "preflight":
            check_baseline(args.checkpoint, args.output)
        else:
            calibration = json.loads(args.calibration.read_text()) if args.calibration else None
            rollout_measurements(args.checkpoint, args.output, args, calibration)


if __name__ == "__main__":
    main()
