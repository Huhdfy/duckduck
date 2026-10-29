# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the frozen MicroDuck H1/H2 study serially, with at most eight formal continuations.

Usage: ``uv run python examples/demos/microduck_reward_research.py --output logs/experiments/NAME``.
``--prepare-only`` validates and collects the existing models; ``--continue-prepared`` starts
the remaining queue from that immutable preparation. A failed queue is never resumed implicitly.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from microduck_ablation import (
    CHECKPOINT,
    ROOT,
    SPEC_ENV,
    digest,
    run_child,
    training_command,
    verify_training,
    write_json,
)

from isaaclab_tasks.contrib.microduck.experiments import config_differences
from isaaclab_tasks.contrib.microduck.reward_research import (
    FORWARD,
    RULES,
    TURNS,
    analyze_evaluation,
    bootstrap_difference,
    h1_gate,
    paired_metric,
    screen_candidate,
    write_report,
)

OLD_BATCH = ROOT / "logs/experiments/microduck_four_way_2026-10-27_v2"


def audit_existing(batch: Path, checkpoint: Path, output: Path) -> dict:
    """Reject undeclared saved-config differences and incompatible continuation provenance."""
    import torch
    import yaml

    old = json.loads((batch / "batch.json").read_text())
    calibration = batch / "calibration/calibration.json"
    if digest(calibration) != old["calibration_sha256"]:
        raise ValueError("Existing joint diagnostic calibration changed.")
    (output / "tracking_calibration.json").write_bytes(calibration.read_bytes())
    if (
        old["status"] != "complete"
        or old["num_envs"] != 4096
        or old["seed"] != 123
        or old["iterations"] != 4000
        or old["checkpoint_sha256"] != digest(checkpoint)
    ):
        raise ValueError("Existing experiment conditions do not match the approved study.")
    selected = {r["variant"]: r for r in old["runs"] if r["variant"] in ("baseline", "velocity_std")}
    if set(selected) != {"baseline", "velocity_std"}:
        raise ValueError("Both completed reference models are required.")
    changes = []
    for kind in ("env", "agent"):
        configs = [
            yaml.load((Path(selected[v]["log_dir"]) / "params" / f"{kind}.yaml").read_text(), Loader=yaml.BaseLoader)
            for v in ("baseline", "velocity_std")
        ]
        for change in config_differences(*configs, kind):
            allowed = change["path"] in {
                "env.rewards.track_lin_vel.params.std",
                "env.rewards.track_ang_vel.params.std",
                "env.log_dir",
                "env.sim.log_dir",
                "agent.run_name",
            }
            changes.append(dict(change, allowed=allowed))
    models = {}
    for variant, record in selected.items():
        folder = Path(record["log_dir"])
        verify_training(folder, 4000, checkpoint)
        state = torch.load(folder / "model_9999.pt", map_location="cpu", weights_only=True)
        if state["iter"] != 9999:
            raise ValueError("Reference endpoint is not iter=9999.")
        models[variant] = {
            "checkpoint": str(folder / "model_9999.pt"),
            "sha256": digest(folder / "model_9999.pt"),
            "spec": {"variant": variant},
        }
    original_source = {}
    intentional_changes = {
        "examples/demos/microduck_ablation.py",
        "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/experiments.py",
        "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/mdp/rewards.py",
        "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/mdp/diagnostics.py",
        "source/isaaclab_tasks/test/test_microduck_velocity.py",
    }
    for relative, checksum in old["source_sha256"].items():
        if Path(relative).suffix not in (".py", ".pyi"):
            continue
        archive = batch / "training_source" / relative
        current = ROOT / relative
        archive_ok = archive.is_file() and digest(archive) == checksum
        current_ok = current.is_file() and (digest(current) == checksum or relative in intentional_changes)
        original_source[relative] = {
            "archive_matches": archive_ok,
            "current_matches": digest(current) == checksum,
            "declared_research_change": relative in intentional_changes,
        }
        if not archive_ok or not current_ok:
            raise ValueError(f"Unexpected source provenance difference: {relative}")
    result = {"models": models, "saved_config_differences": changes, "source_audit": original_source}
    write_json(output / "existing_audit.json", result)
    if any(not item["allowed"] for item in changes):
        raise ValueError("Undeclared difference between existing baseline and dual-std configurations.")
    return models


def freeze_inputs(output: Path, checkpoints: list[Path]) -> dict:
    """Archive all runtime Python sources and exact model/config inputs once."""
    paths = [p for base in (ROOT / "source", ROOT / "examples/demos") for p in base.rglob("*.py")]
    paths += [p for p in (ROOT / "source").rglob("*.pyi")]
    paths += checkpoints + [
        output / "rules.json",
        output / "tracking_calibration.json",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    checksums = {}
    for path in paths:
        checksums[str(path)] = digest(path)
        if path.suffix in (".py", ".pyi"):
            target = output / "source_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    write_json(output / "input_hashes.json", checksums)
    write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "packages": {
                distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            },
        },
    )
    (output / "source.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True))
    return checksums


def verify_inputs(checksums: dict) -> None:
    """Fail closed if sources, checkpoints or frozen screening rules changed mid-queue."""
    changed = [
        name for name, checksum in checksums.items() if not Path(name).is_file() or digest(Path(name)) != checksum
    ]
    if changed:
        raise ValueError(f"Frozen study inputs changed: {changed}")


def run_study(args) -> None:
    """Collect evidence, train isolated ablations, gate interventions, and verify one candidate."""
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    if args.continue_prepared:
        manifest = json.loads((output / "study.json").read_text())
        if manifest["status"] != "prepared" or manifest["checkpoint_sha256"] != digest(checkpoint):
            raise ValueError("Only an unchanged, successfully prepared study may continue.")
        if json.loads((output / "rules.json").read_text()) != RULES:
            raise ValueError("Prepared screening rules differ from current implementation.")
        checksums = json.loads((output / "input_hashes.json").read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "rules.json", RULES)
        manifest = {
            "status": "preparing",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "models": {},
            "runs": [],
            "decisions": {},
            "budget_used": 0,
            "budget_limit": 8,
        }
        checksums = {}

    def status(phase):
        manifest["status"] = phase
        manifest["controller_pid"] = os.getpid()
        manifest["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        write_json(output / "study.json", manifest)
        print(f"研究阶段：{phase}；正式训练预算 {manifest['budget_used']}/8", flush=True)

    def worker(phase, model, folder):
        verify_inputs(checksums)
        folder.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(ROOT / "examples/demos/microduck_ablation.py"),
            "--phase",
            phase,
            "--checkpoint",
            str(model),
            "--output",
            str(folder),
            "--device",
            args.device,
            "--rollout_envs",
            "16",
            "--capture_traces",
        ]
        if phase == "evaluate":
            command += ["--calibration", str(output / "tracking_calibration.json")]
        run_child(command, output / f"{folder.name}.log")

    models = {}

    def evaluate(name, record):
        status(f"evaluating_{name}")
        folder = output / f"evaluation_{name}"
        worker("evaluate", Path(record["checkpoint"]), folder)
        analysis = analyze_evaluation(folder, record["spec"])
        analysis["folder"] = str(folder)
        models[name] = analysis
        record["evaluation"] = str(folder)
        write_report(output, models, manifest["decisions"])

    def train(name, spec, seed=123, smoke=False):
        verify_inputs(checksums)
        if not smoke and manifest["budget_used"] >= 8:
            raise RuntimeError("Formal continuation budget exhausted.")
        spec_path = output / f"{name}.json"
        write_json(
            spec_path,
            dict(
                spec,
                checkpoint_sha256=manifest["checkpoint_sha256"],
                calibration=str(output / "tracking_calibration.json"),
            ),
        )
        job_args = SimpleNamespace(checkpoint=checkpoint, device=args.device, iterations=4000, seed=seed)
        command, folder, environment = training_command(job_args, output, name, len(manifest["runs"]), smoke)
        # The filename selects an immutable spec; its variant may differ for multiseed jobs.
        assert environment[SPEC_ENV] == str(spec_path)
        job = {
            "name": name,
            "spec": spec,
            "seed": seed,
            "smoke": smoke,
            "command": command,
            "log_dir": str(folder),
            "status": "running",
            "spec_sha256": digest(spec_path),
        }
        manifest["runs"].append(job)
        if not smoke:
            manifest["budget_used"] += 1
        status(f"{'smoke' if smoke else 'training'}_{name}")
        run_child(command, output / f"{'smoke' if smoke else 'train'}_{name}.log", environment)
        if digest(spec_path) != job["spec_sha256"]:
            raise ValueError("Training specification changed while its child was running.")
        verify_training(folder, 2 if smoke else 4000, checkpoint, spec.get("competitor"))
        if not smoke:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            events = EventAccumulator(str(folder), size_guidance={"scalars": 0})
            events.Reload()
            early = {
                tag: [{"step": item.step, "value": item.value} for item in events.Scalars(tag) if item.step < 6500]
                for tag in events.Tags()["scalars"]
                if tag.startswith(("优化/", "损失/", "训练/"))
            }
            write_json(output / f"early_training_{name}.json", early)
        verify_inputs(checksums)
        job["status"] = "complete"
        record = {"checkpoint": str(folder / f"model_{6001 if smoke else 9999}.pt"), "spec": spec, "seed": seed}
        record["sha256"] = digest(Path(record["checkpoint"]))
        if not smoke:
            manifest["models"][name] = record
            checksums[record["checkpoint"]] = record["sha256"]
            write_json(output / "input_hashes.json", checksums)
        return record

    try:
        if not args.continue_prepared:
            status("auditing_existing")
            manifest["models"] = audit_existing(args.existing_batch.resolve(), checkpoint, output)
            checksums = freeze_inputs(
                output, [checkpoint] + [Path(r["checkpoint"]) for r in manifest["models"].values()]
            )
            status("baseline_preflight")
            worker("preflight", checkpoint, output / "preflight")
            for name, record in list(manifest["models"].items()):
                evaluate(name, record)
            manifest["decisions"]["offline_h2"] = {name: data["h2"] for name, data in models.items()}
            for variant in ("v_std", "w_std", "w_rational"):
                train(f"check_{variant}", {"variant": variant}, smoke=True)
            status("prepared")
            if args.prepare_only:
                write_report(output, models, manifest["decisions"])
                return
        else:
            verify_inputs(checksums)
            for name, record in manifest["models"].items():
                analysis = json.loads((Path(record["evaluation"]) / "analysis.json").read_text())
                analysis["folder"] = record["evaluation"]
                models[name] = analysis
        for variant in ("v_std", "w_std"):
            record = train(variant, {"variant": variant})
            evaluate(variant, record)
        factorial = {}
        for cases, key in ((FORWARD, "planar_error_mps"), (FORWARD, "yaw_rms_radps"), (TURNS, "angular_rms_radps")):
            values = {
                name: paired_metric(models[name], cases, key) for name in ("baseline", "v_std", "w_std", "velocity_std")
            }
            factorial[key] = bootstrap_difference(
                values["velocity_std"] - values["v_std"], values["w_std"] - values["baseline"]
            )
        manifest["decisions"]["std_interaction"] = factorial
        gate = h1_gate(models["baseline"], models["w_std"])
        manifest["decisions"]["h1"] = gate
        if gate["eligible"]:
            evaluate("w_rational", train("w_rational", {"variant": "w_rational"}))
        # A single consistent competitor across eligible within-model evidence is required.
        h2_by_model = {
            name: data["h2"] for name, data in models.items() if name in ("baseline", "velocity_std", "v_std", "w_std")
        }
        supported = [value for value in h2_by_model.values() if value["eligible"]]
        competitors = {value["competitor"] for value in supported}
        conflicting = any(
            not value["eligible"] and any("difference" in row for row in value["evidence"])
            for value in h2_by_model.values()
        )
        competitor = next(iter(competitors)) if len(competitors) == 1 and not conflicting else None
        manifest["decisions"]["h2"] = {
            "eligible": competitor is not None,
            "competitor": competitor,
            "by_model": h2_by_model,
        }
        if competitor is not None:
            spec = {"variant": "competitor_half", "competitor": competitor}
            train("check_competitor", spec, smoke=True)
            evaluate("competitor_half", train("competitor_half", spec))
        screens = {}
        controls = {"w_rational": "w_std", "competitor_half": "baseline", "v_std": "baseline", "w_std": "baseline"}
        for variant in RULES["priority"]:
            if variant in models:
                screens[variant] = screen_candidate(models[variant], models[controls[variant]], variant)
        manifest["decisions"]["screening"] = screens
        winner = next((v for v in RULES["priority"] if screens.get(v, {}).get("eligible")), None)
        manifest["decisions"]["multiseed_candidate"] = winner
        seed_results = []
        if winner is not None:
            control = controls[winner]
            seed_results.append({"seed": 123, "screen": screens[winner]})
            for seed in (456, 789):
                names = []
                for variant in (control, winner):
                    name = f"{variant}_seed{seed}"
                    spec = copy.deepcopy(manifest["models"][variant]["spec"])
                    evaluate(name, train(name, spec, seed))
                    names.append(name)
                seed_results.append(
                    {"seed": seed, "screen": screen_candidate(models[names[1]], models[names[0]], winner)}
                )
        manifest["decisions"]["multiseed_results"] = seed_results
        write_report(output, models, manifest["decisions"])
        status("complete")
    except BaseException as error:
        manifest["failed_phase"] = manifest["status"]
        manifest["error"] = f"{type(error).__name__}: {error}"
        status("failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--existing-batch", type=Path, default=OLD_BATCH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-prepared", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.continue_prepared:
        parser.error("Preparation and continuation are mutually exclusive.")
    import fcntl

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/experiments/microduck_reward_research.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_study(args)


if __name__ == "__main__":
    main()
