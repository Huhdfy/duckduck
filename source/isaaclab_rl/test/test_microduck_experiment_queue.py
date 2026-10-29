# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The experiment queue stops on a real child-process failure and preserves its failed phase."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_failed_child_stops_the_serial_queue(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("microduck_ablation", root / "examples/demos/microduck_ablation.py")
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint fixture, never loaded because training fails")
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "calibration").mkdir()
    (batch / "calibration/calibration.json").write_text("{}")
    for variant in queue.VARIANTS:
        (batch / f"{variant}.json").write_text("{}")
    queue.write_json(
        batch / "batch.json",
        {
            "status": "prepared",
            "checkpoint_sha256": queue.digest(checkpoint),
            "iterations": 4000,
            "runs": [],
        },
    )
    attempts = tmp_path / "attempts.txt"

    def failing_command(*args, **kwargs):
        code = f"from pathlib import Path; Path({str(attempts)!r}).open('a').write('attempt\\n'); raise SystemExit(3)"
        return [sys.executable, "-c", code], tmp_path / "unused", None

    monkeypatch.setattr(queue, "training_command", failing_command)
    run_child = queue.run_child
    # Configuration validation is a separate contract; exercise real subprocess failure in training.
    monkeypatch.setattr(
        queue, "run_child", lambda command, *args: None if "--phase" in command else run_child(command, *args)
    )
    args = SimpleNamespace(output=batch, checkpoint=checkpoint, continue_prepared=True, iterations=4000, device="cpu")
    with pytest.raises(RuntimeError, match="退出码 3"):
        queue.run_batch(args)
    assert attempts.read_text() == "attempt\n"
    manifest = json.loads((batch / "batch.json").read_text())
    assert manifest["status"] == "failed" and manifest["failed_phase"] == "training_baseline"
    assert len(manifest["runs"]) == 1
