# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record discovery and isolation contracts; no simulator or model inference is required."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.experiment_board.records import collect_records


def _base() -> dict:
    return {
        "id": "dashboard:test-records",
        "generatedAt": "2026-10-29T00:00:00+08:00",
        "queries": {
            key: {"rows": [], "source": {"files": []}}
            for key in ("experiments", "models", "metrics", "reports", "images")
        },
    }


def _write(study: Path, name: str = "新实验") -> None:
    study.mkdir(parents=True)
    (study / "report.md").write_text("# 结果\n\n实际观察：转向改善。\n", encoding="utf-8")
    (study / "experiment.json").write_text(
        json.dumps(
            {
                "schema_version": "microduck.experiment/1.0",
                "study_id": study.name,
                "display_name": name,
                "created_at": "2026-10-29T12:00:00+08:00",
                "status": "completed",
                "summary": "转向改善，保留其他指标的限制。",
                "question": "奖励形状是否影响转向？",
                "reports": [{"path": "report.md", "role": "实验报告"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_refresh_reads_added_changed_and_removed_records(tmp_path):
    """A reader re-scan must see new studies and edits without stale rows or duplicates."""
    base = _base()
    assert collect_records(tmp_path, base)["queries"]["experiments"]["rows"] == []
    study = tmp_path / "logs/experiments/new-study"
    _write(study)
    first = collect_records(tmp_path, base)
    assert [row["title"] for row in first["queries"]["experiments"]["rows"]] == ["新实验"]
    assert first["queries"]["experiments"]["rows"][0]["available"] is False
    assert first["queries"]["reports"]["rows"][0]["text"] == "# 结果\n\n实际观察：转向改善。\n"
    (study / "report.md").write_text("# 修正后的结论\n证据不足。\n", encoding="utf-8")
    second = collect_records(tmp_path, base)
    assert len(second["queries"]["experiments"]["rows"]) == 1
    assert second["queries"]["reports"]["rows"][0]["text"] == "# 修正后的结论\n证据不足。\n"
    (study / "experiment.json").unlink()
    assert collect_records(tmp_path, base)["queries"]["experiments"]["rows"] == []
    assert base["queries"]["experiments"]["rows"] == []


def test_invalid_records_are_visible_without_hiding_valid_records(tmp_path):
    """One broken/unsupported manifest must not suppress independently valid records."""
    for name in ("valid", "broken", "unsupported", "duplicate"):
        _write(tmp_path / "logs/experiments" / name)
    root = tmp_path / "logs/experiments"
    (root / "broken/experiment.json").write_text("{invalid json", encoding="utf-8")
    path = root / "unsupported/experiment.json"
    record = json.loads(path.read_text())
    record["schema_version"] = "microduck.experiment/99.0"
    path.write_text(json.dumps(record))
    path = root / "duplicate/experiment.json"
    record = json.loads(path.read_text())
    record["study_id"] = "valid"
    path.write_text(json.dumps(record))
    snapshot = collect_records(tmp_path, _base())
    rows = snapshot["queries"]["experiments"]["rows"]
    assert len(rows) == 4
    assert len({row["id"] for row in rows}) == 4
    assert sum(row["status"] == "记录错误" for row in rows) == 3
    assert len(snapshot["recordErrors"]) == 3
    assert len(snapshot["queries"]["reports"]["rows"]) == 1


def _add_board(study: Path) -> None:
    (study / "model.pt").write_bytes(b"test artifact: the reader never deserializes a policy")
    checkpoint_sha = hashlib.sha256((study / "model.pt").read_bytes()).hexdigest()
    protocol = {
        "id": "fixed-16x20-v1",
        "metric_version": "microduck-physical/1.0",
        "eval_seed": 42,
        "trials_per_case": 16,
        "duration_s": 20,
        "warmup_s": 3,
        "frame": "root_link_body",
        "backend": "newton-mjwarp",
    }
    (study / "protocol.json").write_text(json.dumps(protocol))
    protocol_sha = hashlib.sha256((study / "protocol.json").read_bytes()).hexdigest()
    board = {
        "schema_version": "microduck.board/1.0",
        "protocol": {**protocol, "sha256": protocol_sha},
        "protocol_file": {"path": "protocol.json"},
        "models": [
            {
                "key": "rational-s123",
                "name": "角速度有理函数",
                "variant": "w-rational",
                "seed": 123,
                "checkpoint": {"path": "model.pt", "sha256": checkpoint_sha},
                "start": 6000,
                "end": 9999,
                "updates": 4000,
                "trainEnvs": 4096,
            }
        ],
        "metrics": [
            {
                "model_key": "rational-s123",
                "case": "forward_0.1",
                "targetVx": 0.1,
                "targetYaw": 0,
                "vx": 0.026,
                "yaw": 0.001,
                "planarError": 0.08,
                "yawRms": 0.05,
                "angularRms": 0.2,
                "jointRms": None,
                "completion": 1,
                "trials": 16,
                "validTrials": 16,
            }
        ],
    }
    (study / "board_data.json").write_text(json.dumps(board))
    path = study / "experiment.json"
    record = json.loads(path.read_text())
    record.update({"board_data": {"path": "board_data.json"}, "quality_status": "passed", "primary_seed": 123})
    path.write_text(json.dumps(record))


def test_ai_authored_metrics_keep_model_identity_protocol_and_numbers(tmp_path):
    """Comparable rows retain AI-provided physical values and checked model/protocol identity."""
    study = tmp_path / "logs/experiments/shape-study"
    _write(study)
    _add_board(study)
    snapshot = collect_records(tmp_path, _base())
    assert snapshot["recordErrors"] == []
    model = snapshot["queries"]["models"]["rows"][0]
    metric = snapshot["queries"]["metrics"]["rows"][0]
    assert (model["start"], model["end"], model["updates"], model["seed"]) == (6000, 9999, 4000, 123)
    assert metric["vx"] == 0.026
    assert metric["jointRms"] is None
    assert metric["evaluationId"] == model["id"]
    assert metric["protocol"] == model["protocol"]
    assert snapshot["queries"]["experiments"]["rows"][0]["available"] is True
    # Replacing a referenced model invalidates its cached hash and removes its comparison rows.
    (study / "model.pt").write_bytes(b"changed model")
    changed = collect_records(tmp_path, _base())
    assert changed["queries"]["metrics"]["rows"] == []
    assert changed["queries"]["experiments"]["rows"][0]["status"] == "记录错误"


@pytest.mark.parametrize("damage", ["traversal", "nonfinite", "protocol_mismatch"])
def test_untrusted_record_references_and_values_cannot_enter_comparisons(tmp_path, damage):
    """Out-of-root paths, non-finite values and mismatched protocol identity fail closed."""
    study = tmp_path / "logs/experiments/invalid-study"
    _write(study)
    _add_board(study)
    if damage == "traversal":
        path = study / "experiment.json"
        record = json.loads(path.read_text())
        record["reports"] = [{"root": "repository", "path": "../secret.md"}]
        path.write_text(json.dumps(record))
    else:
        path = study / "board_data.json"
        board = json.loads(path.read_text())
        if damage == "nonfinite":
            board["metrics"][0]["vx"] = float("nan")
        else:
            board["protocol"]["warmup_s"] = 0
        path.write_text(json.dumps(board))
    snapshot = collect_records(tmp_path, _base())
    assert len(snapshot["recordErrors"]) == 1
    assert snapshot["queries"]["metrics"]["rows"] == []
    assert snapshot["queries"]["reports"]["rows"] == []
