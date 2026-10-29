# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read AI-authored MicroDuck records without running analysis, plotting, or simulation."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

SCHEMA = "microduck.experiment/1.0"
BOARD_SCHEMA = "microduck.board/1.0"
STATUS_LABELS = {
    "planned": "待开始",
    "running": "进行中",
    "completed": "完成",
    "failed": "失败",
    "interrupted": "中断",
}
METRIC_FIELDS = ("vx", "yaw", "planarError", "yawRms", "angularRms", "jointRms")
PROTOCOL_FIELDS = ("id", "sha256", "metric_version", "eval_seed", "trials_per_case", "duration_s", "warmup_s")


def read_json(path: Path) -> dict:
    """Read a finite JSON object, rejecting nonstandard numeric literals."""

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite JSON number: {value}")
        return result

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


@lru_cache(maxsize=256)
def _digest(path: str, modified_ns: int, changed_ns: int, size: int) -> str:
    # File identity invalidates cached checkpoint hashes when an AI replaces an artifact.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_digest(path: Path) -> str:
    """Hash an artifact once per size/mtime/ctime revision rather than once per refresh."""
    stat = path.stat()
    return _digest(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def resolve_ref(reference: dict, study: Path, repository: Path) -> Path:
    """Resolve a record reference inside its explicitly selected local root."""
    if not isinstance(reference, dict):
        raise ValueError("File references must be objects")
    root = {"study": study, "repository": repository}.get(reference.get("root", "study"))
    if root is None:
        raise ValueError("Reference root must be study or repository")
    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Reference path must be relative and cannot contain '..'")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError(f"Missing or out-of-root file: {relative}")
    expected = reference.get("sha256")
    if expected is not None:
        if not isinstance(expected, str) or not re.fullmatch("[a-f0-9]{64}", expected):
            raise ValueError("Invalid SHA256")
        if file_digest(target) != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
    return target


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Identifiers must use ASCII letters, digits, '_' or '-'")
    return value


def _number(value, field: str, *, nullable: bool = False, nonnegative: bool = False):
    if nullable and value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def _report(record: dict, study: Path, repository: Path, eid: str) -> tuple[dict, list[dict]]:
    path = resolve_ref(record, study, repository)
    if path.suffix.lower() != ".md":
        raise ValueError("Reports must be Markdown files")
    text = path.read_text(encoding="utf-8")
    images = []
    for reference in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        # Only bundle local report images; remote links and code are never fetched/executed.
        target = (path.parent / reference).resolve()
        if not target.is_relative_to(repository) or not target.is_file():
            continue
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}.get(
            target.suffix.lower()
        )
        if mime is None or target.stat().st_size > 10 * 1024 * 1024:
            continue
        images.append(
            {
                "path": target.relative_to(repository).as_posix(),
                "data": f"data:{mime};base64," + base64.b64encode(target.read_bytes()).decode(),
            }
        )
    return {
        "id": f"{eid}:{record.get('role', path.stem)}",
        "experiment": eid,
        "role": record.get("role", "实验报告"),
        "filename": path.relative_to(repository).as_posix(),
        "text": text,
    }, images


def _board_rows(board: dict, study: Path, repository: Path, eid: str) -> tuple[list[dict], list[dict]]:
    if board.get("schema_version") != BOARD_SCHEMA:
        raise ValueError(f"Expected {BOARD_SCHEMA}")
    protocol = board["protocol"]
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    for field in PROTOCOL_FIELDS:
        if field not in protocol:
            raise ValueError(f"Missing protocol.{field}")
    if not re.fullmatch("[a-f0-9]{64}", protocol["sha256"]):
        raise ValueError("Protocol must provide its full SHA256")
    protocol_path = resolve_ref(board["protocol_file"], study, repository)
    if file_digest(protocol_path) != protocol["sha256"]:
        raise ValueError("Protocol file SHA256 mismatch")
    protocol_record = read_json(protocol_path)
    for field in PROTOCOL_FIELDS:
        if field != "sha256" and protocol_record.get(field) != protocol[field]:
            raise ValueError(f"protocol.{field} differs from protocol file")
    for field in ("eval_seed", "trials_per_case", "duration_s", "warmup_s"):
        _number(protocol[field], f"protocol.{field}", nonnegative=True)
    if protocol["trials_per_case"] < 1 or protocol["duration_s"] <= protocol["warmup_s"]:
        raise ValueError("Protocol trial count or duration is invalid")
    compatibility = f"{protocol['id']}:{protocol['sha256']}:{protocol['metric_version']}"
    models, metrics, model_map = [], [], {}
    for model in board.get("models", []):
        key = _identifier(model["key"])
        if key in model_map:
            raise ValueError(f"Duplicate model key: {key}")
        checkpoint = resolve_ref(model["checkpoint"], study, repository)
        checksum = model["checkpoint"].get("sha256")
        if not checksum:
            raise ValueError("Model checkpoint requires SHA256")
        for field in ("seed", "end", "updates", "trainEnvs"):
            _number(model[field], field, nonnegative=True)
        _number(model.get("start"), "start", nullable=True, nonnegative=True)
        item = {
            "id": f"{eid}:{key}",
            "experiment": eid,
            "name": _text(model["name"], "model.name"),
            "variant": _identifier(model["variant"]),
            "seed": model["seed"],
            "checkpoint": checkpoint.relative_to(repository).as_posix(),
            "checkpointHash": checksum,
            "start": model.get("start"),
            "end": model["end"],
            "updates": model["updates"],
            "trainEnvs": model["trainEnvs"],
            "evaluation": study.relative_to(repository).as_posix(),
            "protocol": compatibility,
            "protocolLabel": protocol["id"],
            "evalSeed": protocol["eval_seed"],
            "trialsPerCase": protocol["trials_per_case"],
            "duration": protocol["duration_s"],
            "warmup": protocol["warmup_s"],
            "label": model["name"],
            "source": (study / "experiment.json").relative_to(repository).as_posix(),
        }
        models.append(item)
        model_map[key] = item
    seen = set()
    for row in board.get("metrics", []):
        model = model_map[row["model_key"]]
        case = _text(row["case"], "case")
        if (model["id"], case) in seen:
            raise ValueError("Duplicate model/case row")
        seen.add((model["id"], case))
        for field in ("targetVx", "targetYaw", "completion", "trials", "validTrials"):
            _number(row[field], field)
        if not 0 <= row["completion"] <= 1 or not 0 <= row["validTrials"] <= row["trials"]:
            raise ValueError("Invalid completion rate or trial counts")
        if row["trials"] != protocol["trials_per_case"]:
            raise ValueError("Trial count differs from the declared protocol")
        for field in METRIC_FIELDS:
            _number(row.get(field), field, nullable=True, nonnegative=field not in ("vx", "yaw"))
        metrics.append(
            {
                **{field: row.get(field) for field in METRIC_FIELDS},
                **{field: row[field] for field in ("targetVx", "targetYaw", "completion", "trials", "validTrials")},
                "case": case,
                "evaluationId": model["id"],
                "experiment": eid,
                "model": model["name"],
                "protocol": compatibility,
            }
        )
    return models, metrics


def collect_records(repository: Path, base_snapshot: dict) -> dict:
    """Merge current AI-authored records with the retained, reviewed legacy snapshot.

    Malformed records remain visible as errors; their partial metrics cannot enter comparisons.
    No statistical calculations, report generation, or policy execution occur here.
    """
    repository = repository.resolve()
    snapshot = copy.deepcopy(base_snapshot)
    queries = snapshot["queries"]
    errors = []
    known = {row["id"] for row in queries["experiments"]["rows"]}
    references = []
    for path in sorted((repository / "logs/experiments").glob("*/experiment.json")):
        study = path.parent.resolve()
        if not study.is_relative_to(repository / "logs/experiments"):
            errors.append({"file": str(path), "error": "Study directory escapes experiment root"})
            continue
        eid = "record-" + path.parent.name
        record = {}
        try:
            record = read_json(path)
            if record.get("schema_version") != SCHEMA:
                raise ValueError(f"Unsupported schema_version; expected {SCHEMA}")
            eid = _identifier(record["study_id"])
            if eid in known:
                raise ValueError(f"Duplicate study_id: {eid}")
            status = STATUS_LABELS[record["status"]]
            created = datetime.fromisoformat(record["created_at"])
            if created.tzinfo is None:
                raise ValueError("created_at requires a timezone")
            report_rows, image_rows = [], []
            for reference in record.get("reports", []):
                report_row, report_images = _report(reference, study, repository, eid)
                report_rows.append(report_row)
                image_rows.extend(report_images)
            models, metrics = [], []
            if record.get("board_data"):
                board_path = resolve_ref(record["board_data"], study, repository)
                models, metrics = _board_rows(read_json(board_path), study, repository, eid)
                references.append(board_path.relative_to(repository).as_posix())
            quality = record.get("quality_status", "pending")
            if quality not in ("pending", "passed", "failed", "incomplete"):
                raise ValueError("Invalid quality_status")
            # Unfinished or unverified data remain viewable as reports, never as full comparisons.
            comparable = record["status"] == "completed" and quality == "passed" and bool(metrics)
            if not comparable:
                models, metrics = [], []
            experiment = {
                "id": eid,
                "title": _text(record["display_name"], "display_name"),
                "date": created.date().isoformat(),
                "status": status,
                "type": record.get("type", "实验记录"),
                "runs": record.get("training_runs", 0),
                "modelCount": len(models),
                "primarySeed": record.get("primary_seed", models[0]["seed"] if models else 123),
                "verified": record.get("review_status") == "reviewed" and quality == "passed",
                "qualityStatus": quality,
                "folder": study.relative_to(repository).as_posix(),
                "manifest": path.relative_to(repository).as_posix(),
                "summary": _text(record["summary"], "summary"),
                "question": _text(record["question"], "question"),
                "available": comparable,
                "protocol": models[0]["protocol"] if models else None,
            }
            queries["experiments"]["rows"].append(experiment)
            queries["models"]["rows"].extend(models)
            queries["metrics"]["rows"].extend(metrics)
            queries["reports"]["rows"].extend(report_rows)
            queries["images"]["rows"].extend(image_rows)
            references.append(path.relative_to(repository).as_posix())
            known.add(eid)
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
            message = f"{type(error).__name__}: {error}"
            errors.append({"file": path.relative_to(repository).as_posix(), "error": message})
            queries["experiments"]["rows"].append(
                {
                    "id": "invalid-" + path.parent.name,
                    "title": record.get("display_name", path.parent.name),
                    "date": "—",
                    "status": "记录错误",
                    "type": "读取异常",
                    "runs": None,
                    "modelCount": None,
                    "verified": False,
                    "folder": study.relative_to(repository).as_posix(),
                    "manifest": path.relative_to(repository).as_posix(),
                    "summary": message,
                    "question": "记录未通过格式校验；请 AI 修正清单后再刷新。",
                    "available": False,
                    "protocol": None,
                }
            )
    # Keep image lookup deterministic when multiple reports refer to one figure.
    queries["images"]["rows"] = list({row["path"]: row for row in queries["images"]["rows"]}.values())
    for query in queries.values():
        query["source"]["files"] = sorted(set(query["source"].get("files", []) + references))
    snapshot["generatedAt"] = datetime.now().astimezone().isoformat()
    snapshot["buildStatus"] = "complete"
    snapshot["status"] = "本地实验记录 · 点击刷新读取"
    snapshot["recordErrors"] = errors
    snapshot["recordSource"] = "logs/experiments/*/experiment.json"
    return snapshot
