# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve the existing Data app and refresh its AI-authored local experiment records.

Run from the repository root with ``uv run python -m tools.experiment_board.server``.
Only reviewed record loading and app packaging run on refresh, never training or analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .records import collect_records, read_json

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP = ROOT / "logs/apps/microduck-center-demo/app"
DEFAULT_NODE = Path("/usr/lib/chatgpt/resources/cua_node/bin/node")
DEFAULT_BUILDER = (
    Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics/1.1.0-exp-baseline/scripts/data-app.mjs"
)


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, ensure_ascii=False, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _fingerprint(snapshot: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in snapshot.items() if key != "generatedAt"},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


class RecordBoard:
    """Load records and atomically publish the same compiled app under a process lock."""

    def __init__(self, repository: Path, app: Path, node: Path, builder: Path):
        self.repository = repository.resolve()
        self.app = app.resolve()
        self.node = node.resolve()
        self.builder = builder.resolve()
        self.lock = threading.RLock()
        base_path = self.app.parent / "base_snapshot.json"
        if not base_path.exists():
            raise ValueError(f"Retained legacy snapshot is missing: {base_path}")
        self.base = read_json(base_path)
        self.snapshot = read_json(self.app / "src/data.json")
        if self.base["id"] != self.snapshot["id"]:
            raise ValueError("Legacy snapshot and app identity differ")
        self.fingerprint = _fingerprint(self.snapshot)

    def refresh(self, force: bool = False) -> dict:
        """Re-read all record manifests and publish a verified build if needed."""
        with self.lock:
            snapshot = collect_records(self.repository, self.base)
            fingerprint = _fingerprint(snapshot)
            changed = fingerprint != self.fingerprint
            if changed or force:
                self._publish(snapshot)
                self.snapshot = snapshot
                self.fingerprint = fingerprint
            return {
                "ok": True,
                "changed": changed,
                "generatedAt": self.snapshot["generatedAt"],
                "experiments": len(self.snapshot["queries"]["experiments"]["rows"]),
                "models": len(self.snapshot["queries"]["models"]["rows"]),
                "errors": self.snapshot.get("recordErrors", []),
            }

    def _publish(self, snapshot: dict) -> None:
        # Build in a sibling staging project. Failed builds never replace the live app/data.
        with tempfile.TemporaryDirectory(prefix=".record-build-", dir=self.app.parent) as temporary:
            stage = Path(temporary) / "app"
            shutil.copytree(
                self.app,
                stage,
                ignore=shutil.ignore_patterns(
                    "dist",
                    "node_modules",
                    ".data-app-offline",
                    "__pycache__",
                ),
            )
            _atomic_json(stage / "src/data.json", snapshot)
            result = subprocess.run(
                [str(self.node), str(self.builder), "build", "--project-dir", str(stage), "--separate-data"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(f"Data app build failed: {(result.stderr or result.stdout)[-2000:]}")
            distribution = self.app / "dist"
            # Keep previous content-addressed sidecars for an already-open page finishing its load.
            if distribution.exists():
                for old in sorted(distribution.glob("snapshot.*.json"), key=lambda path: path.stat().st_mtime_ns)[-2:]:
                    destination = stage / "dist" / old.name
                    if not destination.exists():
                        shutil.copy2(old, destination)
            previous = Path(temporary) / "previous-dist"
            if distribution.exists():
                distribution.replace(previous)
            try:
                (stage / "dist").replace(distribution)
            except BaseException:
                if previous.exists():
                    previous.replace(distribution)
                raise
            _atomic_json(self.app / "src/data.json", snapshot)


def create_server(board: RecordBoard, port: int) -> ThreadingHTTPServer:
    """Create a loopback-only server with same-origin refresh and bounded static routes."""

    class Handler(BaseHTTPRequestHandler):
        def _allowed(self) -> bool:
            host = f"127.0.0.1:{self.server.server_port}"
            origin = self.headers.get("Origin")
            return (
                self.headers.get("Host") == host
                and (origin is None or origin == f"http://{host}")
                and self.headers.get("Sec-Fetch-Site") != "cross-site"
            )

        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, value: dict) -> None:
            self._reply(
                status,
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def do_GET(self):
            if not self._allowed():
                return self._json(403, {"error": "Forbidden"})
            route = unquote(urlsplit(self.path).path)
            try:
                if route == "/api/records":
                    # This is the current published snapshot, not a hidden refresh or analysis.
                    with board.lock:
                        return self._json(200, board.snapshot)
                if route == "/api/health":
                    return self._json(200, {"ok": True, "service": "microduck-record-board"})
                document = route in ("/", "/index.html") or route.startswith("/_data/")
                if document:
                    board.refresh()
                relative = "index.html" if document else route.lstrip("/")
                if (
                    not relative
                    or ".." in Path(relative).parts
                    or "\\" in relative
                    or any(part.startswith(".") for part in Path(relative).parts)
                ):
                    return self._json(400, {"error": "Invalid asset path"})
                with board.lock:
                    distribution = (board.app / "dist").resolve()
                    asset = (distribution / relative).resolve()
                    if not asset.is_relative_to(distribution) or not asset.is_file():
                        return self._json(404, {"error": "Not found"})
                    mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                    self._reply(200, asset.read_bytes(), mime)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                logger.exception("Unable to read experiment records")
                self._json(503, {"error": str(error)})

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            if not self._allowed():
                return self._json(403, {"error": "Forbidden"})
            if self.path != "/api/refresh":
                return self._json(404, {"error": "Not found"})
            # No request-controlled commands, paths, uploads, or source writes.
            if self.headers.get("Content-Length", "0") != "0" or self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                return self._json(400, {"error": "Refresh does not accept a request body"})
            try:
                self._json(200, board.refresh(force=True))
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                logger.exception("Record refresh failed; retaining the previous published data")
                self._json(503, {"error": str(error)})

        def log_message(self, message, *args):
            logger.info(message, *args)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    """Start the local record board without importing any simulator or RL backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=42843)
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--node", type=Path, default=DEFAULT_NODE)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    board = RecordBoard(ROOT, args.app, args.node, args.builder)
    board.refresh()
    with create_server(board, args.port) as server:
        logger.info("MicroDuck record board: http://127.0.0.1:%s/", server.server_port)
        server.serve_forever()


if __name__ == "__main__":
    main()
