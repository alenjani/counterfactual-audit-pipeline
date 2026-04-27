"""Run manifest — per-run reproducibility metadata stored alongside outputs."""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:
    run_id: str
    config_path: str
    config_resolved: dict[str, Any]
    git_commit: str | None = None
    git_dirty: bool = False
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    seeds: dict[str, int] = field(default_factory=dict)
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=lambda: platform.platform())
    model_versions: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def create(cls, run_id: str, config_path: str, config_resolved: dict[str, Any]) -> "RunManifest":
        commit, dirty = _git_state()
        return cls(
            run_id=run_id,
            config_path=str(config_path),
            config_resolved=config_resolved,
            git_commit=commit,
            git_dirty=dirty,
        )

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def _git_state() -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False
