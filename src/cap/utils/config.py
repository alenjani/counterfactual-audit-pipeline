"""Config loading with YAML + dotted-path overrides."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        node: Any = self.raw
        for part in key.split("."):
            node = node[part]
        return node

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default

    def to_dict(self) -> dict[str, Any]:
        return self.raw


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load YAML config and apply dotted-path overrides."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(data, dotted, value)
    return Config(raw=data)


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
