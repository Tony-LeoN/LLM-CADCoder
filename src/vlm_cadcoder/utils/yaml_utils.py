from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_config(path: str | Path) -> dict[str, Any]:
    """Read YAML if PyYAML is installed; otherwise accept JSON-compatible files."""
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Reading {target} requires PyYAML on the server. "
            "Install the optional yaml extra or provide a JSON config."
        ) from exc

    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {target}")
    return data

