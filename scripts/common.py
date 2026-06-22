"""Shared script helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_dir(config: dict, method: str, seed: int) -> Path:
    base = Path(config.get("output_dir", "artifacts/service_runs"))
    name = config.get("run_name", method)
    path = base / method / f"seed_{seed}" / name
    path.mkdir(parents=True, exist_ok=True)
    return path

