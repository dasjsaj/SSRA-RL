"""Backfill complexity columns for completed continuous-control baselines.

This script does not alter rewards or environment metrics. It only restores the
model-complexity fields required by the TC integrity audit for MADDPG, MATD3,
and MASAC runs that were produced before those fields were emitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ServiceComputing.algorithms.service_continuous_baselines import ServiceContinuousMARLTrainer

CONTINUOUS_ALGOS = {"maddpg", "matd3", "masac"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_path(run_dir: Path) -> Path | None:
    for name in ["checkpoint_best_stochastic.pt", "best_stochastic.pt", "checkpoint_latest.pt"]:
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    return None


def _algo_from_run_dir(root: Path, run_dir: Path) -> str:
    rel = run_dir.relative_to(root)
    return rel.parts[2].lower()


def backfill_run(root: Path, run_dir: Path) -> bool:
    algo = _algo_from_run_dir(root, run_dir)
    if algo not in CONTINUOUS_ALGOS:
        return False
    eval_curve = run_dir / "eval_curve.csv"
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    if not eval_curve.exists() or not config_path.exists() or not summary_path.exists():
        return False

    fields, rows = _read_csv(eval_curve)
    if not rows:
        return False
    needs_csv = any(
        field not in fields
        for field in [
            "parameter_count",
            "inference_time_ms_per_decision",
            "stochastic_parameter_count",
            "stochastic_inference_time_ms_per_decision",
        ]
    )
    summary = _read_json(summary_path)
    summary_eval = summary.get("last_stochastic_eval", {})
    needs_summary = not {
        "parameter_count",
        "inference_time_ms_per_decision",
    }.issubset(summary_eval.keys())
    if not needs_csv and not needs_summary:
        return False

    config = _read_json(config_path)
    trainer = ServiceContinuousMARLTrainer(config, run_dir, algo)
    checkpoint = _checkpoint_path(run_dir)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)
    parameter_count = float(trainer._parameter_count())
    inference_ms = float(trainer._inference_time_ms_per_decision())

    added_fields = [
        "parameter_count",
        "inference_time_ms_per_decision",
        "stochastic_parameter_count",
        "stochastic_inference_time_ms_per_decision",
    ]
    for field in added_fields:
        if field not in fields:
            fields.append(field)
    for row in rows:
        row["parameter_count"] = str(parameter_count)
        row["inference_time_ms_per_decision"] = str(inference_ms)
        row["stochastic_parameter_count"] = str(parameter_count)
        row["stochastic_inference_time_ms_per_decision"] = str(inference_ms)
    _write_csv(eval_curve, fields, rows)

    for key in ["last_stochastic_eval", "best_stochastic_eval"]:
        block = summary.setdefault(key, {})
        block["parameter_count"] = parameter_count
        block["inference_time_ms_per_decision"] = inference_ms
    _write_json(summary_path, summary)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/service_tc")
    args = parser.parse_args()
    root = Path(args.root)
    updated = 0
    for summary_path in sorted(root.rglob("summary.json")):
        if backfill_run(root, summary_path.parent):
            updated += 1
    print(f"Backfilled {updated} continuous-control runs under {root}")


if __name__ == "__main__":
    main()
