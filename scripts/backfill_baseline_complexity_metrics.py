"""Backfill parameter-count and inference-time columns for completed baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ServiceComputing.algorithms.service_continuous_baselines import ServiceContinuousMARLTrainer
from ServiceComputing.algorithms.service_policy_baselines import ServiceOnPolicyMARLTrainer
from ServiceComputing.algorithms.service_value_baselines import ServiceValueMARLTrainer

POLICY_ALGOS = {"coma"}
VALUE_ALGOS = {"madqn", "qmix", "wqmix", "qtran"}
CONTINUOUS_ALGOS = {"maddpg", "matd3", "masac"}
SUPPORTED = POLICY_ALGOS | VALUE_ALGOS | CONTINUOUS_ALGOS


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_dir(root: Path, row: dict[str, str]) -> Path:
    return root / row["difficulty"] / row["scale"] / row["algo"] / f"seed_{row['seed']}" / row["run_name"]


def _checkpoint_path(run_dir: Path) -> Path | None:
    for name in ["checkpoint_best_stochastic.pt", "best_stochastic.pt", "checkpoint_latest.pt"]:
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    return None


def _trainer(algo: str, config: dict[str, Any], run_dir: Path):
    if algo in CONTINUOUS_ALGOS:
        return ServiceContinuousMARLTrainer(config, run_dir, algo=algo)
    if algo in POLICY_ALGOS:
        return ServiceOnPolicyMARLTrainer(config, run_dir, algo=algo)
    if algo in VALUE_ALGOS:
        return ServiceValueMARLTrainer(config, run_dir, algo=algo)
    raise ValueError(f"unsupported algorithm for complexity backfill: {algo}")


def backfill_run(run_dir: Path, algo: str) -> bool:
    if algo not in SUPPORTED:
        return False
    eval_curve = run_dir / "eval_curve.csv"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    if not eval_curve.exists() or not summary_path.exists() or not config_path.exists():
        return False
    fields, rows = _read_csv(eval_curve)
    if not rows:
        return False
    required = [
        "parameter_count",
        "inference_time_ms_per_decision",
        "stochastic_parameter_count",
        "stochastic_inference_time_ms_per_decision",
    ]
    summary = _read_json(summary_path)
    needs_csv = any(field not in fields for field in required)
    needs_summary = not {
        "parameter_count",
        "inference_time_ms_per_decision",
    }.issubset(summary.get("last_stochastic_eval", {}).keys())
    if not needs_csv and not needs_summary:
        return False

    trainer = _trainer(algo, _read_json(config_path), run_dir)
    checkpoint = _checkpoint_path(run_dir)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)
    parameter_count = float(trainer._parameter_count())
    inference_ms = float(trainer._inference_time_ms_per_decision())

    for field in required:
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
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/service_tc")
    args = parser.parse_args()
    root = Path(args.root)
    manifest_path = root / "tc_experiment_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    updated = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            algo = row.get("algo", "").lower()
            if backfill_run(_run_dir(root, row), algo):
                updated += 1
    print(f"Backfilled {updated} baseline runs under {root}")


if __name__ == "__main__":
    main()
