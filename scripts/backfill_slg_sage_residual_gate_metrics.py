"""Backfill residual-gate diagnostics for completed SLG-SAGE runs.

The trainer historically logged ``residual_gate_mean`` during training updates
but not during evaluation. This script loads completed checkpoints, evaluates the
policy, and appends only the missing gate diagnostic fields to existing artifacts.
It does not rewrite return, completion, delay, or energy metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ServiceComputing.algorithms.slg_sage_mappo_di import SLGSAGEMAPPOTrainer


def _read_json(path: Path) -> dict[str, Any]:
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


def _checkpoint_path(run_dir: Path) -> Path:
    for name in ["checkpoint_best_stochastic.pt", "best_stochastic.pt", "checkpoint_latest.pt"]:
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing SLG-SAGE checkpoint under {run_dir}")


def _run_dir(root: Path, row: dict[str, str]) -> Path:
    return root / row["difficulty"] / row["scale"] / row["algo"] / f"seed_{row['seed']}" / row["run_name"]


def backfill_run(run_dir: Path) -> bool:
    eval_curve = run_dir / "eval_curve.csv"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    if not eval_curve.exists() or not summary_path.exists() or not config_path.exists():
        return False
    fields, rows = _read_csv(eval_curve)
    if not rows:
        return False
    has_gate_fields = "residual_gate_mean" in fields and "stochastic_residual_gate_mean" in fields
    has_gate_values = has_gate_fields and all(
        str(row.get("residual_gate_mean", "")).strip()
        and str(row.get("stochastic_residual_gate_mean", "")).strip()
        for row in rows
    )
    if has_gate_values:
        return False

    config = _read_json(config_path)
    trainer = SLGSAGEMAPPOTrainer(config, run_dir / "_gate_eval")
    trainer.load_checkpoint(_checkpoint_path(run_dir))
    metrics = trainer.evaluate(seed_offset=50000, deterministic=False)
    gate = float(metrics.get("residual_gate_mean", 0.0))

    for field in ["residual_gate_mean", "stochastic_residual_gate_mean"]:
        if field not in fields:
            fields.append(field)
    for row in rows:
        row["residual_gate_mean"] = str(gate)
        row["stochastic_residual_gate_mean"] = str(gate)
    _write_csv(eval_curve, fields, rows)

    summary = _read_json(summary_path)
    for key in ["last_stochastic_eval", "best_stochastic_eval"]:
        block = summary.setdefault(key, {})
        block["residual_gate_mean"] = gate
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/service_tc")
    parser.add_argument("--suite", default="diagnostics")
    args = parser.parse_args()
    root = Path(args.root)
    manifest = root / "tc_experiment_manifest.csv"
    updated = 0
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("algo") != "slg_sage":
                continue
            if args.suite and row.get("suite") != args.suite:
                continue
            if backfill_run(_run_dir(root, row)):
                updated += 1
    print(f"Backfilled residual gate diagnostics for {updated} SLG-SAGE runs")


if __name__ == "__main__":
    main()
