"""Audit ServiceComputing paper experiment completion status.

This script does not infer success from intent. It inspects concrete artifacts:
run directories, summaries, best stochastic checkpoints, and best-checkpoint
evaluation rows when available.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ServiceComputing.scripts.paper_experiment_utils import write_rows


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_eval_index(root: Path) -> set[tuple[str, str, str, str, str]]:
    path = root / "paper_best_checkpoint_eval.csv"
    if not path.exists():
        return set()
    rows = csv.DictReader(path.open("r", encoding="utf-8", newline=""))
    return {
        (
            str(row.get("difficulty", "")),
            str(row.get("scale", "")),
            str(row.get("algo", "")),
            str(row.get("seed", "")),
            str(row.get("run_name", "")),
        )
        for row in rows
        if row.get("status") == "completed"
    }


def audit(root: Path) -> list[dict]:
    best_eval_done = _best_eval_index(root)
    rows: list[dict] = []
    for config_path in sorted(root.rglob("config.json")):
        run_dir = config_path.parent
        try:
            rel = run_dir.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 5:
            continue
        difficulty, scale, algo, seed_part, run_name = parts[:5]
        seed = seed_part.replace("seed_", "")
        summary_path = run_dir / "summary.json"
        summary = _read_json(summary_path)
        checkpoint = run_dir / "checkpoints" / "checkpoint_best_stochastic.pt"
        alt_checkpoint = run_dir / "checkpoints" / "best_stochastic.pt"
        eval_curve = run_dir / "eval_curve.csv"
        train_curve = run_dir / "train_curve.csv"
        status = str(summary.get("status", "missing_summary" if not summary_path.exists() else "unknown"))
        best_return = summary.get("best_stochastic_eval_return")
        rows.append(
            {
                "difficulty": difficulty,
                "scale": scale,
                "algo": algo,
                "seed": seed,
                "run_name": run_name,
                "status": status,
                "has_summary": summary_path.exists(),
                "has_eval_curve": eval_curve.exists() and eval_curve.stat().st_size > 0,
                "has_train_curve": train_curve.exists() and train_curve.stat().st_size > 0,
                "has_best_stochastic_checkpoint": checkpoint.exists() or alt_checkpoint.exists() or algo in {"random", "greedy"},
                "has_best_checkpoint_eval": (difficulty, scale, algo, seed, run_name) in best_eval_done,
                "best_stochastic_eval_return": best_return,
                "total_env_steps": summary.get("total_env_steps"),
                "reason": summary.get("reason", ""),
                "run_dir": str(run_dir),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/service_paper")
    args = parser.parse_args()
    root = Path(args.root)
    rows = audit(root)
    out = root / "paper_run_artifact_audit.csv"
    write_rows(out, rows)
    incomplete = [
        row
        for row in rows
        if row["status"] != "completed"
        or not row["has_eval_curve"]
        or not row["has_best_stochastic_checkpoint"]
        or (root / "paper_best_checkpoint_eval.csv").exists()
        and not row["has_best_checkpoint_eval"]
    ]
    print(f"Audited {len(rows)} runs under {root}")
    print(f"Incomplete or unverified rows: {len(incomplete)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
