"""Attribute performance gains within the same SLG trainer configuration family."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev


EVAL_FIELDS = [
    "eval_return",
    "completion_ratio",
    "mean_service_delay",
    "deadline_violation_rate",
    "mean_energy_cost",
    "mean_queue_length",
    "auv_local_ratio_mean",
    "auv_usv_ratio_mean",
    "auv_uav_ratio_mean",
    "auv_tx_power_mean",
]


def parse_named_path(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_run(run_dir: Path) -> dict[str, float]:
    rows = read_csv(run_dir / "eval_curve.csv")
    if not rows:
        raise ValueError(f"No evaluation rows in {run_dir}")
    tail_n = max(1, int(round(len(rows) * 0.2)))
    tail = rows[-tail_n:]
    summary: dict[str, float] = {"eval_points": float(len(rows)), "tail_points": float(tail_n)}
    for field in EVAL_FIELDS:
        values = [float(r[field]) for r in tail if r.get(field, "") not in ("", None)]
        if values:
            summary[f"tail_{field}"] = fmean(values)
            summary[f"tail_{field}_std"] = pstdev(values)
    return summary


def attribute_semantic_source(summaries: dict[str, dict[str, float]]) -> dict[str, float | str]:
    control = summaries["Control-NoSemantic"]
    state = summaries["Semantic-StateOnly"]
    full = summaries["SLG-SAGE-Full"]
    c_return = control["tail_eval_return"]
    s_return = state["tail_eval_return"]
    f_return = full["tail_eval_return"]
    c_comp = control["tail_completion_ratio"]
    s_comp = state["tail_completion_ratio"]
    f_comp = full["tail_completion_ratio"]
    eps = 1e-6
    if f_return > s_return + eps and s_return > c_return + eps and f_comp >= s_comp >= c_comp:
        classification = "semantic_state_and_loss_supported"
    elif f_return > c_return + eps and f_return > s_return + eps and f_comp >= c_comp:
        classification = "semantic_loss_supported_without_state_gain"
    elif f_return <= c_return + eps or f_comp < c_comp - eps:
        classification = "semantic_not_supported_or_harmful"
    else:
        classification = "inconclusive"
    return {
        "classification": classification,
        "full_gain_over_control_return": f_return - c_return,
        "full_gain_over_stateonly_return": f_return - s_return,
        "state_gain_over_control_return": s_return - c_return,
        "full_gain_over_control_completion": f_comp - c_comp,
        "full_gain_over_stateonly_completion": f_comp - s_comp,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    named_runs = [parse_named_path(v) for v in args.run]
    summaries: dict[str, dict[str, float]] = {}
    rows: list[dict] = []
    for label, path in named_runs:
        metrics = summarize_run(path)
        summaries[label] = metrics
        rows.append({"method": label, "run_dir": str(path), **metrics})
    decision = attribute_semantic_source(summaries)
    out_dir = Path(args.output_dir)
    write_csv(rows, out_dir / "semantic_contribution_summary.csv")
    (out_dir / "semantic_contribution_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lines = [
        "# Semantic Contribution Source Ablation",
        "",
        "All compared methods use the same SLG trainer, route-softmax action head, low exploration noise, environment, reward, and evaluation seeds. Only semantic access/loss changes.",
        "",
        "| method | tail return | completion | delay | deadline violation | queue | AUV tx power |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row.get('tail_eval_return', float('nan')):.6f} | "
            f"{row.get('tail_completion_ratio', float('nan')):.6f} | "
            f"{row.get('tail_mean_service_delay', float('nan')):.6f} | "
            f"{row.get('tail_deadline_violation_rate', float('nan')):.6f} | "
            f"{row.get('tail_mean_queue_length', float('nan')):.6f} | "
            f"{row.get('tail_auv_tx_power_mean', float('nan')):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Attribution Decision",
            "",
            f"- Classification: `{decision['classification']}`.",
            f"- Full minus Control return: `{decision['full_gain_over_control_return']:.6f}`.",
            f"- Full minus StateOnly return: `{decision['full_gain_over_stateonly_return']:.6f}`.",
            f"- Full minus Control completion: `{decision['full_gain_over_control_completion']:.6f}`.",
            "",
            "This is an attribution gate, not a multi-seed paper claim.",
        ]
    )
    (out_dir / "semantic_contribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), **decision}, indent=2))


if __name__ == "__main__":
    main()
