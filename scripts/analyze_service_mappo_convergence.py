"""Analyze MAPPO long-test convergence for service offloading runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np

from ServiceComputing.scripts.common import load_json, write_json
from ServiceComputing.scripts.evaluate_random_service_policy import rollout as random_rollout
from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import mean_metrics


def read_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, float]] = []
        for row in reader:
            rows.append({k: to_float(v) for k, v in row.items()})
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def values(rows: list[dict[str, float]], key: str) -> list[float]:
    return [float(r.get(key, math.nan)) for r in rows if math.isfinite(float(r.get(key, math.nan)))]


def evaluation_mode_rows(rows: list[dict[str, float]], policy_mode: str) -> list[dict[str, float]]:
    if policy_mode == "deterministic":
        return rows
    if policy_mode != "stochastic":
        raise ValueError(f"unsupported policy_mode: {policy_mode}")
    required = ["eval_return", "completion_ratio", "mean_service_delay", "deadline_violation_rate", "mean_queue_length"]
    mapped_rows = []
    for row in rows:
        mapped = dict(row)
        for key in required:
            stochastic_key = f"stochastic_{key}"
            if stochastic_key not in row or not math.isfinite(float(row[stochastic_key])):
                raise ValueError(f"missing logged stochastic evaluation metric: {stochastic_key}")
            mapped[key] = row[stochastic_key]
        mapped_rows.append(mapped)
    return mapped_rows


def auc(rows: list[dict[str, float]], key: str) -> float:
    xs = values(rows, "step")
    ys = values(rows, key)
    if len(xs) != len(ys) or len(xs) == 0:
        return math.nan
    if len(xs) == 1 or xs[-1] == xs[0]:
        return float(ys[-1])
    return float(np.trapz(np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)) / (xs[-1] - xs[0]))


def tail_stats(rows: list[dict[str, float]], key: str, frac: float = 0.2) -> tuple[float, float]:
    seq = values(rows, key)
    if not seq:
        return math.nan, math.nan
    n = max(1, int(math.ceil(len(seq) * frac)))
    tail = seq[-n:]
    return float(mean(tail)), float(pstdev(tail) if len(tail) > 1 else 0.0)


def n90(rows: list[dict[str, float]], key: str, higher_is_better: bool = True) -> float:
    seq = values(rows, key)
    steps = values(rows, "step")
    if len(seq) < 2 or len(seq) != len(steps):
        return math.nan
    start = seq[0]
    stable, _ = tail_stats(rows, key, 0.2)
    if not math.isfinite(stable):
        return math.nan
    if higher_is_better:
        target = start + 0.9 * (stable - start)
        for step, val in zip(steps, seq):
            if val >= target:
                return float(step)
    else:
        target = start - 0.9 * (start - stable)
        for step, val in zip(steps, seq):
            if val <= target:
                return float(step)
    return math.nan


def random_baseline(config: dict, episodes: int, seed_offset: int = 10000) -> dict[str, float]:
    env = make_service_env(config)
    base_seed = int(config.get("seed", 0)) + seed_offset
    rows = [random_rollout(env, base_seed + i) for i in range(episodes)]
    summary = mean_metrics(rows)
    summary["episode_return"] = float(np.mean([r["episode_return"] for r in rows]))
    return summary


def safe_ratio(std: float, avg: float) -> float:
    denom = abs(avg)
    if denom < 1e-8:
        return math.inf if std > 0 else 0.0
    return float(std / denom)


def convergence_decision(metrics: dict[str, float], random_metrics: dict[str, float], train_rows: list[dict[str, float]]) -> dict:
    final_return = metrics["last_20pct_eval_return_mean"]
    random_return = random_metrics.get("episode_return", math.nan)
    final_completion = metrics["last_20pct_completion_mean"]
    random_completion = random_metrics.get("completion_ratio", math.nan)
    final_delay = metrics["last_20pct_delay_mean"]
    random_delay = random_metrics.get("mean_service_delay", math.nan)
    entropy_seq = values(train_rows, "entropy_loss")
    first_30 = entropy_seq[: max(1, int(len(entropy_seq) * 0.3))]
    entropy_collapsed = bool(first_30 and min(first_30) < 0.25)
    value_loss_tail, _ = tail_stats(train_rows, "value_loss", 0.2)
    value_loss_first, _ = tail_stats(train_rows[: max(1, int(len(train_rows) * 0.2))], "value_loss", 1.0)
    checks = {
        "return_above_random_30pct": bool(math.isfinite(final_return) and math.isfinite(random_return) and final_return >= random_return * 1.3),
        "completion_above_random_0p10": bool(math.isfinite(final_completion) and math.isfinite(random_completion) and final_completion >= random_completion + 0.10),
        "delay_below_random_20pct": bool(math.isfinite(final_delay) and math.isfinite(random_delay) and final_delay <= random_delay * 0.80),
        "last_20pct_eval_return_stable": bool(metrics["last_20pct_eval_return_cv"] < 0.15),
        "value_loss_not_exploded": bool(metrics["value_loss_max"] < 10.0 and value_loss_tail <= max(10.0, value_loss_first * 3.0 + 1e-6)),
        "entropy_not_early_collapsed": not entropy_collapsed,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "notes": {
            "deadline_zero_is_not_failure": "If deadline_violation_rate remains zero, easy difficulty is SLA-insensitive and should be calibrated next.",
            "value_loss_first_20pct_mean": value_loss_first,
            "value_loss_last_20pct_mean": value_loss_tail,
            "entropy_min_first_30pct": float(min(first_30)) if first_30 else math.nan,
        },
    }


def write_curve_metrics(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(
    run_dir: Path, train_rows: list[dict[str, float]], eval_rows: list[dict[str, float]], suffix: str = ""
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local optional deps
        return [f"plot skipped: {exc}"]

    plots = []

    def plot(rows, key: str, name: str, ylabel: str) -> None:
        xs = values(rows, "step")
        ys = values(rows, key)
        if not xs or not ys:
            return
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker="o", linewidth=1.5)
        plt.xlabel("environment steps")
        plt.ylabel(ylabel)
        plt.title(name)
        plt.grid(True, alpha=0.3)
        out = run_dir / f"{name}{suffix}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        plots.append(str(out))

    plot(eval_rows, "eval_return", "reward_curve", "eval return")
    plot(eval_rows, "completion_ratio", "completion_curve", "completion ratio")
    plot(eval_rows, "mean_service_delay", "delay_curve", "mean service delay")
    plot(eval_rows, "mean_queue_length", "queue_curve", "mean queue length")
    plot(train_rows, "value_loss", "value_loss_curve", "value loss")
    plot(train_rows, "entropy_loss", "entropy_curve", "entropy")
    return plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--random_episodes", type=int, default=20)
    parser.add_argument("--eval_seed_offset", type=int, default=10000)
    parser.add_argument("--policy_mode", choices=["deterministic", "stochastic"], default="deterministic")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / "config.json")
    train_rows = read_csv(run_dir / "train_curve.csv")
    eval_rows = evaluation_mode_rows(read_csv(run_dir / "eval_curve.csv"), args.policy_mode)
    random_metrics = random_baseline(config, args.random_episodes, args.eval_seed_offset)

    metrics = {
        "eval_return_auc": auc(eval_rows, "eval_return"),
        "completion_ratio_auc": auc(eval_rows, "completion_ratio"),
        "mean_delay_auc": auc(eval_rows, "mean_service_delay"),
        "queue_length_auc": auc(eval_rows, "mean_queue_length"),
        "N90_eval_return": n90(eval_rows, "eval_return", True),
        "N90_completion_ratio": n90(eval_rows, "completion_ratio", True),
        "N90_mean_delay": n90(eval_rows, "mean_service_delay", False),
        "value_loss_max": float(max(values(train_rows, "value_loss"))),
        "value_loss_last_mean": tail_stats(train_rows, "value_loss", 0.2)[0],
        "entropy_last_mean": tail_stats(train_rows, "entropy_loss", 0.2)[0],
        "gradient_norm_max": float(max(values(train_rows, "grad_norm"))),
    }
    for name, key in [
        ("eval_return", "eval_return"),
        ("completion", "completion_ratio"),
        ("delay", "mean_service_delay"),
        ("queue", "mean_queue_length"),
    ]:
        avg, std = tail_stats(eval_rows, key, 0.2)
        metrics[f"last_20pct_{name}_mean"] = avg
        metrics[f"last_20pct_{name}_std"] = std
        metrics[f"last_20pct_{name}_cv"] = safe_ratio(std, avg)

    decision = convergence_decision(metrics, random_metrics, train_rows)
    suffix = "" if args.policy_mode == "deterministic" else "_stochastic"
    plot_outputs = maybe_plot(run_dir, train_rows, eval_rows, suffix)
    summary = {
        "run_dir": str(run_dir),
        "policy_mode": args.policy_mode,
        "metrics": metrics,
        "random_baseline": random_metrics,
        "decision": decision,
        "plots": plot_outputs,
    }
    write_json(summary, run_dir / f"mappo_convergence_summary{suffix}.json")
    write_curve_metrics(
        run_dir / f"mappo_curve_metrics{suffix}.csv",
        [{**metrics, **{f"random_{k}": v for k, v in random_metrics.items()}}],
    )

    report = [
        f"# MAPPO Convergence Report ({args.policy_mode})",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Decision",
        "",
        f"Convergence passed: **{decision['passed']}**",
        "",
        "### Checks",
        "",
    ]
    for key, val in decision["checks"].items():
        report.append(f"- {key}: `{val}`")
    report.extend(
        [
            "",
            "## Key Metrics",
            "",
            f"- eval_return_auc: `{metrics['eval_return_auc']:.6f}`",
            f"- completion_ratio_auc: `{metrics['completion_ratio_auc']:.6f}`",
            f"- mean_delay_auc: `{metrics['mean_delay_auc']:.6f}`",
            f"- queue_length_auc: `{metrics['queue_length_auc']:.6f}`",
            f"- N90_eval_return: `{metrics['N90_eval_return']}`",
            f"- last_20pct_eval_return_mean/std: `{metrics['last_20pct_eval_return_mean']:.6f}` / `{metrics['last_20pct_eval_return_std']:.6f}`",
            f"- last_20pct_completion_mean/std: `{metrics['last_20pct_completion_mean']:.6f}` / `{metrics['last_20pct_completion_std']:.6f}`",
            f"- last_20pct_delay_mean/std: `{metrics['last_20pct_delay_mean']:.6f}` / `{metrics['last_20pct_delay_std']:.6f}`",
            f"- value_loss_max: `{metrics['value_loss_max']:.6f}`",
            f"- value_loss_last_mean: `{metrics['value_loss_last_mean']:.6f}`",
            f"- entropy_last_mean: `{metrics['entropy_last_mean']:.6f}`",
            f"- gradient_norm_max: `{metrics['gradient_norm_max']:.6f}`",
            "",
            "## Random Baseline",
            "",
            f"- episode_return: `{random_metrics.get('episode_return', math.nan):.6f}`",
            f"- completion_ratio: `{random_metrics.get('completion_ratio', math.nan):.6f}`",
            f"- mean_service_delay: `{random_metrics.get('mean_service_delay', math.nan):.6f}`",
            f"- deadline_violation_rate: `{random_metrics.get('deadline_violation_rate', math.nan):.6f}`",
            "",
            "## Next Step",
            "",
        ]
    )
    if decision["passed"]:
        report.append("MAPPO passes the easy-mode long-test convergence gate. The next stage should calibrate SLA pressure without enabling semantics.")
    else:
        report.append("MAPPO does not pass all convergence gates yet. Do not enable SLG-SAGE; inspect failed checks and continue environment/MAPPO calibration.")
    (run_dir / f"mappo_convergence_report{suffix}.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"passed": decision["passed"], "run_dir": str(run_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
