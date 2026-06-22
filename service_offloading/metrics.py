"""Metric aggregation helpers for service offloading experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np


ENERGY_ACCOUNTING_KEYS = [
    "total_consumed_energy",
    "completed_task_energy",
    "timeout_task_energy",
    "dropped_task_energy",
    "inflight_task_energy",
    "energy_per_generated_task",
    "energy_per_completed_task",
    "energy_per_successful_completion",
    "key_tx_energy",
    "key_tx_energy_per_successful_completion",
    "excluded_usv_tx_energy",
    "energy_accounting_balance_error",
    "energy_compute_auv",
    "energy_compute_usv",
    "energy_compute_uav",
    "energy_compute_shore",
    "energy_transfer_auv_usv",
    "energy_transfer_usv_uav",
    "energy_transfer_usv_shore",
    "energy_transfer_uav_shore",
]


METRIC_KEYS = [
    "completion_ratio",
    "mean_service_delay",
    "deadline_violation_rate",
    "mean_energy_cost",
    "offload_success_rate",
    "load_balance_index",
    "mean_queue_length",
    "generated_tasks",
    "completed_tasks",
    "timeout_tasks",
    "dropped_tasks",
    "semantic_match_rate",
    "primary_route_ratio_mean",
    "reward_completion",
    "reward_route_progress",
    "reward_delay",
    "reward_deadline",
    "reward_energy",
    "reward_invalid_action",
    "reward_queue_overflow",
    "reward_total_mean",
    "auv_queue_mean",
    "usv_queue_mean",
    "uav_queue_mean",
    "shore_queue_length",
    "virtual_queue_mean",
    "first_hop_offloaded_tasks",
    "weighted_backlog_cost",
    "auv_to_usv_tasks",
    "usv_to_uav_tasks",
    "usv_to_shore_tasks",
    "uav_to_shore_tasks",
    "local_computed_tasks",
    "edge_computed_tasks",
    "reward_backlog",
    "reward_drop",
    "route_progress_per_step",
    *ENERGY_ACCOUNTING_KEYS,
]


def mean_metrics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {k: 0.0 for k in METRIC_KEYS}
    out = {}
    for key in METRIC_KEYS:
        values = [float(r.get(key, 0.0)) for r in rows]
        out[key] = float(np.mean(values))
    return out
