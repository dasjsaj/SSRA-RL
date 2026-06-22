"""Semantic feature extraction and heuristic action priors."""

from __future__ import annotations

import numpy as np


ROLE_TO_ID = {"auv": 0, "usv": 1, "uav": 2, "shore": 3}


def semantic_features(raw: dict) -> np.ndarray:
    task_deadline = float(raw.get("task_deadline", 1.0))
    data_size = float(raw.get("task_data_size", 0.0))
    cpu_cycles = float(raw.get("task_cpu_cycles", 0.0))
    link_reliability = float(raw.get("link_reliability", 1.0))
    queue_pressure = float(raw.get("neighbor_queue_pressure", raw.get("queue_length", 0.0)))
    energy_pressure = 1.0 - float(raw.get("remaining_energy", 1.0))
    cpu_load = float(raw.get("cpu_load", 0.0))
    role = int(raw.get("role_id", 0))
    role_onehot = np.zeros(3, dtype=np.float32)
    if 0 <= role < 3:
        role_onehot[role] = 1.0
    task_urgency = float(np.clip(1.0 - task_deadline, 0.0, 1.0))
    compute_intensity = float(np.clip(cpu_cycles, 0.0, 1.0))
    data_intensity = float(np.clip(data_size, 0.0, 1.0))
    resource_pressure = float(np.clip(0.5 * queue_pressure + 0.3 * cpu_load + 0.2 * energy_pressure, 0.0, 1.0))
    return np.concatenate(
        [
            np.array(
                [
                    task_urgency,
                    compute_intensity,
                    data_intensity,
                    link_reliability,
                    queue_pressure,
                    energy_pressure,
                    resource_pressure,
                ],
                dtype=np.float32,
            ),
            role_onehot,
        ],
        axis=0,
    )


def semantic_prior_action(raw: dict) -> np.ndarray:
    """Rule prior used only as a soft learning target, not as an action override."""
    role = int(raw.get("role_id", 0))
    sem = raw.get("semantic", np.zeros(10, dtype=np.float32))
    urgency = float(sem[0])
    compute = float(sem[1])
    data = float(sem[2])
    queue_pressure = float(sem[4]) if len(sem) > 4 else float(raw.get("queue_length", 0.0))
    resource_pressure = float(sem[6]) if len(sem) > 6 else queue_pressure
    link_rel = float(raw.get("link_reliability", 0.5))
    queue = float(raw.get("queue_length", 0.0))
    prior = np.zeros(6, dtype=np.float32)
    if role == 0:  # AUV
        offload_need = np.clip(0.45 * compute + 0.35 * data + 0.2 * urgency, 0.0, 1.0)
        link_rate = np.asarray(raw.get("link_rate", np.array([0.2, 0.5, 1.0], dtype=np.float32)), dtype=np.float32)
        cpu_capacity = float(raw.get("cpu_capacity", 0.5))
        cpu_load = float(raw.get("cpu_load", 0.0))
        local_delay = compute / max(0.05, cpu_capacity * (1.0 - cpu_load))
        usv_delay = data / max(0.05, float(link_rate[0])) + 0.6 * compute + 0.6 * queue_pressure
        uav_delay = data / max(0.05, float(link_rate[1] if link_rate.size > 1 else link_rate[0])) + 0.5 * compute + 0.5 * queue_pressure
        delay_estimates = np.array([local_delay, usv_delay, uav_delay], dtype=np.float32)
        intensity_bias = np.array([0.15 * offload_need, 0.0, 0.05 * urgency], dtype=np.float32)
        primary = int(np.argmin(delay_estimates + intensity_bias))
        route_prior = np.full(3, 0.12, dtype=np.float32)
        route_prior[primary] = 0.76
        prior[:3] = route_prior
        offload_primary = primary in (1, 2)
        tx_floor = 0.55 if offload_primary else 0.40
        prior[3] = np.clip(
            tx_floor
            + 0.18 * urgency
            + 0.18 * resource_pressure
            + 0.18 * data
            + 0.22 * (1.0 - link_rel),
            0.0,
            0.95,
        )
        prior[4] = np.clip(0.2 + 0.5 * data, 0.0, 1.0)
        prior[5] = 0.5
    elif role == 1:  # USV
        prior[0] = np.clip(1.0 - queue, 0.1, 1.0)
        prior[1] = np.clip(0.4 + compute, 0.0, 1.0)
        prior[2] = np.clip(0.25 + urgency, 0.0, 1.0)
        prior[3] = np.clip(0.25 + data, 0.0, 1.0)
        prior[4] = 0.5
        prior[5] = 0.5
    else:  # UAV
        prior[0] = np.clip(1.0 - queue, 0.1, 1.0)
        prior[1] = np.clip(0.5 + compute, 0.0, 1.0)
        prior[2] = np.clip(0.35 + data, 0.0, 1.0)
        prior[3] = np.clip(0.5 + urgency, 0.0, 1.0)
        prior[4] = 0.5
        prior[5] = 0.5
    return prior.astype(np.float32)
