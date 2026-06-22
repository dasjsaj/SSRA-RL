"""Scenario primitives for cross-domain service offloading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TASK_TYPES = ["tracking_update", "anomaly_detection", "sonar_recognition", "path_replanning"]
ROLES = ["auv", "usv", "uav"]


@dataclass
class NodeState:
    node_id: str
    role: Literal["auv", "usv", "uav", "shore"]
    position: np.ndarray
    velocity: np.ndarray
    remaining_energy: float
    cpu_capacity: float
    cpu_load: float = 0.0
    queue_length: float = 0.0
    availability: float = 1.0


@dataclass
class ServiceTask:
    task_id: int
    source_agent_id: str
    task_type: int
    task_data_size: float
    task_cpu_cycles: float
    task_deadline: float
    task_priority: float
    task_age: float = 0.0


def make_node(node_id: str, role: str, rng: np.random.Generator) -> NodeState:
    if role == "auv":
        pos = rng.uniform([0.0, 0.0, -1.0], [1.0, 1.0, -0.2]).astype(np.float32)
        cpu = float(rng.uniform(0.35, 0.55))
        energy = float(rng.uniform(0.65, 1.0))
    elif role == "usv":
        pos = rng.uniform([0.0, 0.0, 0.0], [1.0, 1.0, 0.0]).astype(np.float32)
        cpu = float(rng.uniform(0.65, 0.9))
        energy = float(rng.uniform(0.75, 1.0))
    elif role == "uav":
        pos = rng.uniform([0.0, 0.0, 0.5], [1.0, 1.0, 1.0]).astype(np.float32)
        cpu = float(rng.uniform(0.75, 1.0))
        energy = float(rng.uniform(0.7, 1.0))
    else:
        pos = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        cpu = 1.4
        energy = 1.0
    return NodeState(
        node_id=node_id,
        role=role,  # type: ignore[arg-type]
        position=pos,
        velocity=np.zeros(3, dtype=np.float32),
        remaining_energy=energy,
        cpu_capacity=cpu,
    )


def generate_task(task_id: int, source_agent_id: str, rng: np.random.Generator) -> ServiceTask:
    task_type = int(rng.integers(0, len(TASK_TYPES)))
    # Normalized task profiles. Data/cycles/deadline are in abstract units.
    profiles = {
        0: (0.20, 0.25, 0.90, 1.2),  # tracking_update
        1: (0.45, 0.45, 1.20, 0.9),  # anomaly_detection
        2: (0.75, 0.75, 1.60, 0.8),  # sonar_recognition
        3: (0.50, 0.85, 1.05, 1.1),  # path_replanning
    }
    data, cycles, deadline, priority = profiles[task_type]
    jitter = rng.uniform(0.85, 1.15, size=3)
    return ServiceTask(
        task_id=task_id,
        source_agent_id=source_agent_id,
        task_type=task_type,
        task_data_size=float(np.clip(data * jitter[0], 0.05, 1.0)),
        task_cpu_cycles=float(np.clip(cycles * jitter[1], 0.05, 1.0)),
        task_deadline=float(np.clip(deadline * jitter[2], 0.40, 2.0)),
        task_priority=float(priority),
    )
