"""Queue-aware two-hop AUV-USV-UAV/shore service offloading environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .scenario import TASK_TYPES, NodeState, ServiceTask, generate_task, make_node


@dataclass
class QueuedTask:
    """A compute task that remains identifiable while moving through queues."""

    task_id: int
    source_agent_id: str
    task_type: int
    data_size: float
    remaining_cycles: float
    deadline: float
    priority: float
    elapsed: float = 0.0
    energy: float = 0.0
    hops: int = 0


class DualHopQueueServiceOffloadingEnv:
    """Cooperative queueing environment inspired by UAV-assisted MEC models.

    AUVs generate tasks. An offloaded task must first reach a USV; USVs can
    process it locally or relay it to a UAV/shore node. UAVs process or forward
    tasks to shore. Shore is a fixed non-agent compute resource.
    """

    env_model = "dual_hop_queue"
    action_dim = 4

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        cfg = self.config.get("env", self.config)
        self.env_cfg = cfg
        self.n_auv = int(cfg.get("n_auv", 4))
        self.n_usv = int(cfg.get("n_usv", 2))
        self.n_uav = int(cfg.get("n_uav", 2))
        self.episode_length = int(cfg.get("episode_length", 100))
        self.action_mode = str(cfg.get("action_mode", "simple"))
        self.action_dim = 4
        self.obs_mode = str(cfg.get("obs_mode", "minimal"))
        self.reward_mode = str(cfg.get("reward_mode", "stable_v1"))
        self.use_semantic_side_channel = bool(cfg.get("use_semantic_side_channel", False))
        self.difficulty = str(cfg.get("difficulty", "easy"))
        self.use_mobility_control = False
        self.use_semantic_reward = False
        self.task_arrival_rate = float(cfg.get("task_arrival_rate", 0.22))
        self.initial_tasks_per_auv = int(cfg.get("initial_tasks_per_auv", 0))
        self.queue_capacity = int(cfg.get("queue_capacity", 24))
        self.task_data_scale = float(cfg.get("task_data_scale", 1.0))
        self.task_cpu_scale = float(cfg.get("task_cpu_scale", 1.0))
        self.auv_cpu_scale = float(cfg.get("auv_cpu_scale", 1.0))
        self.usv_cpu_scale = float(cfg.get("usv_cpu_scale", 1.0))
        self.uav_cpu_scale = float(cfg.get("uav_cpu_scale", 1.0))
        self.shore_cpu_scale = float(cfg.get("shore_cpu_scale", 1.0))
        self.deadline_scale = float(cfg.get("deadline_scale", 2.5))
        self.slot_duration = float(cfg.get("slot_duration", 0.1))
        self.virtual_queue_threshold = float(cfg.get("virtual_queue_threshold", 5.0))
        self.compute_service_scale = float(cfg.get("compute_service_scale", 1.6))
        self.backlog_penalty_weight = float(cfg.get("backlog_penalty_weight", 0.25))
        self.use_route_progress_reward = bool(cfg.get("use_route_progress_reward", False))
        self.route_progress_reward_weights = {
            "auv_to_usv_tasks": float(cfg.get("reward_auv_to_usv", 0.08)),
            "usv_to_uav_tasks": float(cfg.get("reward_usv_to_uav", 0.06)),
            "usv_to_shore_tasks": float(cfg.get("reward_usv_to_shore", 0.05)),
            "uav_to_shore_tasks": float(cfg.get("reward_uav_to_shore", 0.04)),
        }
        self.backlog_role_weights = {
            "auv": float(cfg.get("auv_backlog_weight", 1.0)),
            "usv": float(cfg.get("usv_backlog_weight", 0.65)),
            # Reaching a USV is useful access progress, but further relays are
            # not progress until downstream compute actually pays off.
            "uav": float(cfg.get("uav_backlog_weight", 0.65)),
            "shore": float(cfg.get("shore_backlog_weight", 0.65)),
        }
        self.route_delay_normalizer = float(cfg.get("route_delay_normalizer", 8.0))
        self.semantic_usv_forward_coordination_penalty = float(
            cfg.get("semantic_usv_forward_coordination_penalty", 0.0)
        )
        self.use_task_aware_semantic_teacher = bool(cfg.get("use_task_aware_semantic_teacher", False))
        self.semantic_task_compatibility_weight = float(cfg.get("semantic_task_compatibility_weight", 0.15))
        self.semantic_convergence_bonus_weight = float(cfg.get("semantic_convergence_bonus_weight", 0.10))
        self.semantic_energy_cost_weight = float(cfg.get("semantic_energy_cost_weight", 0.0))
        self.use_qos_tie_energy_teacher = bool(cfg.get("use_qos_tie_energy_teacher", False))
        self.semantic_qos_tie_threshold = float(cfg.get("semantic_qos_tie_threshold", 0.05))
        self.semantic_qos_tie_energy_weight = float(cfg.get("semantic_qos_tie_energy_weight", 0.10))
        self.use_deadline_capacity_route_gate = bool(cfg.get("use_deadline_capacity_route_gate", False))
        self.semantic_deadline_capacity_gate_weight = float(cfg.get("semantic_deadline_capacity_gate_weight", 0.20))
        self.use_downstream_aware_semantic_teacher = bool(
            cfg.get("use_downstream_aware_semantic_teacher", False)
        )
        self.use_downstream_coordination_teacher = bool(cfg.get("use_downstream_coordination_teacher", False))
        self.semantic_downstream_delay_weight = float(cfg.get("semantic_downstream_delay_weight", 0.25))
        self.semantic_downstream_energy_weight = float(cfg.get("semantic_downstream_energy_weight", 0.08))
        self.semantic_downstream_queue_weight = float(cfg.get("semantic_downstream_queue_weight", 0.08))
        self.semantic_downstream_parallelism_weight = float(
            cfg.get("semantic_downstream_parallelism_weight", 0.20)
        )
        self.semantic_extra_hop_penalty_weight = float(cfg.get("semantic_extra_hop_penalty_weight", 0.0))
        self.use_marginal_completion_teacher = bool(cfg.get("use_marginal_completion_teacher", False))
        self.semantic_marginal_completion_weight = float(cfg.get("semantic_marginal_completion_weight", 0.25))
        self.semantic_deadline_risk_weight = float(cfg.get("semantic_deadline_risk_weight", 0.10))
        self.semantic_terminal_compute_bonus_weight = float(
            cfg.get("semantic_terminal_compute_bonus_weight", 0.0)
        )
        self.obs_dim = 21
        self.global_extra_dim = 6
        self.global_state_dim = self.obs_dim * (self.n_auv + self.n_usv + self.n_uav) + self.global_extra_dim
        downstream_dim = 12 if self.use_downstream_aware_semantic_teacher else 0
        coordination_dim = 4 if self.use_downstream_coordination_teacher else 0
        marginal_dim = 12 if self.use_marginal_completion_teacher else 0
        self.semantic_dim = 17 + downstream_dim + coordination_dim + marginal_dim if self.use_semantic_side_channel else 0
        self.seed_value = 0
        self.rng = np.random.default_rng(0)
        self.nodes: dict[str, NodeState] = {}
        self.agent_ids: list[str] = []
        self.shore = make_node("shore_0", "shore", self.rng)
        self.queues: dict[str, deque[QueuedTask]] = {}
        self.shore_queue: deque[QueuedTask] = deque()
        self.virtual_queues: dict[str, float] = {}
        self.step_count = 0
        self.task_counter = 0
        self.generated_total = 0
        self.completed_total = 0
        self.timeout_total = 0
        self.dropped_total = 0
        self.first_hop_offloaded_total = 0
        self.completed_delays: list[float] = []
        self.completed_energy: list[float] = []
        self.total_consumed_energy = 0.0
        self.completed_task_energy_total = 0.0
        self.timeout_task_energy_total = 0.0
        self.dropped_task_energy_total = 0.0
        self.energy_breakdown_total: dict[str, float] = {}

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.task_counter = 0
        self.generated_total = 0
        self.completed_total = 0
        self.timeout_total = 0
        self.dropped_total = 0
        self.first_hop_offloaded_total = 0
        self.completed_delays = []
        self.completed_energy = []
        self.total_consumed_energy = 0.0
        self.completed_task_energy_total = 0.0
        self.timeout_task_energy_total = 0.0
        self.dropped_task_energy_total = 0.0
        self.energy_breakdown_total = {
            "energy_compute_auv": 0.0,
            "energy_compute_usv": 0.0,
            "energy_compute_uav": 0.0,
            "energy_compute_shore": 0.0,
            "energy_transfer_auv_usv": 0.0,
            "energy_transfer_usv_uav": 0.0,
            "energy_transfer_usv_shore": 0.0,
            "energy_transfer_uav_shore": 0.0,
        }
        self.nodes = {}
        for i in range(self.n_auv):
            self.nodes[f"auv_{i}"] = make_node(f"auv_{i}", "auv", self.rng)
            self.nodes[f"auv_{i}"].cpu_capacity *= self.auv_cpu_scale
        for i in range(self.n_usv):
            self.nodes[f"usv_{i}"] = make_node(f"usv_{i}", "usv", self.rng)
            self.nodes[f"usv_{i}"].cpu_capacity *= self.usv_cpu_scale
        for i in range(self.n_uav):
            self.nodes[f"uav_{i}"] = make_node(f"uav_{i}", "uav", self.rng)
            self.nodes[f"uav_{i}"].cpu_capacity *= self.uav_cpu_scale
        self.shore = make_node("shore_0", "shore", self.rng)
        self.shore.cpu_capacity *= self.shore_cpu_scale
        self.agent_ids = list(self.nodes.keys())
        self.queues = {aid: deque() for aid in self.agent_ids}
        self.shore_queue = deque()
        self.virtual_queues = {aid: 0.0 for aid in self.agent_ids}
        for aid in self._role_ids("auv"):
            for _ in range(self.initial_tasks_per_auv):
                self._enqueue_generated_task(aid)
        return self._obs_dict(), {"agent_ids": list(self.agent_ids), "env_model": self.env_model}

    def step(self, actions: dict[str, np.ndarray]):
        self.step_count += 1
        transitions = {
            "auv_to_usv_tasks": 0.0,
            "usv_to_uav_tasks": 0.0,
            "usv_to_shore_tasks": 0.0,
            "uav_to_shore_tasks": 0.0,
            "local_computed_tasks": 0.0,
            "edge_computed_tasks": 0.0,
        }
        step_energy = 0.0
        step_completed = 0
        step_timeout = 0
        step_dropped = 0
        per_agent = {
            aid: {
                "moved_task": 0.0,
                "completed_task": 0.0,
                "route_progress": 0.0,
                "energy_used": 0.0,
            }
            for aid in self.agent_ids
        }
        self._generate_arrivals()

        completed, energy = self._service_shore(transitions)
        step_completed += completed
        step_energy += energy
        for aid in self._role_ids("uav"):
            moved_before = transitions["uav_to_shore_tasks"]
            completed, energy = self._service_uav(aid, actions.get(aid), transitions)
            moved = transitions["uav_to_shore_tasks"] - moved_before
            per_agent[aid] = self._agent_outcome(moved, completed, energy)
            step_completed += completed
            step_energy += energy
        for aid in self._role_ids("usv"):
            moved_before = transitions["usv_to_uav_tasks"] + transitions["usv_to_shore_tasks"]
            completed, energy = self._service_usv(aid, actions.get(aid), transitions)
            moved = transitions["usv_to_uav_tasks"] + transitions["usv_to_shore_tasks"] - moved_before
            per_agent[aid] = self._agent_outcome(moved, completed, energy)
            step_completed += completed
            step_energy += energy
        for aid in self._role_ids("auv"):
            moved_before = transitions["auv_to_usv_tasks"]
            completed, energy = self._service_auv(aid, actions.get(aid), transitions)
            moved = transitions["auv_to_usv_tasks"] - moved_before
            per_agent[aid] = self._agent_outcome(moved, completed, energy)
            step_completed += completed
            step_energy += energy

        step_timeout, step_dropped = self._age_and_expire_tasks()
        self.completed_total += step_completed
        self.timeout_total += step_timeout
        self.dropped_total += step_dropped
        self._update_virtual_queues()
        metrics, reward = self._metrics_and_reward(
            transitions, step_completed, step_timeout, step_dropped, step_energy
        )
        rewards = {aid: reward for aid in self.agent_ids}
        done = self.step_count >= self.episode_length
        return self._obs_dict(), rewards, done, False, {"metrics": metrics, "per_agent": per_agent}

    @staticmethod
    def _agent_outcome(moved: float, completed: int, energy: float) -> dict[str, float]:
        return {
            "moved_task": float(moved),
            "completed_task": float(completed),
            "route_progress": float(moved + completed),
            "energy_used": float(energy),
        }

    def get_global_state(self) -> np.ndarray:
        local = [self._agent_obs(aid)["obs"] for aid in self.agent_ids]
        all_queues = [len(self.queues[aid]) for aid in self.agent_ids] + [len(self.shore_queue)]
        extras = np.array(
            [
                np.mean(all_queues) / max(1, self.queue_capacity),
                self.generated_total / max(1, self.episode_length * max(1, self.n_auv)),
                self.completed_total / max(1, self.generated_total),
                self.timeout_total / max(1, self.generated_total),
                len(self.shore_queue) / max(1, self.queue_capacity),
                self.step_count / max(1, self.episode_length),
            ],
            dtype=np.float32,
        )
        return np.concatenate(local + [extras], axis=0).astype(np.float32)

    def _obs_dict(self) -> dict[str, dict]:
        return {aid: self._agent_obs(aid) for aid in self.agent_ids}

    def _agent_obs(self, aid: str) -> dict[str, Any]:
        node = self.nodes[aid]
        queue = self.queues[aid]
        task = queue[0] if queue else None
        role_vec = np.zeros(3, dtype=np.float32)
        role_vec[{"auv": 0, "usv": 1, "uav": 2}[node.role]] = 1.0
        link_rate, link_loss, downstream_queue = self._next_hop_features(aid)
        route_delay_estimates = self._route_delay_estimates(aid, task)
        slack = 1.0 if task is None else np.clip((task.deadline - task.elapsed) / max(task.deadline, 1e-6), 0.0, 1.0)
        obs = np.array(
            [
                *role_vec,
                node.remaining_energy,
                np.clip(node.cpu_capacity / 1.4, 0.0, 1.0),
                node.cpu_load,
                len(queue) / max(1, self.queue_capacity),
                0.0 if task is None else task.data_size,
                0.0 if task is None else task.remaining_cycles,
                slack,
                0.0 if task is None else task.priority / 1.2,
                link_rate,
                link_loss,
                downstream_queue,
                len(self.shore_queue) / max(1, self.queue_capacity),
                self.virtual_queues[aid] / max(1.0, self.queue_capacity),
                self.generated_total / max(1, self.episode_length * max(1, self.n_auv)),
                self.step_count / max(1, self.episode_length),
                *route_delay_estimates,
            ],
            dtype=np.float32,
        )
        action_mask = self._role_action_mask(aid, task is not None)
        side_channel = self._semantic_side_channel(aid, task, route_delay_estimates, action_mask)
        return {
            "obs": obs,
            "raw": obs,
            "action_mask": action_mask,
            "semantic": side_channel["semantic"],
            "semantic_prior": side_channel["semantic_prior"],
            "semantic_route_costs": side_channel["route_costs"],
            "semantic_action_mask": side_channel["action_mask"],
            "semantic_task_compatibility": side_channel["task_compatibility"],
            "semantic_convergence_bonus": side_channel["convergence_bonus"],
            "semantic_energy_cost": side_channel["energy_cost"],
            "semantic_key_energy_cost": side_channel["key_energy_cost"],
            "semantic_route_gate_penalty": side_channel["route_gate_penalty"],
            "semantic_downstream_delay": side_channel["downstream_delay"],
            "semantic_downstream_energy": side_channel["downstream_energy"],
            "semantic_downstream_queue": side_channel["downstream_queue"],
            "semantic_downstream_parallelism_bonus": side_channel["downstream_parallelism_bonus"],
            "semantic_extra_hop_penalty": side_channel["extra_hop_penalty"],
            "semantic_marginal_completion_value": side_channel["marginal_completion_value"],
            "semantic_deadline_risk": side_channel["deadline_risk"],
            "semantic_terminal_compute_bonus": side_channel["terminal_compute_bonus"],
            "agent_id": aid,
            "role": node.role,
            "route_delay_estimates": route_delay_estimates,
        }

    def _semantic_side_channel(
        self, aid: str, task: QueuedTask | None, route_delay_estimates: np.ndarray, action_mask: np.ndarray
    ) -> dict[str, np.ndarray]:
        if not self.use_semantic_side_channel:
            return {
                "semantic": np.zeros(0, dtype=np.float32),
                "semantic_prior": np.zeros(self.action_dim, dtype=np.float32),
                "route_costs": np.zeros(self.action_dim, dtype=np.float32),
                "action_mask": action_mask.astype(np.float32),
                "task_compatibility": np.zeros(self.action_dim, dtype=np.float32),
                "convergence_bonus": np.zeros(self.action_dim, dtype=np.float32),
                "energy_cost": np.zeros(self.action_dim, dtype=np.float32),
                "key_energy_cost": np.zeros(self.action_dim, dtype=np.float32),
                "route_gate_penalty": np.zeros(self.action_dim, dtype=np.float32),
                "downstream_delay": np.zeros(self.action_dim, dtype=np.float32),
                "downstream_energy": np.zeros(self.action_dim, dtype=np.float32),
                "downstream_queue": np.zeros(self.action_dim, dtype=np.float32),
                "downstream_parallelism_bonus": np.zeros(self.action_dim, dtype=np.float32),
                "extra_hop_penalty": np.zeros(self.action_dim, dtype=np.float32),
                "marginal_completion_value": np.zeros(self.action_dim, dtype=np.float32),
                "deadline_risk": np.zeros(self.action_dim, dtype=np.float32),
                "terminal_compute_bonus": np.zeros(self.action_dim, dtype=np.float32),
            }
        node = self.nodes[aid]
        role_vec = np.zeros(3, dtype=np.float32)
        role_vec[{"auv": 0, "usv": 1, "uav": 2}[node.role]] = 1.0
        task_vec = np.zeros(4, dtype=np.float32)
        if task is not None:
            task_vec[int(task.task_type)] = 1.0
        has_task = float(task is not None)
        urgency = 0.0 if task is None else float(
            np.clip(task.elapsed / max(task.deadline, 1e-6), 0.0, 1.0)
        )
        data_intensity = 0.0 if task is None else float(np.clip(task.data_size, 0.0, 1.0))
        compute_intensity = 0.0 if task is None else float(np.clip(task.remaining_cycles / 2.0, 0.0, 1.0))
        queue_pressure = float(len(self.queues[aid]) / max(1, self.queue_capacity))
        _, _, downstream_pressure = self._next_hop_features(aid)
        downstream_delay, downstream_energy, downstream_queue = self._semantic_downstream_components(aid, task)
        downstream_parallelism_bonus = self._semantic_downstream_parallelism_bonus(aid, task)
        extra_hop_penalty = self._semantic_extra_hop_penalty(aid, task)
        marginal_completion_value, deadline_risk, terminal_compute_bonus = self._semantic_marginal_completion_components(
            aid, task
        )
        semantic_values = [
            *role_vec,
            *task_vec,
            has_task,
            urgency,
            data_intensity,
            compute_intensity,
            queue_pressure,
            downstream_pressure,
            node.remaining_energy,
            *route_delay_estimates,
        ]
        if self.use_downstream_aware_semantic_teacher:
            semantic_values.extend([*downstream_delay, *downstream_energy, *downstream_queue])
        if self.use_downstream_coordination_teacher:
            semantic_values.extend([*downstream_parallelism_bonus])
        if self.use_marginal_completion_teacher:
            semantic_values.extend([*marginal_completion_value, *deadline_risk, *terminal_compute_bonus])
        semantic = np.asarray(
            [
                *semantic_values,
            ],
            dtype=np.float32,
        )
        route_costs, task_compatibility, convergence_bonus, energy_cost = self._semantic_route_components(
            aid, task, route_delay_estimates, downstream_pressure, urgency, action_mask
        )
        key_energy_cost = self._semantic_key_tx_energy_costs(aid, task)
        route_gate_penalty = self._deadline_capacity_route_gate(aid, task)
        prior_logits = -route_costs
        prior_logits = np.where(action_mask > 0.0, prior_logits, -1e9)
        prior = np.exp(prior_logits - np.max(prior_logits))
        prior = prior / max(1e-8, float(prior.sum()))
        return {
            "semantic": semantic,
            "semantic_prior": prior.astype(np.float32),
            "route_costs": route_costs.astype(np.float32),
            "action_mask": action_mask.astype(np.float32),
            "task_compatibility": task_compatibility.astype(np.float32),
            "convergence_bonus": convergence_bonus.astype(np.float32),
            "energy_cost": energy_cost.astype(np.float32),
            "key_energy_cost": key_energy_cost.astype(np.float32),
            "route_gate_penalty": route_gate_penalty.astype(np.float32),
            "downstream_delay": downstream_delay.astype(np.float32),
            "downstream_energy": downstream_energy.astype(np.float32),
            "downstream_queue": downstream_queue.astype(np.float32),
            "downstream_parallelism_bonus": downstream_parallelism_bonus.astype(np.float32),
            "extra_hop_penalty": extra_hop_penalty.astype(np.float32),
            "marginal_completion_value": marginal_completion_value.astype(np.float32),
            "deadline_risk": deadline_risk.astype(np.float32),
            "terminal_compute_bonus": terminal_compute_bonus.astype(np.float32),
        }

    def _role_action_mask(self, aid: str, has_task: bool) -> np.ndarray:
        if self.action_mode != "discrete_route":
            return np.ones(self.action_dim, dtype=np.float32)
        if not has_task:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        role = self.nodes[aid].role
        if role == "auv":
            return np.array(
                [1.0, float(self.n_usv >= 1), float(self.n_usv >= 2), 1.0], dtype=np.float32
            )
        if role == "usv":
            return np.array([1.0, float(bool(self._role_ids("uav"))), 1.0, 1.0], dtype=np.float32)
        return np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)

    def _semantic_action_mask(self, aid: str, has_task: bool) -> np.ndarray:
        """Backward-compatible name for the shared role-validity mask."""
        return self._role_action_mask(aid, has_task)

    def _semantic_route_costs(
        self,
        aid: str,
        task: QueuedTask | None,
        route_delay_estimates: np.ndarray,
        downstream_pressure: float,
        urgency: float,
        mask: np.ndarray,
    ) -> np.ndarray:
        return self._semantic_route_components(
            aid, task, route_delay_estimates, downstream_pressure, urgency, mask
        )[0]

    def _semantic_route_components(
        self,
        aid: str,
        task: QueuedTask | None,
        route_delay_estimates: np.ndarray,
        downstream_pressure: float,
        urgency: float,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if task is None:
            zeros = np.zeros(self.action_dim, dtype=np.float32)
            return np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32), zeros, zeros, zeros
        costs = np.ones(self.action_dim, dtype=np.float32)
        costs[:3] = route_delay_estimates + 0.2 * downstream_pressure
        costs[:3] += 0.35 * urgency * route_delay_estimates
        if self.nodes[aid].role in {"auv", "usv"}:
            costs[1:3] += 0.04 * float(np.clip(task.data_size, 0.0, 1.0))
        if self.nodes[aid].role == "usv":
            # Forwarding adds a coordination stage; for non-urgent work a capable
            # USV should not be guided downstream solely by nominal edge speed.
            costs[1:3] += self.semantic_usv_forward_coordination_penalty * (1.0 - urgency)
        task_compatibility = np.zeros(self.action_dim, dtype=np.float32)
        convergence_bonus = np.zeros(self.action_dim, dtype=np.float32)
        energy_cost = self._semantic_route_energy_costs(aid, task)
        downstream_delay, downstream_energy, downstream_queue = self._semantic_downstream_components(aid, task)
        downstream_parallelism_bonus = self._semantic_downstream_parallelism_bonus(aid, task)
        extra_hop_penalty = self._semantic_extra_hop_penalty(aid, task)
        marginal_completion_value, deadline_risk, terminal_compute_bonus = self._semantic_marginal_completion_components(
            aid, task
        )
        if self.use_task_aware_semantic_teacher:
            task_compatibility = self._task_route_compatibility(aid, task.task_type)
            completion_proximity = np.clip(1.0 - route_delay_estimates, 0.0, 1.0)
            convergence_bonus[:3] = completion_proximity * (0.5 + 0.5 * urgency)
            costs -= self.semantic_task_compatibility_weight * task_compatibility
            costs -= self.semantic_convergence_bonus_weight * convergence_bonus
            costs += self.semantic_energy_cost_weight * energy_cost
            if self.use_downstream_aware_semantic_teacher:
                costs += self.semantic_downstream_delay_weight * downstream_delay
                costs += self.semantic_downstream_energy_weight * downstream_energy
                costs += self.semantic_downstream_queue_weight * downstream_queue
            if self.use_downstream_coordination_teacher:
                costs -= self.semantic_downstream_parallelism_weight * downstream_parallelism_bonus
                costs += self.semantic_extra_hop_penalty_weight * extra_hop_penalty
            if self.use_marginal_completion_teacher:
                costs -= self.semantic_marginal_completion_weight * marginal_completion_value
                costs -= self.semantic_terminal_compute_bonus_weight * terminal_compute_bonus
                costs += self.semantic_deadline_risk_weight * deadline_risk
        if self.use_deadline_capacity_route_gate:
            costs += self.semantic_deadline_capacity_gate_weight * self._deadline_capacity_route_gate(aid, task)
        costs[3] = 0.8 + 0.7 * urgency + 0.2 * float(len(self.queues[aid]) > 0)
        costs[mask <= 0.0] = 10.0
        return costs, task_compatibility, convergence_bonus, energy_cost

    def _semantic_marginal_completion_components(
        self, aid: str, task: QueuedTask | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value = np.zeros(self.action_dim, dtype=np.float32)
        deadline_risk = np.ones(self.action_dim, dtype=np.float32)
        terminal_bonus = np.zeros(self.action_dim, dtype=np.float32)
        if task is None:
            return value, np.zeros(self.action_dim, dtype=np.float32), terminal_bonus
        delays, progress = self._semantic_completion_delay_and_progress(aid, task)
        slack = max(task.deadline - task.elapsed, 1e-6)
        fit = np.clip((slack - delays) / max(slack, 1e-6), 0.0, 1.0)
        deadline_risk = np.clip((delays - slack) / max(task.deadline, 1e-6), 0.0, 1.0)
        terminal_bonus[0] = float(progress[0] >= 0.999 and fit[0] > 0.0)
        hop_efficiency = np.ones(self.action_dim, dtype=np.float32)
        if task.hops >= 2:
            hop_efficiency[1:3] *= 0.45
        elif task.hops == 1:
            hop_efficiency[1:3] *= 0.75
        value[:3] = 0.50 * fit[:3] + 0.35 * progress[:3] + 0.15 * hop_efficiency[:3] + 0.20 * terminal_bonus[:3]
        if terminal_bonus[0] > 0.0:
            value[1:3] *= 0.80
        value[3] = 0.0
        deadline_risk[3] = 1.0
        return (
            np.clip(value, 0.0, 1.0).astype(np.float32),
            deadline_risk.astype(np.float32),
            terminal_bonus.astype(np.float32),
        )

    def _semantic_completion_delay_and_progress(self, aid: str, task: QueuedTask) -> tuple[np.ndarray, np.ndarray]:
        node = self.nodes[aid]
        unavailable = self.route_delay_normalizer
        delays = np.full(self.action_dim, unavailable, dtype=np.float32)
        progress = np.zeros(self.action_dim, dtype=np.float32)

        def compute_progress(target: NodeState, queue_len: int, shore: bool = False) -> tuple[float, float]:
            delay = self._estimated_compute_delay(task, target, queue_len, shore=shore)
            budget = max(0.02, target.cpu_capacity * self.compute_service_scale)
            fraction = float(np.clip(budget / max(task.remaining_cycles, 1e-6), 0.0, 1.0))
            return delay, fraction

        if node.role == "auv":
            delays[0], progress[0] = compute_progress(node, len(self.queues[aid]))
            usvs = self._role_ids("usv")
            for action_index in (1, 2):
                if not usvs:
                    continue
                usv = self.nodes[usvs[min(action_index - 1, len(usvs) - 1)]]
                downstream_delay, _, _ = self._best_downstream_completion_from_node(task, usv.node_id)
                delays[action_index] = self._estimated_transfer_delay(task, node, usv, "auv_usv") + downstream_delay
                progress[action_index] = 0.45 + 0.55 * self._downstream_completion_progress(task, usv.node_id)
        elif node.role == "usv":
            delays[0], progress[0] = compute_progress(node, len(self.queues[aid]))
            uavs = self._role_ids("uav")
            if uavs:
                uav = min(
                    (self.nodes[target] for target in uavs),
                    key=lambda x: self._estimated_transfer_delay(task, node, x, "usv_uav")
                    + self._estimated_compute_delay(task, x, len(self.queues[x.node_id])),
                )
                compute_delay, compute_fraction = compute_progress(uav, len(self.queues[uav.node_id]))
                delays[1] = self._estimated_transfer_delay(task, node, uav, "usv_uav") + compute_delay
                progress[1] = 0.45 + 0.55 * compute_fraction
            shore_delay, shore_fraction = compute_progress(self.shore, len(self.shore_queue), shore=True)
            delays[2] = self._estimated_transfer_delay(task, node, self.shore, "usv_shore") + shore_delay
            progress[2] = 0.45 + 0.55 * shore_fraction
        else:
            delays[0], progress[0] = compute_progress(node, len(self.queues[aid]))
            shore_delay, shore_fraction = compute_progress(self.shore, len(self.shore_queue), shore=True)
            delays[1] = self._estimated_transfer_delay(task, node, self.shore, "uav_shore") + shore_delay
            progress[1] = 0.35 + 0.45 * shore_fraction
        return delays, progress

    def _downstream_completion_progress(self, task: QueuedTask, node_id: str) -> float:
        node = self.nodes[node_id]
        local_budget = max(0.02, node.cpu_capacity * self.compute_service_scale)
        best = float(np.clip(local_budget / max(task.remaining_cycles, 1e-6), 0.0, 1.0))
        if node.role == "usv":
            for uav_id in self._role_ids("uav"):
                uav_budget = max(0.02, self.nodes[uav_id].cpu_capacity * self.compute_service_scale)
                best = max(best, float(np.clip(uav_budget / max(task.remaining_cycles, 1e-6), 0.0, 1.0)))
            shore_budget = max(0.02, self.shore.cpu_capacity * self.compute_service_scale)
            best = max(best, float(np.clip(shore_budget / max(task.remaining_cycles, 1e-6), 0.0, 1.0)))
        return best

    def _semantic_extra_hop_penalty(self, aid: str, task: QueuedTask | None) -> np.ndarray:
        penalty = np.zeros(self.action_dim, dtype=np.float32)
        if task is None:
            return penalty
        role = self.nodes[aid].role
        if role == "uav" and task.hops >= 2:
            local_delay = self._estimated_compute_delay(task, self.nodes[aid], len(self.queues[aid]))
            shore_delay = self._estimated_transfer_delay(task, self.nodes[aid], self.shore, "uav_shore") + self._estimated_compute_delay(
                task, self.shore, len(self.shore_queue), shore=True
            )
            if shore_delay >= 0.8 * local_delay:
                penalty[1] = 1.0
        if role == "usv" and task.hops >= 2:
            penalty[1:3] = 0.5
        return penalty

    def _semantic_downstream_parallelism_bonus(self, aid: str, task: QueuedTask | None) -> np.ndarray:
        """Reward routes that use idle downstream capacity when the current node is backed up."""
        bonus = np.zeros(self.action_dim, dtype=np.float32)
        if task is None:
            return bonus
        node = self.nodes[aid]
        own_pressure = min(1.0, len(self.queues[aid]) / max(1.0, 0.5 * self.queue_capacity))
        urgency = float(np.clip(task.elapsed / max(task.deadline, 1e-6), 0.0, 1.0))
        if node.role == "auv":
            for action_index in (1, 2):
                usvs = self._role_ids("usv")
                if not usvs:
                    continue
                usv_id = usvs[min(action_index - 1, len(usvs) - 1)]
                bonus[action_index] = 0.5 * self._downstream_idle_compute_bonus(usv_id, task) + 0.3 * urgency
        elif node.role == "usv":
            uavs = self._role_ids("uav")
            if uavs:
                best_uav = min(
                    (self.nodes[uav_id] for uav_id in uavs),
                    key=lambda uav: self._estimated_transfer_delay(task, node, uav, "usv_uav")
                    + self._estimated_compute_delay(task, uav, len(self.queues[uav.node_id])),
                )
                best_uav_idle = 1.0 - len(self.queues[best_uav.node_id]) / max(1.0, 0.5 * self.queue_capacity)
                capacity_ratio = best_uav.cpu_capacity / max(node.cpu_capacity, 1e-6)
                shore_pressure = min(1.0, len(self.shore_queue) / max(1.0, 0.5 * self.queue_capacity))
                uav_delay = self._estimated_transfer_delay(task, node, best_uav, "usv_uav") + self._estimated_compute_delay(
                    task, best_uav, len(self.queues[best_uav.node_id])
                )
                shore_delay = self._estimated_transfer_delay(task, node, self.shore, "usv_shore") + self._estimated_compute_delay(
                    task, self.shore, len(self.shore_queue), shore=True
                )
                delay_competitiveness = max(0.0, (1.15 * shore_delay - uav_delay) / max(shore_delay, 1e-6))
                necessity_gate = max(delay_competitiveness, shore_pressure)
                bonus[1] = np.clip(
                    (0.45 * own_pressure + 0.35 * best_uav_idle + 0.20 * min(1.0, capacity_ratio))
                    * necessity_gate,
                    0.0,
                    1.0,
                )
            if len(self.shore_queue) < len(self.queues[aid]):
                shore_idle = 1.0 - len(self.shore_queue) / max(1.0, 0.5 * self.queue_capacity)
                bonus[2] = np.clip(0.35 * own_pressure + 0.25 * shore_idle, 0.0, 0.8)
        else:
            shore_idle = 1.0 - len(self.shore_queue) / max(1.0, 0.5 * self.queue_capacity)
            bonus[1] = np.clip(0.4 * own_pressure + 0.3 * shore_idle, 0.0, 0.8)
        return bonus.astype(np.float32)

    def _downstream_idle_compute_bonus(self, node_id: str, task: QueuedTask) -> float:
        node = self.nodes[node_id]
        local_idle = 1.0 - len(self.queues[node_id]) / max(1.0, 0.5 * self.queue_capacity)
        local_capacity = min(1.0, node.cpu_capacity / max(task.remaining_cycles, 1e-6))
        best = 0.4 * max(0.0, local_idle) + 0.3 * local_capacity
        if node.role == "usv":
            for uav_id in self._role_ids("uav"):
                uav = self.nodes[uav_id]
                uav_idle = 1.0 - len(self.queues[uav_id]) / max(1.0, 0.5 * self.queue_capacity)
                uav_capacity = min(1.0, uav.cpu_capacity / max(task.remaining_cycles, 1e-6))
                best = max(best, 0.5 * max(0.0, uav_idle) + 0.3 * uav_capacity)
            shore_idle = 1.0 - len(self.shore_queue) / max(1.0, 0.5 * self.queue_capacity)
            best = max(best, 0.45 * max(0.0, shore_idle) + 0.3)
        return float(np.clip(best, 0.0, 1.0))

    def _semantic_downstream_components(self, aid: str, task: QueuedTask | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate route-to-completion pressure for each valid route.

        These estimates are semantic side-channel features only. They do not
        execute the environment or access future rollout outcomes.
        """
        zeros = np.zeros(self.action_dim, dtype=np.float32)
        if task is None:
            return zeros, zeros, zeros
        node = self.nodes[aid]
        unavailable = self.route_delay_normalizer
        raw_delay = np.full(self.action_dim, unavailable, dtype=np.float32)
        raw_energy = np.ones(self.action_dim, dtype=np.float32)
        raw_queue = np.ones(self.action_dim, dtype=np.float32)

        if node.role == "auv":
            raw_delay[0] = self._estimated_compute_delay(task, node, len(self.queues[aid]))
            raw_energy[0] = self._estimated_compute_energy(task, node)
            raw_queue[0] = len(self.queues[aid]) / max(1, self.queue_capacity)
            usvs = self._role_ids("usv")
            for action_index in (1, 2):
                if not usvs:
                    continue
                usv = self.nodes[usvs[min(action_index - 1, len(usvs) - 1)]]
                transfer_delay = self._estimated_transfer_delay(task, node, usv, "auv_usv")
                transfer_energy = self._estimated_transfer_energy(task, "auv_usv")
                completion_delay, completion_energy, completion_queue = self._best_downstream_completion_from_node(
                    task, usv.node_id
                )
                raw_delay[action_index] = transfer_delay + completion_delay
                raw_energy[action_index] = transfer_energy + completion_energy
                raw_queue[action_index] = max(
                    len(self.queues[usv.node_id]) / max(1, self.queue_capacity),
                    completion_queue,
                )
        elif node.role == "usv":
            raw_delay[0] = self._estimated_compute_delay(task, node, len(self.queues[aid]))
            raw_energy[0] = self._estimated_compute_energy(task, node)
            raw_queue[0] = len(self.queues[aid]) / max(1, self.queue_capacity)
            uavs = self._role_ids("uav")
            if uavs:
                uav = min((self.nodes[target] for target in uavs), key=lambda x: np.linalg.norm(node.position - x.position))
                raw_delay[1] = (
                    self._estimated_transfer_delay(task, node, uav, "usv_uav")
                    + self._estimated_compute_delay(task, uav, len(self.queues[uav.node_id]))
                )
                raw_energy[1] = self._estimated_transfer_energy(task, "usv_uav") + self._estimated_compute_energy(task, uav)
                raw_queue[1] = len(self.queues[uav.node_id]) / max(1, self.queue_capacity)
            raw_delay[2] = (
                self._estimated_transfer_delay(task, node, self.shore, "usv_shore")
                + self._estimated_compute_delay(task, self.shore, len(self.shore_queue), shore=True)
            )
            raw_energy[2] = self._estimated_transfer_energy(task, "usv_shore") + self._estimated_compute_energy(
                task, self.shore, shore=True
            )
            raw_queue[2] = len(self.shore_queue) / max(1, self.queue_capacity)
        else:
            raw_delay[0] = self._estimated_compute_delay(task, node, len(self.queues[aid]))
            raw_energy[0] = self._estimated_compute_energy(task, node)
            raw_queue[0] = len(self.queues[aid]) / max(1, self.queue_capacity)
            raw_delay[1] = (
                self._estimated_transfer_delay(task, node, self.shore, "uav_shore")
                + self._estimated_compute_delay(task, self.shore, len(self.shore_queue), shore=True)
            )
            raw_energy[1] = self._estimated_transfer_energy(task, "uav_shore") + self._estimated_compute_energy(
                task, self.shore, shore=True
            )
            raw_queue[1] = len(self.shore_queue) / max(1, self.queue_capacity)

        delay = np.clip(raw_delay / max(1e-6, self.route_delay_normalizer), 0.0, 1.0)
        energy_normalizer = max(float(np.max(raw_energy[:3])), 1e-8)
        energy = np.clip(raw_energy / energy_normalizer, 0.0, 1.0)
        queue = np.clip(raw_queue, 0.0, 1.0)
        return delay.astype(np.float32), energy.astype(np.float32), queue.astype(np.float32)

    def _best_downstream_completion_from_node(self, task: QueuedTask, node_id: str) -> tuple[float, float, float]:
        node = self.nodes[node_id]
        if node.role == "usv":
            options: list[tuple[float, float, float]] = [
                (
                    self._estimated_compute_delay(task, node, len(self.queues[node_id])),
                    self._estimated_compute_energy(task, node),
                    len(self.queues[node_id]) / max(1, self.queue_capacity),
                )
            ]
            for uav_id in self._role_ids("uav"):
                uav = self.nodes[uav_id]
                options.append(
                    (
                        self._estimated_transfer_delay(task, node, uav, "usv_uav")
                        + self._estimated_compute_delay(task, uav, len(self.queues[uav_id])),
                        self._estimated_transfer_energy(task, "usv_uav") + self._estimated_compute_energy(task, uav),
                        len(self.queues[uav_id]) / max(1, self.queue_capacity),
                    )
                )
            options.append(
                (
                    self._estimated_transfer_delay(task, node, self.shore, "usv_shore")
                    + self._estimated_compute_delay(task, self.shore, len(self.shore_queue), shore=True),
                    self._estimated_transfer_energy(task, "usv_shore") + self._estimated_compute_energy(task, self.shore, shore=True),
                    len(self.shore_queue) / max(1, self.queue_capacity),
                )
            )
            return min(options, key=lambda item: item[0])
        if node.role == "uav":
            local = (
                self._estimated_compute_delay(task, node, len(self.queues[node_id])),
                self._estimated_compute_energy(task, node),
                len(self.queues[node_id]) / max(1, self.queue_capacity),
            )
            shore = (
                self._estimated_transfer_delay(task, node, self.shore, "uav_shore")
                + self._estimated_compute_delay(task, self.shore, len(self.shore_queue), shore=True),
                self._estimated_transfer_energy(task, "uav_shore") + self._estimated_compute_energy(task, self.shore, shore=True),
                len(self.shore_queue) / max(1, self.queue_capacity),
            )
            return min([local, shore], key=lambda item: item[0])
        return (
            self._estimated_compute_delay(task, node, len(self.queues[node_id])),
            self._estimated_compute_energy(task, node),
            len(self.queues[node_id]) / max(1, self.queue_capacity),
        )

    def _semantic_route_energy_costs(self, aid: str, task: QueuedTask) -> np.ndarray:
        """Estimate immediate action energy for teacher calibration using current state only."""
        role = self.nodes[aid].role
        compute_energy = 0.04 * min(task.remaining_cycles, self.nodes[aid].cpu_capacity * self.compute_service_scale)
        transfer_energy = 0.03 * task.data_size * 1.5
        if role == "auv":
            raw = np.array([compute_energy, transfer_energy, transfer_energy, 0.0], dtype=np.float32)
        elif role == "usv":
            raw = np.array([compute_energy, transfer_energy, transfer_energy, 0.0], dtype=np.float32)
        else:
            raw = np.array([compute_energy, transfer_energy, 0.0, 0.0], dtype=np.float32)
        normalizer = max(float(np.max(raw[:3])), 1e-8)
        return (raw / normalizer).astype(np.float32)

    def _semantic_key_tx_energy_costs(self, aid: str, task: QueuedTask | None) -> np.ndarray:
        """Estimate the paper's primary mobile transmission energy per action."""
        raw = np.zeros(self.action_dim, dtype=np.float32)
        if task is None:
            return raw
        role = self.nodes[aid].role
        if role == "auv":
            raw[1:3] = self._estimated_transfer_energy(task, "auv_usv")
        elif role == "uav":
            raw[1] = self._estimated_transfer_energy(task, "uav_shore")
        normalizer = max(float(np.max(raw)), 1e-8)
        return (raw / normalizer).astype(np.float32)

    def _deadline_capacity_route_gate(self, aid: str, task: QueuedTask | None) -> np.ndarray:
        """Softly penalize long/downstream routes under deadline or capacity pressure."""
        penalty = np.zeros(self.action_dim, dtype=np.float32)
        if task is None:
            return penalty
        role = self.nodes[aid].role
        slack = max(task.deadline - task.elapsed, 1e-6)
        deadline_risk = float(np.clip(1.0 - slack / max(task.deadline, 1e-6), 0.0, 1.0))
        own_pressure = float(len(self.queues[aid]) / max(1, self.queue_capacity))
        _, _, downstream_pressure = self._next_hop_features(aid)
        pressure = float(np.clip(0.45 * deadline_risk + 0.35 * downstream_pressure + 0.20 * own_pressure, 0.0, 1.0))
        if role == "auv":
            penalty[1:3] = pressure
        elif role == "usv":
            penalty[1:3] = pressure
        else:
            penalty[1] = pressure
        return penalty.astype(np.float32)

    @staticmethod
    def _estimated_transfer_energy(task: QueuedTask, link_type: str) -> float:
        return float(0.03 * task.data_size * 1.5)

    @staticmethod
    def _estimated_compute_energy(task: QueuedTask, node: NodeState, shore: bool = False) -> float:
        rate = 0.025 if shore or node.role == "shore" else 0.04
        return float(rate * task.remaining_cycles)

    def _task_route_compatibility(self, aid: str, task_type: int) -> np.ndarray:
        """Return task-aware route preferences in action order for an agent role."""
        role = self.nodes[aid].role
        profiles = {
            "auv": {
                "tracking_update": [0.16, 0.18, 0.18, 0.0],
                "anomaly_detection": [0.10, 0.17, 0.17, 0.0],
                "sonar_recognition": [-0.06, 0.24, 0.24, 0.0],
                "path_replanning": [0.15, 0.19, 0.19, 0.0],
            },
            "usv": {
                "tracking_update": [0.20, 0.14, 0.04, 0.0],
                "anomaly_detection": [0.14, 0.18, 0.12, 0.0],
                "sonar_recognition": [0.02, 0.20, 0.25, 0.0],
                "path_replanning": [0.20, 0.16, 0.04, 0.0],
            },
            "uav": {
                "tracking_update": [0.20, 0.04, 0.0, 0.0],
                "anomaly_detection": [0.18, 0.10, 0.0, 0.0],
                "sonar_recognition": [0.14, 0.22, 0.0, 0.0],
                "path_replanning": [0.20, 0.05, 0.0, 0.0],
            },
        }
        task_name = TASK_TYPES[int(task_type)]
        return np.asarray(profiles[role][task_name], dtype=np.float32)

    def _generate_arrivals(self) -> None:
        for aid in self._role_ids("auv"):
            if self.rng.random() < self.task_arrival_rate:
                self._enqueue_generated_task(aid)

    def _enqueue_generated_task(self, aid: str) -> None:
        self.task_counter += 1
        source = generate_task(self.task_counter, aid, self.rng)
        task = QueuedTask(
            task_id=source.task_id,
            source_agent_id=aid,
            task_type=source.task_type,
            data_size=float(np.clip(source.task_data_size * self.task_data_scale, 0.02, 1.5)),
            remaining_cycles=float(np.clip(source.task_cpu_cycles * self.task_cpu_scale, 0.01, 2.0)),
            deadline=float(np.clip(source.task_deadline * self.deadline_scale, 0.2, 8.0)),
            priority=source.task_priority,
        )
        if len(self.queues[aid]) >= self.queue_capacity:
            self.dropped_total += 1
            return
        self.queues[aid].append(task)
        self.generated_total += 1

    def _service_auv(self, aid: str, raw_action: np.ndarray | None, transitions: dict[str, float]) -> tuple[int, float]:
        queue = self.queues[aid]
        if not queue:
            return 0, 0.0
        if self.action_mode == "discrete_route":
            route = self._discrete_route(raw_action)
            if route == 3:
                return 0, 0.0
            if route == 0:
                completed, energy = self._compute_head(aid, 1.0)
                if completed:
                    transitions["local_computed_tasks"] += float(completed)
                return completed, energy
            target_ids = self._role_ids("usv")
            if not target_ids:
                return 0, 0.0
            target = target_ids[min(route - 1, len(target_ids) - 1)]
            moved, energy = self._transfer_head(aid, target, "auv_usv", 1.0)
            if moved:
                transitions["auv_to_usv_tasks"] += 1.0
                self.first_hop_offloaded_total += 1
            return 0, energy
        action = self._action(raw_action)
        task = queue[0]
        if action[0] >= action[1]:
            completed, energy = self._compute_head(aid, max(action[0], 0.05))
            if completed:
                transitions["local_computed_tasks"] += float(completed)
            return completed, energy
        target_ids = self._role_ids("usv")
        if not target_ids:
            return 0, 0.0
        target = target_ids[min(int(action[2] * len(target_ids)), len(target_ids) - 1)]
        moved, energy = self._transfer_head(aid, target, "auv_usv", max(action[3], 0.1))
        if moved:
            transitions["auv_to_usv_tasks"] += 1.0
            self.first_hop_offloaded_total += 1
        return 0, energy

    def _service_usv(self, aid: str, raw_action: np.ndarray | None, transitions: dict[str, float]) -> tuple[int, float]:
        if not self.queues[aid]:
            return 0, 0.0
        if self.action_mode == "discrete_route":
            mode = self._discrete_route(raw_action)
            if mode == 3:
                return 0, 0.0
            power = 1.0
        else:
            action = self._action(raw_action)
            mode = int(np.argmax(action[:3]))
            power = max(action[3], 0.1)
        if mode == 0:
            effort = 1.0 if self.action_mode == "discrete_route" else max(action[0], 0.05)
            completed, energy = self._compute_head(aid, effort)
            if completed:
                transitions["edge_computed_tasks"] += float(completed)
            return completed, energy
        if mode == 1 and self._role_ids("uav"):
            candidates = self._role_ids("uav")
            if self.action_mode == "discrete_route":
                target = min(candidates, key=lambda target_id: len(self.queues[target_id]))
            else:
                target = candidates[min(int(action[3] * len(candidates)), len(candidates) - 1)]
            moved, energy = self._transfer_head(aid, target, "usv_uav", power)
            if moved:
                transitions["usv_to_uav_tasks"] += 1.0
            return 0, energy
        moved, energy = self._transfer_head(aid, "shore_0", "usv_shore", power)
        if moved:
            transitions["usv_to_shore_tasks"] += 1.0
        return 0, energy

    def _service_uav(self, aid: str, raw_action: np.ndarray | None, transitions: dict[str, float]) -> tuple[int, float]:
        if not self.queues[aid]:
            return 0, 0.0
        if self.action_mode == "discrete_route":
            route = self._discrete_route(raw_action)
            if route >= 2:
                return 0, 0.0
            if route == 0:
                completed, energy = self._compute_head(aid, 1.0)
                if completed:
                    transitions["edge_computed_tasks"] += float(completed)
                return completed, energy
            moved, energy = self._transfer_head(aid, "shore_0", "uav_shore", 1.0)
            if moved:
                transitions["uav_to_shore_tasks"] += 1.0
            return 0, energy
        action = self._action(raw_action)
        if action[0] >= action[1]:
            completed, energy = self._compute_head(aid, max(action[2], 0.1))
            if completed:
                transitions["edge_computed_tasks"] += float(completed)
            return completed, energy
        moved, energy = self._transfer_head(aid, "shore_0", "uav_shore", max(action[3], 0.1))
        if moved:
            transitions["uav_to_shore_tasks"] += 1.0
        return 0, energy

    def _service_shore(self, transitions: dict[str, float]) -> tuple[int, float]:
        if not self.shore_queue:
            return 0, 0.0
        budget = self.shore.cpu_capacity * self.compute_service_scale
        completed = 0
        energy = 0.0
        queue_delay = 0.02 + 0.02 * len(self.shore_queue)
        while self.shore_queue and budget > 1e-8:
            task = self.shore_queue[0]
            consumed = min(task.remaining_cycles, budget)
            task.remaining_cycles -= consumed
            budget -= consumed
            task.elapsed += queue_delay
            used_energy = 0.025 * consumed
            task.energy += used_energy
            energy += used_energy
            self._record_energy("energy_compute_shore", used_energy)
            if task.remaining_cycles <= 1e-8:
                self.shore_queue.popleft()
                self._record_completion(task)
                completed += 1
            else:
                break
        transitions["edge_computed_tasks"] += float(completed)
        return completed, energy

    def _compute_head(self, aid: str, effort: float) -> tuple[int, float]:
        node = self.nodes[aid]
        queue = self.queues[aid]
        budget = node.cpu_capacity * self.compute_service_scale * float(np.clip(effort, 0.05, 1.0))
        completed = 0
        energy = 0.0
        queue_delay = 0.03 + 0.04 * len(queue)
        while queue and budget > 1e-8:
            task = queue[0]
            consumed = min(task.remaining_cycles, budget)
            task.remaining_cycles -= consumed
            budget -= consumed
            task.elapsed += queue_delay
            used_energy = 0.04 * consumed
            task.energy += used_energy
            energy += used_energy
            self._record_energy(f"energy_compute_{node.role}", used_energy)
            if task.remaining_cycles <= 1e-8:
                queue.popleft()
                self._record_completion(task)
                completed += 1
            else:
                break
        node.cpu_load = float(np.clip(0.7 * node.cpu_load + 0.3 * effort, 0.0, 1.0))
        node.remaining_energy = float(np.clip(node.remaining_energy - energy * 0.01, 0.0, 1.0))
        return completed, energy

    def _transfer_head(self, src_id: str, dst_id: str, link_type: str, power: float) -> tuple[bool, float]:
        source_queue = self.queues[src_id]
        destination = self.shore_queue if dst_id == "shore_0" else self.queues[dst_id]
        if not source_queue or len(destination) >= self.queue_capacity:
            return False, 0.0
        src = self.nodes[src_id]
        dst = self.shore if dst_id == "shore_0" else self.nodes[dst_id]
        rate, loss, prop = self._link_features(src, dst, link_type)
        task = source_queue[0]
        effective_rate = max(0.02, rate * (0.5 + 0.5 * power))
        transfer_delay = task.data_size / effective_rate + prop
        energy = 0.03 * task.data_size * (0.5 + power)
        self._record_energy(f"energy_transfer_{link_type}", energy)
        # Link loss raises delay/cost; only a severe outage blocks transfer.
        if loss >= 0.85:
            task.elapsed += self.slot_duration
            task.energy += energy
            return False, energy
        source_queue.popleft()
        task.elapsed += transfer_delay
        task.energy += energy
        task.hops += 1
        destination.append(task)
        return True, energy

    def _age_and_expire_tasks(self) -> tuple[int, int]:
        timeouts = 0
        drops = 0
        for queue in list(self.queues.values()) + [self.shore_queue]:
            kept: deque[QueuedTask] = deque()
            while queue:
                task = queue.popleft()
                task.elapsed += self.slot_duration
                if task.elapsed > task.deadline:
                    self.timeout_task_energy_total += task.energy
                    timeouts += 1
                else:
                    kept.append(task)
            queue.extend(kept)
        return timeouts, drops

    def _record_completion(self, task: QueuedTask) -> None:
        self.completed_delays.append(task.elapsed)
        self.completed_energy.append(task.energy)
        self.completed_task_energy_total += task.energy

    def _record_energy(self, component: str, value: float) -> None:
        amount = float(value)
        self.total_consumed_energy += amount
        self.energy_breakdown_total[component] = self.energy_breakdown_total.get(component, 0.0) + amount

    def _inflight_energy(self) -> float:
        tasks = [task for queue in list(self.queues.values()) + [self.shore_queue] for task in queue]
        return float(sum(task.energy for task in tasks))

    def _update_virtual_queues(self) -> None:
        for aid in self.agent_ids:
            backlog = float(len(self.queues[aid]))
            self.virtual_queues[aid] = max(0.0, self.virtual_queues[aid] + backlog - self.virtual_queue_threshold)

    def _metrics_and_reward(
        self,
        transitions: dict[str, float],
        step_completed: int,
        step_timeout: int,
        step_dropped: int,
        step_energy: float,
    ) -> tuple[dict[str, float], float]:
        auv_q = np.array([len(self.queues[aid]) for aid in self._role_ids("auv")], dtype=np.float32)
        usv_q = np.array([len(self.queues[aid]) for aid in self._role_ids("usv")], dtype=np.float32)
        uav_q = np.array([len(self.queues[aid]) for aid in self._role_ids("uav")], dtype=np.float32)
        all_q = np.concatenate([auv_q, usv_q, uav_q, np.array([len(self.shore_queue)], dtype=np.float32)])
        backlog = float(all_q.mean() / max(1, self.queue_capacity))
        weighted_backlog = (
            self.backlog_role_weights["auv"] * float(np.sum(auv_q))
            + self.backlog_role_weights["usv"] * float(np.sum(usv_q))
            + self.backlog_role_weights["uav"] * float(np.sum(uav_q))
            + self.backlog_role_weights["shore"] * float(len(self.shore_queue))
        ) / max(1.0, float(self.n_auv))
        delay = float(np.mean(self.completed_delays[-max(1, step_completed) :])) if step_completed else backlog
        reward_completion = float(step_completed) / max(1, self.n_auv)
        reward_backlog = -self.backlog_penalty_weight * weighted_backlog
        reward_delay = -0.20 * float(np.clip(delay / 4.0, 0.0, 1.0))
        reward_deadline = -1.0 * float(step_timeout) / max(1, self.n_auv)
        reward_energy = -0.10 * float(np.clip(step_energy, 0.0, 2.0))
        reward_drop = -0.50 * float(step_dropped) / max(1, self.n_auv)
        route_progress_per_step = float(
            transitions["auv_to_usv_tasks"]
            + transitions["usv_to_uav_tasks"]
            + transitions["usv_to_shore_tasks"]
            + transitions["uav_to_shore_tasks"]
        )
        reward_route_progress = 0.0
        if self.use_route_progress_reward:
            reward_route_progress = sum(
                self.route_progress_reward_weights[key] * transitions[key]
                for key in self.route_progress_reward_weights
            ) / max(1, self.n_auv)
        reward = float(
            np.clip(
                reward_completion
                + reward_route_progress
                + reward_backlog
                + reward_delay
                + reward_deadline
                + reward_energy
                + reward_drop,
                -2.0,
                2.0,
            )
        )
        mean_energy = float(np.mean(self.completed_energy)) if self.completed_energy else float(step_energy)
        inflight_energy = self._inflight_energy()
        key_tx_energy = float(
            self.energy_breakdown_total.get("energy_transfer_auv_usv", 0.0)
            + self.energy_breakdown_total.get("energy_transfer_uav_shore", 0.0)
        )
        excluded_usv_tx_energy = float(
            self.energy_breakdown_total.get("energy_transfer_usv_uav", 0.0)
            + self.energy_breakdown_total.get("energy_transfer_usv_shore", 0.0)
        )
        energy_balance_error = abs(
            self.total_consumed_energy
            - self.completed_task_energy_total
            - self.timeout_task_energy_total
            - self.dropped_task_energy_total
            - inflight_energy
        )
        metrics = {
            "completion_ratio": self.completed_total / max(1, self.generated_total),
            "mean_service_delay": float(np.mean(self.completed_delays)) if self.completed_delays else delay,
            "deadline_violation_rate": self.timeout_total / max(1, self.generated_total),
            "mean_energy_cost": mean_energy,
            "total_consumed_energy": self.total_consumed_energy,
            "completed_task_energy": self.completed_task_energy_total,
            "timeout_task_energy": self.timeout_task_energy_total,
            "dropped_task_energy": self.dropped_task_energy_total,
            "inflight_task_energy": inflight_energy,
            "energy_per_generated_task": self.total_consumed_energy / max(1, self.generated_total),
            "energy_per_completed_task": self.completed_task_energy_total / max(1, self.completed_total),
            "energy_per_successful_completion": self.total_consumed_energy / max(1, self.completed_total),
            "key_tx_energy": key_tx_energy,
            "key_tx_energy_per_successful_completion": key_tx_energy / max(1, self.completed_total),
            "excluded_usv_tx_energy": excluded_usv_tx_energy,
            "energy_accounting_balance_error": energy_balance_error,
            "offload_success_rate": (
                self.first_hop_offloaded_total / max(1.0, float(self.generated_total))
            ),
            "load_balance_index": float(np.std(all_q) / (np.mean(all_q) + 1e-6)) if float(np.mean(all_q)) > 0 else 0.0,
            "mean_queue_length": float(np.mean(all_q)),
            "generated_tasks": float(self.generated_total),
            "completed_tasks": float(self.completed_total),
            "timeout_tasks": float(self.timeout_total),
            "dropped_tasks": float(self.dropped_total),
            "semantic_match_rate": 0.0,
            "primary_route_ratio_mean": 0.0,
            "auv_queue_mean": float(np.mean(auv_q)) if auv_q.size else 0.0,
            "usv_queue_mean": float(np.mean(usv_q)) if usv_q.size else 0.0,
            "uav_queue_mean": float(np.mean(uav_q)) if uav_q.size else 0.0,
            "shore_queue_length": float(len(self.shore_queue)),
            "virtual_queue_mean": float(np.mean(list(self.virtual_queues.values()))) if self.virtual_queues else 0.0,
            "weighted_backlog_cost": float(weighted_backlog),
            "first_hop_offloaded_tasks": float(self.first_hop_offloaded_total),
            "reward_completion": reward_completion,
            "reward_route_progress": reward_route_progress,
            "reward_backlog": reward_backlog,
            "reward_delay": reward_delay,
            "reward_deadline": reward_deadline,
            "reward_energy": reward_energy,
            "reward_drop": reward_drop,
            "reward_invalid_action": 0.0,
            "reward_queue_overflow": reward_drop,
            "reward_total_mean": reward,
            "route_progress_per_step": route_progress_per_step,
            **self.energy_breakdown_total,
            **transitions,
        }
        return metrics, reward

    def _role_ids(self, role: str) -> list[str]:
        return [aid for aid, node in self.nodes.items() if node.role == role]

    @staticmethod
    def _action(action: np.ndarray | None) -> np.ndarray:
        if action is None:
            return np.zeros(4, dtype=np.float32)
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size < 4:
            values = np.pad(values, (0, 4 - values.size))
        return np.clip(values[:4], 0.0, 1.0)

    def _discrete_route(self, action: int | np.ndarray | None) -> int:
        values = np.asarray(0 if action is None else action).reshape(-1)
        return int(np.clip(int(values[0]), 0, self.action_dim - 1))

    def _next_hop_features(self, aid: str) -> tuple[float, float, float]:
        node = self.nodes[aid]
        if node.role == "auv":
            targets = self._role_ids("usv")
            link_type = "auv_usv"
        elif node.role == "usv":
            targets = self._role_ids("uav")
            link_type = "usv_uav"
        else:
            targets = []
            link_type = "uav_shore"
        if targets:
            dst = min((self.nodes[target] for target in targets), key=lambda x: np.linalg.norm(node.position - x.position))
            queue = len(self.queues[dst.node_id]) / max(1, self.queue_capacity)
        else:
            dst = self.shore
            queue = len(self.shore_queue) / max(1, self.queue_capacity)
        rate, loss, _ = self._link_features(node, dst, link_type)
        return rate, loss, float(queue)

    def _route_delay_estimates(self, aid: str, task: QueuedTask | None) -> np.ndarray:
        """Estimate each role's available execution routes from current state."""
        if task is None:
            return np.zeros(3, dtype=np.float32)
        node = self.nodes[aid]
        unavailable = self.route_delay_normalizer
        if node.role == "auv":
            local = self._estimated_compute_delay(task, node, len(self.queues[aid]))
            usvs = self._role_ids("usv")
            if not usvs:
                routes = [local, unavailable, unavailable]
            else:
                access_routes = []
                for index in range(2):
                    usv = self.nodes[usvs[min(index, len(usvs) - 1)]]
                    access_routes.append(
                        self._estimated_transfer_delay(task, node, usv, "auv_usv")
                        + self._estimated_compute_delay(task, usv, len(self.queues[usv.node_id]))
                    )
                routes = [local, *access_routes]
        elif node.role == "usv":
            local = self._estimated_compute_delay(task, node, len(self.queues[aid]))
            uavs = self._role_ids("uav")
            if uavs:
                uav = min((self.nodes[target] for target in uavs), key=lambda x: np.linalg.norm(node.position - x.position))
                via_uav = self._estimated_transfer_delay(task, node, uav, "usv_uav") + self._estimated_compute_delay(
                    task, uav, len(self.queues[uav.node_id])
                )
            else:
                via_uav = unavailable
            via_shore = self._estimated_transfer_delay(task, node, self.shore, "usv_shore") + self._estimated_compute_delay(
                task, self.shore, len(self.shore_queue), shore=True
            )
            routes = [local, via_uav, via_shore]
        else:
            local = self._estimated_compute_delay(task, node, len(self.queues[aid]))
            via_shore = self._estimated_transfer_delay(task, node, self.shore, "uav_shore") + self._estimated_compute_delay(
                task, self.shore, len(self.shore_queue), shore=True
            )
            routes = [local, via_shore, unavailable]
        return np.clip(np.asarray(routes, dtype=np.float32) / max(1e-6, self.route_delay_normalizer), 0.0, 1.0)

    def _estimated_compute_delay(self, task: QueuedTask, node: NodeState, queue_len: int, shore: bool = False) -> float:
        service_rate = max(0.02, node.cpu_capacity * self.compute_service_scale)
        queue_delay = (0.02 + 0.02 * (queue_len + 1)) if shore else (0.03 + 0.04 * (queue_len + 1))
        return float(task.remaining_cycles / service_rate + queue_delay)

    def _estimated_transfer_delay(self, task: QueuedTask, src: NodeState, dst: NodeState, link_type: str) -> float:
        rate, _, propagation = self._link_features(src, dst, link_type)
        effective_rate = max(0.02, rate * 0.8)
        return float(task.data_size / effective_rate + propagation)

    def _link_features(self, src: NodeState, dst: NodeState, link_type: str) -> tuple[float, float, float]:
        distance = float(np.linalg.norm(src.position - dst.position))
        if link_type == "auv_usv":
            base_rate, base_loss, prop = 0.36, 0.12, 0.12
        elif link_type in {"usv_uav", "usv_shore"}:
            base_rate, base_loss, prop = 1.10, 0.035, 0.025
        else:
            base_rate, base_loss, prop = 1.45, 0.02, 0.015
        difficulty_multiplier = {"easy": 1.2, "medium": 1.0, "hard": 0.72}.get(self.difficulty, 1.0)
        rate = float(np.clip(base_rate * difficulty_multiplier / (1.0 + distance), 0.02, 2.0))
        loss = float(np.clip(base_loss + 0.12 * distance, 0.0, 0.95))
        propagation = float(prop + 0.04 * distance)
        return rate, loss, propagation
