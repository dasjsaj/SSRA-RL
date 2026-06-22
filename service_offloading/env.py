"""Simplified UAV/USV/AUV edge offloading MARL environment.

The environment is independent of Tracking/AUV6DOF and uses abstract motion,
link, queue, and service-computing models designed for fast algorithm work.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .scenario import ROLES, TASK_TYPES, NodeState, ServiceTask, generate_task, make_node
from .semantic import ROLE_TO_ID, semantic_features, semantic_prior_action


class CrossDomainServiceOffloadingEnv:
    action_dim = 6

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        env_cfg = self.config.get("env", self.config)
        self.env_cfg = env_cfg
        self.n_auv = int(env_cfg.get("n_auv", 4))
        self.n_usv = int(env_cfg.get("n_usv", 2))
        self.n_uav = int(env_cfg.get("n_uav", 2))
        self.episode_length = int(env_cfg.get("episode_length", 100))
        self.difficulty = str(env_cfg.get("difficulty", "medium")).lower()
        self.action_mode = str(env_cfg.get("action_mode", "full")).lower()
        self.obs_mode = str(env_cfg.get("obs_mode", "rich")).lower()
        self.reward_mode = str(env_cfg.get("reward_mode", "original")).lower()
        self.use_mobility_control = bool(env_cfg.get("use_mobility_control", True))
        self.use_semantic_side_channel = bool(env_cfg.get("use_semantic_side_channel", self.obs_mode != "minimal"))
        self.task_profile_mode = str(
            env_cfg.get("task_profile_mode", "easy_two_types" if self.difficulty == "easy" else "full")
        ).lower()
        self.task_data_scale = float(env_cfg.get("task_data_scale", 1.0))
        self.task_cpu_scale = float(env_cfg.get("task_cpu_scale", 1.0))
        self.task_type_data_scales = self._fit_task_type_scales(env_cfg.get("task_type_data_scales"))
        self.task_type_cpu_scales = self._fit_task_type_scales(env_cfg.get("task_type_cpu_scales"))
        self.auv_cpu_scale = float(env_cfg.get("auv_cpu_scale", 1.0))
        self.edge_cpu_scale = float(env_cfg.get("edge_cpu_scale", 1.0))
        self.local_execution_delay_scale = float(env_cfg.get("local_execution_delay_scale", 0.35))
        self.transmission_delay_scale = float(env_cfg.get("transmission_delay_scale", 0.45))
        self.edge_execution_delay_scale = float(env_cfg.get("edge_execution_delay_scale", 0.35))
        default_arrival = {"easy": 0.20, "medium": 0.55, "hard": 0.85}.get(self.difficulty, 0.55)
        default_queue = {"easy": 20.0, "medium": 14.0, "hard": 10.0}.get(self.difficulty, 14.0)
        self.task_generation_prob = float(env_cfg.get("task_generation_prob", default_arrival))
        self.max_queue = float(env_cfg.get("max_queue", default_queue))
        default_focus = {"easy": 0.52, "medium": 0.58, "hard": 0.64}.get(self.difficulty, 0.58)
        self.min_primary_route_ratio = float(env_cfg.get("min_primary_route_ratio", default_focus))
        self.deadline_scale = float(env_cfg.get("deadline_scale", 1.0))
        self.action_dim = 4 if self.action_mode == "simple" else 6
        self.seed_value = 0
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.task_counter = 0
        self.nodes: dict[str, NodeState] = {}
        self.agent_ids: list[str] = []
        self.pending_tasks: dict[str, ServiceTask | None] = {}
        self.shore = make_node("shore_0", "shore", self.rng)
        self.semantic_dim = 10 if self.use_semantic_side_channel else 0
        self.obs_dim = 17 if self.obs_mode == "minimal" else 43
        self.global_extra_dim = 5 if self.obs_mode == "minimal" else 0
        self.global_state_dim = self.obs_dim * (self.n_auv + self.n_usv + self.n_uav) + self.global_extra_dim

    @staticmethod
    def _fit_task_type_scales(scales) -> np.ndarray:
        if scales is None:
            return np.ones(len(TASK_TYPES), dtype=np.float32)
        values = np.asarray(scales, dtype=np.float32).reshape(-1)
        fitted = np.ones(len(TASK_TYPES), dtype=np.float32)
        fitted[: min(values.size, fitted.size)] = values[: fitted.size]
        return np.clip(fitted, 0.05, 5.0)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.task_counter = 0
        self.nodes = {}
        for i in range(self.n_auv):
            self.nodes[f"auv_{i}"] = make_node(f"auv_{i}", "auv", self.rng)
            self.nodes[f"auv_{i}"].cpu_capacity *= self.auv_cpu_scale
        for i in range(self.n_usv):
            self.nodes[f"usv_{i}"] = make_node(f"usv_{i}", "usv", self.rng)
            self.nodes[f"usv_{i}"].cpu_capacity *= self.edge_cpu_scale
        for i in range(self.n_uav):
            self.nodes[f"uav_{i}"] = make_node(f"uav_{i}", "uav", self.rng)
            self.nodes[f"uav_{i}"].cpu_capacity *= self.edge_cpu_scale
        self.shore = make_node("shore_0", "shore", self.rng)
        self.shore.cpu_capacity *= self.edge_cpu_scale
        self.agent_ids = list(self.nodes.keys())
        self.pending_tasks = {aid: None for aid in self.agent_ids}
        self._generate_tasks()
        return self._obs_dict(), {"agent_ids": self.agent_ids}

    def step(self, actions: dict[str, np.ndarray]):
        self.step_count += 1
        processed = []
        rewards = {}
        self._move_nodes(actions)
        for aid in self.agent_ids:
            node = self.nodes[aid]
            action = self._legalize_action(actions.get(aid, np.zeros(self.action_dim, dtype=np.float32)), node.role)
            rewards[aid] = 0.0
            if node.role == "auv":
                result = self._process_auv_task(node, action)
            else:
                result = self._process_edge_node(node, action)
            processed.append(result)
            rewards[aid] = float(result["reward"])
        self._age_and_generate()
        metrics = self._aggregate_step_metrics(processed)
        done = self.step_count >= self.episode_length
        return self._obs_dict(), rewards, done, False, {"metrics": metrics, "per_agent": processed}

    def get_global_state(self) -> np.ndarray:
        obs = self._obs_dict()
        parts = [obs[aid]["obs"] for aid in self.agent_ids]
        if self.global_extra_dim:
            parts.append(self._system_stats_vector())
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _obs_dict(self) -> dict[str, dict]:
        return {aid: self._agent_obs(aid) for aid in self.agent_ids}

    def _agent_obs(self, aid: str) -> dict:
        node = self.nodes[aid]
        task = self.pending_tasks.get(aid) if node.role == "auv" else self._nearest_auv_task(node)
        raw_dict = self._raw_feature_dict(node, task)
        if self.obs_mode == "minimal":
            raw = self._minimal_vector(raw_dict)
            if self.use_semantic_side_channel:
                sem = semantic_features(raw_dict)
                raw_dict["semantic"] = sem
                prior = self._fit_action_dim(semantic_prior_action(raw_dict))
            else:
                sem = np.zeros(0, dtype=np.float32)
                prior = np.zeros(self.action_dim, dtype=np.float32)
            return {
                "obs": raw,
                "raw": raw,
                "semantic": sem,
                "semantic_prior": prior,
                "agent_id": aid,
                "role": node.role,
                "role_id": ROLE_TO_ID[node.role],
                "task_label_success": np.float32(self._heuristic_success_label(raw_dict)),
                "task_label_deadline_violation": np.float32(self._heuristic_deadline_label(raw_dict)),
                "task_label_delay": np.float32(self._heuristic_delay_label(raw_dict)),
            }
        sem = semantic_features(raw_dict)
        raw_dict["semantic"] = sem
        prior = self._fit_action_dim(semantic_prior_action(raw_dict))
        raw = self._raw_vector(raw_dict)
        obs = np.concatenate([raw, sem], axis=0).astype(np.float32)
        return {
            "obs": obs,
            "raw": raw,
            "semantic": sem,
            "semantic_prior": prior,
            "agent_id": aid,
            "role": node.role,
            "role_id": ROLE_TO_ID[node.role],
            "task_label_success": np.float32(self._heuristic_success_label(raw_dict)),
            "task_label_deadline_violation": np.float32(self._heuristic_deadline_label(raw_dict)),
            "task_label_delay": np.float32(self._heuristic_delay_label(raw_dict)),
        }

    def _fit_action_dim(self, prior: np.ndarray) -> np.ndarray:
        fitted = np.zeros(self.action_dim, dtype=np.float32)
        n = min(self.action_dim, int(prior.shape[0]))
        fitted[:n] = prior[:n]
        return np.clip(fitted, 0.0, 1.0).astype(np.float32)

    def _raw_feature_dict(self, node: NodeState, task: ServiceTask | None) -> dict:
        task_type = task.task_type if task is not None else 0
        task_onehot = np.zeros(len(TASK_TYPES), dtype=np.float32)
        task_onehot[task_type] = 1.0
        nearest_usv = self._nearest_role(node, "usv")
        nearest_uav = self._nearest_role(node, "uav")
        auv_usv = self._link_features(node, nearest_usv, "auv_usv")
        usv_uav = self._link_features(nearest_usv or node, nearest_uav, "usv_uav")
        uav_shore = self._link_features(nearest_uav or node, self.shore, "uav_shore")
        neighbor_queue = np.mean([n.queue_length for n in self.nodes.values() if n.node_id != node.node_id]) / self.max_queue
        neighbor_load = np.mean([n.cpu_load for n in self.nodes.values() if n.node_id != node.node_id])
        neighbor_energy = np.mean([1.0 - n.remaining_energy for n in self.nodes.values() if n.node_id != node.node_id])
        return {
            "position": node.position.astype(np.float32),
            "velocity": node.velocity.astype(np.float32),
            "remaining_energy": node.remaining_energy,
            "cpu_capacity": node.cpu_capacity,
            "cpu_load": node.cpu_load,
            "queue_length": node.queue_length / self.max_queue,
            "task_data_size": 0.0 if task is None else task.task_data_size,
            "task_cpu_cycles": 0.0 if task is None else task.task_cpu_cycles,
            "task_deadline": 1.0 if task is None else task.task_deadline,
            "task_priority": 0.0 if task is None else task.task_priority,
            "task_age": 0.0 if task is None else task.task_age,
            "task_onehot": task_onehot,
            "link_rate": np.array([auv_usv[0], usv_uav[0], uav_shore[0]], dtype=np.float32),
            "packet_loss": np.array([auv_usv[1], usv_uav[1], uav_shore[1]], dtype=np.float32),
            "prop_delay": np.array([auv_usv[2], usv_uav[2], uav_shore[2]], dtype=np.float32),
            "link_reliability": float(np.clip(1.0 - np.mean([auv_usv[1], usv_uav[1], uav_shore[1]]), 0.0, 1.0)),
            "neighbor_queue_pressure": float(neighbor_queue),
            "neighbor_cpu_load": float(neighbor_load),
            "neighbor_energy_pressure": float(neighbor_energy),
            "neighbor_availability": float(np.mean([n.availability for n in self.nodes.values()])),
            "role_id": ROLE_TO_ID[node.role],
        }

    def _minimal_vector(self, raw: dict) -> np.ndarray:
        role_onehot = np.zeros(3, dtype=np.float32)
        role_id = int(raw["role_id"])
        if role_id < 3:
            role_onehot[role_id] = 1.0
        task_deadline = float(np.clip(raw["task_deadline"] / 2.0, 0.0, 1.0))
        task_priority = float(np.clip(raw["task_priority"] / 1.5, 0.0, 1.0))
        local_delay = raw["task_cpu_cycles"] / max(0.05, raw["cpu_capacity"] * (1.0 - raw["cpu_load"]))
        best_usv_rate = float(raw["link_rate"][0])
        best_uav_rate = float(raw["link_rate"][1])
        best_offload_delay = raw["task_data_size"] / max(0.05, max(best_usv_rate, best_uav_rate)) + raw["task_cpu_cycles"]
        # Queue pressure is already normalized in raw features at neighborhood level; use role-specific estimates below.
        usv_queues = [n.queue_length / self.max_queue for n in self.nodes.values() if n.role == "usv"]
        uav_queues = [n.queue_length / self.max_queue for n in self.nodes.values() if n.role == "uav"]
        vec = np.array(
            [
                *role_onehot,
                raw["remaining_energy"],
                raw["cpu_capacity"],
                raw["cpu_load"],
                raw["queue_length"],
                raw["task_data_size"],
                raw["task_cpu_cycles"],
                task_deadline,
                task_priority,
                np.clip(best_usv_rate / 1.5, 0.0, 1.0),
                np.clip(best_uav_rate / 1.5, 0.0, 1.0),
                np.clip(min(usv_queues) if usv_queues else 1.0, 0.0, 1.0),
                np.clip(min(uav_queues) if uav_queues else 1.0, 0.0, 1.0),
                np.clip(local_delay / 2.0, 0.0, 1.0),
                np.clip(best_offload_delay / 2.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return vec[: self.obs_dim].astype(np.float32)

    def _system_stats_vector(self) -> np.ndarray:
        queues = np.array([n.queue_length / self.max_queue for n in self.nodes.values()], dtype=np.float32)
        links = []
        for aid in self.agent_ids:
            node = self.nodes[aid]
            usv = self._nearest_role(node, "usv")
            uav = self._nearest_role(node, "uav")
            links.extend([self._link_features(node, usv, "auv_usv")[0], self._link_features(node, uav, "usv_uav")[0]])
        completed = sum(1.0 for task in self.pending_tasks.values() if task is None)
        return np.array(
            [
                float(np.mean(queues)) if queues.size else 0.0,
                float(np.clip(np.mean(links) / 1.5, 0.0, 1.0)) if links else 0.0,
                float(self.task_counter / max(1, self.episode_length * max(1, self.n_auv))),
                float(completed / max(1, len(self.pending_tasks))),
                float(self.step_count / max(1, self.episode_length)),
            ],
            dtype=np.float32,
        )

    def _raw_vector(self, raw: dict) -> np.ndarray:
        role_onehot = np.zeros(3, dtype=np.float32)
        role_id = int(raw["role_id"])
        if role_id < 3:
            role_onehot[role_id] = 1.0
        pieces = [
            raw["position"],
            raw["velocity"],
            np.array(
                [
                    raw["remaining_energy"],
                    raw["cpu_capacity"],
                    raw["cpu_load"],
                    raw["queue_length"],
                    raw["task_data_size"],
                    raw["task_cpu_cycles"],
                    raw["task_deadline"],
                    raw["task_priority"],
                    raw["task_age"],
                ],
                dtype=np.float32,
            ),
            raw["task_onehot"],
            raw["link_rate"],
            raw["packet_loss"],
            raw["prop_delay"],
            np.array(
                [
                    raw["link_reliability"],
                    raw["neighbor_queue_pressure"],
                    raw["neighbor_cpu_load"],
                    raw["neighbor_energy_pressure"],
                    raw["neighbor_availability"],
                ],
                dtype=np.float32,
            ),
            role_onehot,
        ]
        raw_vec = np.concatenate(pieces, axis=0)
        if raw_vec.size < self.obs_dim - self.semantic_dim:
            raw_vec = np.pad(raw_vec, (0, self.obs_dim - self.semantic_dim - raw_vec.size))
        return raw_vec[: self.obs_dim - self.semantic_dim].astype(np.float32)

    def _legalize_action(self, action: np.ndarray, role: str) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size < self.action_dim:
            a = np.pad(a, (0, self.action_dim - a.size))
        a = np.clip(a[: self.action_dim], 0.0, 1.0)
        if role == "auv":
            total = float(a[0] + a[1] + a[2])
            if total > 1e-6:
                a[:3] = a[:3] / max(1.0, total)
        return a.astype(np.float32)

    def _process_auv_task(self, node: NodeState, action: np.ndarray) -> dict:
        task = self.pending_tasks.get(node.node_id)
        if task is None:
            return self._result(0.0, 0.0, 0.0, False, False, False, 0.0, 1.0, self._zero_reward_components())
        local, to_usv, to_uav = action[:3]
        tx_power = action[3]
        compression = action[4] if action.size > 4 else 0.0
        compressed_data = task.task_data_size * (1.0 - 0.45 * compression)
        local_delay = self.local_execution_delay_scale * local * task.task_cpu_cycles / max(0.05, node.cpu_capacity * (1.0 - node.cpu_load))
        local_energy = 0.28 * local * task.task_cpu_cycles
        usv = self._nearest_role(node, "usv")
        uav = self._nearest_role(node, "uav")
        usv_delay, usv_energy, usv_success = self._offload_delay(node, usv, compressed_data * to_usv, task.task_cpu_cycles * to_usv, "auv_usv", tx_power)
        uav_delay, uav_energy, uav_success = self._offload_delay(node, uav, compressed_data * to_uav, task.task_cpu_cycles * to_uav, "usv_uav", tx_power)
        delay = float(local_delay + usv_delay + uav_delay + 0.03 * task.task_age)
        energy = float(local_energy + usv_energy + uav_energy + 0.05 * tx_power)
        primary_route_ratio = float(max(local, to_usv, to_uav))
        completed = bool(
            delay <= task.task_deadline
            and (local + to_usv * usv_success + to_uav * uav_success) >= 0.72
            and primary_route_ratio >= self.min_primary_route_ratio
        )
        violation = bool(delay > task.task_deadline)
        semantic_match = self._semantic_match(task, action, node.role)
        if completed:
            self.pending_tasks[node.node_id] = None
        else:
            node.queue_length = min(self.max_queue, node.queue_length + 0.25)
        queue_overflow = float(node.queue_length >= self.max_queue - 1e-6)
        node.remaining_energy = float(np.clip(node.remaining_energy - energy * 0.04, 0.0, 1.0))
        reward, components = self._reward(completed, delay, energy, violation, semantic_match, queue_overflow=queue_overflow)
        return self._result(
            reward,
            delay,
            energy,
            completed,
            violation,
            False,
            semantic_match,
            float(usv_success or uav_success),
            components,
            primary_route_ratio=primary_route_ratio,
        )

    def _process_edge_node(self, node: NodeState, action: np.ndarray) -> dict:
        accept, cpu_alloc = float(action[0]), float(action[1])
        node.cpu_load = float(np.clip(0.75 * node.cpu_load + 0.25 * (accept * cpu_alloc + node.queue_length / self.max_queue), 0.0, 1.0))
        node.queue_length = float(np.clip(node.queue_length * (1.0 - 0.25 * cpu_alloc), 0.0, self.max_queue))
        energy = 0.02 + 0.03 * cpu_alloc
        node.remaining_energy = float(np.clip(node.remaining_energy - energy * 0.02, 0.0, 1.0))
        reward, components = self._reward(False, 0.0, energy, False, 0.5)
        reward *= 0.2
        components = {k: v * 0.2 for k, v in components.items()}
        return self._result(reward, 0.0, energy, False, False, False, 0.5, 1.0, components)

    def _offload_delay(self, src: NodeState, dst: NodeState | None, data: float, cycles: float, link_type: str, tx_power: float):
        if dst is None or data <= 1e-6:
            return 0.0, 0.0, 1.0
        rate, loss, prop = self._link_features(src, dst, link_type)
        rate = rate * (0.8 + 0.4 * tx_power)
        tx_delay = self.transmission_delay_scale * data / max(0.05, rate) + prop
        queue_delay = 0.35 * dst.queue_length / self.max_queue
        exec_delay = self.edge_execution_delay_scale * cycles / max(0.05, dst.cpu_capacity * (1.0 - dst.cpu_load))
        success = float(loss < 0.45 and dst.remaining_energy > 0.05)
        dst.queue_length = float(np.clip(dst.queue_length + data + cycles * 0.25, 0.0, self.max_queue))
        return float(tx_delay + queue_delay + exec_delay), float(0.08 * data * (0.5 + tx_power)), success

    def _reward(
        self,
        completed: bool,
        delay: float,
        energy: float,
        violation: bool,
        semantic_match: float,
        invalid_action: float = 0.0,
        queue_overflow: float = 0.0,
    ) -> tuple[float, dict[str, float]]:
        if self.reward_mode == "stable_v1":
            completion = 1.0 * float(completed)
            delay_penalty = -float(np.clip(delay / 2.0, 0.0, 1.0))
            deadline_penalty = -1.0 * float(violation)
            energy_penalty = -0.2 * float(np.clip(energy, 0.0, 1.0))
            invalid_penalty = -0.5 * float(np.clip(invalid_action, 0.0, 1.0))
            queue_penalty = -0.5 * float(np.clip(queue_overflow, 0.0, 1.0))
            reward = completion + delay_penalty + deadline_penalty + energy_penalty + invalid_penalty + queue_penalty
            reward = float(np.clip(reward, -2.0, 2.0))
            return reward, {
                "reward_completion": completion,
                "reward_delay": delay_penalty,
                "reward_deadline": deadline_penalty,
                "reward_energy": energy_penalty,
                "reward_invalid_action": invalid_penalty,
                "reward_queue_overflow": queue_penalty,
            }
        w = self.config.get("reward", {})
        comp = float(completed)
        delay_cost = float(np.clip(delay, 0.0, 2.0)) / 2.0
        energy_cost = float(np.clip(energy, 0.0, 1.0))
        balance = 1.0 - self._load_balance_index()
        semantic = semantic_match if self.config.get("reward", {}).get("use_semantic_reward", True) else 0.0
        reward = (
            float(w.get("w_comp", 0.35)) * comp
            - float(w.get("w_delay", 0.25)) * delay_cost
            - float(w.get("w_energy", 0.15)) * energy_cost
            + float(w.get("w_balance", 0.10)) * balance
            + float(w.get("w_semantic", 0.10)) * semantic
            - float(w.get("w_violation", 0.05)) * float(violation)
        )
        reward = float(np.clip(reward, -5.0, 5.0))
        return reward, {
            "reward_completion": float(w.get("w_comp", 0.35)) * comp,
            "reward_delay": -float(w.get("w_delay", 0.25)) * delay_cost,
            "reward_deadline": -float(w.get("w_violation", 0.05)) * float(violation),
            "reward_energy": -float(w.get("w_energy", 0.15)) * energy_cost,
            "reward_invalid_action": 0.0,
            "reward_queue_overflow": 0.0,
        }

    def _zero_reward_components(self) -> dict[str, float]:
        return {
            "reward_completion": 0.0,
            "reward_delay": 0.0,
            "reward_deadline": 0.0,
            "reward_energy": 0.0,
            "reward_invalid_action": 0.0,
            "reward_queue_overflow": 0.0,
        }

    def _result(
        self,
        reward,
        delay,
        energy,
        completed,
        violation,
        dropped,
        semantic_match,
        offload_success,
        reward_components=None,
        primary_route_ratio=0.0,
    ):
        row = {
            "reward": float(reward),
            "delay": float(delay),
            "energy": float(energy),
            "completed": float(completed),
            "deadline_violation": float(violation),
            "dropped": float(dropped),
            "semantic_match": float(semantic_match),
            "offload_success": float(offload_success),
            "generated": 0.0,
            "primary_route_ratio": float(primary_route_ratio),
        }
        row.update(reward_components or self._zero_reward_components())
        return row

    def _aggregate_step_metrics(self, rows: list[dict]) -> dict:
        auv_rows = rows[: self.n_auv]
        completed = float(sum(r["completed"] for r in rows))
        timeout = float(sum(r["deadline_violation"] for r in rows))
        dropped = float(sum(r["dropped"] for r in rows))
        pending = float(sum(1.0 for aid in list(self.nodes)[: self.n_auv] if self.pending_tasks.get(aid) is not None))
        generated = max(1.0, pending + completed, completed + timeout + dropped)
        queues = np.array([n.queue_length for n in self.nodes.values()], dtype=np.float32)
        metrics = {
            "completion_ratio": completed / max(1.0, generated),
            "mean_service_delay": float(np.mean([r["delay"] for r in rows])),
            "deadline_violation_rate": timeout / max(1.0, generated),
            "mean_energy_cost": float(np.mean([r["energy"] for r in rows])),
            "offload_success_rate": float(np.mean([r["offload_success"] for r in rows])),
            "load_balance_index": self._load_balance_index(),
            "mean_queue_length": float(np.mean(queues)),
            "generated_tasks": generated,
            "completed_tasks": completed,
            "timeout_tasks": timeout,
            "dropped_tasks": dropped,
            "semantic_match_rate": float(np.mean([r["semantic_match"] for r in rows])),
            "primary_route_ratio_mean": float(np.mean([r.get("primary_route_ratio", 0.0) for r in auv_rows])),
        }
        for key in self._zero_reward_components():
            metrics[key] = float(np.mean([r.get(key, 0.0) for r in rows]))
        metrics["reward_total_mean"] = float(np.mean([r.get("reward", 0.0) for r in rows]))
        return metrics

    def _load_balance_index(self) -> float:
        loads = np.array([n.cpu_load + n.queue_length / self.max_queue for n in self.nodes.values()], dtype=np.float32)
        if loads.size == 0 or float(loads.mean()) <= 1e-6:
            return 0.0
        return float(np.clip(loads.std() / (loads.mean() + 1e-6), 0.0, 1.0))

    def _semantic_match(self, task: ServiceTask, action: np.ndarray, role: str) -> float:
        if role == "auv":
            prior = semantic_prior_action({"semantic": np.array([1.0 - task.task_deadline, task.task_cpu_cycles, task.task_data_size, 0.8, 0, 0, 0, 1, 0, 0], dtype=np.float32), "role_id": 0})
            k = min(5, action.size, prior.size)
            return float(1.0 - np.mean(np.abs(prior[:k] - action[:k])))
        return 0.5

    def _generate_tasks(self):
        for aid, node in self.nodes.items():
            if node.role == "auv" and self.pending_tasks.get(aid) is None and self.rng.random() < self.task_generation_prob:
                self.task_counter += 1
                self.pending_tasks[aid] = self._calibrate_task(generate_task(self.task_counter, aid, self.rng))

    def _calibrate_task(self, task: ServiceTask) -> ServiceTask:
        if self.difficulty == "easy":
            if self.task_profile_mode == "easy_two_types" and task.task_type > 1:
                task.task_type = int(self.rng.integers(0, 2))
                if task.task_type == 0:
                    task.task_data_size, task.task_cpu_cycles, task.task_priority = 0.18, 0.22, 1.0
                else:
                    task.task_data_size, task.task_cpu_cycles, task.task_priority = 0.35, 0.35, 0.9
            task.task_deadline = float(np.clip(task.task_deadline * 1.35 + 0.25, 0.8, 2.4))
        elif self.difficulty == "hard":
            task.task_deadline = float(np.clip(task.task_deadline * 0.85, 0.35, 1.6))
        task_type = int(np.clip(task.task_type, 0, len(TASK_TYPES) - 1))
        task.task_data_size = float(
            np.clip(task.task_data_size * self.task_data_scale * self.task_type_data_scales[task_type], 0.02, 1.0)
        )
        task.task_cpu_cycles = float(
            np.clip(task.task_cpu_cycles * self.task_cpu_scale * self.task_type_cpu_scales[task_type], 0.05, 1.0)
        )
        task.task_deadline = float(np.clip(task.task_deadline * self.deadline_scale, 0.25, 2.4))
        return task

    def _age_and_generate(self):
        for aid, task in list(self.pending_tasks.items()):
            if task is not None:
                task.task_age += 1.0 / max(1, self.episode_length)
                if task.task_age > 1.0:
                    self.pending_tasks[aid] = None
        self._generate_tasks()

    def _move_nodes(self, actions: dict[str, np.ndarray]):
        if not self.use_mobility_control or self.action_mode == "simple":
            for node in self.nodes.values():
                node.position[:2] = np.clip(node.position[:2] + self.rng.normal(0.0, 0.001, size=2), 0.0, 1.0)
            return
        for aid, node in self.nodes.items():
            a = self._legalize_action(actions.get(aid, np.zeros(self.action_dim, dtype=np.float32)), node.role)
            if node.role == "auv":
                node.position[0] = np.clip(node.position[0] + 0.01 * (a[5] - 0.5), 0.0, 1.0)
            else:
                node.position[0] = np.clip(node.position[0] + 0.015 * (a[4] - 0.5), 0.0, 1.0)
                node.position[1] = np.clip(node.position[1] + 0.015 * (a[5] - 0.5), 0.0, 1.0)

    def _nearest_role(self, node: NodeState, role: str) -> NodeState | None:
        candidates = [n for n in self.nodes.values() if n.role == role]
        if not candidates:
            return None
        return min(candidates, key=lambda n: float(np.linalg.norm(node.position - n.position)))

    def _nearest_auv_task(self, node: NodeState) -> ServiceTask | None:
        tasks = [t for t in self.pending_tasks.values() if t is not None]
        if not tasks:
            return None
        return tasks[0]

    def _link_features(self, src: NodeState, dst: NodeState | None, link_type: str):
        if dst is None:
            return 0.05, 0.8, 1.0
        dist = float(np.linalg.norm(src.position - dst.position))
        if link_type == "auv_usv":
            base_rate, base_loss, prop = 0.22, 0.22, 0.25
        elif link_type == "usv_uav":
            base_rate, base_loss, prop = 0.75, 0.08, 0.06
        else:
            base_rate, base_loss, prop = 1.1, 0.04, 0.04
        rate = float(np.clip(base_rate / (1.0 + dist), 0.03, 1.5))
        loss = float(np.clip(base_loss + 0.18 * dist, 0.0, 0.95))
        delay = float(np.clip(prop + 0.1 * dist, 0.0, 1.0))
        if self.difficulty == "easy":
            rate = float(np.clip(rate * 1.8, 0.05, 1.8))
            loss = float(np.clip(loss * 0.25, 0.0, 0.35))
            delay = float(np.clip(delay * 0.55, 0.0, 0.8))
        elif self.difficulty == "hard":
            rate = float(np.clip(rate * 0.75, 0.02, 1.2))
            loss = float(np.clip(loss * 1.35, 0.0, 0.98))
            delay = float(np.clip(delay * 1.2, 0.0, 1.2))
        return rate, loss, delay

    def _heuristic_success_label(self, raw: dict) -> float:
        return float(raw["task_deadline"] > 0.35 and raw["link_reliability"] > 0.55 and raw["neighbor_queue_pressure"] < 0.7)

    def _heuristic_deadline_label(self, raw: dict) -> float:
        return float(raw["task_deadline"] < 0.35 or raw["neighbor_queue_pressure"] > 0.75)

    def _heuristic_delay_label(self, raw: dict) -> float:
        return float(np.clip(raw["task_data_size"] / max(0.05, raw["link_rate"].mean()) + raw["task_cpu_cycles"], 0.0, 2.0) / 2.0)
