"""Semantic-loss-guided residual extension of the discrete DI-MAPPO baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ServiceComputing.algorithms.service_mappo_di import RolloutBatch, ServiceMAPPOTrainer
from ServiceComputing.models.service_semantic_guidance import (
    DiscreteSemanticGuidance,
    semantic_teacher_distribution,
)
from ServiceComputing.models.structured_policy import StructuredServiceActor
from ServiceComputing.service_offloading.metrics import ENERGY_ACCOUNTING_KEYS, mean_metrics


@dataclass
class SemanticRolloutBatch(RolloutBatch):
    semantic: np.ndarray
    route_costs: np.ndarray
    semantic_masks: np.ndarray
    key_energy_costs: np.ndarray
    completion_labels: np.ndarray
    deadline_labels: np.ndarray
    delay_labels: np.ndarray
    key_energy_labels: np.ndarray
    congestion_labels: np.ndarray


class SLGSAGEMAPPOTrainer(ServiceMAPPOTrainer):
    """MAPPO with zero-initialized semantic residual logits and decaying losses."""

    TASK_PRESENT_INDEX = 7

    def __init__(self, config: dict[str, Any], run_dir: Path):
        if not bool(config.get("env", {}).get("use_semantic_side_channel", False)):
            raise ValueError("SLG-SAGE requires env.use_semantic_side_channel=true")
        super().__init__(config, run_dir)
        if not self.discrete_actions:
            raise ValueError("SLG-SAGE DI trainer currently supports action_mode=discrete_route only")
        scfg = config.get("semantic", {})
        mcfg = config.get("mappo", {})
        self.guidance = DiscreteSemanticGuidance(
            semantic_dim=self.env.semantic_dim,
            action_dim=self.env.action_dim,
            hidden_dim=int(scfg.get("hidden_dim", 64)),
            zero_init_output=bool(scfg.get("zero_init_semantic_output", True)),
        ).to(self.device)
        self.use_structured_actor = bool(scfg.get("use_structured_actor", False))
        self.structured_actor = (
            StructuredServiceActor(
                obs_dim=self.env.obs_dim,
                semantic_dim=self.env.semantic_dim,
                action_dim=self.env.action_dim,
                hidden_dim=int(scfg.get("structured_hidden_dim", mcfg.get("hidden_dim", 128))),
                use_role_embedding=bool(scfg.get("use_role_embedding", True)),
                fusion_mode=str(scfg.get("fusion_mode", "gated")),
            ).to(self.device)
            if self.use_structured_actor
            else None
        )
        extra_params = list(self.guidance.parameters())
        if self.structured_actor is not None:
            extra_params += list(self.structured_actor.parameters())
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + extra_params,
            lr=float(mcfg.get("learning_rate", 3e-4)),
        )
        self.lambda_prior_0 = float(scfg.get("lambda_prior_0", 0.08))
        self.lambda_guide_0 = float(scfg.get("lambda_guide_0", 0.03))
        self.lambda_aux = float(scfg.get("lambda_aux", 0.03))
        self.lambda_deterministic_distill = float(scfg.get("lambda_deterministic_distill", 0.0))
        self.lambda_completion_aux = float(scfg.get("lambda_completion_aux", 1.0))
        self.lambda_deadline_aux = float(scfg.get("lambda_deadline_aux", 1.0))
        self.lambda_delay_aux = float(scfg.get("lambda_delay_aux", 1.0))
        self.lambda_key_energy_aux = float(scfg.get("lambda_key_energy_aux", 1.0))
        self.lambda_congestion_aux = float(scfg.get("lambda_congestion_aux", 1.0))
        self.guide_updates_base_actor = bool(scfg.get("guide_updates_base_actor", True))
        self.semantic_input_mode = str(scfg.get("semantic_input_mode", "full"))
        self.semantic_input_mask = self._build_semantic_input_mask(self.semantic_input_mode)
        self.use_residual_gate = bool(scfg.get("use_residual_gate", False))
        self.use_qos_tie_energy_teacher = bool(scfg.get("use_qos_tie_energy_teacher", False))
        self.qos_tie_threshold = float(scfg.get("qos_tie_threshold", self.env.semantic_qos_tie_threshold))
        self.qos_tie_energy_weight = float(scfg.get("qos_tie_energy_weight", self.env.semantic_qos_tie_energy_weight))
        self.prior_decay_steps = float(scfg.get("prior_decay_steps", 40000))
        self.guide_decay_steps = float(scfg.get("guide_decay_steps", 25000))
        self.teacher_temperature = float(scfg.get("teacher_temperature", 0.5))
        self.semantic_logit_scale_max = float(scfg.get("semantic_logit_scale_max", 0.30))
        self.semantic_min_logit_scale = float(scfg.get("semantic_min_logit_scale", 0.0))
        self.semantic_residual_warmup_steps = float(scfg.get("semantic_residual_warmup_steps", 5000))
        self.semantic_residual_decay_start_steps = scfg.get("semantic_residual_decay_start_steps")
        self.semantic_residual_decay_start_steps = (
            None
            if self.semantic_residual_decay_start_steps is None
            else float(self.semantic_residual_decay_start_steps)
        )
        self.semantic_residual_decay_steps = float(scfg.get("semantic_residual_decay_steps", 1.0))
        self.semantic_step = 0

    def _semantic_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target_env = env or self.env
        semantic = np.stack([obs[aid]["semantic"] for aid in target_env.agent_ids], axis=0).astype(np.float32)
        if self.semantic_input_mask is not None:
            semantic = semantic * self.semantic_input_mask
        return semantic

    def _build_semantic_input_mask(self, mode: str) -> np.ndarray | None:
        mode = mode.lower()
        if mode in {"full", "full_semantics"}:
            return None
        mask = np.zeros(self.env.semantic_dim, dtype=np.float32)
        if mode in {"raw_only", "none"}:
            return mask
        keep: list[tuple[int, int]] = []
        if mode in {"task", "task_only", "link", "task_link", "queue", "task_link_queue"}:
            keep.extend([(3, 11)])
        if mode in {"link", "task_link", "queue", "task_link_queue"}:
            keep.extend([(12, 17)])
        if mode in {"queue", "task_link_queue"}:
            keep.extend([(11, 13), (25, 29)])
        for start, end in keep:
            mask[start : min(end, mask.shape[0])] = 1.0
        return mask

    def _semantic_cost_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target_env = env or self.env
        return np.stack([obs[aid]["semantic_route_costs"] for aid in target_env.agent_ids], axis=0).astype(np.float32)

    def _semantic_mask_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target_env = env or self.env
        return np.stack([obs[aid]["semantic_action_mask"] for aid in target_env.agent_ids], axis=0).astype(np.float32)

    def _semantic_key_energy_cost_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target_env = env or self.env
        return np.stack(
            [obs[aid].get("semantic_key_energy_cost", np.zeros(target_env.action_dim, dtype=np.float32)) for aid in target_env.agent_ids],
            axis=0,
        ).astype(np.float32)

    def _auxiliary_label_arrays(self, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        metrics = info["metrics"]
        per_agent = info.get("per_agent", {})
        global_completion = float(metrics["reward_completion"] > 0.0)
        global_deadline = float(metrics["reward_deadline"] < 0.0)
        delay_value = float(
            np.clip(metrics["mean_service_delay"] / self.env.route_delay_normalizer, 0.0, 1.0)
        )
        key_energy_value = float(np.clip(metrics.get("key_tx_energy_per_successful_completion", 0.0), 0.0, 1.0))
        congestion_value = float(np.clip(metrics.get("mean_queue_length", 0.0) / max(1.0, self.env.queue_capacity), 0.0, 1.0))
        completion = np.array(
            [float(per_agent.get(aid, {}).get("completed_task", global_completion)) for aid in self.env.agent_ids],
            dtype=np.float32,
        )
        # Use attributable timeouts when exposed; otherwise fall back to the
        # system event and let the configured weight limit this coarse target.
        deadline = np.array(
            [float(per_agent.get(aid, {}).get("caused_timeout", global_deadline)) for aid in self.env.agent_ids],
            dtype=np.float32,
        )
        delay = np.full(len(self.env.agent_ids), delay_value, dtype=np.float32)
        key_energy = np.full(len(self.env.agent_ids), key_energy_value, dtype=np.float32)
        congestion = np.full(len(self.env.agent_ids), congestion_value, dtype=np.float32)
        return completion, deadline, delay, key_energy, congestion

    def _semantic_policy_factor(self) -> float:
        if self.semantic_residual_decay_start_steps is None:
            return 1.0
        elapsed = max(0.0, self.semantic_step - self.semantic_residual_decay_start_steps)
        return float(max(0.0, 1.0 - elapsed / max(1.0, self.semantic_residual_decay_steps)))

    def semantic_logit_scale(self) -> float:
        progress = min(1.0, self.semantic_step / max(1.0, self.semantic_residual_warmup_steps))
        scale = self.semantic_logit_scale_max * progress * self._semantic_policy_factor()
        if self.env.use_semantic_side_channel and self.semantic_min_logit_scale > 0.0:
            scale = max(scale, self.semantic_min_logit_scale)
        return float(scale)

    def semantic_coefficients(self) -> tuple[float, float]:
        prior = self.lambda_prior_0 * np.exp(-self.semantic_step / max(1.0, self.prior_decay_steps))
        guide = (
            self.lambda_guide_0
            * np.exp(-self.semantic_step / max(1.0, self.guide_decay_steps))
            * self._semantic_policy_factor()
        )
        return float(prior), float(guide)

    def _active_semantic_mask(self, semantic: torch.Tensor) -> torch.Tensor:
        return semantic[..., self.TASK_PRESENT_INDEX].clamp(0.0, 1.0)

    def _active_semantic_mean(self, per_agent_values: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        active = self._active_semantic_mask(semantic)
        return (per_agent_values * active).sum() / active.sum().clamp_min(1.0)

    def _active_centered_residual(self, residual: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        centered = residual - residual.mean(dim=-1, keepdim=True)
        return centered * self._active_semantic_mask(semantic).unsqueeze(-1)

    def _gated_centered_residual(
        self, residual: torch.Tensor, semantic: torch.Tensor, residual_gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        centered = self._active_centered_residual(residual, semantic)
        if self.use_residual_gate and residual_gate is not None:
            centered = centered * residual_gate.unsqueeze(-1).clamp(0.0, 1.0)
        return centered

    def _semantic_guidance_kl(
        self, base_logits: torch.Tensor, residual_logits: torch.Tensor, semantic: torch.Tensor, teacher: torch.Tensor
    ) -> torch.Tensor:
        guided_base_logits = base_logits if self.guide_updates_base_actor else base_logits.detach()
        semantic_only_logits = guided_base_logits + self.semantic_logit_scale() * self._gated_centered_residual(
            residual_logits, semantic
        )
        per_agent = F.kl_div(F.log_softmax(semantic_only_logits, dim=-1), teacher, reduction="none").sum(dim=-1)
        return self._active_semantic_mean(per_agent, semantic)

    def _semantic_row_metrics(
        self, base_logits: torch.Tensor, final_logits: torch.Tensor, teacher: torch.Tensor, semantic: torch.Tensor
    ) -> dict[str, float]:
        active = self._active_semantic_mask(semantic)
        changed = (torch.argmax(base_logits, dim=-1) != torch.argmax(final_logits, dim=-1)).float()
        matched = (torch.argmax(teacher, dim=-1) == torch.argmax(final_logits, dim=-1)).float()
        active_count = active.sum().clamp_min(1.0)
        return {
            "semantic_changed_argmax_rate": float(changed.mean().detach().cpu()),
            "teacher_policy_top1_match_rate": float(matched.mean().detach().cpu()),
            "teacher_top_action_match_rate": float(matched.mean().detach().cpu()),
            "semantic_active_fraction": float(active.mean().detach().cpu()),
            "semantic_changed_argmax_rate_active": float(((changed * active).sum() / active_count).detach().cpu()),
            "teacher_policy_top1_match_rate_active": float(((matched * active).sum() / active_count).detach().cpu()),
        }

    def _guided_forward(
        self,
        obs_arr: np.ndarray,
        global_arr: np.ndarray,
        semantic_arr: np.ndarray,
        action_mask_arr: np.ndarray | None = None,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        semantic_t = torch.as_tensor(semantic_arr, dtype=torch.float32, device=self.device)
        if self.structured_actor is not None:
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
            mask_t = None if action_mask_arr is None else torch.as_tensor(action_mask_arr, dtype=torch.float32, device=self.device)
            base_logits = self.structured_actor(obs_t, semantic_t, mask_t)["logits"]
        else:
            base_logits, _, _ = self._forward(obs_arr, global_arr, action_mask_arr)
        guidance = self.guidance(semantic_t, actions)
        centered = self._gated_centered_residual(
            guidance["residual_logits"], semantic_t, guidance.get("residual_gate")
        )
        final_logits = base_logits + self.semantic_logit_scale() * centered
        if action_mask_arr is not None:
            action_mask_t = torch.as_tensor(action_mask_arr, dtype=torch.float32, device=self.device)
            final_logits = final_logits.masked_fill(action_mask_t <= 0.0, -1e9)
        return base_logits, final_logits, guidance

    def _guided_batch(
        self,
        obs: torch.Tensor,
        global_obs: torch.Tensor,
        semantic: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if action_masks is None:
            action_masks = torch.ones((*obs.shape[:2], self.env.action_dim), dtype=torch.float32, device=self.device)
        inputs = {
            "agent_state": obs,
            "global_state": global_obs,
            "action_mask": action_masks,
        }
        actor_critic = self.model(inputs, mode="compute_actor_critic")
        if self.structured_actor is not None:
            base_logits = self.structured_actor(obs, semantic, action_masks)["logits"]
        else:
            base_logits = actor_critic["logit"].masked_fill(action_masks <= 0.0, -1e9)
        flat_semantic = semantic.reshape(-1, semantic.shape[-1])
        flat_actions = actions.reshape(-1)
        guidance_flat = self.guidance(flat_semantic, flat_actions)
        residual = guidance_flat["residual_logits"].reshape(*base_logits.shape)
        residual_gate = guidance_flat["residual_gate"].reshape(*actions.shape)
        centered = self._gated_centered_residual(residual, semantic, residual_gate)
        final_logits = base_logits + self.semantic_logit_scale() * centered
        final_logits = final_logits.masked_fill(action_masks <= 0.0, -1e9)
        guidance = {
            key: value.reshape(*actions.shape) if key != "residual_logits" else residual
            for key, value in guidance_flat.items()
        }
        guidance["residual_gate"] = residual_gate
        return base_logits, final_logits, actor_critic["value"], guidance

    def _parameter_count(self) -> int:
        total = super()._parameter_count() + sum(parameter.numel() for parameter in self.guidance.parameters())
        if self.structured_actor is not None:
            total += sum(parameter.numel() for parameter in self.structured_actor.parameters())
        return int(total)

    def _inference_time_ms_per_decision(self, env=None) -> float:
        target_env = env or self.eval_env
        obs, _ = target_env.reset(seed=self.seed + 777000)
        obs_arr = self._obs_array(obs, target_env)
        global_arr = self._global_array(target_env)
        semantic_arr = self._semantic_array(obs, target_env)
        action_masks = self._action_mask_array(obs, target_env)
        iterations = int(self.config.get("mappo", {}).get("inference_benchmark_iters", 10))
        if iterations <= 0:
            return 0.0
        with torch.no_grad():
            self._guided_forward(obs_arr, global_arr, semantic_arr, action_masks)
            start = time.perf_counter()
            for _ in range(iterations):
                self._guided_forward(obs_arr, global_arr, semantic_arr, action_masks)
            elapsed = time.perf_counter() - start
        return float(1000.0 * elapsed / max(1, iterations * len(target_env.agent_ids)))

    def collect(self) -> SemanticRolloutBatch:
        obs_buf, global_buf, action_mask_buf, action_buf, logp_buf, value_buf = [], [], [], [], [], []
        reward_buf, done_buf, metrics_rows, action_rows = [], [], [], []
        semantic_buf, cost_buf, mask_buf, key_energy_cost_buf = [], [], [], []
        completion_buf, deadline_buf, delay_buf, key_energy_buf, congestion_buf = [], [], [], [], []
        semantic_rows = []
        train_return = 0.0
        for _ in range(self.rollout_steps):
            obs_arr = self._obs_array(self._obs)
            global_arr = self._global_array()
            semantic_arr = self._semantic_array(self._obs)
            costs = self._semantic_cost_array(self._obs)
            masks = self._semantic_mask_array(self._obs)
            key_energy_costs = self._semantic_key_energy_cost_array(self._obs)
            action_masks = self._action_mask_array(self._obs)
            with torch.no_grad():
                base_logits, final_logits, _ = self._guided_forward(obs_arr, global_arr, semantic_arr, action_masks)
                value = self._forward(obs_arr, global_arr, action_masks)[2]
                dist = Categorical(logits=final_logits)
                raw_action = dist.sample()
                log_prob = dist.log_prob(raw_action)
                teacher = semantic_teacher_distribution(
                    torch.as_tensor(costs, device=self.device),
                    torch.as_tensor(masks, device=self.device),
                    self.teacher_temperature,
                    torch.as_tensor(key_energy_costs, device=self.device) if self.use_qos_tie_energy_teacher else None,
                    self.qos_tie_threshold,
                    self.qos_tie_energy_weight,
                )
            raw_np = raw_action.cpu().numpy()
            env_actions = {aid: int(raw_np[i]) for i, aid in enumerate(self.env.agent_ids)}
            next_obs, rewards, done, _, info = self.env.step(env_actions)
            reward_arr = np.array([float(rewards[aid]) for aid in self.env.agent_ids], dtype=np.float32)
            completion_labels, deadline_labels, delay_labels, key_energy_labels, congestion_labels = self._auxiliary_label_arrays(info)
            obs_buf.append(obs_arr)
            global_buf.append(global_arr)
            action_mask_buf.append(action_masks)
            semantic_buf.append(semantic_arr)
            cost_buf.append(costs)
            mask_buf.append(masks)
            key_energy_cost_buf.append(key_energy_costs)
            action_buf.append(raw_np.astype(np.int64))
            logp_buf.append(log_prob.cpu().numpy().astype(np.float32))
            value_buf.append(value.cpu().numpy().astype(np.float32))
            reward_buf.append(reward_arr)
            done_buf.append(np.full(len(self.env.agent_ids), float(done), dtype=np.float32))
            completion_buf.append(completion_labels)
            deadline_buf.append(deadline_labels)
            delay_buf.append(delay_labels)
            key_energy_buf.append(key_energy_labels)
            congestion_buf.append(congestion_labels)
            metrics_rows.append(info["metrics"])
            action_rows.append(self._action_metrics(env_actions))
            semantic_rows.append(self._semantic_row_metrics(base_logits, final_logits, teacher, torch.as_tensor(semantic_arr, device=self.device)))
            train_return += float(np.mean(reward_arr))
            self._obs = next_obs
            if done:
                self.episode += 1
                self._obs, _ = self.env.reset(seed=self.seed + self.episode)

        with torch.no_grad():
            last_value = self._forward(
                self._obs_array(self._obs), self._global_array(), self._action_mask_array(self._obs)
            )[2].cpu().numpy().astype(np.float32)
        rewards_arr = np.asarray(reward_buf, dtype=np.float32)
        dones_arr = np.asarray(done_buf, dtype=np.float32)
        values_arr = np.asarray(value_buf, dtype=np.float32)
        advantages = np.zeros_like(rewards_arr, dtype=np.float32)
        last_gae = np.zeros(rewards_arr.shape[1], dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            next_value = last_value if t == self.rollout_steps - 1 else values_arr[t + 1]
            nonterminal = 1.0 - dones_arr[t]
            delta = rewards_arr[t] + self.gamma * next_value * nonterminal - values_arr[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
        metrics = mean_metrics(metrics_rows)
        metrics.update(self._mean_dicts(action_rows))
        metrics.update(self._mean_dicts(semantic_rows))
        return SemanticRolloutBatch(
            obs=np.asarray(obs_buf, dtype=np.float32),
            global_obs=np.asarray(global_buf, dtype=np.float32),
            action_masks=np.asarray(action_mask_buf, dtype=np.float32),
            actions=np.asarray(action_buf, dtype=np.int64),
            log_probs=np.asarray(logp_buf, dtype=np.float32),
            values=values_arr,
            rewards=rewards_arr,
            dones=dones_arr,
            advantages=advantages,
            returns=advantages + values_arr,
            metrics=metrics,
            train_return=train_return,
            semantic=np.asarray(semantic_buf, dtype=np.float32),
            route_costs=np.asarray(cost_buf, dtype=np.float32),
            semantic_masks=np.asarray(mask_buf, dtype=np.float32),
            key_energy_costs=np.asarray(key_energy_cost_buf, dtype=np.float32),
            completion_labels=np.asarray(completion_buf, dtype=np.float32),
            deadline_labels=np.asarray(deadline_buf, dtype=np.float32),
            delay_labels=np.asarray(delay_buf, dtype=np.float32),
            key_energy_labels=np.asarray(key_energy_buf, dtype=np.float32),
            congestion_labels=np.asarray(congestion_buf, dtype=np.float32),
        )

    def update(self, batch: SemanticRolloutBatch) -> dict[str, float]:
        indices = np.arange(batch.obs.shape[0])
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        lambda_prior, lambda_guide = self.semantic_coefficients()
        losses = []
        for _ in range(self.update_epochs):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_steps):
                mb = indices[start : start + self.minibatch_steps]
                obs = torch.as_tensor(batch.obs[mb], dtype=torch.float32, device=self.device)
                global_obs = torch.as_tensor(batch.global_obs[mb], dtype=torch.float32, device=self.device)
                semantic = torch.as_tensor(batch.semantic[mb], dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(batch.actions[mb], dtype=torch.long, device=self.device)
                old_logp = torch.as_tensor(batch.log_probs[mb], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(batch.returns[mb], dtype=torch.float32, device=self.device)
                route_costs = torch.as_tensor(batch.route_costs[mb], dtype=torch.float32, device=self.device)
                masks = torch.as_tensor(batch.semantic_masks[mb], dtype=torch.float32, device=self.device)
                key_energy_costs = torch.as_tensor(batch.key_energy_costs[mb], dtype=torch.float32, device=self.device)
                action_masks = torch.as_tensor(batch.action_masks[mb], dtype=torch.float32, device=self.device)
                completion = torch.as_tensor(batch.completion_labels[mb], dtype=torch.float32, device=self.device)
                deadline = torch.as_tensor(batch.deadline_labels[mb], dtype=torch.float32, device=self.device)
                delay = torch.as_tensor(batch.delay_labels[mb], dtype=torch.float32, device=self.device)
                key_energy = torch.as_tensor(batch.key_energy_labels[mb], dtype=torch.float32, device=self.device)
                congestion = torch.as_tensor(batch.congestion_labels[mb], dtype=torch.float32, device=self.device)
                base_logits, final_logits, values, guidance = self._guided_batch(
                    obs, global_obs, semantic, actions, action_masks
                )
                dist = Categorical(logits=final_logits)
                logp = dist.log_prob(actions)
                ratio = torch.exp(logp - old_logp)
                adv = advantages[mb]
                policy_loss = -torch.min(
                    ratio * adv, torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                ).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_loss = dist.entropy().mean()
                teacher = semantic_teacher_distribution(
                    route_costs,
                    masks,
                    self.teacher_temperature,
                    key_energy_costs if self.use_qos_tie_energy_teacher else None,
                    self.qos_tie_threshold,
                    self.qos_tie_energy_weight,
                )
                semantic_prior_loss = self._active_semantic_mean(
                    F.kl_div(F.log_softmax(guidance["residual_logits"], dim=-1), teacher, reduction="none").sum(dim=-1),
                    semantic,
                )
                semantic_guidance_loss = self._semantic_guidance_kl(
                    base_logits, guidance["residual_logits"], semantic, teacher
                )
                completion_loss = self._active_semantic_mean(
                    F.binary_cross_entropy_with_logits(guidance["completion_logit"], completion, reduction="none"),
                    semantic,
                )
                deadline_loss = self._active_semantic_mean(
                    F.binary_cross_entropy_with_logits(guidance["deadline_logit"], deadline, reduction="none"),
                    semantic,
                )
                delay_loss = self._active_semantic_mean(
                    F.smooth_l1_loss(guidance["delay_pred"], delay, reduction="none"), semantic
                )
                key_energy_loss = self._active_semantic_mean(
                    F.smooth_l1_loss(guidance["key_energy_pred"], key_energy, reduction="none"), semantic
                )
                congestion_loss = self._active_semantic_mean(
                    F.binary_cross_entropy_with_logits(
                        guidance["congestion_logit"], congestion, reduction="none"
                    ),
                    semantic,
                )
                semantic_aux_loss = (
                    self.lambda_completion_aux * completion_loss
                    + self.lambda_deadline_aux * deadline_loss
                    + self.lambda_delay_aux * delay_loss
                    + self.lambda_key_energy_aux * key_energy_loss
                    + self.lambda_congestion_aux * congestion_loss
                )
                teacher_top_action = torch.argmax(teacher, dim=-1)
                deterministic_distill_loss = self._active_semantic_mean(
                    F.cross_entropy(
                        final_logits.reshape(-1, self.env.action_dim),
                        teacher_top_action.reshape(-1),
                        reduction="none",
                    ).reshape(teacher_top_action.shape),
                    semantic,
                )
                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_loss
                    + lambda_prior * semantic_prior_loss
                    + lambda_guide * semantic_guidance_loss
                    + self.lambda_aux * semantic_aux_loss
                    + self.lambda_deterministic_distill * deterministic_distill_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                params = list(self.model.parameters()) + list(self.guidance.parameters())
                if self.structured_actor is not None:
                    params += list(self.structured_actor.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()
                losses.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy_loss": float(entropy_loss.detach().cpu()),
                        "total_loss": float(total_loss.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu()),
                        "semantic_prior_loss": float(semantic_prior_loss.detach().cpu()),
                        "semantic_guidance_loss": float(semantic_guidance_loss.detach().cpu()),
                        "semantic_aux_loss": float(semantic_aux_loss.detach().cpu()),
                        "deterministic_distill_loss": float(deterministic_distill_loss.detach().cpu()),
                        "completion_prediction_loss": float(completion_loss.detach().cpu()),
                        "deadline_prediction_loss": float(deadline_loss.detach().cpu()),
                        "delay_prediction_loss": float(delay_loss.detach().cpu()),
                        "key_energy_prediction_loss": float(key_energy_loss.detach().cpu()),
                        "congestion_prediction_loss": float(congestion_loss.detach().cpu()),
                        "lambda_prior": lambda_prior,
                        "lambda_guide": lambda_guide,
                        "lambda_deterministic_distill": self.lambda_deterministic_distill,
                        "semantic_logit_scale": self.semantic_logit_scale(),
                        "semantic_policy_factor": self._semantic_policy_factor(),
                        "semantic_min_logit_scale": self.semantic_min_logit_scale,
                        "semantic_residual_active": float(self.semantic_logit_scale() > 0.0),
                        "residual_gate_mean": float(guidance["residual_gate"].mean().detach().cpu()),
                        "use_structured_actor": float(self.structured_actor is not None),
                        **self._semantic_row_metrics(base_logits, final_logits, teacher, semantic),
                        "teacher_prior_entropy": float(
                            (-(teacher * torch.log(teacher.clamp_min(1e-8))).sum(dim=-1).mean()).detach().cpu()
                        ),
                    }
                )
        self.semantic_step += self.rollout_steps
        return {key: float(np.mean([row[key] for row in losses])) for key in losses[0]}

    def evaluate(self, seed_offset: int = 10000, deterministic: bool = True) -> dict[str, float]:
        env = self.eval_env
        rows, terminal_rows, action_rows, semantic_rows, returns = [], [], [], [], []
        rng_state = torch.random.get_rng_state()
        try:
            if not deterministic:
                torch.manual_seed(self.seed + seed_offset + 900000)
            for ep in range(self.eval_episodes):
                obs, _ = env.reset(seed=self.seed + seed_offset + ep)
                done = False
                episode_return = 0.0
                while not done:
                    obs_arr = self._obs_array(obs, env)
                    global_arr = self._global_array(env)
                    semantic_arr = self._semantic_array(obs, env)
                    costs = self._semantic_cost_array(obs, env)
                    masks = self._semantic_mask_array(obs, env)
                    key_energy_costs = self._semantic_key_energy_cost_array(obs, env)
                    action_masks = self._action_mask_array(obs, env)
                    with torch.no_grad():
                        base_logits, final_logits, guidance = self._guided_forward(
                            obs_arr, global_arr, semantic_arr, action_masks
                        )
                        action_t = torch.argmax(final_logits, dim=-1) if deterministic else Categorical(logits=final_logits).sample()
                        teacher = semantic_teacher_distribution(
                            torch.as_tensor(costs, device=self.device),
                            torch.as_tensor(masks, device=self.device),
                            self.teacher_temperature,
                            torch.as_tensor(key_energy_costs, device=self.device) if self.use_qos_tie_energy_teacher else None,
                            self.qos_tie_threshold,
                            self.qos_tie_energy_weight,
                        )
                    actions = {aid: int(action_t[i].cpu()) for i, aid in enumerate(env.agent_ids)}
                    obs, rewards, done, _, info = env.step(actions)
                    episode_return += float(np.mean(list(rewards.values())))
                    rows.append(info["metrics"])
                    if done:
                        terminal_rows.append(info["metrics"])
                    action_rows.append(self._action_metrics(actions, env))
                    semantic_row = self._semantic_row_metrics(
                        base_logits, final_logits, teacher, torch.as_tensor(semantic_arr, device=self.device)
                    )
                    semantic_row["semantic_logit_scale"] = self.semantic_logit_scale()
                    semantic_row["semantic_policy_factor"] = self._semantic_policy_factor()
                    semantic_row["semantic_min_logit_scale"] = self.semantic_min_logit_scale
                    semantic_row["semantic_residual_active"] = float(self.semantic_logit_scale() > 0.0)
                    semantic_row["residual_gate_mean"] = float(guidance["residual_gate"].mean().detach().cpu())
                    semantic_row["use_structured_actor"] = float(self.structured_actor is not None)
                    semantic_rows.append(semantic_row)
                returns.append(episode_return)
        finally:
            torch.random.set_rng_state(rng_state)
        metrics = mean_metrics(rows)
        terminal_metrics = mean_metrics(terminal_rows)
        metrics.update({key: terminal_metrics[key] for key in ENERGY_ACCOUNTING_KEYS})
        metrics.update(self._mean_dicts(action_rows))
        metrics.update(self._mean_dicts(semantic_rows))
        metrics["eval_return"] = float(np.mean(returns))
        metrics["parameter_count"] = float(self._parameter_count())
        metrics["inference_time_ms_per_decision"] = self._inference_time_ms_per_decision(env)
        return metrics

    def _save_checkpoint(self, step: int, eval_metrics: dict[str, float], filename: str = "checkpoint_latest.pt") -> Path:
        target = self.run_dir / "checkpoints" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": int(step),
                "model_state_dict": self.model.state_dict(),
                "guidance_state_dict": self.guidance.state_dict(),
                "structured_actor_state_dict": self.structured_actor.state_dict() if self.structured_actor is not None else None,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "eval_metrics": dict(eval_metrics),
                "semantic_step": int(self.semantic_step),
                "config": self.config,
            },
            target,
        )
        return target

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model_state_dict"])
        if "guidance_state_dict" in payload:
            self.guidance.load_state_dict(payload["guidance_state_dict"])
        if self.structured_actor is not None and payload.get("structured_actor_state_dict") is not None:
            self.structured_actor.load_state_dict(payload["structured_actor_state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.initial_env_steps = int(payload.get("step", 0))
        self.semantic_step = int(payload.get("semantic_step", self.initial_env_steps))
        self.best_eval_metrics = dict(payload.get("eval_metrics", {}))
        self.best_eval_return = float(self.best_eval_metrics.get("eval_return", -float("inf")))
        self.best_eval_step = self.initial_env_steps
