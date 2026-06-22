"""Lightweight SLG-SAGE-MAPPO trainer for the service offloading environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out_dim))


class ServiceSemanticEncoder(nn.Module):
    def __init__(self, semantic_dim: int, hidden_dim: int):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        self.net = mlp(semantic_dim, hidden_dim, hidden_dim) if semantic_dim > 0 else None

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return torch.zeros((semantic.shape[0], self.hidden_dim), dtype=semantic.dtype, device=semantic.device)
        return self.net(semantic)


class SemanticPriorHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim), nn.Sigmoid())

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class AuxiliaryHeads(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.success = nn.Linear(hidden_dim, 1)
        self.deadline = nn.Linear(hidden_dim, 1)
        self.delay = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "success_logit": self.success(embedding).squeeze(-1),
            "deadline_logit": self.deadline(embedding).squeeze(-1),
            "delay": self.delay(embedding).squeeze(-1),
        }


class SLGSAGEActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        semantic_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        global_state_dim: int | None = None,
        initial_log_std: float = -0.7,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.semantic_dim = semantic_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.raw_encoder = mlp(obs_dim, hidden_dim, hidden_dim)
        self.semantic_encoder = ServiceSemanticEncoder(semantic_dim, hidden_dim)
        self.actor_body = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh())
        if action_dim >= 4:
            self.route_head = nn.Linear(hidden_dim, 3)
            self.control_head = nn.Linear(hidden_dim, action_dim - 3)
        else:
            self.route_head = None
            self.control_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
        self.global_state_dim = int(global_state_dim or obs_dim * n_agents)
        self.critic = mlp(self.global_state_dim, hidden_dim, 1)
        self.prior_head = SemanticPriorHead(hidden_dim, action_dim)
        self.aux_heads = AuxiliaryHeads(hidden_dim)

    def policy(self, obs: torch.Tensor, semantic: torch.Tensor):
        raw_h = self.raw_encoder(obs)
        sem_h = self.semantic_encoder(semantic)
        actor_h = self.actor_body(torch.cat([raw_h, sem_h], dim=-1))
        if self.route_head is not None:
            route = F.softmax(self.route_head(actor_h), dim=-1)
            control = torch.sigmoid(self.control_head(actor_h))
            mean = torch.cat([route, control], dim=-1)
        else:
            mean = torch.sigmoid(self.control_head(actor_h))
        std = self.log_std.exp().expand_as(mean)
        prior = self.prior_head(sem_h)
        aux = self.aux_heads(sem_h)
        return mean, std, prior, aux

    def value(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.critic(global_state).squeeze(-1)


@dataclass
class RolloutBatch:
    obs: list
    semantic: list
    global_state: list
    actions: list
    log_probs: list
    rewards: list
    dones: list
    values: list
    success_labels: list
    deadline_labels: list
    delay_labels: list


class SLGSAGEMAPPO:
    def __init__(self, env, config: dict[str, Any]):
        self.env = env
        cfg = config.get("algo", config)
        self.device = torch.device(cfg.get("device", "cpu"))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))
        self.clip_ratio = float(cfg.get("clip_ratio", 0.2))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.value_coef = float(cfg.get("value_coef", 0.5))
        self.lambda_prior_0 = float(cfg.get("lambda_prior_0", 0.2))
        self.lambda_aux = float(cfg.get("lambda_aux", 0.05))
        self.lambda_cons = float(cfg.get("lambda_cons", 0.05))
        self.decay_steps = float(cfg.get("decay_steps", 100000))
        self.update_epochs = int(cfg.get("update_epochs", 4))
        self.minibatch_size = int(cfg.get("minibatch_size", 512))
        self.global_step = 0
        if not env.agent_ids:
            env.reset(seed=int(config.get("seed", 0)))
        self.model = SLGSAGEActorCritic(
            env.obs_dim,
            env.semantic_dim,
            env.action_dim,
            len(env.agent_ids),
            int(cfg.get("hidden_dim", 128)),
            global_state_dim=env.global_state_dim,
            initial_log_std=float(cfg.get("initial_log_std", -0.7)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg.get("learning_rate", 3e-4)))

    def semantic_prior_coef(self) -> float:
        return float(self.lambda_prior_0 * np.exp(-self.global_step / max(1.0, self.decay_steps)))

    @torch.no_grad()
    def act(self, obs_dict: dict, deterministic: bool = False):
        obs = torch.tensor(np.stack([obs_dict[aid]["obs"] for aid in self.env.agent_ids]), dtype=torch.float32, device=self.device)
        sem = torch.tensor(np.stack([obs_dict[aid]["semantic"] for aid in self.env.agent_ids]), dtype=torch.float32, device=self.device)
        mean, std, _, _ = self.model.policy(obs, sem)
        dist = Normal(mean, std)
        action_t = mean if deterministic else dist.rsample()
        action_t = action_t.clamp(0.0, 1.0)
        action_t = self._postprocess_action_tensor(action_t)
        log_prob = dist.log_prob(action_t).sum(-1)
        global_state = torch.tensor(self.env.get_global_state(), dtype=torch.float32, device=self.device).unsqueeze(0)
        value = self.model.value(global_state).item()
        actions = {aid: action_t[i].cpu().numpy() for i, aid in enumerate(self.env.agent_ids)}
        records = {
            "obs": obs.cpu().numpy(),
            "semantic": sem.cpu().numpy(),
            "global_state": global_state.squeeze(0).cpu().numpy(),
            "actions": action_t.cpu().numpy(),
            "log_probs": log_prob.cpu().numpy(),
            "value": value,
            "success_labels": np.array([obs_dict[aid]["task_label_success"] for aid in self.env.agent_ids], dtype=np.float32),
            "deadline_labels": np.array([obs_dict[aid]["task_label_deadline_violation"] for aid in self.env.agent_ids], dtype=np.float32),
            "delay_labels": np.array([obs_dict[aid]["task_label_delay"] for aid in self.env.agent_ids], dtype=np.float32),
        }
        return actions, records

    def _postprocess_action_tensor(self, action_t: torch.Tensor) -> torch.Tensor:
        if self.env.action_mode != "simple" or action_t.shape[-1] < 3:
            return action_t
        processed = action_t.clone()
        auv_idx = [i for i, aid in enumerate(self.env.agent_ids) if self.env.nodes[aid].role == "auv"]
        if auv_idx:
            route = processed[auv_idx, :3]
            route_sum = route.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            processed[auv_idx, :3] = route / route_sum
        return processed

    def collect_episode(self, seed: int | None = None, train: bool = True):
        obs, _ = self.env.reset(seed=seed)
        batch = RolloutBatch([], [], [], [], [], [], [], [], [], [], [])
        metrics_rows = []
        done = False
        while not done:
            actions, rec = self.act(obs, deterministic=not train)
            next_obs, rewards, done, _, info = self.env.step(actions)
            reward = float(np.mean(list(rewards.values())))
            batch.obs.append(rec["obs"])
            batch.semantic.append(rec["semantic"])
            batch.global_state.append(rec["global_state"])
            batch.actions.append(rec["actions"])
            batch.log_probs.append(rec["log_probs"])
            batch.values.append(rec["value"])
            batch.rewards.append(reward)
            batch.dones.append(float(done))
            batch.success_labels.append(rec["success_labels"])
            batch.deadline_labels.append(rec["deadline_labels"])
            batch.delay_labels.append(rec["delay_labels"])
            metrics_rows.append({"reward": reward, **info["metrics"], **self._action_metrics(rec["actions"])})
            obs = next_obs
            self.global_step += len(self.env.agent_ids)
        return batch, metrics_rows

    def _action_metrics(self, actions: np.ndarray) -> dict[str, float]:
        if actions.size == 0:
            return {}
        metrics = {
            "action_local_ratio_mean": float(np.mean(actions[:, 0])) if actions.shape[1] > 0 else 0.0,
            "action_usv_or_accept_mean": float(np.mean(actions[:, 1])) if actions.shape[1] > 1 else 0.0,
            "action_uav_or_relay_mean": float(np.mean(actions[:, 2])) if actions.shape[1] > 2 else 0.0,
            "action_tx_power_mean": float(np.mean(actions[:, 3])) if actions.shape[1] > 3 else 0.0,
        }
        auv_idx = [i for i, aid in enumerate(self.env.agent_ids) if self.env.nodes[aid].role == "auv"]
        if auv_idx:
            auv_actions = actions[auv_idx]
            metrics.update(
                {
                    "auv_local_ratio_mean": float(np.mean(auv_actions[:, 0])) if auv_actions.shape[1] > 0 else 0.0,
                    "auv_usv_ratio_mean": float(np.mean(auv_actions[:, 1])) if auv_actions.shape[1] > 1 else 0.0,
                    "auv_uav_ratio_mean": float(np.mean(auv_actions[:, 2])) if auv_actions.shape[1] > 2 else 0.0,
                    "auv_tx_power_mean": float(np.mean(auv_actions[:, 3])) if auv_actions.shape[1] > 3 else 0.0,
                }
            )
        return metrics

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        rewards = np.asarray(batch.rewards, dtype=np.float32)
        dones = np.asarray(batch.dones, dtype=np.float32)
        values = np.asarray(batch.values + [0.0], dtype=np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n_agents = len(self.env.agent_ids)
        obs = torch.tensor(np.concatenate(batch.obs, axis=0), dtype=torch.float32, device=self.device)
        sem = torch.tensor(np.concatenate(batch.semantic, axis=0), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.concatenate(batch.actions, axis=0), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(np.concatenate(batch.log_probs, axis=0), dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(np.repeat(adv, n_agents), dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        global_state = torch.tensor(np.stack(batch.global_state), dtype=torch.float32, device=self.device)
        success = torch.tensor(np.concatenate(batch.success_labels), dtype=torch.float32, device=self.device)
        deadline = torch.tensor(np.concatenate(batch.deadline_labels), dtype=torch.float32, device=self.device)
        delay = torch.tensor(np.concatenate(batch.delay_labels), dtype=torch.float32, device=self.device)
        prior_coef = self.semantic_prior_coef()
        last = {}
        for _ in range(self.update_epochs):
            mean, std, prior, aux = self.model.policy(obs, sem)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(log_probs - old_log_probs)
            policy_loss = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t).mean()
            values_pred = self.model.value(global_state)
            value_loss = F.mse_loss(values_pred, returns_t)
            entropy = dist.entropy().sum(-1).mean()
            semantic_prior_loss = F.mse_loss(mean, prior.detach())
            success_loss = F.binary_cross_entropy_with_logits(aux["success_logit"], success)
            deadline_loss = F.binary_cross_entropy_with_logits(aux["deadline_logit"], deadline)
            delay_loss = F.mse_loss(aux["delay"], delay)
            semantic_aux_loss = success_loss + deadline_loss + delay_loss
            semantic_consistency_loss = F.mse_loss(mean.detach(), prior)
            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
                + prior_coef * semantic_prior_loss
                + self.lambda_aux * semantic_aux_loss
                + self.lambda_cons * semantic_consistency_loss
            )
            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            last = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy_loss": float(entropy.detach().cpu()),
                "semantic_prior_loss": float(semantic_prior_loss.detach().cpu()),
                "semantic_aux_loss": float(semantic_aux_loss.detach().cpu()),
                "semantic_consistency_loss": float(semantic_consistency_loss.detach().cpu()),
                "semantic_prior_coef": float(prior_coef),
                "total_loss": float(total_loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
        return last

    def evaluate(self, episodes: int = 5, seed: int = 0) -> dict[str, float]:
        rows = []
        for i in range(episodes):
            _, metrics = self.collect_episode(seed=seed + 10000 + i, train=False)
            rows.extend(metrics)
        return {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in rows[0]} if rows else {}
