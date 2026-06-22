"""MAPPO baseline for service offloading using DI-engine's MAVAC backbone.

This module intentionally contains no semantic prior, semantic reward, or
auxiliary semantic loss. It is a small training harness around DI-engine's
native multi-agent actor-critic model so the service environment can be audited
before adding SLG-SAGE mechanisms.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal

from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import ENERGY_ACCOUNTING_KEYS, mean_metrics

ROOT = Path(__file__).resolve().parents[2]
DI_ENGINE = ROOT / "DI-engine-main"
if DI_ENGINE.exists() and str(DI_ENGINE) not in sys.path:
    sys.path.insert(0, str(DI_ENGINE))

from ding.model import MAVAC  # noqa: E402


@dataclass
class RolloutBatch:
    obs: np.ndarray
    global_obs: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    metrics: dict[str, float]
    train_return: float


class ServiceMAPPOTrainer:
    def __init__(self, config: dict[str, Any], run_dir: Path):
        self.config = config
        self.env = make_service_env(config)
        self.eval_env = make_service_env(config)
        self.seed = int(config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.device = torch.device(config.get("device", "cpu"))
        mcfg = config.get("mappo", {})
        hidden_dim = int(mcfg.get("hidden_dim", 128))
        self.discrete_actions = getattr(self.env, "action_mode", "simple") == "discrete_route"
        self.model = MAVAC(
            agent_obs_shape=self.env.obs_dim,
            global_obs_shape=self.env.global_state_dim,
            action_shape=self.env.action_dim,
            agent_num=len(self.env.agent_ids) if self.env.agent_ids else self.env.n_auv + self.env.n_usv + self.env.n_uav,
            actor_head_hidden_size=hidden_dim,
            critic_head_hidden_size=hidden_dim,
            action_space="discrete" if self.discrete_actions else "continuous",
            sigma_type="independent",
            bound_type="tanh",
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(mcfg.get("learning_rate", 3e-4)))
        self.gamma = float(mcfg.get("gamma", 0.99))
        self.gae_lambda = float(mcfg.get("gae_lambda", 0.95))
        self.clip_ratio = float(mcfg.get("clip_ratio", 0.2))
        self.entropy_coef = float(mcfg.get("entropy_coef", 0.01))
        self.value_loss_coef = float(mcfg.get("value_loss_coef", 0.5))
        self.max_grad_norm = float(mcfg.get("max_grad_norm", 0.5))
        self.update_epochs = int(mcfg.get("update_epochs", 4))
        self.rollout_steps = int(mcfg.get("rollout_steps", 256))
        self.minibatch_steps = int(mcfg.get("minibatch_steps", min(128, self.rollout_steps)))
        self.eval_episodes = int(mcfg.get("eval_episodes", 5))
        self.eval_freq = int(mcfg.get("eval_freq", 10000))
        self.total_env_steps = int(mcfg.get("total_env_steps", 100000))
        self.report_stochastic_eval = bool(mcfg.get("report_stochastic_eval", False))
        self.run_dir = run_dir
        self.train_csv = run_dir / "train_curve.csv"
        self.eval_csv = run_dir / "eval_curve.csv"
        self.best_eval_return = -float("inf")
        self.best_eval_metrics: dict[str, float] = {}
        self.best_eval_step = 0
        self.best_stochastic_eval_return = -float("inf")
        self.best_stochastic_eval_metrics: dict[str, float] = {}
        self.best_stochastic_eval_step = 0
        self.initial_env_steps = 0
        self.anneal_after_positive_eval_drop = bool(mcfg.get("anneal_after_positive_eval_drop", False))
        self.anneal_factor = float(mcfg.get("anneal_factor", 0.2))
        self.anneal_min_learning_rate = float(mcfg.get("anneal_min_learning_rate", 1e-5))
        self.learning_rate_annealed = False
        self._obs, _ = self.env.reset(seed=self.seed)
        self.episode = 0

    def _obs_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target_env = env or self.env
        return np.stack([obs[aid]["obs"] for aid in target_env.agent_ids], axis=0).astype(np.float32)

    def _global_array(self, env=None) -> np.ndarray:
        target_env = env or self.env
        state = target_env.get_global_state().astype(np.float32)
        return np.repeat(state[None, :], len(target_env.agent_ids), axis=0)

    def _action_mask_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target_env = env or self.env
        return np.stack(
            [
                obs[aid].get("action_mask", np.ones(target_env.action_dim, dtype=np.float32))
                for aid in target_env.agent_ids
            ],
            axis=0,
        ).astype(np.float32)

    def _forward(self, obs_arr: np.ndarray, global_arr: np.ndarray, action_mask_arr: np.ndarray | None = None):
        obs_t = torch.as_tensor(obs_arr[None, ...], dtype=torch.float32, device=self.device)
        global_t = torch.as_tensor(global_arr[None, ...], dtype=torch.float32, device=self.device)
        inputs = {"agent_state": obs_t, "global_state": global_t}
        if self.discrete_actions:
            if action_mask_arr is None:
                action_mask_arr = np.ones((obs_arr.shape[0], self.env.action_dim), dtype=np.float32)
            action_mask_t = torch.as_tensor(action_mask_arr[None, ...], dtype=torch.float32, device=self.device)
            inputs["action_mask"] = action_mask_t
        out = self.model(inputs, mode="compute_actor_critic")
        if self.discrete_actions:
            logits = out["logit"].squeeze(0).masked_fill(action_mask_t.squeeze(0) <= 0.0, -1e9)
            return logits, None, out["value"].squeeze(0)
        mu = out["logit"]["mu"].squeeze(0)
        sigma = out["logit"]["sigma"].squeeze(0).clamp_min(1e-4)
        value = out["value"].squeeze(0)
        return mu, sigma, value

    @staticmethod
    def _env_action(raw_action: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-raw_action))).astype(np.float32)

    def _action_metrics(self, actions: dict[str, np.ndarray], env=None) -> dict[str, float]:
        target_env = env or self.env
        if self.discrete_actions:
            role_actions = {
                role: [int(actions[aid]) for aid in target_env.agent_ids if target_env.nodes[aid].role == role]
                for role in ["auv", "usv", "uav"]
            }
            return {
                "auv_local_ratio_mean": float(np.mean([a == 0 for a in role_actions["auv"]])) if role_actions["auv"] else 0.0,
                "auv_upload_usv_ratio_mean": float(np.mean([a in {1, 2} for a in role_actions["auv"]])) if role_actions["auv"] else 0.0,
                "auv_idle_ratio_mean": float(np.mean([a == 3 for a in role_actions["auv"]])) if role_actions["auv"] else 0.0,
                "usv_local_compute_preference_mean": float(np.mean([a == 0 for a in role_actions["usv"]])) if role_actions["usv"] else 0.0,
                "usv_forward_uav_preference_mean": float(np.mean([a == 1 for a in role_actions["usv"]])) if role_actions["usv"] else 0.0,
                "usv_forward_shore_preference_mean": float(np.mean([a == 2 for a in role_actions["usv"]])) if role_actions["usv"] else 0.0,
                "usv_idle_ratio_mean": float(np.mean([a == 3 for a in role_actions["usv"]])) if role_actions["usv"] else 0.0,
                "uav_local_compute_preference_mean": float(np.mean([a == 0 for a in role_actions["uav"]])) if role_actions["uav"] else 0.0,
                "uav_forward_shore_preference_mean": float(np.mean([a == 1 for a in role_actions["uav"]])) if role_actions["uav"] else 0.0,
                "uav_idle_ratio_mean": float(np.mean([a == 3 for a in role_actions["uav"]])) if role_actions["uav"] else 0.0,
            }
        matrix = np.stack([actions[aid] for aid in target_env.agent_ids], axis=0)
        metrics = {
            "action_local_ratio_mean": float(matrix[:, 0].mean()) if matrix.shape[1] > 0 else 0.0,
            "action_usv_or_accept_mean": float(matrix[:, 1].mean()) if matrix.shape[1] > 1 else 0.0,
            "action_uav_or_relay_mean": float(matrix[:, 2].mean()) if matrix.shape[1] > 2 else 0.0,
            "action_tx_power_mean": float(matrix[:, 3].mean()) if matrix.shape[1] > 3 else 0.0,
        }
        dual_hop = getattr(target_env, "env_model", "legacy") == "dual_hop_queue"
        auv_ids = [aid for aid in target_env.agent_ids if target_env.nodes[aid].role == "auv"]
        if auv_ids:
            auv = np.stack([actions[aid] for aid in auv_ids], axis=0)
            if dual_hop:
                denom = np.clip(auv[:, :2].sum(axis=1, keepdims=True), 1e-6, None)
                ratios = auv[:, :2] / denom
                metrics.update(
                    {
                        "auv_local_ratio_mean": float(ratios[:, 0].mean()),
                        "auv_upload_usv_ratio_mean": float(ratios[:, 1].mean()),
                        "auv_usv_selector_mean": float(auv[:, 2].mean()),
                        "auv_tx_power_mean": float(auv[:, 3].mean()),
                    }
                )
            else:
                denom = np.clip(auv[:, :3].sum(axis=1, keepdims=True), 1e-6, None)
                ratios = auv[:, :3] / denom
                metrics.update(
                    {
                        "auv_local_ratio_mean": float(ratios[:, 0].mean()),
                        "auv_usv_ratio_mean": float(ratios[:, 1].mean()),
                        "auv_uav_ratio_mean": float(ratios[:, 2].mean()),
                        "auv_tx_power_mean": float(auv[:, 3].mean()) if auv.shape[1] > 3 else 0.0,
                    }
                )
        usv_ids = [aid for aid in target_env.agent_ids if target_env.nodes[aid].role == "usv"]
        if usv_ids:
            usv = np.stack([actions[aid] for aid in usv_ids], axis=0)
            if dual_hop:
                metrics.update(
                    {
                        "usv_local_compute_preference_mean": float(usv[:, 0].mean()),
                        "usv_forward_uav_preference_mean": float(usv[:, 1].mean()),
                        "usv_forward_shore_preference_mean": float(usv[:, 2].mean()),
                    }
                )
            else:
                metrics.update(
                    {
                        "usv_accept_ratio_mean": float(usv[:, 0].mean()),
                        "usv_cpu_allocation_ratio_mean": float(usv[:, 1].mean()) if usv.shape[1] > 1 else 0.0,
                        "usv_relay_ratio_mean": float(usv[:, 2].mean()) if usv.shape[1] > 2 else 0.0,
                    }
                )
        uav_ids = [aid for aid in target_env.agent_ids if target_env.nodes[aid].role == "uav"]
        if uav_ids:
            uav = np.stack([actions[aid] for aid in uav_ids], axis=0)
            if dual_hop:
                metrics.update(
                    {
                        "uav_local_compute_preference_mean": float(uav[:, 0].mean()),
                        "uav_forward_shore_preference_mean": float(uav[:, 1].mean()),
                        "uav_cpu_effort_mean": float(uav[:, 2].mean()),
                    }
                )
            else:
                metrics.update(
                    {
                        "uav_accept_ratio_mean": float(uav[:, 0].mean()),
                        "uav_cpu_allocation_ratio_mean": float(uav[:, 1].mean()) if uav.shape[1] > 1 else 0.0,
                        "uav_relay_ratio_mean": float(uav[:, 2].mean()) if uav.shape[1] > 2 else 0.0,
                    }
                )
        return metrics

    def _parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.model.parameters()))

    def _inference_time_ms_per_decision(self, env=None) -> float:
        target_env = env or self.eval_env
        obs, _ = target_env.reset(seed=self.seed + 777000)
        obs_arr = self._obs_array(obs, target_env)
        global_arr = self._global_array(target_env)
        action_masks = self._action_mask_array(obs, target_env)
        iterations = int(self.config.get("mappo", {}).get("inference_benchmark_iters", 10))
        if iterations <= 0:
            return 0.0
        with torch.no_grad():
            self._forward(obs_arr, global_arr, action_masks)
            start = time.perf_counter()
            for _ in range(iterations):
                self._forward(obs_arr, global_arr, action_masks)
            elapsed = time.perf_counter() - start
        return float(1000.0 * elapsed / max(1, iterations * len(target_env.agent_ids)))

    @staticmethod
    def _mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        return {k: float(np.mean([row.get(k, 0.0) for row in rows])) for k in rows[0].keys()}

    def collect(self) -> RolloutBatch:
        obs_buf, global_buf, mask_buf, action_buf, logp_buf, value_buf = [], [], [], [], [], []
        reward_buf, done_buf, metrics_rows, action_rows = [], [], [], []
        train_return = 0.0
        for _ in range(self.rollout_steps):
            obs_arr = self._obs_array(self._obs)
            global_arr = self._global_array()
            action_masks = self._action_mask_array(self._obs)
            with torch.no_grad():
                policy_out, sigma, value = self._forward(obs_arr, global_arr, action_masks)
                if self.discrete_actions:
                    dist = Categorical(logits=policy_out)
                    raw_action = dist.sample()
                else:
                    dist = Independent(Normal(policy_out, sigma), 1)
                    raw_action = dist.sample()
                log_prob = dist.log_prob(raw_action)
            raw_np = raw_action.cpu().numpy()
            if self.discrete_actions:
                env_actions = {aid: int(raw_np[i]) for i, aid in enumerate(self.env.agent_ids)}
            else:
                env_actions = {aid: self._env_action(raw_np[i]) for i, aid in enumerate(self.env.agent_ids)}
            next_obs, rewards, done, _, info = self.env.step(env_actions)
            reward_arr = np.array([float(rewards[aid]) for aid in self.env.agent_ids], dtype=np.float32)
            obs_buf.append(obs_arr)
            global_buf.append(global_arr)
            mask_buf.append(action_masks)
            action_buf.append(raw_np.astype(np.float32))
            logp_buf.append(log_prob.cpu().numpy().astype(np.float32))
            value_buf.append(value.cpu().numpy().astype(np.float32))
            reward_buf.append(reward_arr)
            done_buf.append(np.full(len(self.env.agent_ids), float(done), dtype=np.float32))
            metrics_rows.append(info["metrics"])
            action_rows.append(self._action_metrics(env_actions))
            train_return += float(np.mean(reward_arr))
            self._obs = next_obs
            if done:
                self.episode += 1
                self._obs, _ = self.env.reset(seed=self.seed + self.episode)

        with torch.no_grad():
            last_value = self._forward(
                self._obs_array(self._obs), self._global_array(), self._action_mask_array(self._obs)
            )[2].cpu().numpy().astype(np.float32)
        rewards = np.asarray(reward_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)
        values = np.asarray(value_buf, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            next_value = last_value if t == self.rollout_steps - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        metrics = mean_metrics(metrics_rows)
        metrics.update(self._mean_dicts(action_rows))
        return RolloutBatch(
            obs=np.asarray(obs_buf, dtype=np.float32),
            global_obs=np.asarray(global_buf, dtype=np.float32),
            action_masks=np.asarray(mask_buf, dtype=np.float32),
            actions=np.asarray(action_buf, dtype=np.int64 if self.discrete_actions else np.float32),
            log_probs=np.asarray(logp_buf, dtype=np.float32),
            values=values,
            rewards=rewards,
            dones=dones,
            advantages=advantages,
            returns=returns,
            metrics=metrics,
            train_return=train_return,
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        T = batch.obs.shape[0]
        indices = np.arange(T)
        adv_all = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
        losses = []
        for _ in range(self.update_epochs):
            self.rng.shuffle(indices)
            for start in range(0, T, self.minibatch_steps):
                mb = indices[start : start + self.minibatch_steps]
                obs = torch.as_tensor(batch.obs[mb], dtype=torch.float32, device=self.device)
                global_obs = torch.as_tensor(batch.global_obs[mb], dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(
                    batch.actions[mb], dtype=torch.long if self.discrete_actions else torch.float32, device=self.device
                )
                action_masks = torch.as_tensor(batch.action_masks[mb], dtype=torch.float32, device=self.device)
                old_logp = torch.as_tensor(batch.log_probs[mb], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(batch.returns[mb], dtype=torch.float32, device=self.device)
                adv = adv_all[mb]
                inputs = {"agent_state": obs, "global_state": global_obs}
                if self.discrete_actions:
                    inputs["action_mask"] = action_masks
                out = self.model(inputs, mode="compute_actor_critic")
                values = out["value"]
                if self.discrete_actions:
                    logits = out["logit"].masked_fill(action_masks <= 0.0, -1e9)
                    dist = Categorical(logits=logits)
                else:
                    mu = out["logit"]["mu"]
                    sigma = out["logit"]["sigma"].clamp_min(1e-4)
                    dist = Independent(Normal(mu, sigma), 1)
                logp = dist.log_prob(actions)
                ratio = torch.exp(logp - old_logp)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_loss = dist.entropy().mean()
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "entropy_loss": float(entropy_loss.detach().cpu()),
                        "total_loss": float(total_loss.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                    }
                )
        return {k: float(np.mean([row[k] for row in losses])) for k in losses[0]}

    def evaluate(self, seed_offset: int = 10000, deterministic: bool = True) -> dict[str, float]:
        env = self.eval_env
        rows = []
        terminal_rows = []
        action_rows = []
        returns = []
        rng_state = torch.random.get_rng_state()
        try:
            if not deterministic:
                torch.manual_seed(self.seed + seed_offset + 900000)
            for ep in range(self.eval_episodes):
                obs, _ = env.reset(seed=self.seed + seed_offset + ep)
                done = False
                ep_return = 0.0
                while not done:
                    obs_arr = self._obs_array(obs, env)
                    global_arr = self._global_array(env)
                    action_masks = self._action_mask_array(obs, env)
                    with torch.no_grad():
                        policy_out, sigma, _ = self._forward(obs_arr, global_arr, action_masks)
                    if self.discrete_actions:
                        action_t = torch.argmax(policy_out, dim=-1) if deterministic else Categorical(logits=policy_out).sample()
                        raw_np = action_t.cpu().numpy()
                        env_actions = {aid: int(raw_np[i]) for i, aid in enumerate(env.agent_ids)}
                    else:
                        raw_t = policy_out if deterministic else Independent(Normal(policy_out, sigma), 1).sample()
                        raw_np = raw_t.cpu().numpy()
                        env_actions = {aid: self._env_action(raw_np[i]) for i, aid in enumerate(env.agent_ids)}
                    obs, rewards, done, _, info = env.step(env_actions)
                    ep_return += float(np.mean(list(rewards.values())))
                    rows.append(info["metrics"])
                    action_rows.append(self._action_metrics(env_actions, env))
                    if done:
                        terminal_rows.append(info["metrics"])
                returns.append(ep_return)
        finally:
            torch.random.set_rng_state(rng_state)
        out = mean_metrics(rows)
        terminal_metrics = mean_metrics(terminal_rows)
        out.update({key: terminal_metrics[key] for key in ENERGY_ACCOUNTING_KEYS})
        out.update(self._mean_dicts(action_rows))
        out["eval_return"] = float(np.mean(returns))
        out["parameter_count"] = float(self._parameter_count())
        out["inference_time_ms_per_decision"] = self._inference_time_ms_per_decision(env)
        return out

    @staticmethod
    def _append_csv(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        existing_rows: list[dict[str, str]] = []
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                prior_fields = list(reader.fieldnames or [])
                existing_rows = list(reader)
            fieldnames = prior_fields + [field for field in fieldnames if field not in prior_fields]
            if fieldnames != prior_fields:
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not existing_rows and f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def _save_checkpoint(self, step: int, eval_metrics: dict[str, float], filename: str = "checkpoint_latest.pt") -> Path:
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        target = checkpoint_dir / filename
        torch.save(
            {
                "step": int(step),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "eval_metrics": dict(eval_metrics),
                "config": self.config,
            },
            target,
        )
        return target

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.initial_env_steps = int(payload.get("step", 0))
        self.best_eval_metrics = dict(payload.get("eval_metrics", {}))
        self.best_eval_return = float(self.best_eval_metrics.get("eval_return", -float("inf")))
        self.best_eval_step = self.initial_env_steps

    def _maybe_anneal_learning_rate(self, eval_return: float) -> float:
        if (
            self.anneal_after_positive_eval_drop
            and not self.learning_rate_annealed
            and self.best_eval_return > 0.0
            and eval_return < self.best_eval_return
        ):
            for group in self.optimizer.param_groups:
                group["lr"] = max(self.anneal_min_learning_rate, float(group["lr"]) * self.anneal_factor)
            self.learning_rate_annealed = True
        return float(self.optimizer.param_groups[0]["lr"])

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        env_steps = self.initial_env_steps
        last_eval: dict[str, float] = {}
        last_stochastic_eval: dict[str, float] = {}
        next_eval = env_steps
        while env_steps < self.total_env_steps:
            batch = self.collect()
            losses = self.update(batch)
            env_steps += self.rollout_steps
            row = {
                "step": env_steps,
                "episode": self.episode,
                "train_return": batch.train_return,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                **losses,
                **batch.metrics,
            }
            self._append_csv(self.train_csv, row)
            if env_steps >= next_eval:
                last_eval = self.evaluate()
                if self.report_stochastic_eval:
                    last_stochastic_eval = self.evaluate(deterministic=False)
                eval_row = {"step": env_steps, **last_eval}
                eval_row.update({f"stochastic_{key}": value for key, value in last_stochastic_eval.items()})
                if last_stochastic_eval:
                    eval_row["stochastic_deterministic_gap"] = float(
                        last_stochastic_eval.get("eval_return", 0.0) - last_eval.get("eval_return", 0.0)
                    )
                self._append_csv(self.eval_csv, eval_row)
                self._save_checkpoint(env_steps, last_eval)
                eval_return = float(last_eval.get("eval_return", -float("inf")))
                if eval_return > self.best_eval_return:
                    self.best_eval_return = eval_return
                    self.best_eval_metrics = dict(last_eval)
                    self.best_eval_step = env_steps
                    self._save_checkpoint(env_steps, last_eval, "checkpoint_best.pt")
                else:
                    self._maybe_anneal_learning_rate(eval_return)
                if last_stochastic_eval:
                    stochastic_return = float(last_stochastic_eval.get("eval_return", -float("inf")))
                    if stochastic_return > self.best_stochastic_eval_return:
                        self.best_stochastic_eval_return = stochastic_return
                        self.best_stochastic_eval_metrics = dict(last_stochastic_eval)
                        self.best_stochastic_eval_step = env_steps
                        self._save_checkpoint(env_steps, last_stochastic_eval, "checkpoint_best_stochastic.pt")
                next_eval += self.eval_freq
                print(
                    f"[service_mappo] step={env_steps} train_return={batch.train_return:.3f} "
                    f"eval_return={last_eval.get('eval_return', 0.0):.3f} "
                    f"stochastic_eval_return={last_stochastic_eval.get('eval_return', float('nan')):.3f} "
                    f"completion={last_eval.get('completion_ratio', 0.0):.3f} "
                    f"deadline={last_eval.get('deadline_violation_rate', 0.0):.3f} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
        summary = {
            "total_env_steps": env_steps,
            "episodes": self.episode,
            "last_eval": last_eval,
            "best_eval_return": self.best_eval_return,
            "best_eval_step": self.best_eval_step,
            "best_eval": self.best_eval_metrics,
            "last_stochastic_eval": last_stochastic_eval,
            "best_stochastic_eval_return": self.best_stochastic_eval_return,
            "best_stochastic_eval_step": self.best_stochastic_eval_step,
            "best_stochastic_eval": self.best_stochastic_eval_metrics,
            "final_learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "env": {
                "agent_num": len(self.env.agent_ids),
                "obs_shape": self.env.obs_dim,
                "global_obs_shape": self.env.global_state_dim,
                "action_shape": self.env.action_dim,
                "action_mode": self.env.action_mode,
                "obs_mode": self.env.obs_mode,
                "reward_mode": self.env.reward_mode,
                "difficulty": self.env.difficulty,
            },
        }
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
