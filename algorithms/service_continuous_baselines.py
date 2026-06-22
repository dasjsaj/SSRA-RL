"""Continuous-control MARL baselines for ServiceComputing paper experiments.

These adapters cover MADDPG, MASAC, and MATD3 under the environment's simple
continuous action mode. They intentionally live outside DI-engine internals and
share the same run/evaluation/checkpoint format as the MAPPO paper runner.
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import ENERGY_ACCOUNTING_KEYS, mean_metrics


@dataclass
class ContinuousBatch:
    obs: torch.Tensor
    global_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    next_global_obs: torch.Tensor
    dones: torch.Tensor


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs))


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        return self.mean(hidden), self.log_std(hidden).clamp(-5.0, 1.5)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action = torch.sigmoid(raw)
        log_prob = normal.log_prob(raw) - torch.log(action * (1.0 - action) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.sigmoid(mean)


class CentralCritic(nn.Module):
    def __init__(self, global_dim: int, agent_num: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim + agent_num * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        flat_actions = actions.flatten(start_dim=1)
        return self.net(torch.cat([global_obs, flat_actions], dim=-1)).view(-1)


class ServiceContinuousMARLTrainer:
    """Compact continuous trainer for MADDPG/MASAC/MATD3-style baselines."""

    def __init__(self, config: dict[str, Any], run_dir: Path, algo: str):
        self.config = config
        self.algo = algo.lower()
        self.env = make_service_env(config)
        self.eval_env = make_service_env(config)
        if getattr(self.env, "action_mode", "simple") == "discrete_route":
            raise ValueError(f"{self.algo} adapter requires a continuous/simple action mode")
        self.seed = int(config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.device = torch.device(config.get("device", "cpu"))
        mcfg = config.get("mappo", {})
        ccfg = config.get("continuous_baseline", {})
        self.total_env_steps = int(mcfg.get("total_env_steps", 100000))
        self.rollout_steps = int(mcfg.get("rollout_steps", 256))
        self.eval_freq = int(mcfg.get("eval_freq", 5000))
        self.eval_episodes = int(mcfg.get("eval_episodes", 8))
        self.gamma = float(mcfg.get("gamma", 0.99))
        self.hidden_dim = int(mcfg.get("hidden_dim", 128))
        self.learning_rate = float(mcfg.get("learning_rate", 1e-4))
        self.batch_size = int(ccfg.get("batch_size", 128))
        self.buffer_size = int(ccfg.get("buffer_size", 100000))
        self.updates_per_collect = int(ccfg.get("updates_per_collect", 1))
        self.tau = float(ccfg.get("target_tau", 0.01))
        self.exploration_noise = float(ccfg.get("exploration_noise", 0.12))
        self.policy_noise = float(ccfg.get("policy_noise", 0.08))
        self.policy_noise_clip = float(ccfg.get("policy_noise_clip", 0.20))
        self.policy_delay = int(ccfg.get("policy_delay", 2))
        self.alpha = float(ccfg.get("sac_alpha", 0.05))
        self.agent_num = self.env.n_auv + self.env.n_usv + self.env.n_uav
        self.action_dim = self.env.action_dim
        self.actor: nn.Module
        self.target_actor: nn.Module
        if self.algo == "masac":
            self.actor = SquashedGaussianActor(self.env.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
            self.target_actor = SquashedGaussianActor(self.env.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        else:
            self.actor = DeterministicActor(self.env.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
            self.target_actor = DeterministicActor(self.env.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic1 = CentralCritic(self.env.global_state_dim, self.agent_num, self.action_dim, self.hidden_dim).to(
            self.device
        )
        self.target_critic1 = CentralCritic(
            self.env.global_state_dim, self.agent_num, self.action_dim, self.hidden_dim
        ).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.critic2: CentralCritic | None = None
        self.target_critic2: CentralCritic | None = None
        if self.algo in {"masac", "matd3"}:
            self.critic2 = CentralCritic(
                self.env.global_state_dim, self.agent_num, self.action_dim, self.hidden_dim
            ).to(self.device)
            self.target_critic2 = CentralCritic(
                self.env.global_state_dim, self.agent_num, self.action_dim, self.hidden_dim
            ).to(self.device)
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        critic_params = list(self.critic1.parameters())
        if self.critic2 is not None:
            critic_params += list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.learning_rate)
        self.replay: deque[tuple[np.ndarray, ...]] = deque(maxlen=self.buffer_size)
        self.run_dir = run_dir
        self.train_csv = run_dir / "train_curve.csv"
        self.eval_csv = run_dir / "eval_curve.csv"
        self.best_stochastic_eval_return = -float("inf")
        self.best_stochastic_eval_metrics: dict[str, float] = {}
        self.best_stochastic_eval_step = 0
        self.update_step = 0
        self._obs, _ = self.env.reset(seed=self.seed)
        self.episode = 0

    def _obs_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target_env = env or self.env
        return np.stack([obs[aid]["obs"] for aid in target_env.agent_ids], axis=0).astype(np.float32)

    def _state_array(self, env=None) -> np.ndarray:
        target_env = env or self.env
        return target_env.get_global_state().astype(np.float32)

    @staticmethod
    def _mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        return {k: float(np.mean([row.get(k, 0.0) for row in rows])) for k in rows[0].keys()}

    def _actor_actions(self, obs_arr: np.ndarray, stochastic: bool, noise_scale: float = 0.0) -> np.ndarray:
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.algo == "masac":
                assert isinstance(self.actor, SquashedGaussianActor)
                actions = self.actor.sample(obs_t)[0] if stochastic else self.actor.mean_action(obs_t)
            else:
                assert isinstance(self.actor, DeterministicActor)
                actions = self.actor(obs_t)
                if stochastic and noise_scale > 0.0:
                    actions = actions + torch.randn_like(actions) * noise_scale
            return actions.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

    def _parameter_count(self) -> int:
        modules: list[nn.Module] = [self.actor, self.critic1]
        if self.critic2 is not None:
            modules.append(self.critic2)
        return int(sum(parameter.numel() for module in modules for parameter in module.parameters()))

    def _inference_time_ms_per_decision(self, env=None) -> float:
        target_env = env or self.eval_env
        obs, _ = target_env.reset(seed=self.seed + 777000)
        obs_arr = self._obs_array(obs, target_env)
        iterations = int(self.config.get("mappo", {}).get("inference_benchmark_iters", 10))
        if iterations <= 0:
            return 0.0
        with torch.no_grad():
            self._actor_actions(obs_arr, stochastic=False)
            start = time.perf_counter()
            for _ in range(iterations):
                self._actor_actions(obs_arr, stochastic=False)
            elapsed = time.perf_counter() - start
        return float(1000.0 * elapsed / max(1, iterations * len(target_env.agent_ids)))

    def _action_metrics(self, actions: dict[str, np.ndarray], env=None) -> dict[str, float]:
        target_env = env or self.env
        matrix = np.stack([actions[aid] for aid in target_env.agent_ids], axis=0)
        metrics = {
            "action_local_ratio_mean": float(matrix[:, 0].mean()) if matrix.shape[1] > 0 else 0.0,
            "action_usv_or_accept_mean": float(matrix[:, 1].mean()) if matrix.shape[1] > 1 else 0.0,
            "action_uav_or_relay_mean": float(matrix[:, 2].mean()) if matrix.shape[1] > 2 else 0.0,
            "action_tx_power_mean": float(matrix[:, 3].mean()) if matrix.shape[1] > 3 else 0.0,
        }
        for role in ["auv", "usv", "uav"]:
            ids = [aid for aid in target_env.agent_ids if target_env.nodes[aid].role == role]
            if ids:
                values = np.stack([actions[aid] for aid in ids], axis=0)
                metrics[f"{role}_action_mean_0"] = float(values[:, 0].mean())
                metrics[f"{role}_action_mean_1"] = float(values[:, 1].mean()) if values.shape[1] > 1 else 0.0
                metrics[f"{role}_action_mean_2"] = float(values[:, 2].mean()) if values.shape[1] > 2 else 0.0
                metrics[f"{role}_action_mean_3"] = float(values[:, 3].mean()) if values.shape[1] > 3 else 0.0
        return metrics

    def collect(self, env_steps: int) -> tuple[dict[str, float], float]:
        metrics_rows, action_rows = [], []
        train_return = 0.0
        noise = self.exploration_noise * max(0.05, 1.0 - env_steps / max(1, self.total_env_steps))
        for _ in range(self.rollout_steps):
            obs_arr = self._obs_array(self._obs)
            state = self._state_array()
            action_arr = self._actor_actions(obs_arr, stochastic=True, noise_scale=noise)
            actions = {aid: action_arr[index] for index, aid in enumerate(self.env.agent_ids)}
            next_obs, rewards, done, _, info = self.env.step(actions)
            reward = float(np.mean(list(rewards.values())))
            self.replay.append(
                (
                    obs_arr,
                    state,
                    action_arr,
                    np.asarray(reward, dtype=np.float32),
                    self._obs_array(next_obs),
                    self._state_array(),
                    np.asarray(float(done), dtype=np.float32),
                )
            )
            train_return += reward
            metrics_rows.append(info["metrics"])
            action_rows.append(self._action_metrics(actions))
            self._obs = next_obs
            if done:
                self.episode += 1
                self._obs, _ = self.env.reset(seed=self.seed + self.episode)
        metrics = mean_metrics(metrics_rows)
        metrics.update(self._mean_dicts(action_rows))
        metrics["exploration_noise"] = noise
        return metrics, train_return

    def _sample_batch(self) -> ContinuousBatch | None:
        if len(self.replay) < self.batch_size:
            return None
        indices = self.rng.choice(len(self.replay), size=self.batch_size, replace=False)
        rows = [self.replay[int(i)] for i in indices]
        stacked = [np.stack([row[i] for row in rows], axis=0) for i in range(7)]
        return ContinuousBatch(
            obs=torch.as_tensor(stacked[0], dtype=torch.float32, device=self.device),
            global_obs=torch.as_tensor(stacked[1], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(stacked[2], dtype=torch.float32, device=self.device),
            rewards=torch.as_tensor(stacked[3], dtype=torch.float32, device=self.device).view(-1),
            next_obs=torch.as_tensor(stacked[4], dtype=torch.float32, device=self.device),
            next_global_obs=torch.as_tensor(stacked[5], dtype=torch.float32, device=self.device),
            dones=torch.as_tensor(stacked[6], dtype=torch.float32, device=self.device).view(-1),
        )

    def _target_actions(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.algo == "masac":
            assert isinstance(self.target_actor, SquashedGaussianActor)
            actions, logp = self.target_actor.sample(obs)
            return actions, logp
        assert isinstance(self.target_actor, DeterministicActor)
        actions = self.target_actor(obs)
        if self.algo == "matd3":
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
            actions = (actions + noise).clamp(0.0, 1.0)
        logp = torch.zeros(actions.shape[:-1], dtype=torch.float32, device=self.device)
        return actions, logp

    def _policy_actions(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.algo == "masac":
            assert isinstance(self.actor, SquashedGaussianActor)
            return self.actor.sample(obs)
        assert isinstance(self.actor, DeterministicActor)
        actions = self.actor(obs)
        return actions, torch.zeros(actions.shape[:-1], dtype=torch.float32, device=self.device)

    def update(self) -> dict[str, float]:
        losses: list[dict[str, float]] = []
        for _ in range(self.updates_per_collect):
            batch = self._sample_batch()
            if batch is None:
                losses.append(
                    {"critic_loss": 0.0, "actor_loss": 0.0, "total_loss": 0.0, "grad_norm": 0.0}
                )
                continue
            with torch.no_grad():
                next_actions, next_logp = self._target_actions(batch.next_obs)
                target_q1 = self.target_critic1(batch.next_global_obs, next_actions)
                if self.target_critic2 is not None:
                    target_q2 = self.target_critic2(batch.next_global_obs, next_actions)
                    target_q = torch.minimum(target_q1, target_q2)
                else:
                    target_q = target_q1
                if self.algo == "masac":
                    target_q = target_q - self.alpha * next_logp.sum(dim=-1)
                td_target = batch.rewards + self.gamma * (1.0 - batch.dones) * target_q

            pred_q1 = self.critic1(batch.global_obs, batch.actions)
            critic_loss = F.mse_loss(pred_q1, td_target)
            if self.critic2 is not None:
                pred_q2 = self.critic2(batch.global_obs, batch.actions)
                critic_loss = critic_loss + F.mse_loss(pred_q2, td_target)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_params = list(self.critic1.parameters())
            if self.critic2 is not None:
                critic_params += list(self.critic2.parameters())
            critic_grad = torch.nn.utils.clip_grad_norm_(critic_params, 0.5)
            self.critic_optimizer.step()

            actor_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            should_update_actor = self.algo != "matd3" or self.update_step % self.policy_delay == 0
            if should_update_actor:
                policy_actions, logp = self._policy_actions(batch.obs)
                q = self.critic1(batch.global_obs, policy_actions)
                if self.critic2 is not None:
                    q = torch.minimum(q, self.critic2(batch.global_obs, policy_actions))
                actor_loss = (self.alpha * logp.sum(dim=-1) - q).mean() if self.algo == "masac" else -q.mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                self._soft_update(self.actor, self.target_actor)
                self._soft_update(self.critic1, self.target_critic1)
                if self.critic2 is not None and self.target_critic2 is not None:
                    self._soft_update(self.critic2, self.target_critic2)
            else:
                actor_grad = torch.tensor(0.0)
            self.update_step += 1
            total_loss = critic_loss.detach() + actor_loss.detach()
            losses.append(
                {
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "total_loss": float(total_loss.cpu()),
                    "grad_norm": float(
                        max(
                            float(critic_grad.detach().cpu() if torch.is_tensor(critic_grad) else critic_grad),
                            float(actor_grad.detach().cpu() if torch.is_tensor(actor_grad) else actor_grad),
                        )
                    ),
                }
            )
        return {key: float(np.mean([row[key] for row in losses])) for key in losses[0]}

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for src, dst in zip(source.parameters(), target.parameters()):
                dst.mul_(1.0 - self.tau).add_(src, alpha=self.tau)

    def evaluate(self, seed_offset: int = 10000, deterministic: bool = False) -> dict[str, float]:
        env = self.eval_env
        rows, terminal_rows, action_rows, returns = [], [], [], []
        for ep in range(self.eval_episodes):
            obs, _ = env.reset(seed=self.seed + seed_offset + ep)
            done = False
            ep_return = 0.0
            while not done:
                obs_arr = self._obs_array(obs, env)
                action_arr = self._actor_actions(
                    obs_arr,
                    stochastic=not deterministic,
                    noise_scale=0.04 if not deterministic and self.algo != "masac" else 0.0,
                )
                actions = {aid: action_arr[index] for index, aid in enumerate(env.agent_ids)}
                obs, rewards, done, _, info = env.step(actions)
                ep_return += float(np.mean(list(rewards.values())))
                rows.append(info["metrics"])
                action_rows.append(self._action_metrics(actions, env))
                if done:
                    terminal_rows.append(info["metrics"])
            returns.append(ep_return)
        out = mean_metrics(rows)
        terminal = mean_metrics(terminal_rows)
        out.update({key: terminal[key] for key in ENERGY_ACCOUNTING_KEYS})
        out.update(self._mean_dicts(action_rows))
        out["eval_return"] = float(np.mean(returns))
        out["parameter_count"] = float(self._parameter_count())
        out["inference_time_ms_per_decision"] = self._inference_time_ms_per_decision(env)
        return out

    @staticmethod
    def _append_csv(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(row.keys())
        existing = []
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                old_fields = list(reader.fieldnames or [])
                existing = list(reader)
            fields = old_fields + [field for field in fields if field not in old_fields]
            if fields != old_fields:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(existing)
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not existing and handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def _save_checkpoint(self, step: int, eval_metrics: dict[str, float], filename: str) -> Path:
        path = self.run_dir / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "algo": self.algo,
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "eval_metrics": dict(eval_metrics),
            "config": self.config,
        }
        if self.critic2 is not None:
            payload["critic2"] = self.critic2.state_dict()
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(payload["actor"])
        self.target_actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.target_critic1.load_state_dict(payload["critic1"])
        if self.critic2 is not None and "critic2" in payload:
            self.critic2.load_state_dict(payload["critic2"])
            assert self.target_critic2 is not None
            self.target_critic2.load_state_dict(payload["critic2"])

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        env_steps = 0
        next_eval = 0
        last_eval: dict[str, float] = {}
        while env_steps < self.total_env_steps:
            metrics, train_return = self.collect(env_steps)
            env_steps += self.rollout_steps
            losses = self.update()
            self._append_csv(
                self.train_csv,
                {
                    "step": env_steps,
                    "episode": self.episode,
                    "train_return": train_return,
                    **losses,
                    **metrics,
                },
            )
            if env_steps >= next_eval:
                last_eval = self.evaluate(deterministic=False)
                eval_row = {
                    "step": env_steps,
                    "parameter_count": last_eval.get("parameter_count", 0.0),
                    "inference_time_ms_per_decision": last_eval.get("inference_time_ms_per_decision", 0.0),
                    **{f"stochastic_{k}": v for k, v in last_eval.items()},
                }
                self._append_csv(self.eval_csv, eval_row)
                self._save_checkpoint(env_steps, last_eval, "checkpoint_latest.pt")
                stochastic_return = float(last_eval.get("eval_return", -float("inf")))
                if stochastic_return > self.best_stochastic_eval_return:
                    self.best_stochastic_eval_return = stochastic_return
                    self.best_stochastic_eval_metrics = dict(last_eval)
                    self.best_stochastic_eval_step = env_steps
                    self._save_checkpoint(env_steps, last_eval, "checkpoint_best_stochastic.pt")
                next_eval += self.eval_freq
                print(
                    f"[{self.algo}] step={env_steps} train_return={train_return:.3f} "
                    f"stochastic_eval_return={last_eval.get('eval_return', 0.0):.3f} "
                    f"completion={last_eval.get('completion_ratio', 0.0):.3f}",
                    flush=True,
                )
        summary = {
            "algo": self.algo,
            "status": "completed",
            "total_env_steps": env_steps,
            "episodes": self.episode,
            "last_stochastic_eval": last_eval,
            "best_stochastic_eval_return": self.best_stochastic_eval_return,
            "best_stochastic_eval_step": self.best_stochastic_eval_step,
            "best_stochastic_eval": self.best_stochastic_eval_metrics,
            "env": {
                "agent_num": len(self.env.agent_ids),
                "obs_shape": self.env.obs_dim,
                "global_obs_shape": self.env.global_state_dim,
                "action_shape": self.env.action_dim,
                "action_mode": self.env.action_mode,
                "difficulty": self.env.difficulty,
            },
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
