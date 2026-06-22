"""COMA/HAPPO-style discrete policy baselines for ServiceComputing."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import ENERGY_ACCOUNTING_KEYS, mean_metrics


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(obs).masked_fill(mask <= 0.0, -1e9)


class CentralValue(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


class CounterfactualQ(nn.Module):
    def __init__(self, global_dim: int, agent_num: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(global_dim + agent_num, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        batch = global_obs.shape[0]
        agent_eye = torch.eye(self.agent_num, device=global_obs.device).unsqueeze(0).expand(batch, -1, -1)
        state = global_obs.unsqueeze(1).expand(-1, self.agent_num, -1)
        return self.net(torch.cat([state, agent_eye], dim=-1))


@dataclass
class PolicyBatch:
    obs: np.ndarray
    global_obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    metrics: dict[str, float]
    train_return: float


class ServiceOnPolicyMARLTrainer:
    """Compact COMA/HAPPO-style trainer for discrete service routing."""

    def __init__(self, config: dict[str, Any], run_dir: Path, algo: str):
        self.config = config
        self.algo = algo.lower()
        self.env = make_service_env(config)
        self.eval_env = make_service_env(config)
        if getattr(self.env, "action_mode", "simple") != "discrete_route":
            raise ValueError(f"{self.algo} adapter requires action_mode=discrete_route")
        self.seed = int(config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.device = torch.device(config.get("device", "cpu"))
        mcfg = config.get("mappo", {})
        self.hidden_dim = int(mcfg.get("hidden_dim", 128))
        self.agent_num = self.env.n_auv + self.env.n_usv + self.env.n_uav
        self.action_dim = self.env.action_dim
        self.actor = ActorNet(self.env.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.value = CentralValue(self.env.global_state_dim, self.hidden_dim).to(self.device)
        self.coma_q = CounterfactualQ(self.env.global_state_dim, self.agent_num, self.action_dim, self.hidden_dim).to(self.device)
        modules: list[nn.Module] = [self.actor, self.value]
        if self.algo == "coma":
            modules.append(self.coma_q)
        self.optimizer = torch.optim.Adam([p for module in modules for p in module.parameters()], lr=float(mcfg.get("learning_rate", 1e-4)))
        self.gamma = float(mcfg.get("gamma", 0.99))
        self.gae_lambda = float(mcfg.get("gae_lambda", 0.95))
        self.clip_ratio = float(mcfg.get("clip_ratio", 0.2))
        self.entropy_coef = float(mcfg.get("entropy_coef", 0.001))
        self.value_loss_coef = float(mcfg.get("value_loss_coef", 0.5))
        self.max_grad_norm = float(mcfg.get("max_grad_norm", 0.5))
        self.update_epochs = int(mcfg.get("update_epochs", 4))
        self.rollout_steps = int(mcfg.get("rollout_steps", 256))
        self.minibatch_steps = int(mcfg.get("minibatch_steps", 128))
        self.eval_episodes = int(mcfg.get("eval_episodes", 8))
        self.eval_freq = int(mcfg.get("eval_freq", 5000))
        self.total_env_steps = int(mcfg.get("total_env_steps", 100000))
        self.run_dir = run_dir
        self.train_csv = run_dir / "train_curve.csv"
        self.eval_csv = run_dir / "eval_curve.csv"
        self.best_stochastic_eval_return = -float("inf")
        self.best_stochastic_eval_metrics: dict[str, float] = {}
        self.best_stochastic_eval_step = 0
        self._obs, _ = self.env.reset(seed=self.seed)
        self.episode = 0

    def _obs_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target = env or self.env
        return np.stack([obs[aid]["obs"] for aid in target.agent_ids], axis=0).astype(np.float32)

    def _global_array(self, env=None) -> np.ndarray:
        target = env or self.env
        return target.get_global_state().astype(np.float32)

    def _global_repeat(self, env=None) -> np.ndarray:
        target = env or self.env
        state = target.get_global_state().astype(np.float32)
        return np.repeat(state[None, :], len(target.agent_ids), axis=0)

    def _mask_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target = env or self.env
        return np.stack([obs[aid]["action_mask"] for aid in target.agent_ids], axis=0).astype(np.float32)

    @staticmethod
    def _mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        return {k: float(np.mean([row.get(k, 0.0) for row in rows])) for k in rows[0]}

    def _action_metrics(self, actions: dict[str, int], env=None) -> dict[str, float]:
        target = env or self.env
        role_actions = {
            role: [int(actions[aid]) for aid in target.agent_ids if target.nodes[aid].role == role]
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

    def _policy(self, obs_arr: np.ndarray, masks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        logits = self.actor(obs_t, masks_t)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions), logits

    def _parameter_count(self) -> int:
        modules: list[nn.Module] = [self.actor, self.value]
        if self.algo == "coma":
            modules.append(self.coma_q)
        return int(sum(parameter.numel() for module in modules for parameter in module.parameters()))

    def _inference_time_ms_per_decision(self, env=None) -> float:
        target = env or self.eval_env
        obs, _ = target.reset(seed=self.seed + 777000)
        obs_arr = self._obs_array(obs, target)
        masks = self._mask_array(obs, target)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        iterations = int(self.config.get("mappo", {}).get("inference_benchmark_iters", 10))
        if iterations <= 0:
            return 0.0
        with torch.no_grad():
            self.actor(obs_t, masks_t)
            start = time.perf_counter()
            for _ in range(iterations):
                self.actor(obs_t, masks_t)
            elapsed = time.perf_counter() - start
        return float(1000.0 * elapsed / max(1, iterations * len(target.agent_ids)))

    def collect(self) -> PolicyBatch:
        obs_buf, global_buf, mask_buf, action_buf, logp_buf, value_buf = [], [], [], [], [], []
        reward_buf, done_buf, metrics_rows, action_rows = [], [], [], []
        train_return = 0.0
        for _ in range(self.rollout_steps):
            obs_arr = self._obs_array(self._obs)
            state = self._global_array()
            state_rep = self._global_repeat()
            masks = self._mask_array(self._obs)
            with torch.no_grad():
                actions_t, logp_t, _ = self._policy(obs_arr, masks)
                value = self.value(torch.as_tensor(state_rep, dtype=torch.float32, device=self.device))
            actions = {aid: int(actions_t.cpu().numpy()[i]) for i, aid in enumerate(self.env.agent_ids)}
            next_obs, rewards, done, _, info = self.env.step(actions)
            reward = np.array([float(rewards[aid]) for aid in self.env.agent_ids], dtype=np.float32)
            obs_buf.append(obs_arr)
            global_buf.append(state_rep)
            mask_buf.append(masks)
            action_buf.append(actions_t.cpu().numpy().astype(np.int64))
            logp_buf.append(logp_t.cpu().numpy().astype(np.float32))
            value_buf.append(value.cpu().numpy().astype(np.float32))
            reward_buf.append(reward)
            done_buf.append(np.full(len(self.env.agent_ids), float(done), dtype=np.float32))
            metrics_rows.append(info["metrics"])
            action_rows.append(self._action_metrics(actions))
            train_return += float(np.mean(reward))
            self._obs = next_obs
            if done:
                self.episode += 1
                self._obs, _ = self.env.reset(seed=self.seed + self.episode)
        with torch.no_grad():
            last_value = self.value(
                torch.as_tensor(self._global_repeat(), dtype=torch.float32, device=self.device)
            ).cpu().numpy()
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
        return PolicyBatch(
            obs=np.asarray(obs_buf, dtype=np.float32),
            global_obs=np.asarray(global_buf, dtype=np.float32),
            masks=np.asarray(mask_buf, dtype=np.float32),
            actions=np.asarray(action_buf, dtype=np.int64),
            log_probs=np.asarray(logp_buf, dtype=np.float32),
            values=values,
            rewards=rewards,
            dones=dones,
            advantages=advantages,
            returns=returns,
            metrics=metrics,
            train_return=train_return,
        )

    def update(self, batch: PolicyBatch) -> dict[str, float]:
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
                masks = torch.as_tensor(batch.masks[mb], dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(batch.actions[mb], dtype=torch.long, device=self.device)
                old_logp = torch.as_tensor(batch.log_probs[mb], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(batch.returns[mb], dtype=torch.float32, device=self.device)
                adv = adv_all[mb]
                logits = self.actor(obs, masks)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                values = self.value(global_obs)
                value_loss = F.mse_loss(values, returns)
                if self.algo == "coma":
                    q = self.coma_q(global_obs[:, 0, :])
                    q_taken = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                    baseline = (dist.probs * q).sum(dim=-1)
                    coma_adv = (q_taken - baseline).detach()
                    q_target = returns.detach()
                    q_loss = F.mse_loss(q_taken, q_target)
                    policy_loss = -(logp * coma_adv).mean()
                    total_loss = policy_loss + self.value_loss_coef * value_loss + 0.5 * q_loss - self.entropy_coef * entropy
                else:
                    ratio = torch.exp(logp - old_logp)
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    q_loss = torch.zeros((), device=self.device)
                    total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                params = list(self.actor.parameters()) + list(self.value.parameters())
                if self.algo == "coma":
                    params += list(self.coma_q.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()
                losses.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "q_loss": float(q_loss.detach().cpu()),
                        "entropy_loss": float(entropy.detach().cpu()),
                        "total_loss": float(total_loss.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                    }
                )
        return {key: float(np.mean([row[key] for row in losses])) for key in losses[0]}

    def evaluate(self, seed_offset: int = 10000, deterministic: bool = False) -> dict[str, float]:
        env = self.eval_env
        rows, terminal_rows, action_rows, returns = [], [], [], []
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed + seed_offset + 900000)
        try:
            for ep in range(self.eval_episodes):
                obs, _ = env.reset(seed=self.seed + seed_offset + ep)
                done = False
                ep_return = 0.0
                while not done:
                    obs_arr = self._obs_array(obs, env)
                    masks = self._mask_array(obs, env)
                    with torch.no_grad():
                        logits = self.actor(
                            torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device),
                            torch.as_tensor(masks, dtype=torch.float32, device=self.device),
                        )
                        action_t = torch.argmax(logits, dim=-1) if deterministic else Categorical(logits=logits).sample()
                    actions = {aid: int(action_t.cpu().numpy()[i]) for i, aid in enumerate(env.agent_ids)}
                    obs, rewards, done, _, info = env.step(actions)
                    ep_return += float(np.mean(list(rewards.values())))
                    rows.append(info["metrics"])
                    action_rows.append(self._action_metrics(actions, env))
                    if done:
                        terminal_rows.append(info["metrics"])
                returns.append(ep_return)
        finally:
            torch.random.set_rng_state(rng_state)
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
        torch.save(
            {
                "step": int(step),
                "algo": self.algo,
                "actor": self.actor.state_dict(),
                "value": self.value.state_dict(),
                "coma_q": self.coma_q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "eval_metrics": dict(eval_metrics),
                "config": self.config,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(payload["actor"])
        self.value.load_state_dict(payload["value"])
        if "coma_q" in payload:
            self.coma_q.load_state_dict(payload["coma_q"])

    def train(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        env_steps = 0
        next_eval = 0
        last_eval: dict[str, float] = {}
        while env_steps < self.total_env_steps:
            batch = self.collect()
            losses = self.update(batch)
            env_steps += self.rollout_steps
            self._append_csv(
                self.train_csv,
                {"step": env_steps, "episode": self.episode, "train_return": batch.train_return, **losses, **batch.metrics},
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
                score = float(last_eval.get("eval_return", -float("inf")))
                if score > self.best_stochastic_eval_return:
                    self.best_stochastic_eval_return = score
                    self.best_stochastic_eval_metrics = dict(last_eval)
                    self.best_stochastic_eval_step = env_steps
                    self._save_checkpoint(env_steps, last_eval, "checkpoint_best_stochastic.pt")
                next_eval += self.eval_freq
                print(
                    f"[{self.algo}] step={env_steps} train_return={batch.train_return:.3f} "
                    f"stochastic_eval_return={score:.3f} completion={last_eval.get('completion_ratio', 0.0):.3f}",
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
