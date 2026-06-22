"""Lightweight discrete MARL value baselines for ServiceComputing.

The implementations here are intentionally compact experiment adapters for the
project's discrete two-hop route action space. They are not DI-engine internals
and do not alter the environment or MAPPO/SAGE trainers.
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
from torch.distributions import Categorical

from ServiceComputing.service_offloading import make_service_env
from ServiceComputing.service_offloading.metrics import ENERGY_ACCOUNTING_KEYS, mean_metrics


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    global_obs: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_masks: torch.Tensor
    dones: torch.Tensor


class AgentQNet(nn.Module):
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
        return self.net(obs)


class MonotonicMixer(nn.Module):
    def __init__(self, agent_num: int, global_dim: int, hidden_dim: int):
        super().__init__()
        self.agent_num = agent_num
        self.hyper_w1 = nn.Linear(global_dim, agent_num * hidden_dim)
        self.hyper_b1 = nn.Linear(global_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(global_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, agent_q: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        batch = agent_q.shape[0]
        w1 = torch.abs(self.hyper_w1(global_obs)).view(batch, self.agent_num, -1)
        b1 = self.hyper_b1(global_obs).view(batch, 1, -1)
        hidden = F.elu(torch.bmm(agent_q.view(batch, 1, self.agent_num), w1) + b1)
        w2 = torch.abs(self.hyper_w2(global_obs)).view(batch, -1, 1)
        b2 = self.hyper_b2(global_obs).view(batch, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(batch)


class QTranJointNet(nn.Module):
    def __init__(self, global_dim: int, agent_num: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.net = nn.Sequential(
            nn.Linear(global_dim + agent_num * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.v = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, global_obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = F.one_hot(actions.long(), num_classes=self.action_dim).float().flatten(start_dim=1)
        return self.net(torch.cat([global_obs, one_hot], dim=-1)).view(-1), self.v(global_obs).view(-1)


class ServiceValueMARLTrainer:
    """Shared off-policy trainer for MADQN/QMIX/WQMIX/QTRAN-style baselines."""

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
        acfg = config.get("value_baseline", {})
        self.total_env_steps = int(mcfg.get("total_env_steps", 100000))
        self.rollout_steps = int(mcfg.get("rollout_steps", 256))
        self.eval_freq = int(mcfg.get("eval_freq", 5000))
        self.eval_episodes = int(mcfg.get("eval_episodes", 8))
        self.gamma = float(mcfg.get("gamma", 0.99))
        self.hidden_dim = int(mcfg.get("hidden_dim", 128))
        self.batch_size = int(acfg.get("batch_size", 128))
        self.buffer_size = int(acfg.get("buffer_size", 50000))
        self.learning_rate = float(mcfg.get("learning_rate", 1e-4))
        self.target_update_interval = int(acfg.get("target_update_interval", 1000))
        self.epsilon_start = float(acfg.get("epsilon_start", 1.0))
        self.epsilon_end = float(acfg.get("epsilon_end", 0.05))
        self.epsilon_decay_steps = float(acfg.get("epsilon_decay_steps", max(1, self.total_env_steps * 0.6)))
        self.double_q = bool(acfg.get("double_q", True))
        self.agent_num = len(self.env.agent_ids) if self.env.agent_ids else self.env.n_auv + self.env.n_usv + self.env.n_uav
        self.action_dim = self.env.action_dim
        self.q_net = AgentQNet(self.env.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_q_net = AgentQNet(self.env.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        modules: list[nn.Module] = [self.q_net]
        self.mixer: MonotonicMixer | None = None
        self.target_mixer: MonotonicMixer | None = None
        self.qtran: QTranJointNet | None = None
        self.target_qtran: QTranJointNet | None = None
        if self.algo in {"qmix", "wqmix"}:
            self.mixer = MonotonicMixer(self.agent_num, self.env.global_state_dim, self.hidden_dim).to(self.device)
            self.target_mixer = MonotonicMixer(self.agent_num, self.env.global_state_dim, self.hidden_dim).to(self.device)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            modules.append(self.mixer)
        if self.algo == "qtran":
            self.qtran = QTranJointNet(self.env.global_state_dim, self.agent_num, self.action_dim, self.hidden_dim).to(
                self.device
            )
            self.target_qtran = QTranJointNet(
                self.env.global_state_dim, self.agent_num, self.action_dim, self.hidden_dim
            ).to(self.device)
            self.target_qtran.load_state_dict(self.qtran.state_dict())
            modules.append(self.qtran)
        self.optimizer = torch.optim.Adam([p for module in modules for p in module.parameters()], lr=self.learning_rate)
        self.replay: deque[tuple[np.ndarray, ...]] = deque(maxlen=self.buffer_size)
        self.run_dir = run_dir
        self.train_csv = run_dir / "train_curve.csv"
        self.eval_csv = run_dir / "eval_curve.csv"
        self.best_stochastic_eval_return = -float("inf")
        self.best_stochastic_eval_metrics: dict[str, float] = {}
        self.best_stochastic_eval_step = 0
        self._obs, _ = self.env.reset(seed=self.seed)
        self.episode = 0

    def _obs_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target_env = env or self.env
        return np.stack([obs[aid]["obs"] for aid in target_env.agent_ids], axis=0).astype(np.float32)

    def _global_array(self, env=None) -> np.ndarray:
        target_env = env or self.env
        state = target_env.get_global_state().astype(np.float32)
        return np.repeat(state[None, :], len(target_env.agent_ids), axis=0)

    def _state_array(self, env=None) -> np.ndarray:
        target_env = env or self.env
        return target_env.get_global_state().astype(np.float32)

    def _action_mask_array(self, obs: dict[str, dict], env=None) -> np.ndarray:
        target_env = env or self.env
        return np.stack([obs[aid]["action_mask"] for aid in target_env.agent_ids], axis=0).astype(np.float32)

    def _epsilon(self, step: int) -> float:
        frac = min(1.0, step / max(1.0, self.epsilon_decay_steps))
        return float(self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start))

    def _select_actions(self, obs: dict[str, dict], env=None, epsilon: float = 0.0) -> dict[str, int]:
        target_env = env or self.env
        obs_arr = self._obs_array(obs, target_env)
        masks = self._action_mask_array(obs, target_env)
        with torch.no_grad():
            q = self.q_net(torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device))
            q = q.masked_fill(torch.as_tensor(masks, dtype=torch.float32, device=self.device) <= 0.0, -1e9)
            probs = torch.softmax(q, dim=-1)
            greedy = torch.argmax(q, dim=-1).cpu().numpy()
            sampled = Categorical(probs=probs).sample().cpu().numpy()
        actions = {}
        for index, aid in enumerate(target_env.agent_ids):
            valid = np.flatnonzero(masks[index] > 0.0)
            if self.rng.random() < epsilon:
                actions[aid] = int(self.rng.choice(valid))
            elif epsilon <= 0.0:
                actions[aid] = int(sampled[index])
            else:
                actions[aid] = int(greedy[index])
        return actions

    def _parameter_count(self) -> int:
        modules: list[nn.Module] = [self.q_net]
        if self.mixer is not None:
            modules.append(self.mixer)
        if self.qtran is not None:
            modules.append(self.qtran)
        return int(sum(parameter.numel() for module in modules for parameter in module.parameters()))

    def _inference_time_ms_per_decision(self, env=None) -> float:
        target_env = env or self.eval_env
        obs, _ = target_env.reset(seed=self.seed + 777000)
        obs_arr = self._obs_array(obs, target_env)
        masks = self._action_mask_array(obs, target_env)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        iterations = int(self.config.get("mappo", {}).get("inference_benchmark_iters", 10))
        if iterations <= 0:
            return 0.0
        with torch.no_grad():
            self.q_net(obs_t).masked_fill(masks_t <= 0.0, -1e9)
            start = time.perf_counter()
            for _ in range(iterations):
                self.q_net(obs_t).masked_fill(masks_t <= 0.0, -1e9)
            elapsed = time.perf_counter() - start
        return float(1000.0 * elapsed / max(1, iterations * len(target_env.agent_ids)))

    @staticmethod
    def _mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        return {k: float(np.mean([row.get(k, 0.0) for row in rows])) for k in rows[0].keys()}

    def _action_metrics(self, actions: dict[str, int], env=None) -> dict[str, float]:
        target_env = env or self.env
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

    def collect(self, env_steps: int) -> tuple[dict[str, float], float]:
        metrics_rows, action_rows = [], []
        train_return = 0.0
        epsilon = self._epsilon(env_steps)
        for _ in range(self.rollout_steps):
            obs_arr = self._obs_array(self._obs)
            state = self._state_array()
            masks = self._action_mask_array(self._obs)
            actions = self._select_actions(self._obs, epsilon=epsilon)
            action_arr = np.asarray([actions[aid] for aid in self.env.agent_ids], dtype=np.int64)
            next_obs, rewards, done, _, info = self.env.step(actions)
            reward = float(np.mean(list(rewards.values())))
            self.replay.append(
                (
                    obs_arr,
                    state,
                    masks,
                    action_arr,
                    np.asarray(reward, dtype=np.float32),
                    self._obs_array(next_obs),
                    self._state_array(),
                    self._action_mask_array(next_obs),
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
        metrics["epsilon"] = epsilon
        return metrics, train_return

    def _sample_batch(self) -> ReplayBatch | None:
        if len(self.replay) < self.batch_size:
            return None
        indices = self.rng.choice(len(self.replay), size=self.batch_size, replace=False)
        rows = [self.replay[int(i)] for i in indices]
        stacked = [np.stack([row[i] for row in rows], axis=0) for i in range(9)]
        return ReplayBatch(
            obs=torch.as_tensor(stacked[0], dtype=torch.float32, device=self.device),
            global_obs=torch.as_tensor(stacked[1], dtype=torch.float32, device=self.device),
            masks=torch.as_tensor(stacked[2], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(stacked[3], dtype=torch.long, device=self.device),
            rewards=torch.as_tensor(stacked[4], dtype=torch.float32, device=self.device).view(-1),
            next_obs=torch.as_tensor(stacked[5], dtype=torch.float32, device=self.device),
            next_global_obs=torch.as_tensor(stacked[6], dtype=torch.float32, device=self.device),
            next_masks=torch.as_tensor(stacked[7], dtype=torch.float32, device=self.device),
            dones=torch.as_tensor(stacked[8], dtype=torch.float32, device=self.device).view(-1),
        )

    def _mix(self, agent_q: torch.Tensor, global_obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        if self.algo == "madqn":
            return agent_q.mean(dim=-1)
        if self.algo in {"qmix", "wqmix"}:
            mixer = self.target_mixer if target else self.mixer
            assert mixer is not None
            return mixer(agent_q, global_obs)
        if self.algo == "qtran":
            assert self.qtran is not None and self.target_qtran is not None
            joint, _ = (self.target_qtran if target else self.qtran)(global_obs, agent_q.long())
            return joint
        return agent_q.mean(dim=-1)

    def update(self) -> dict[str, float]:
        batch = self._sample_batch()
        if batch is None:
            return {"td_loss": 0.0, "total_loss": 0.0, "grad_norm": 0.0}
        q_all = self.q_net(batch.obs).masked_fill(batch.masks <= 0.0, -1e9)
        chosen_q = q_all.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            target_next_all = self.target_q_net(batch.next_obs).masked_fill(batch.next_masks <= 0.0, -1e9)
            if self.double_q:
                online_next = self.q_net(batch.next_obs).masked_fill(batch.next_masks <= 0.0, -1e9)
                next_actions = torch.argmax(online_next, dim=-1)
                next_agent_q = target_next_all.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            else:
                next_agent_q = target_next_all.max(dim=-1).values
            if self.algo == "qtran":
                target_actions = next_actions if self.double_q else torch.argmax(target_next_all, dim=-1)
                target_joint = self._mix(target_actions, batch.next_global_obs, target=True)
            else:
                target_joint = self._mix(next_agent_q, batch.next_global_obs, target=True)
            td_target = batch.rewards + self.gamma * (1.0 - batch.dones) * target_joint

        if self.algo == "qtran":
            assert self.qtran is not None
            pred_joint, state_v = self.qtran(batch.global_obs, batch.actions)
            td_loss = F.mse_loss(pred_joint, td_target)
            sum_q = chosen_q.sum(dim=-1)
            opt_actions = torch.argmax(q_all, dim=-1)
            opt_joint, opt_v = self.qtran(batch.global_obs, opt_actions)
            optimality = F.mse_loss(opt_joint - opt_v, q_all.max(dim=-1).values.sum(dim=-1).detach())
            non_opt = F.relu(sum_q.detach() - state_v - pred_joint).mean()
            loss = td_loss + 0.1 * optimality + 0.1 * non_opt
        else:
            pred_joint = self._mix(chosen_q, batch.global_obs)
            if self.algo == "wqmix":
                weights = torch.where(td_target > pred_joint.detach(), torch.full_like(td_target, 1.5), torch.ones_like(td_target))
                td_loss = (weights * (pred_joint - td_target).pow(2)).mean()
            else:
                td_loss = F.mse_loss(pred_joint, td_target)
            loss = td_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(self.q_net.parameters())
        if self.mixer is not None:
            params += list(self.mixer.parameters())
        if self.qtran is not None:
            params += list(self.qtran.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 0.5)
        self.optimizer.step()
        return {
            "td_loss": float(td_loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
        }

    def _sync_targets(self) -> None:
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        if self.mixer is not None and self.target_mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.qtran is not None and self.target_qtran is not None:
            self.target_qtran.load_state_dict(self.qtran.state_dict())

    def evaluate(self, seed_offset: int = 10000, deterministic: bool = False) -> dict[str, float]:
        env = self.eval_env
        rows, terminal_rows, action_rows, returns = [], [], [], []
        for ep in range(self.eval_episodes):
            obs, _ = env.reset(seed=self.seed + seed_offset + ep)
            done = False
            ep_return = 0.0
            while not done:
                actions = self._select_actions(obs, env=env, epsilon=0.0 if deterministic else 0.0)
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
            "q_net": self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "eval_metrics": dict(eval_metrics),
            "config": self.config,
        }
        if self.mixer is not None:
            payload["mixer"] = self.mixer.state_dict()
        if self.qtran is not None:
            payload["qtran"] = self.qtran.state_dict()
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(payload["q_net"])
        self.target_q_net.load_state_dict(payload["q_net"])
        if self.mixer is not None and "mixer" in payload:
            self.mixer.load_state_dict(payload["mixer"])
            assert self.target_mixer is not None
            self.target_mixer.load_state_dict(payload["mixer"])
        if self.qtran is not None and "qtran" in payload:
            self.qtran.load_state_dict(payload["qtran"])
            assert self.target_qtran is not None
            self.target_qtran.load_state_dict(payload["qtran"])

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
            if env_steps % self.target_update_interval < self.rollout_steps:
                self._sync_targets()
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
