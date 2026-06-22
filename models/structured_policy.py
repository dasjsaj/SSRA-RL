"""Structured service-routing actor for enhanced SLG-SAGE variants."""

from __future__ import annotations

import torch
from torch import nn


class StructuredServiceActor(nn.Module):
    """Role-aware multi-branch actor with action-conditioned scoring.

    The input layout follows ``DualHopQueueServiceOffloadingEnv``:
    task features live in the local observation, resource/link features are
    split by stable indices, and optional semantic features are fused through a
    learned gate.
    """

    def __init__(
        self,
        obs_dim: int,
        semantic_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        use_role_embedding: bool = True,
        fusion_mode: str = "gated",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.semantic_dim = int(semantic_dim)
        self.action_dim = int(action_dim)
        self.use_role_embedding = bool(use_role_embedding)
        self.fusion_mode = str(fusion_mode)
        self.role_embedding = nn.Linear(3, hidden_dim)
        self.task_branch = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.resource_branch = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.link_branch = nn.Sequential(nn.Linear(6, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.semantic_branch = nn.Sequential(
            nn.Linear(max(1, semantic_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4))
        self.mean_fusion = nn.Linear(hidden_dim * 4, hidden_dim)
        self.attention_query = nn.Parameter(torch.zeros(hidden_dim))
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _pad_or_trim(values: torch.Tensor, width: int) -> torch.Tensor:
        if values.shape[-1] == width:
            return values
        if values.shape[-1] > width:
            return values[..., :width]
        pad = torch.zeros(*values.shape[:-1], width - values.shape[-1], dtype=values.dtype, device=values.device)
        return torch.cat([values, pad], dim=-1)

    def forward(
        self,
        obs: torch.Tensor,
        semantic: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        original_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, obs.shape[-1])
        role = self._pad_or_trim(flat_obs[..., 0:3], 3)
        task = self._pad_or_trim(flat_obs[..., 7:11], 4)
        resource = self._pad_or_trim(flat_obs[..., [3, 4, 5, 6, 15]], 5)
        link = self._pad_or_trim(flat_obs[..., [11, 12, 13, 14, 18, 19]], 6)
        if semantic is None or self.semantic_dim <= 0:
            flat_semantic = torch.zeros(flat_obs.shape[0], max(1, self.semantic_dim), dtype=flat_obs.dtype, device=flat_obs.device)
        else:
            flat_semantic = self._pad_or_trim(semantic.reshape(-1, semantic.shape[-1]), max(1, self.semantic_dim))

        task_features = self.task_branch(task)
        if self.use_role_embedding:
            task_features = task_features + self.role_embedding(role)
        branches = torch.stack(
            [task_features, self.resource_branch(resource), self.link_branch(link), self.semantic_branch(flat_semantic)],
            dim=-2,
        )
        flat_branches = branches.reshape(branches.shape[0], -1)
        if self.fusion_mode == "gated":
            gate_logits = self.fusion_gate(flat_branches)
            gate = torch.softmax(gate_logits, dim=-1)
            fused = (branches * gate.unsqueeze(-1)).sum(dim=-2)
        elif self.fusion_mode == "attention":
            query = self.attention_query.to(dtype=branches.dtype, device=branches.device)
            gate_logits = torch.matmul(branches, query)
            gate = torch.softmax(gate_logits, dim=-1)
            fused = (branches * gate.unsqueeze(-1)).sum(dim=-2)
        else:
            gate = torch.full((branches.shape[0], 4), 0.25, dtype=branches.dtype, device=branches.device)
            fused = torch.relu(self.mean_fusion(flat_branches))
        action_ids = torch.arange(self.action_dim, device=obs.device)
        action_features = self.action_embedding(action_ids).unsqueeze(0).expand(fused.shape[0], -1, -1)
        fused_per_action = fused.unsqueeze(1).expand(-1, self.action_dim, -1)
        scores = self.scorer(torch.cat([fused_per_action, action_features], dim=-1)).squeeze(-1)
        if action_mask is not None:
            flat_mask = action_mask.reshape(-1, action_mask.shape[-1])
            scores = scores.masked_fill(flat_mask <= 0.0, -1e9)
        logits = scores.reshape(*original_shape, self.action_dim)
        return {
            "logits": logits,
            "action_scores": logits,
            "fusion_gate": gate.reshape(*original_shape, 4),
        }
