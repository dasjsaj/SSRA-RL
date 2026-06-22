"""Semantic residual policy guidance for discrete service routes."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ServiceSemanticEncoder(nn.Module):
    def __init__(self, semantic_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        return self.net(semantic)


class DiscreteSemanticGuidance(nn.Module):
    """Generate policy residual logits and observable-outcome estimates."""

    def __init__(self, semantic_dim: int, action_dim: int, hidden_dim: int = 64, zero_init_output: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = ServiceSemanticEncoder(semantic_dim, hidden_dim)
        self.residual_head = nn.Linear(hidden_dim, action_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.outcome_body = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.Tanh(),
        )
        self.completion_head = nn.Linear(hidden_dim, 1)
        self.deadline_head = nn.Linear(hidden_dim, 1)
        self.delay_head = nn.Linear(hidden_dim, 1)
        self.key_energy_head = nn.Linear(hidden_dim, 1)
        self.congestion_head = nn.Linear(hidden_dim, 1)
        if zero_init_output:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    def forward(self, semantic: torch.Tensor, actions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.encoder(semantic)
        residual_logits = self.residual_head(hidden)
        if actions is None:
            actions = torch.zeros(semantic.shape[0], dtype=torch.long, device=semantic.device)
        action_one_hot = F.one_hot(actions.long(), num_classes=self.action_dim).to(dtype=hidden.dtype)
        outcome_hidden = self.outcome_body(torch.cat([hidden, action_one_hot], dim=-1))
        return {
            "residual_logits": residual_logits,
            "completion_logit": self.completion_head(outcome_hidden).squeeze(-1),
            "deadline_logit": self.deadline_head(outcome_hidden).squeeze(-1),
            "delay_pred": torch.sigmoid(self.delay_head(outcome_hidden).squeeze(-1)),
            "key_energy_pred": torch.sigmoid(self.key_energy_head(outcome_hidden).squeeze(-1)),
            "congestion_logit": self.congestion_head(outcome_hidden).squeeze(-1),
            "residual_gate": torch.sigmoid(self.gate_head(hidden).squeeze(-1)),
        }


def semantic_teacher_distribution(
    route_costs: torch.Tensor,
    action_mask: torch.Tensor,
    temperature: float = 0.5,
    key_energy_costs: torch.Tensor | None = None,
    qos_tie_threshold: float = 0.0,
    energy_tie_weight: float = 0.0,
) -> torch.Tensor:
    """Turn current-observation route estimates into a masked soft prior."""
    effective_costs = route_costs
    if key_energy_costs is not None and energy_tie_weight > 0.0 and qos_tie_threshold > 0.0:
        masked_costs = route_costs.masked_fill(action_mask <= 0.0, 1e9)
        best_cost = masked_costs.min(dim=-1, keepdim=True).values
        tie_mask = ((masked_costs - best_cost).abs() <= float(qos_tie_threshold)).to(route_costs.dtype)
        effective_costs = route_costs + float(energy_tie_weight) * key_energy_costs * tie_mask
    logits = -effective_costs / max(float(temperature), 1e-6)
    logits = logits.masked_fill(action_mask <= 0.0, -1e9)
    return torch.softmax(logits, dim=-1)
