"""Learned service-computing model extensions."""

from .service_semantic_guidance import DiscreteSemanticGuidance, semantic_teacher_distribution
from .structured_policy import StructuredServiceActor

__all__ = ["DiscreteSemanticGuidance", "StructuredServiceActor", "semantic_teacher_distribution"]
