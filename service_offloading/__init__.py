"""Cross-domain UAV/USV/AUV service offloading environment."""

from .env import CrossDomainServiceOffloadingEnv
from .queue_env import DualHopQueueServiceOffloadingEnv


def make_service_env(config=None):
    """Construct the selected environment while preserving legacy configs."""
    cfg = (config or {}).get("env", config or {})
    if str(cfg.get("env_model", "legacy")).lower() == "dual_hop_queue":
        return DualHopQueueServiceOffloadingEnv(config)
    return CrossDomainServiceOffloadingEnv(config)


__all__ = ["CrossDomainServiceOffloadingEnv", "DualHopQueueServiceOffloadingEnv", "make_service_env"]
