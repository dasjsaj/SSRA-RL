# SSRA-RL: Service-Semantic Role-Aware Reinforcement Learning

This repository provides the core implementation of **SSRA-RL**, a service-semantic role-aware reinforcement learning framework for role-aware service-chain scheduling in heterogeneous mobile edge computing (MEC). The code supports queue-aware task generation, role-dependent forwarding and computation, action masking, semantic policy guidance, and comparative MARL experiments.

> Paper: *Service-Semantic Reinforcement Learning for Role-Aware Service-Chain Scheduling in Heterogeneous Mobile Edge Computing*

## Overview

Heterogeneous MEC systems often require tasks to be admitted, forwarded, processed, and delivered across nodes with different service roles. In this setting, a scheduling action is not only a discrete action ID; it also has service-level consequences, such as completion likelihood, remaining delay, downstream congestion, deadline risk, and key mobile transmission energy.

SSRA-RL addresses this issue by introducing service-semantic guidance into MARL-based service-chain scheduling. The framework contains:

- **Role-heterogeneous service-chain environment** with source, gateway, mobile-edge, and fixed-edge/cloud roles.
- **Queue-aware task lifecycle** including stochastic arrivals, forwarding, partial computation, completion, timeout, and overflow dropping.
- **Role-constrained action masks** to prevent infeasible service-chain actions.
- **Service-semantic decision variables** describing task urgency, role context, queue pressure, downstream state, route delay, deadline risk, and key energy cost.
- **Masked semantic action prior** over role-feasible actions.
- **Gated residual logit guidance** for controlled semantic intervention without rule-based action replacement.
- **Auxiliary service-outcome prediction** for completion, deadline risk, delay, congestion, and key transmission energy.

## DI-engine Integration

The comparative experiments are built on top of the **DI-engine** library. In particular, the MAPPO baseline uses DI-engine's `MAVAC` backbone through the adapter in:

```text
algorithms/service_mappo_di.py
```

The proposed semantic-guided method extends this DI-engine-based MAPPO interface through:

```text
algorithms/slg_sage_mappo_di.py
```

The repository also provides adapter-style interfaces for other comparison algorithms, including value-based, policy-gradient, and continuous-control baselines. These adapters are designed to connect the proposed service-chain environment with different MARL algorithms under the same observation, action-mask, reward, and metric interfaces.

If a local DI-engine source tree is placed under:

```text
DI-engine-main/
```

then `service_mappo_di.py` automatically adds it to `sys.path`. Otherwise, please install DI-engine according to its official installation instructions.

## Repository Structure

```text
SSRA-RL/
├── algorithms/
│   ├── service_mappo_di.py              # DI-engine MAPPO baseline interface
│   ├── slg_sage_mappo_di.py             # SSRA-RL semantic-guided MAPPO extension
│   ├── service_value_baselines.py        # MADQN / QMIX / WQMIX / QTRAN-style baselines
│   ├── service_policy_baselines.py       # COMA / HAPPO-style policy baselines
│   ├── service_continuous_baselines.py   # MADDPG / MATD3 / MASAC-style baselines
│   └── slg_sage_mappo.py                # Lightweight semantic-guided trainer
│
├── models/
│   ├── service_semantic_guidance.py      # Semantic encoder, residual logits, outcome heads
│   └── structured_policy.py             # Optional structured actor module
│
├── service_offloading/
│   ├── queue_env.py                      # Queue-aware role-heterogeneous service-chain environment
│   ├── env.py                            # Legacy cross-domain service offloading environment
│   ├── scenario.py                       # Node and task primitives
│   ├── semantic.py                       # Semantic feature extraction and heuristic priors
│   └── metrics.py                        # Metric aggregation helpers
│
└── scripts/
    ├── analyze_semantic_contribution.py
    ├── analyze_service_mappo_convergence.py
    ├── audit_paper_run_status.py
    ├── backfill_baseline_complexity_metrics.py
    ├── backfill_continuous_complexity_metrics.py
    └── backfill_slg_sage_residual_gate_metrics.py
```

## Main Components

### 1. Role-Heterogeneous Service-Chain Environment

The main environment is:

```python
from ServiceComputing.service_offloading import make_service_env
```

It supports a queue-aware dual-hop service-chain model:

```python
config = {
    "env": {
        "env_model": "dual_hop_queue",
        "n_auv": 4,
        "n_usv": 2,
        "n_uav": 2,
        "episode_length": 150,
        "queue_capacity": 30,
        "task_arrival_rate": 0.28,
        "action_mode": "discrete_route",
        "use_semantic_side_channel": True,
    }
}

env = make_service_env(config)
obs, info = env.reset(seed=1)
```

The representative implementation uses AUV-like sources, USV-like gateways, UAV-like mobile edge nodes, and a shore-side fixed edge/cloud node. The model is role-level and can be adapted to other heterogeneous MEC service-chain systems.

### 2. Action Masking

Each agent receives an action mask in its observation dictionary:

```python
action_mask = obs[agent_id]["action_mask"]
```

The mask excludes infeasible decisions caused by role constraints, empty queues, unavailable forwarding directions, or invalid service-chain transitions. Both the learned actor and the semantic action prior operate on the masked feasible action set.

### 3. Semantic Side Channel

When `use_semantic_side_channel=True`, each agent additionally receives semantic variables, semantic priors, route costs, and auxiliary service signals:

```python
semantic_state = obs[agent_id]["semantic"]
semantic_prior = obs[agent_id]["semantic_prior"]
route_costs = obs[agent_id]["semantic_route_costs"]
semantic_mask = obs[agent_id]["semantic_action_mask"]
```

These variables are used by SSRA-RL to construct service-aware policy guidance.

### 4. Metrics

The environment records service and resource metrics, including:

- average return;
- completion ratio;
- mean service delay;
- deadline-violation rate;
- mean queue length;
- generated, completed, timeout, and dropped tasks;
- route progress;
- key mobile transmission energy;
- detailed energy-accounting terms.

Metric aggregation utilities are provided in:

```text
service_offloading/metrics.py
```

## Installation

Create a Python environment and install the required scientific-computing dependencies:

```bash
conda create -n ssra-rl python=3.10 -y
conda activate ssra-rl

pip install numpy torch
```

Install or prepare DI-engine for the MAPPO-based comparison experiments:

```bash
# Option 1: install DI-engine using its official instructions.
# Option 2: place the DI-engine source tree under the repository root:
# SSRA-RL/DI-engine-main/
```

The code imports the project package as `ServiceComputing`. If your cloned repository is named differently, make sure the package directory is importable under this name, or update the import paths consistently.

A typical local setup is:

```bash
# assume the parent directory contains the package folder ServiceComputing/
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Quick Start

### Run the DI-engine MAPPO Baseline

```python
from pathlib import Path
from ServiceComputing.algorithms.service_mappo_di import ServiceMAPPOTrainer

config = {
    "seed": 1,
    "device": "cpu",
    "env": {
        "env_model": "dual_hop_queue",
        "n_auv": 2,
        "n_usv": 1,
        "n_uav": 1,
        "episode_length": 150,
        "queue_capacity": 30,
        "task_arrival_rate": 0.28,
        "action_mode": "discrete_route",
        "use_semantic_side_channel": False,
    },
    "mappo": {
        "hidden_dim": 128,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "rollout_steps": 256,
        "total_env_steps": 100000,
        "eval_freq": 5000,
        "eval_episodes": 10,
    },
}

trainer = ServiceMAPPOTrainer(config, Path("runs/mappo_small"))
summary = trainer.train()
print(summary)
```

### Run SSRA-RL

The current implementation keeps the legacy trainer class name `SLGSAGEMAPPOTrainer` for compatibility, while the corresponding method in the paper is referred to as **SSRA-RL**.

```python
from pathlib import Path
from ServiceComputing.algorithms.slg_sage_mappo_di import SLGSAGEMAPPOTrainer

config = {
    "seed": 1,
    "device": "cpu",
    "env": {
        "env_model": "dual_hop_queue",
        "n_auv": 2,
        "n_usv": 1,
        "n_uav": 1,
        "episode_length": 150,
        "queue_capacity": 30,
        "task_arrival_rate": 0.28,
        "action_mode": "discrete_route",
        "use_semantic_side_channel": True,
        "use_task_aware_semantic_teacher": True,
        "use_downstream_aware_semantic_teacher": True,
        "use_marginal_completion_teacher": True,
    },
    "mappo": {
        "hidden_dim": 128,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "rollout_steps": 256,
        "total_env_steps": 100000,
        "eval_freq": 5000,
        "eval_episodes": 10,
    },
    "semantic": {
        "hidden_dim": 64,
        "zero_init_semantic_output": True,
        "lambda_prior_0": 0.08,
        "lambda_guide_0": 0.03,
        "lambda_aux": 0.03,
        "teacher_temperature": 0.5,
        "semantic_logit_scale_max": 0.30,
        "semantic_min_logit_scale": 0.05,
        "semantic_residual_warmup_steps": 5000,
        "prior_decay_steps": 40000,
        "guide_decay_steps": 25000,
    },
}

trainer = SLGSAGEMAPPOTrainer(config, Path("runs/ssra_rl_small"))
summary = trainer.train()
print(summary)
```

Training outputs are saved under the specified run directory, including:

```text
config.json
train_curve.csv
eval_curve.csv
summary.json
checkpoints/
```

## Baseline Interfaces

This repository provides unified trainer interfaces for comparison experiments:

| File | Algorithms / Purpose |
|---|---|
| `algorithms/service_mappo_di.py` | DI-engine MAPPO baseline with MAVAC backbone |
| `algorithms/slg_sage_mappo_di.py` | SSRA-RL semantic-guided MAPPO extension |
| `algorithms/service_value_baselines.py` | MADQN, QMIX, WQMIX, QTRAN-style baselines |
| `algorithms/service_policy_baselines.py` | COMA, HAPPO-style policy baselines |
| `algorithms/service_continuous_baselines.py` | MADDPG, MATD3, MASAC-style baselines |

All baseline adapters interact with the same environment interface and report the same metrics whenever applicable. This makes the comparison experiments consistent in terms of scenario configuration, action feasibility, queue dynamics, and service-chain reward definition.

## Analysis Utilities

The `scripts/` directory contains utilities for checking experiment status, analyzing semantic contributions, and backfilling complexity or diagnostic metrics:

```bash
python -m ServiceComputing.scripts.audit_paper_run_status --help
python -m ServiceComputing.scripts.analyze_semantic_contribution --help
python -m ServiceComputing.scripts.analyze_service_mappo_convergence --help
```

## Notes for Reproducibility

- Use the same scenario configuration, random seeds, and evaluation episodes when comparing algorithms.
- For discrete-route experiments, set `action_mode="discrete_route"`.
- For SSRA-RL, set `use_semantic_side_channel=True`.
- The semantic residual head is zero-initialized by default, so the semantic branch does not change the initial base policy.
- The semantic guidance coefficients are decayed during training to keep the final policy primarily return-optimized.
- The implementation reports key mobile transmission energy separately from other accounting terms.


## Acknowledgement

We thank the DI-engine developers for providing an open reinforcement learning framework.

## License

Please add the license file that matches your intended release policy.
