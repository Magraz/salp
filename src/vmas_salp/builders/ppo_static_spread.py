from vmas_salp.learning.dataclasses import (
    ExperimentConfig,
)

from dataclasses import asdict

BATCH = "static_spread"

# DEFAULTS
N_GENS_BETWEEN_SAVE = 20

OUTPUT_MULTIPLIER = 1.0

# EXPERIMENTS
PPO_G = ExperimentConfig(
    team_size=3,
)

EXP_DICTS = [
    {"name": "ppo_g", "config": asdict(PPO_G)},
]
