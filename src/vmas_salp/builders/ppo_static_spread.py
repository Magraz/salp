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
    use_teaming=True,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    team_size=3,
)

EXP_DICTS = [
    {"name": "ppo_g", "config": asdict(PPO_G)},
]
