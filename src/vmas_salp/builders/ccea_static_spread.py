from pathlib import Path

from vmas_salp.learning.dataclasses import (
    ExperimentConfig,
    PolicyConfig,
    CCEAConfig,
)
from vmas_salp.learning.ccea.types import (
    FitnessShapingEnum,
    FitnessCalculationEnum,
    SelectionEnum,
    PolicyEnum,
    InitializationEnum,
)
from copy import deepcopy
from dataclasses import asdict

BATCH = "static_spread"

# DEFAULTS
N_STEPS = 200
N_GENS = 5000
SUBPOP_SIZE = 100
N_GENS_BETWEEN_SAVE = 20
GRU_POLICY_LAYERS = [83]
MLP_POLICY_LAYERS = [64, 64]

OUTPUT_MULTIPLIER = 1.0
WEIGHT_INITIALIZATION = InitializationEnum.KAIMING
FITNESS_CALC = FitnessCalculationEnum.LAST
MEAN = 0.0
MIN_STD_DEV = 0.05
MAX_STD_DEV = 0.25

GRU_POLICY_CONFIG = PolicyConfig(
    type=PolicyEnum.GRU,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=GRU_POLICY_LAYERS,
    output_multiplier=OUTPUT_MULTIPLIER,
)

MLP_POLICY_CONFIG = PolicyConfig(
    type=PolicyEnum.MLP,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=MLP_POLICY_LAYERS,
    output_multiplier=OUTPUT_MULTIPLIER,
)

CNN_POLICY_CONFIG = PolicyConfig(
    type=PolicyEnum.CNN,
    weight_initialization=WEIGHT_INITIALIZATION,
    output_multiplier=OUTPUT_MULTIPLIER,
    hidden_layers=0,
)

G_CCEA = CCEAConfig(
    n_steps=N_STEPS,
    n_gens=N_GENS,
    fitness_shaping=FitnessShapingEnum.G,
    selection=SelectionEnum.SOFTMAX,
    subpopulation_size=SUBPOP_SIZE,
    fitness_calculation=FITNESS_CALC,
    mutation={
        "mean": MEAN,
        "min_std_deviation": MIN_STD_DEV,
        "max_std_deviation": MAX_STD_DEV,
    },
)

D_CCEA = deepcopy(G_CCEA)
D_CCEA.fitness_shaping = FitnessShapingEnum.D

FC_CCEA = deepcopy(G_CCEA)
FC_CCEA.fitness_shaping = FitnessShapingEnum.FC


# EXPERIMENTS
G_MLP = ExperimentConfig(
    use_teaming=True,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=G_CCEA,
    team_size=6,
)
G_GRU = deepcopy(G_MLP)
G_GRU.policy_config = GRU_POLICY_CONFIG

G_CNN = deepcopy(G_MLP)
G_CNN.policy_config = CNN_POLICY_CONFIG

EXP_DICTS = [
    {"name": "g_gru", "config": asdict(G_GRU)},
    {"name": "g_mlp", "config": asdict(G_MLP)},
    {"name": "g_cnn", "config": asdict(G_CNN)},
]
