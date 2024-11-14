from pathlib import Path

from vmas_rovers.learning.ccea.dataclasses import (
    ExperimentConfig,
    PolicyConfig,
    CCEAConfig,
)
from vmas_rovers.learning.ccea.types import (
    FitnessShapingEnum,
    FitnessCalculationEnum,
    SelectionEnum,
    PolicyEnum,
    InitializationEnum,
)

from dataclasses import asdict

BATCH = Path(__file__).stem
EXPERIMENTS = ["d_gru", "d_mlp", "g_gru", "g_mlp"]

# DEFAULTS
N_STEPS = 40
N_GENS = 500
SUBPOP_SIZE = 100
N_GENS_BETWEEN_SAVE = 20
GRU_NEURONS = 8
MLP_NEURONS = 12
OUTPUT_MULTIPLIER = 1.0
WEIGHT_INITIALIZATION = InitializationEnum.KAIMING
FITNESS_CALC = FitnessCalculationEnum.LAST
MEAN = 0.0
MIN_STD_DEV = 0.05
MAX_STD_DEV = 0.25

GRU_POLICY_CONFIG = PolicyConfig(
    type=PolicyEnum.GRU,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=[GRU_NEURONS],
    output_multiplier=OUTPUT_MULTIPLIER,
)

MLP_POLICY_CONFIG = PolicyConfig(
    type=PolicyEnum.MLP,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=[MLP_NEURONS, MLP_NEURONS],
    output_multiplier=OUTPUT_MULTIPLIER,
)

D_CCEA = CCEAConfig(
    n_steps=N_STEPS,
    n_gens=N_GENS,
    fitness_shaping=FitnessShapingEnum.D,
    selection=SelectionEnum.SOFTMAX,
    subpopulation_size=SUBPOP_SIZE,
    fitness_calculation=FITNESS_CALC,
    mutation={
        "mean": MEAN,
        "min_std_deviation": MIN_STD_DEV,
        "max_std_deviation": MAX_STD_DEV,
    },
)

G_CCEA = CCEAConfig(
    n_steps=N_STEPS,
    n_gens=N_GENS,
    fitness_shaping=FitnessShapingEnum.D,
    selection=SelectionEnum.SOFTMAX,
    subpopulation_size=SUBPOP_SIZE,
    fitness_calculation=FITNESS_CALC,
    mutation={
        "mean": MEAN,
        "min_std_deviation": MIN_STD_DEV,
        "max_std_deviation": MAX_STD_DEV,
    },
)

# EXPERIMENTS
D_GRU = ExperimentConfig(
    use_teaming=False,
    use_fc=False,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=GRU_POLICY_CONFIG,
    ccea_config=D_CCEA,
)

D_MLP = ExperimentConfig(
    use_teaming=False,
    use_fc=False,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=D_CCEA,
)

G_MLP = ExperimentConfig(
    use_teaming=False,
    use_fc=False,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=G_CCEA,
)

G_GRU = ExperimentConfig(
    use_teaming=False,
    use_fc=False,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=GRU_POLICY_CONFIG,
    ccea_config=G_CCEA,
)


EXP_DICTS = [
    {"name": "d_gru", "config": asdict(D_GRU)},
    {"name": "d_mlp", "config": asdict(D_MLP)},
    {"name": "g_gru", "config": asdict(G_GRU)},
    {"name": "g_mlp", "config": asdict(G_MLP)},
]
