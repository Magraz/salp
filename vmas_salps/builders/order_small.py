from pathlib import Path

from vmas_rovers.learning.ccea.dataclasses import (
    ExperimentConfig,
    PolicyConfig,
    CCEAConfig,
    FitnessCriticConfig,
)
from vmas_rovers.learning.ccea.types import (
    FitnessShapingEnum,
    FitnessCalculationEnum,
    FitnessCriticError,
    FitnessCriticType,
    SelectionEnum,
    PolicyEnum,
    InitializationEnum,
)

from dataclasses import asdict

BATCH = Path(__file__).stem

# DEFAULTS
N_STEPS = 100
N_GENS = 5000
SUBPOP_SIZE = 100
N_GENS_BETWEEN_SAVE = 20
GRU_POLICY_LAYERS = [12]
MLP_POLICY_LAYERS = [23, 23]
FC_MLP = [80, 80]
FC_GRU = [44]
FC_EPOCHS = 20

OUTPUT_MULTIPLIER = 1.0
WEIGHT_INITIALIZATION = InitializationEnum.KAIMING
FITNESS_CALC = FitnessCalculationEnum.AGG
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

FC_CCEA = CCEAConfig(
    n_steps=N_STEPS,
    n_gens=N_GENS,
    fitness_shaping=FitnessShapingEnum.FC,
    selection=SelectionEnum.SOFTMAX,
    subpopulation_size=SUBPOP_SIZE,
    fitness_calculation=FITNESS_CALC,
    mutation={
        "mean": MEAN,
        "min_std_deviation": MIN_STD_DEV,
        "max_std_deviation": MAX_STD_DEV,
    },
)

MLP_FC = FitnessCriticConfig(
    epochs=FC_EPOCHS,
    type=FitnessCriticType.MLP,
    loss_type=FitnessCriticError.MSE_MAE,
    hidden_layers=FC_MLP,
)

GRU_FC = FitnessCriticConfig(
    epochs=FC_EPOCHS,
    type=FitnessCriticType.GRU,
    loss_type=FitnessCriticError.MSE_MAE,
    hidden_layers=FC_GRU,
)

# EXPERIMENTS
D_GRU = ExperimentConfig(
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=GRU_POLICY_CONFIG,
    ccea_config=D_CCEA,
)

D_MLP = ExperimentConfig(
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=D_CCEA,
)

G_MLP = ExperimentConfig(
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=G_CCEA,
)

G_GRU = ExperimentConfig(
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=GRU_POLICY_CONFIG,
    ccea_config=G_CCEA,
)

MLP_FC_MLP = ExperimentConfig(
    use_fc=True,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=FC_CCEA,
    fc_config=MLP_FC,
)

GRU_FC_MLP = ExperimentConfig(
    use_fc=True,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=FC_CCEA,
    fc_config=GRU_FC,
)


EXP_DICTS = [
    {"name": "d_gru", "config": asdict(D_GRU)},
    {"name": "d_mlp", "config": asdict(D_MLP)},
    {"name": "g_gru", "config": asdict(G_GRU)},
    {"name": "g_mlp", "config": asdict(G_MLP)},
    {"name": "mlp_fc_mlp", "config": asdict(MLP_FC_MLP)},
    {"name": "gru_fc_mlp", "config": asdict(GRU_FC_MLP)},
]
