from enum import StrEnum
from vmas_salp.learning.types import Team


class EvalInfo:
    def __init__(
        self,
        team: Team,
        team_fitness: float,
        agent_fitnesses: list[float],
        joint_traj: list,
    ):
        self.team = team
        self.agent_fitnesses = agent_fitnesses
        self.team_fitness = team_fitness
        self.joint_traj = joint_traj


class InitializationEnum(StrEnum):
    KAIMING = "kaiming"


class PolicyEnum(StrEnum):
    GRU = "GRU"
    MLP = "MLP"
    CNN = "CNN"


class FitnessShapingEnum(StrEnum):
    D = "difference"
    G = "global"
    HOF = "hof_difference"
    FC = "fitness_critics"


class FitnessCriticError(StrEnum):
    MSE = "MSE"
    MAE = "MAE"
    MSE_MAE = "MSE+MAE"


class FitnessCriticType(StrEnum):
    MLP = "MLP"
    GRU = "GRU"
    ATT = "ATT"


class SelectionEnum(StrEnum):
    SOFTMAX = "softmax"
    EPSILON = "epsilon"
    BINARY = "binary"
    TOURNAMENT = "tournament"


class FitnessCalculationEnum(StrEnum):
    AGG = "aggregate"
    LAST = "last_step"
