from dataclasses import dataclass


@dataclass
class CCEAConfig:
    n_gens: int
    n_steps: int
    subpopulation_size: int
    selection: str
    fitness_shaping: str
    fitness_calculation: str
    mutation: dict


@dataclass
class FitnessCriticConfig:
    epochs: int
    type: str
    loss_type: str
    hidden_layers: tuple[int]


@dataclass
class PolicyConfig:
    weight_initialization: str
    type: str
    hidden_layers: tuple[int]
    output_multiplier: float


@dataclass
class ExperimentConfig:
    use_teaming: bool = False
    use_fc: bool = False
    ccea_config: CCEAConfig = None
    policy_config: PolicyConfig = None
    fc_config: FitnessCriticConfig = None
    n_gens_between_save: int = 0


@dataclass
class PositionConfig:
    spawn_rule: str
    coordinates: tuple[int]


@dataclass
class RoversConfig:
    observation_radius: int
    type: int
    color: str
    position: PositionConfig


@dataclass
class POIConfig:
    value: float
    coupling: int
    observation_radius: float
    type: int
    order: int
    position: PositionConfig


@dataclass
class EnvironmentConfig:
    map_size: tuple[int]
    use_order: bool
    rovers: list[RoversConfig]
    pois: list[POIConfig]
    obs_space_dim: int
    action_space_dim: int
