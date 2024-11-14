import os
import yaml
import torch
from pathlib import Path
from vmas_rovers.learning.ccea.ccea import CooperativeCoevolutionaryAlgorithm
from vmas_rovers.learning.ccea.dataclasses import ExperimentConfig, EnvironmentConfig
from dataclasses import asdict


def runCCEA(batch_dir: str, batch_name: str, experiment_name: str, trial_id: int):

    exp_file = os.path.join(batch_dir, f"{experiment_name}.yaml")

    with open(str(exp_file), "r") as file:
        exp_dict = yaml.unsafe_load(file)

    env_file = os.path.join(batch_dir, "_env.yaml")

    with open(str(env_file), "r") as file:
        env_dict = yaml.safe_load(file)

    env_config = EnvironmentConfig(**env_dict)
    exp_config = ExperimentConfig(**exp_dict)

    ccea = CooperativeCoevolutionaryAlgorithm(
        batch_dir=batch_dir,
        trials_dir=Path(batch_dir).parents[1]
        / "results"
        / batch_name
        / experiment_name,
        trial_id=trial_id,
        trial_name=Path(exp_file).stem,
        video_name=f"{experiment_name}_{trial_id}",
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Environment Data
        map_size=env_config.map_size,
        observation_size=env_config.obs_space_dim,
        action_size=env_config.action_space_dim,
        n_agents=len(env_config.rovers),
        n_pois=len(env_config.pois),
        # Experiment Data
        **asdict(exp_config),
    )
    return ccea.run()
