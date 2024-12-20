import os
from pathlib import Path
import yaml

from vmas import make_env
from vmas.simulator.environment import Environment

from vmas_salp.domain.salp_domain import SalpDomain
from torchrl.envs.libs.vmas import VmasEnv


def create_env(
    batch_dir,
    n_envs: int,
    device: str,
    n_agents: int,
    benchmark: bool = False,
    **kwargs
) -> Environment:

    env_filename = "_env.yaml"

    if benchmark:
        env_filename = "salp.yaml"

    env_file = os.path.join(batch_dir, env_filename)

    with open(str(env_file), "r") as file:
        env_config = yaml.safe_load(file)

    # Environment data
    map_size = env_config["map_size"]

    # Agent data
    agents_colors = [
        agent["color"] if agent.get("color") else "BLUE"
        for agent in env_config["agents"]
    ]
    agents_positions = [poi["position"]["coordinates"] for poi in env_config["agents"]]
    lidar_range = [rover["observation_radius"] for rover in env_config["agents"]]
    shuffle_agents_positions = env_config["shuffle_agents_positions"]

    # POIs data
    n_pois = len(env_config["targets"])
    poi_positions = [poi["position"]["coordinates"] for poi in env_config["targets"]]
    poi_values = [poi["value"] for poi in env_config["targets"]]
    poi_colors = [
        poi["color"] if poi.get("color") else "GREEN" for poi in env_config["targets"]
    ]
    coupling = [poi["coupling"] for poi in env_config["targets"]]
    obs_radius = [poi["observation_radius"] for poi in env_config["targets"]]

    if benchmark:
        # Set up the enviornment
        env = VmasEnv(
            scenario=SalpDomain(),
            num_envs=n_envs,
            device=device,
            seed=None,
            # Environment specific variables
            n_agents=n_agents,
            n_targets=n_pois,
            agents_positions=agents_positions,
            agents_colors=agents_colors,
            targets_positions=poi_positions,
            targets_values=poi_values,
            targets_colors=poi_colors,
            x_semidim=map_size[0],
            y_semidim=map_size[1],
            agents_per_target=coupling[0],
            covering_range=obs_radius[0],
            lidar_range=lidar_range[0],
            viewer_zoom=kwargs.pop("viewer_zoom", 1.8),
            shuffle_agents_positions=shuffle_agents_positions,
        )
    else:
        env = make_env(
            scenario=SalpDomain(),
            num_envs=n_envs,
            device=device,
            seed=None,
            # Environment specific variables
            n_agents=n_agents,
            n_targets=n_pois,
            agents_positions=agents_positions,
            agents_colors=agents_colors,
            targets_positions=poi_positions,
            targets_values=poi_values,
            targets_colors=poi_colors,
            x_semidim=map_size[0],
            y_semidim=map_size[1],
            agents_per_target=coupling[0],
            covering_range=obs_radius[0],
            lidar_range=lidar_range[0],
            viewer_zoom=kwargs.pop("viewer_zoom", 1.8),
            shuffle_agents_positions=shuffle_agents_positions,
        )

    return env
