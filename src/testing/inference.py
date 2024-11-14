import pickle
import sys
import os
from pathlib import Path
import torch
import yaml

sys.path.insert(0, "./src")

from learning.ccea.ccea import CooperativeCoevolutionaryAlgorithm
from domain.create_env import create_env

batch_name = "order"
experiment_name = "d_gru"
trial_id = 0
checkpoint_path = f"./src/testing/checkpoint.pickle"
batch_dir = f"./src/experiments/yamls/{batch_name}"
config_dir = os.path.join(batch_dir, f"{experiment_name}.yaml")

with open(str(config_dir), "r") as file:
    config = yaml.safe_load(file)

env_file = os.path.join(batch_dir, "_env.yaml")

with open(str(env_file), "r") as file:
    env_config = yaml.safe_load(file)

best_team = None

with open(checkpoint_path, "rb") as handle:
    checkpoint = pickle.load(handle)
    best_team = checkpoint["best_team"]


ccea = CooperativeCoevolutionaryAlgorithm(
    batch_dir=batch_dir,
    trials_dir=os.path.join(
        Path(batch_dir).parents[1],
        "results",
        "_".join((batch_name, experiment_name)),
    ),
    trial_id=trial_id,
    trial_name=Path(config_dir).stem,
    video_name=f"{experiment_name}_{trial_id}",
    device="cuda" if torch.cuda.is_available() else "cpu",
    # Environment Data
    map_size=env_config["env"]["map_size"],
    n_steps=config["ccea"]["n_steps"],
    observation_size=8,
    action_size=2,
    # Flags
    use_teaming=config["use_teaming"],
    use_fc=config["use_fc"],
    # Agent data
    n_agents=len(env_config["env"]["rovers"]),
    policy_type=config["policy"]["type"],
    policy_n_hidden=config["policy"]["hidden_layers"],
    weight_initialization=config["policy"]["weight_initialization"],
    output_multiplier=config["policy"]["output_multiplier"],
    # POIs data
    n_pois=len(env_config["env"]["pois"]),
    # Learning data
    n_gens=config["ccea"]["n_gens"],
    subpop_size=config["ccea"]["subpopulation_size"],
    selection_method=config["ccea"]["selection"],
    mutation_mean=config["ccea"]["mutation"]["mean"],
    max_std_deviation=config["ccea"]["mutation"]["max_std_deviation"],
    min_std_deviation=config["ccea"]["mutation"]["min_std_deviation"],
    fitness_shaping_method=config["ccea"]["fitness_shaping"],
    team_size=config["teaming"]["team_size"],
    fc_n_epochs=config["fitness_critic"]["epochs"],
    fc_loss_type=config["fitness_critic"]["loss_type"],
    fc_type=config["fitness_critic"]["type"],
    fc_n_hidden=config["fitness_critic"]["hidden_layers"],
    fitness_calculation=config["ccea"]["fitness_calculation"],
    n_gens_between_save=config["data"]["n_gens_between_save"],
)

eval_infos = ccea.evaluateTeams(
    create_env(batch_dir=batch_dir, n_envs=1, device=ccea.device, viewer_zoom=2),
    [best_team],
    render=True,
    save_render=True,
)
