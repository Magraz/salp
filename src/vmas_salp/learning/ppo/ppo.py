from deap import base
from deap import creator
from deap import tools
import torch

import random

from vmas.simulator.environment import Environment
from vmas.simulator.utils import save_video

from vmas_salp.domain.create_env import create_env

from vmas_salp.learning.utils import (
    CCEAConfig,
    PolicyConfig,
)

from copy import deepcopy
import numpy as np
import os
from pathlib import Path
import yaml
import logging
import pickle
import csv

from itertools import combinations

# Create and configure logger
logging.basicConfig(format="%(asctime)s %(message)s")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


class PPO:
    def __init__(
        self,
        batch_dir: str,
        trials_dir: str,
        trial_id: int,
        trial_name: str,
        video_name: str,
        device: str,
        **kwargs,
    ):

        self.batch_dir = batch_dir
        self.trials_dir = trials_dir
        self.trial_name = trial_name
        self.trial_id = trial_id
        self.video_name = video_name

        # Environment data
        self.device = device
        self.map_size = kwargs.pop("map_size", [])
        self.observation_size = kwargs.pop("observation_size", 0)
        self.action_size = kwargs.pop("action_size", 0)
        self.n_agents = kwargs.pop("n_agents", 0)
        self.n_pois = kwargs.pop("n_pois", 0)

        # Experiment Data
        self.n_gens_between_save = kwargs.pop("n_gens_between_save", 0)

        # Flags
        self.use_teaming = kwargs.pop("use_teaming", False)
        self.use_fc = kwargs.pop("use_fc", False)

        self.team_size = (
            kwargs.pop("team_size", 0) if self.use_teaming else self.n_agents
        )
        self.team_combinations = [
            combo for combo in combinations(range(self.n_agents), self.team_size)
        ]

    def formTeams(self, population, joint_policies: int) -> list[Team]:
        # Start a list of teams
        teams = []

        # For each row in the population of subpops (grabs an individual from each row in the subpops)
        for i in range(joint_policies):

            # Get agents in this row of subpopulations
            agents = [subpop[i] for subpop in population]

            # Put the i'th individual on the team if it is inside our team combinations
            combination = random.sample(self.team_combinations, 1)[0]

            teams.append(
                Team(
                    idx=i,
                    individuals=[agents[idx] for idx in combination],
                    combination=combination,
                )
            )

        return teams

    def evaluateTeams(
        self,
        env: Environment,
        teams: list[Team],
        render: bool = False,
        save_render: bool = False,
    ):
        # Set up models
        joint_policies = [
            [self.generateTemplateNN() for _ in range(self.team_size)] for _ in teams
        ]

        # Load in the weights
        for i, team in enumerate(teams):
            for agent_nn, individual in zip(joint_policies[i], team.individuals):
                agent_nn.set_params(torch.from_numpy(individual).to(self.device))

        # Get initial observations per agent
        observations = env.reset()

        G_list = []
        frame_list = []

        # Start evaluation
        for step in range(self.n_steps):

            stacked_obs = torch.stack(observations, -1)

            actions = [
                torch.empty((0, self.action_size)).to(self.device)
                for _ in range(self.team_size)
            ]

            for observation, joint_policy in zip(stacked_obs, joint_policies):

                for i, policy in enumerate(joint_policy):
                    policy_output = policy.forward(observation[:, i])
                    actions[i] = torch.cat(
                        (
                            actions[i],
                            policy_output * self.output_multiplier,
                        ),
                        dim=0,
                    )

            observations, rewards, _, _ = env.step(actions)

            G_list.append(torch.stack([g[: len(teams)] for g in rewards], dim=0)[0])

            # Visualization
            if render:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )
                if save_render:
                    frame_list.append(frame)

        # Save video
        if render and save_render:
            save_video(self.video_name, frame_list, fps=1 / env.scenario.world.dt)

        return

    def load_checkpoint(
        self,
        checkpoint_name: str,
        fitness_dir: str,
        trial_dir: str,
    ):
        # Load checkpoint file
        with open(checkpoint_name, "rb") as handle:
            checkpoint = pickle.load(handle)
            pop = checkpoint["population"]
            checkpoint_gen = checkpoint["gen"]

        # Set fitness csv file to checkpoint
        new_fit_path = os.path.join(trial_dir, "fitness_edit.csv")
        with open(fitness_dir, "r") as inp, open(new_fit_path, "w") as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                if row[0].isdigit():
                    gen = int(row[0])
                    if gen <= checkpoint_gen:
                        writer.writerow(row)
                else:
                    writer.writerow(row)

        # Remove old fitness file
        os.remove(fitness_dir)
        # Rename new fitness file
        os.rename(new_fit_path, fitness_dir)

        return pop, checkpoint_gen

    def run(self):

        # Set trial directory name
        trial_folder_name = "_".join(("trial", str(self.trial_id)))
        trial_dir = os.path.join(self.trials_dir, trial_folder_name)
        fitness_dir = f"{trial_dir}/fitness.csv"
        checkpoint_name = os.path.join(trial_dir, "checkpoint.pickle")

        # Create directory for saving data
        if not os.path.isdir(trial_dir):
            os.makedirs(trial_dir)

        checkpoint_exists = Path(checkpoint_name).is_file()
        pop = None

        # Load checkpoint
        checkpoint_gen = 0

        # Create environment
        env = create_env(
            self.batch_dir,
            n_envs=10,
            device=self.device,
            team_size=self.team_size,
            benchmark=False,
        )
