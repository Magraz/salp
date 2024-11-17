from deap import base
from deap import creator
from deap import tools
import torch

import random

from vmas.simulator.environment import Environment
from vmas.simulator.utils import save_video

from vmas_salp.domain.create_env import create_env
from vmas_salp.learning.ppo.policy import Agent

import numpy as np

import os
from pathlib import Path
import logging
import pickle
import csv
import time

from itertools import combinations

from vmas_salp.learning.types import (
    Team,
)

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

        # Flags
        self.use_teaming = kwargs.pop("use_teaming", False)

        self.team_size = (
            kwargs.pop("team_size", 0) if self.use_teaming else self.n_agents
        )
        self.team_combinations = [
            combo for combo in combinations(range(self.n_agents), self.team_size)
        ]

        # PPO Args
        self.seed = 1
        self.torch_deterministic = True
        self.total_timesteps = 1000000
        self.learning_rate = 3e-4
        self.num_envs = 1
        self.num_steps = 2048
        self.episodes = 1000
        self.anneal_lr = True
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.num_minibatches = 32
        self.update_epochs = 10
        self.norm_adv = True
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None

        self.batch_size = 0
        self.minibatch_size = 0
        self.num_iterations = 0

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

    def run(self):
        # TRY NOT TO MODIFY: seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        # Create environment
        env = create_env(
            self.batch_dir,
            device=self.device,
            n_envs=self.num_envs,
            n_agents=self.team_size,
        )

        joint_policies = [
            [
                Agent(self.observation_size, self.action_size, lr=self.learning_rate)
                for _ in self.team_size
            ]
        ]

        # ALGO Logic: Storage setup
        obs = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        values = []

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = env.reset()
        next_done = torch.zeros(self.num_envs).to(self.device)

        # Start evaluation
        for iteration in range(1, self.num_iterations + 1):

            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                for joint_policy in joint_policies:
                    for agent in joint_policy:
                        agent.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):

                global_step += self.num_envs
                obs.append(next_obs)
                dones.append(next_done)

                ##MY CODE
                stacked_obs = torch.stack(next_obs, -1)

                joint_actions = [
                    torch.empty((0, self.action_size)).to(self.device)
                    for _ in range(self.team_size)
                ]

                for observation, joint_policy in zip(stacked_obs, joint_policies):

                    for i, agent in enumerate(joint_policy):

                        with torch.no_grad():
                            action, logprob, _, value = agent.get_action_and_value(
                                observation[:, i]
                            )
                            values.append(value.flatten())
                        actions.append(action)
                        logprobs.append(logprob)

                        joint_actions[i] = torch.cat(
                            (
                                joint_actions[i],
                                action,
                            ),
                            dim=0,
                        )

                next_obs, rew, next_done, info = env.step(joint_actions)
                rewards.append(rew)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + (self.observation_size))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + (self.action_size))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
