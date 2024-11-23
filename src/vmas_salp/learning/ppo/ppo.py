from deap import base
from deap import creator
from deap import tools
import torch
from torch.nn import functional as F

import random

# from vmas.simulator.environment import Environment
# from vmas.simulator.utils import save_video

from vmas_salp.domain.create_env import create_env
from vmas_salp.learning.ppo.policy import Agent

import numpy as np

import os
from pathlib import Path
import logging
import pickle
import csv
import time
from tqdm import tqdm

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
        self.num_steps = 2048 # Noah: I am taking this as the number of steps in rollout
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
    
    def seed_PPO(self):
        # TRY NOT TO MODIFY: seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

    def reset_buffers(self):
        # ALGO Logic: Storage setup
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def init_learning(self):
        # Create environment
        self.env = create_env(
            self.batch_dir,
            device=self.device,
            n_envs=self.num_envs,
            n_agents=self.team_size,
        )

        self.joint_policies = [
            [
                Agent(self.observation_size, self.action_size, lr=self.learning_rate)
                for _ in self.team_size
            ]
        ]

        # Initialize buffers to empty
        self.reset_buffers()

        # TRY NOT TO MODIFY: start the game
        self.global_step = 0
        self.start_time = time.time()
        self.next_obs, _ = self.env.reset()
        self.next_done = torch.zeros(self.num_envs).to(self.device)

    def set_model_modes(self, do_eval):
        for a in self.joint_policies:
            if do_eval:
                a.actor_mean.eval()
            else:
                a.actor_mean.train()

    def calculate_returns_and_advantage(self, last_values, dones):
        '''
        Calculate the returns and advantages from the just collected rollout data
        See the calculate_returns_and_advantage method under RolloutBuffer in https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
        '''
        self.returns = np.zeros((self.n_agents, len(self.values[0])))
        self.advantages = np.zeros((self.n_agents, len(self.values[0])))

        last_gae_lam = 0
        # For each agent
        for i in range(self.n_agents):
            last_gae_lam = 0
            # Go through the buffer for the current agent 
            # and calculate the returns and advantages for just that agent
            for j in reversed(range(len(self.values[0]))):
                if j == len(self.values) - 1:
                    next_non_terminal = 1.0 - int(dones[i])
                    next_values = last_values[i][j]
                else:
                    next_non_terminal = 1.0 - self.dones[i][j + 1]
                    next_values = self.values[i][j + 1]

                # Calculate TD error
                delta = self.rewards[i][j] + self.gamma * next_values * next_non_terminal - self.values[i][j]
                # Calculate GAE Lambda
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                # Calculate GAE
                self.advantages[i, j] = last_gae_lam
            # Calculate returns for current agent
            self.returns[i] = self.advantages + np.array(self.values[i])

    def collect_rollouts(self):
        '''
        Add data to the data buffers
        See the collect_rollouts method in https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py#L21
        '''
        # Set all models to eval
        self.set_model_modes(True)
        # Reset Buffers
        self.reset_buffers()

        # TODO: Need to init last_obs
        
        for i in range(self.num_steps):
            with torch.no_grad():
                temp_acts = []
                temp_vals = []
                temp_log_probs = []
                # For each policy, get the action, value, and lob probability
                for j, agent in enumerate(self.joint_policies):
                    action, log_prob, _, value = agent.get_action_and_value(torch.tensor(self.last_obs[j], device=self.device)) 
                    temp_acts.append(action.cpu())
                    temp_vals.append(value.cpu())
                    temp_log_probs.append(log_prob.cpu())

            

            # Take a step in the env with the just computed actions
            obs, rewards, dones, infos = self.env.step(temp_acts)


            # After getting all the agent's stuff, add to the real buffers
            self.obs.append(self.last_obs)
            self.actions.append(temp_acts)
            self.rewards.append(rewards)
            self.dones.append(dones)
            self.values.append(temp_vals)
            self.logprobs.append(temp_log_probs)
            

            # Update the last observation for the next iteration
            self.last_obs = obs

        with torch.no_grad():
            temp_final_val = []
            # Get the value of the last obs so we can compute returns and advantage
            for k, agent in enumerate(self.joint_policies):
                temp_final_val.append(agent.get_value(torch.tensor(obs[k], device=self.device)).cpu())

        self.calculate_returns_and_advantage(last_values=temp_final_val, dones=dones)

        return True



    def train(self):
        '''
        Using data from the buffers, update each policy
        See the train method in https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
        '''
        # Set models to train
        self.set_model_modes(False)

        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        batch_size = self.num_steps / self.num_minibatches

        for i in range(self.n_agents):
            # TODO: Currently these lists aren't kept track of from agent to agent
            entropy_losses = []
            pg_losses = []
            value_losses = []
            clip_fractions = []
            continue_training = True
            for epoch in range(self.update_epochs):
                approx_kl_divergences = []
                start_index = 0
                for stop_index in range(batch_size, self.num_steps, batch_size):
                    # Grab batch_size actions from the current agent buffer
                    actions = self.actions[i][start_index:stop_index]

                    # Grab observations for the current agents
                    obs = self.obs[i][start_index:stop_index]

                    _, log_probs, entropy, values = self.joint_policies[i].get_action_and_value(torch.tensor(obs, device=self.device), actions) 

                    #Grab advantages (self.advantages is np array currently)
                    advantages = self.advantages[i, start_index:stop_index]

                    if self.norm_adv and len(advantages) > 1:
                        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

                    old_log_probs = torch.tensor(self.logprobs[i][start_index:stop_index])

                    # ratio of the old policy and the new (will be 1 for the first iteration (duh))
                    ratio = torch.exp(log_probs - old_log_probs)

                    # Calculate the clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # Keep track of some stuff for logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_coef).float()).item()
                    clip_fractions.append(clip_fraction)

                    # Calculate Value loss
                    old_values = self.values[i][start_index:stop_index]
                    if self.clip_vloss is None:
                        values_pred = values
                    else:
                        values_pred = old_values + torch.clamp(values - old_values, -self.clip_vloss, self.clip_vloss)

                    # Grab the returns for the current agent (self.returns is currently a np array)
                    returns = self.returns[i, start_index:stop_index]
                    value_loss = F.mse_loss(returns, values_pred)
                    # Keep track of for logging purposes
                    value_losses.append(value_loss)

                    # Now let's do entropy loss
                    if entropy is None:
                        entropy_loss = -torch.mean(-log_probs)
                    else:
                        entropy_loss = -torch.mean(entropy)
                    # Keep track of for logging purposes
                    entropy_losses.append(entropy_loss)

                    # Get the total loss
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Calculate Approx KL divergence
                    with torch.no_grad():
                        log_ratio = log_probs - old_log_probs
                        approx_kl_divergence = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        # Keep track of the kl_divergence for logging purposes
                        approx_kl_divergences.append(approx_kl_divergence)

                    # Don't know if we want this, but I added it in for completeness.
                    if self.target_kl is not None and approx_kl_divergence > 1.5 * self.target_kl:
                        continue_training = False
                        print(f'Early stopping at step {epoch} for agent {i} due to reaching max KL divergence {approx_kl_divergence}')
                        break

                    # Do the actual optimization
                    self.joint_policies[i].optimizer.zero_grad()
                    loss.backward
                    # Clip the gradient norm
                    torch.nn.utils.clip_grad_norm_(self.joint_policies[i].parameters(), self.max_grad_norm)
                    self.joint_policies[i].optimizer.step()

                    
                    # Update the starting index
                    start_index = stop_index
                # If the KL is too high for the current agent, move on to the next agent
                if not continue_training:
                    break
        

    
    def run_PPO(self):
        '''
        Use this to train policies for all agents
        '''
        # Seed probabilities
        self.seed_PPO()
        # Add members for all the stuff PPO needs
        self.init_learning()
        # TODO: Initialize last_obs to whatever a good initial input is
        self.last_obs = None

        # Train for total_timesteps iterations
        for i in tqdm(range(self.total_timesteps)):
            # Collect data with current policy
            continue_training = self.collect_rollouts()

            if not continue_training:
                break

            # Train on the data just collected
            self.train()
        
        return self.joint_policies


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
