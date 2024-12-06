from deap import base
from deap import creator
from deap import tools
import torch
from torch.nn import functional as F
# torch.autograd.set_detect_anomaly(True)

import random

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
        self.total_timesteps = 10
        self.learning_rate = 3e-4
        self.num_envs = 1
        self.num_steps = 128  # 2048 # Noah: I am taking this as the number of steps in rollout
        self.episodes = 1000
        self.anneal_lr = True
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.num_minibatches = 4  # 32
        self.update_epochs = 10
        self.norm_adv = True
        self.clip_coef = 0.2
        self.clip_vloss = None
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
        self.obs = torch.zeros(
            (self.n_agents, self.num_envs, self.num_steps, self.observation_size),
            device=self.device,
        )
        self.actions = torch.zeros(
            (self.n_agents, self.num_envs, self.num_steps, self.action_size),
            device=self.device,
        )
        self.logprobs = torch.zeros(
            (self.n_agents, self.num_envs, self.num_steps), device=self.device
        )
        self.rewards = torch.zeros(
            (self.n_agents, self.num_envs, self.num_steps), device=self.device
        )
        self.dones = torch.zeros(
            (self.n_agents, self.num_envs, self.num_steps), device=self.device
        )
        self.values = torch.zeros(
            (self.n_agents, self.num_envs, self.num_steps), device=self.device
        )

    def init_learning(self):
        # Create environment
        self.env = create_env(
            self.batch_dir,
            device=self.device,
            n_envs=self.num_envs,
            n_agents=self.team_size,
        )

        self.joint_policies = [
            Agent(self.observation_size, self.action_size, lr=self.learning_rate)
            for _ in range(self.team_size)
        ]

        # Initialize buffers to empty
        self.reset_buffers()

        # TRY NOT TO MODIFY: start the game
        self.global_step = 0
        self.start_time = time.time()
        self.next_obs = self.env.reset()
        self.next_done = torch.zeros(self.num_envs).to(self.device)

    def set_model_modes(self, do_eval):
        for a in self.joint_policies:
            if do_eval:
                a.actor_mean.eval()
            else:
                a.actor_mean.train()

    def calculate_returns_and_advantage(self, last_values, dones):
        """
        Calculate the returns and advantages from the just collected rollout data
        See the calculate_returns_and_advantage method under RolloutBuffer in https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
        """
        self.returns = torch.zeros(
            (self.n_agents, self.num_envs, self.num_steps), device=self.device
        )
        self.advantages = torch.zeros(
            (self.n_agents, self.num_envs, self.num_steps), device=self.device
        )

        # For each agent
        # for i in range(self.n_agents):
        last_gae_lam = 0
        # Go through the buffer for the current agent
        # and calculate the returns and advantages for just that agent
        # TODO: May need to reduce the dimensions of dones, rewards, and values
        for k in reversed(range(self.num_steps)):
            if k == self.num_steps - 1:
                next_non_terminal = 1.0 - int(dones[0])
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[:, :, k + 1]
                next_values = self.values[:, :, k + 1]

            # Calculate TD error
            delta = (
                self.rewards[:, :, k]
                + self.gamma * next_values * next_non_terminal
                - self.values[:, :, k]
            )
            # Calculate GAE Lambda
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            # Calculate GAE
            # q = self.advantages[:, :, k]
            # qq = last_gae_lam
            self.advantages[:, :, k] = last_gae_lam
        # Calculate returns for current agent
        self.returns = self.advantages + self.values
        # q = ""

    def evaluate(self, render=True, save_render=True):
        # Seed probabilities
        self.seed_PPO()
        # Add members for all the stuff PPO needs
        self.init_learning()
        self.last_obs = self.env.reset()

        # Set all models to eval
        self.set_model_modes(True)
        # Reset Buffers
        self.reset_buffers()

        frame_list = []
        rew = 0

        # Do rollout for num_steps for every agents
        for i in range(self.num_steps):
            with torch.no_grad():
                temp_acts = torch.zeros(
                    (self.n_agents, self.num_envs, self.action_size)
                )
                temp_vals = torch.zeros((self.n_agents, self.num_envs))
                temp_log_probs = torch.zeros((self.n_agents, self.num_envs))
                # For each policy, get the action, value, and lob probability
                for j, agent in enumerate(self.joint_policies):
                    action, log_prob, _, value = agent.get_action_and_value(
                        self.last_obs[j]
                    )
                    # t = action.cpu().data.numpy()
                    # if action > 1:
                    #     action = torch.tensor([[1.0]], device=self.device)
                    # elif action < -1:
                    #     action = torch.tensor([[-1.0]], device=self.device)

                    temp_acts[j] = action
                    temp_vals[j] = value
                    temp_log_probs[j] = log_prob

            # Take a step in the env with the just computed actions
            obs, rewards, dones, infos = self.env.step(temp_acts)

            rew = rewards[0]

            # Update the last observation for the next iteration
            self.last_obs = obs

            # Visualization
            if render:
                frame = self.env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )
                if save_render:
                    frame_list.append(frame)

        # Save video
        if render and save_render:
            save_video(self.video_name, frame_list, fps=1 / self.env.scenario.world.dt)

        return rew

    def collect_rollouts(self):
        """
        Add data to the data buffers
        See the collect_rollouts method in https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py#L21
        """
        # Set all models to eval
        self.set_model_modes(True)
        # Reset Buffers
        self.reset_buffers()

        # Do rollout for num_steps for every agents
        for i in range(self.num_steps):
            temp_acts = torch.zeros((self.n_agents, self.num_envs, self.action_size))
            temp_vals = torch.zeros((self.n_agents, self.num_envs))
            temp_log_probs = torch.zeros((self.n_agents, self.num_envs))
            # For each policy, get the action, value, and log probability
            for j, agent in enumerate(self.joint_policies):
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(
                        self.last_obs[j]
                    )
                # t = action.cpu().data.numpy()
                # if action > 1:
                #     action = torch.tensor([[1.0]], device=self.device)
                # elif action < -1:
                #     action = torch.tensor([[-1.0]], device=self.device)

                temp_acts[j] = action
                temp_vals[j] = value
                temp_log_probs[j] = log_prob
                # temp_acts.append(action.cpu())
                # temp_vals.append(value.cpu())
                # temp_log_probs.append(log_prob.cpu())

            # Take a step in the env with clipped actions (note, we save the unclipped actions)
            clipped_action = torch.clamp(temp_acts, 0.0, 1.0)
            obs, rewards, dones, infos = self.env.step(clipped_action)

            # After getting all the agent's stuff, add to the real buffers
            self.obs[:, :, i] = torch.stack(self.last_obs)
            self.actions[:, :, i] = temp_acts
            self.rewards[:, :, i] = torch.stack(rewards)
            self.dones[:, :, i] = dones
            self.values[:, :, i] = temp_vals
            self.logprobs[:, :, i] = temp_log_probs

            # self.obs.append(self.last_obs)
            # self.actions.append(temp_acts)
            # self.rewards.append(rewards)
            # self.dones.append(dones)
            # self.values.append(temp_vals)
            # self.logprobs.append(temp_log_probs)

            # Update the last observation for the next iteration
            self.last_obs = obs

        with torch.no_grad():
            temp_final_val = torch.zeros(
                (self.n_agents, self.num_envs), device=self.device
            )
            # Get the value of the last obs so we can compute returns and advantage
            for k, agent in enumerate(self.joint_policies):
                temp_final_val[k] = agent.get_value(obs[k])

        self.calculate_returns_and_advantage(last_values=temp_final_val, dones=dones)
        
        # TODO: This is dumb. The training algorithm should not have to reset the environment.
        # Create a brand new env
        self.env = create_env(
            self.batch_dir,
            device=self.device,
            n_envs=self.num_envs,
            n_agents=self.team_size,
        )
        self.last_obs = self.env.reset()

        return True

    def train(self):
        """
        Using data from the buffers, update each policy
        See the train method in https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
        """
        # Set models to train
        self.set_model_modes(False)

        # entropy_losses = []
        # pg_losses = []
        # value_losses = []
        batch_size = self.num_steps // self.num_minibatches

        # TODO: Just keep track of losses
        entropy_losses = []
        pg_losses = []
        value_losses = []
        continue_training = True
        for epoch in range(self.update_epochs):
            approx_kl_divergences = []
            start_index = 0
            # Use these indices to "shuffle" the buffers
            indices = np.random.permutation(self.num_steps)
            for stop_index in range(batch_size, self.num_steps, batch_size):
                for cur_agent_index, cur_agent_policy in enumerate(self.joint_policies):
                    # Grab batch_size actions for all agents from the current agent buffer
                    actions = self.actions[cur_agent_index, :, indices[start_index:stop_index]]

                    # Grab observations for all agents (num_agents x num_envs x batch_size x obs_size)
                    obs = self.obs[cur_agent_index, :, indices[start_index:stop_index]]

                    # Do the forward pass for each agent
                    # temp_log_probs = []
                    # temp_entropy = []
                    # temp_values = []
                    # for ii, agent in enumerate(self.joint_policies):
                    _, cur_log_probs, cur_entropy, cur_values = cur_agent_policy.get_action_and_value(obs, actions)
                    
                    # temp_log_probs.append(cur_log_probs)
                    # temp_entropy.append(cur_entropy)
                    # temp_values.append(cur_values.squeeze(-1))

                    # Convert to tensors (may need to put on cuda)
                    # Size: num_agents x num_envs x batch
                    log_probs = cur_log_probs
                    entropy = cur_entropy
                    values = cur_values.squeeze(-1)
                    # log_probs = torch.stack(temp_log_probs)
                    # entropy = torch.stack(temp_entropy)
                    # values = torch.stack(temp_values)

                    # Grab advantages (size: num_agents x num_envs x batch)
                    advantages = self.advantages[cur_agent_index, :, indices[start_index:stop_index]]

                    if self.norm_adv and advantages.shape[1] > 1:
                        advantages = (advantages - torch.mean(advantages)) / (
                            torch.std(advantages) + 1e-8
                        )

                    old_log_probs = self.logprobs[cur_agent_index, :, indices[start_index:stop_index]]

                    # ratio of the old policy and the new (will be 1 for the first iteration (duh))
                    # TODO: Make sure the first iteration is all ones. It currently has some non-one entries
                    ratio = torch.exp(log_probs - old_log_probs)

                    # Calculate the clipped surrogate loss for each agent
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    # Should be size num_agents x 1 (mean over the batch dimension)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean(dim=-1)

                    # Keep track of some stuff for logging
                    pg_losses.append(policy_loss)
                    # clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_coef).float()).item()
                    # clip_fractions.append(clip_fraction)

                    # Calculate Value loss (num_agents x num_envs x batch_size)
                    old_values = self.values[cur_agent_index, :, indices[start_index:stop_index]]
                    if self.clip_vloss is None:
                        values_pred = values
                    else:
                        values_pred = old_values + torch.clamp(
                            values - old_values, -self.clip_vloss, self.clip_vloss
                        )

                    # Grab the returns for the current agent
                    returns = self.returns[cur_agent_index, :, indices[start_index:stop_index]]
                    # Should be size num_agents x 1
                    value_loss = F.mse_loss(returns, values_pred, reduction="none")
                    value_loss = torch.mean(value_loss, dim=-1)
                    # Keep track of for logging purposes
                    value_losses.append(value_loss)

                    # Now let's do entropy loss (should be num_agents x 1)
                    if entropy is None:
                        entropy_loss = -torch.mean(-log_probs, dim=-1)
                    else:
                        entropy_loss = -torch.mean(entropy, dim=-1)
                    # Keep track of for logging purposes
                    entropy_losses.append(entropy_loss)

                    # Get the total loss for the current agent.
                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                    )

                    # Calculate Approx KL divergence
                    with torch.no_grad():
                        log_ratio = log_probs - old_log_probs
                        approx_kl_divergence = (
                            torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        )
                        # Keep track of the kl_divergence for logging purposes
                        approx_kl_divergences.append(approx_kl_divergence)

                    # Don't know if we want this, but I added it in for completeness.
                    if (
                        self.target_kl is not None
                        and approx_kl_divergence > 1.5 * self.target_kl
                    ):
                        continue_training = False
                        print(
                            f"Early stopping at step {epoch} for due to reaching max KL divergence {approx_kl_divergence}"
                        )
                        break

                    # Do the actual optimization for the current agent agent
                    # for j, a in enumerate(self.joint_policies):
                    cur_agent_policy.optimizer.zero_grad()
                    # cur_agent_loss = loss[j]
                    # cur_agent_loss.backward()
                    loss.backward()
                    # Clip the gradient norm
                    torch.nn.utils.clip_grad_norm_(cur_agent_policy.parameters(), self.max_grad_norm)
                    cur_agent_policy.optimizer.step()

                # Update the starting index
                start_index = stop_index
            # If the KL is too high for the current agent, move on to the next agent
            if not continue_training:
                break
        # t = self.rewards[0, 0]
        return pg_losses, value_losses, entropy_losses, torch.mean(self.rewards[0, 0])
    
    '''
    OG train
    def train(self):
        """
        Using data from the buffers, update each policy
        See the train method in https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
        """
        # Set models to train
        self.set_model_modes(False)

        # entropy_losses = []
        # pg_losses = []
        # value_losses = []
        batch_size = self.num_steps // self.num_minibatches

        # TODO: Just keep track of losses
        entropy_losses = []
        pg_losses = []
        value_losses = []
        continue_training = True
        for epoch in range(self.update_epochs):
            approx_kl_divergences = []
            start_index = 0
            # Use these indices to "shuffle" the buffers
            indices = np.random.permutation(self.num_steps)
            for stop_index in range(batch_size, self.num_steps, batch_size):
                # Grab batch_size actions for all agents from the current agent buffer
                actions = self.actions[:, :, indices[start_index:stop_index]]

                # Grab observations for all agents (num_agents x num_envs x batch_size x obs_size)
                obs = self.obs[:, :, indices[start_index:stop_index]]

                # Do the forward pass for each agent
                temp_log_probs = []
                temp_entropy = []
                temp_values = []
                for ii, agent in enumerate(self.joint_policies):
                    t = obs[ii]
                    _, cur_log_probs, cur_entropy, cur_values = (
                        agent.get_action_and_value(obs[ii], actions[ii])
                    )
                    temp_log_probs.append(cur_log_probs)
                    temp_entropy.append(cur_entropy)
                    temp_values.append(cur_values.squeeze(-1))

                # Convert to tensors (may need to put on cuda)
                # Size: num_agents x num_envs x batch
                log_probs = torch.stack(temp_log_probs)
                entropy = torch.stack(temp_entropy)
                values = torch.stack(temp_values)

                # Grab advantages (size: num_agents x num_envs x batch)
                advantages = self.advantages[:, :, indices[start_index:stop_index]]

                if self.norm_adv and len(advantages) > 1:
                    advantages = (advantages - torch.mean(advantages)) / (
                        torch.std(advantages) + 1e-8
                    )

                old_log_probs = self.logprobs[:, :, indices[start_index:stop_index]]

                # ratio of the old policy and the new (will be 1 for the first iteration (duh))
                # TODO: Make sure the first iteration is all ones. It currently has some non-one entries
                ratio = torch.exp(log_probs - old_log_probs)

                # Calculate the clipped surrogate loss for each agent
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                # Should be size num_agents x 1 (mean over the batch dimension)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean(dim=-1)

                # Keep track of some stuff for logging
                pg_losses.append(policy_loss)
                # clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_coef).float()).item()
                # clip_fractions.append(clip_fraction)

                # Calculate Value loss (num_agents x num_envs x batch_size)
                old_values = self.values[:, :, indices[start_index:stop_index]]
                if self.clip_vloss is None:
                    values_pred = values
                else:
                    values_pred = old_values + torch.clamp(
                        values - old_values, -self.clip_vloss, self.clip_vloss
                    )

                # Grab the returns for the current agent
                returns = self.returns[:, :, indices[start_index:stop_index]]
                # Should be size num_agents x 1
                value_loss = F.mse_loss(returns, values_pred, reduction="none")
                value_loss = torch.mean(value_loss, dim=-1)
                # Keep track of for logging purposes
                value_losses.append(value_loss)

                # Now let's do entropy loss (should be num_agents x 1)
                if entropy is None:
                    entropy_loss = -torch.mean(-log_probs, dim=-1)
                else:
                    entropy_loss = -torch.mean(entropy, dim=-1)
                # Keep track of for logging purposes
                entropy_losses.append(entropy_loss)

                # Get the total loss. Should be size (num_agents x 1)
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate Approx KL divergence
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs
                    approx_kl_divergence = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    # Keep track of the kl_divergence for logging purposes
                    approx_kl_divergences.append(approx_kl_divergence)

                # Don't know if we want this, but I added it in for completeness.
                if (
                    self.target_kl is not None
                    and approx_kl_divergence > 1.5 * self.target_kl
                ):
                    continue_training = False
                    print(
                        f"Early stopping at step {epoch} for due to reaching max KL divergence {approx_kl_divergence}"
                    )
                    break

                # Do the actual optimization for each agent
                for j, a in enumerate(self.joint_policies):
                    a.optimizer.zero_grad()
                    # cur_agent_loss = loss[j]
                    # cur_agent_loss.backward()
                    loss[j].backward()
                    # Clip the gradient norm
                    torch.nn.utils.clip_grad_norm_(a.parameters(), self.max_grad_norm)
                    a.optimizer.step()

                # Update the starting index
                start_index = stop_index
            # If the KL is too high for the current agent, move on to the next agent
            if not continue_training:
                break
        return pg_losses, value_losses, entropy_losses, self.rewards
    '''

    def run(self):
        """
        Use this to train policies for all agents
        """
        # Set trial directory name
        trial_folder_name = "_".join(("trial", str(self.trial_id)))
        trial_dir = os.path.join(self.trials_dir, trial_folder_name)

        # Create directory for saving data
        if not os.path.isdir(trial_dir):
            os.makedirs(trial_dir)

        # Seed probabilities
        self.seed_PPO()
        # Add members for all the stuff PPO needs
        self.init_learning()
        self.last_obs = self.env.reset()
        rew = []
        # Train for total_timesteps iterations
        for i in tqdm(range(self.total_timesteps)):
            # Collect data with current policy
            continue_training = self.collect_rollouts()

            if not continue_training:
                break

            # Train on the data just collected
            pg_losses, value_losses, entropy_losses, rewards = self.train()

            # Save in some way or plot
            # rew.append(rewards[0, 0, 0].detach().item())
            rew.append(rewards.item())

            if i % 10 == 0:
                # Save checkpoint
                with open(os.path.join(trial_dir, "checkpoint.pickle"), "wb") as handle:
                    pickle.dump(
                        {
                            "joint_policies": self.joint_policies,
                            "rew": rew,
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
        # print(rew)
        return self.joint_policies, rew

    # def run(self):
    #     # TRY NOT TO MODIFY: seeding
    #     random.seed(self.seed)
    #     np.random.seed(self.seed)
    #     torch.manual_seed(self.seed)
    #     torch.backends.cudnn.deterministic = self.torch_deterministic

    #     # Create environment
    #     env = create_env(
    #         self.batch_dir,
    #         device=self.device,
    #         n_envs=self.num_envs,
    #         n_agents=self.team_size,
    #     )

    #     joint_policies = [
    #         [
    #             Agent(self.observation_size, self.action_size, lr=self.learning_rate)
    #             for _ in self.team_size
    #         ]
    #     ]

    #     # ALGO Logic: Storage setup
    #     obs = []
    #     actions = []
    #     logprobs = []
    #     rewards = []
    #     dones = []
    #     values = []

    #     # TRY NOT TO MODIFY: start the game
    #     global_step = 0
    #     start_time = time.time()
    #     next_obs, _ = env.reset()
    #     next_done = torch.zeros(self.num_envs).to(self.device)

    #     # Start evaluation
    #     for iteration in range(1, self.num_iterations + 1):

    #         # Annealing the rate if instructed to do so.
    #         if self.anneal_lr:
    #             frac = 1.0 - (iteration - 1.0) / self.num_iterations
    #             lrnow = frac * self.learning_rate
    #             for joint_policy in joint_policies:
    #                 for agent in joint_policy:
    #                     agent.optimizer.param_groups[0]["lr"] = lrnow

    #         for step in range(0, self.num_steps):

    #             global_step += self.num_envs
    #             obs.append(next_obs)
    #             dones.append(next_done)

    #             ##MY CODE
    #             stacked_obs = torch.stack(next_obs, -1)

    #             joint_actions = [
    #                 torch.empty((0, self.action_size)).to(self.device)
    #                 for _ in range(self.team_size)
    #             ]

    #             for observation, joint_policy in zip(stacked_obs, joint_policies):

    #                 for i, agent in enumerate(joint_policy):

    #                     with torch.no_grad():
    #                         action, logprob, _, value = agent.get_action_and_value(
    #                             observation[:, i]
    #                         )
    #                         values.append(value.flatten())
    #                     actions.append(action)
    #                     logprobs.append(logprob)

    #                     joint_actions[i] = torch.cat(
    #                         (
    #                             joint_actions[i],
    #                             action,
    #                         ),
    #                         dim=0,
    #                     )

    #             next_obs, rew, next_done, info = env.step(joint_actions)
    #             rewards.append(rew)

    #         # bootstrap value if not done
    #         with torch.no_grad():
    #             next_value = agent.get_value(next_obs).reshape(1, -1)
    #             advantages = torch.zeros_like(rewards).to(self.device)
    #             lastgaelam = 0
    #             for t in reversed(range(self.num_steps)):
    #                 if t == self.num_steps - 1:
    #                     nextnonterminal = 1.0 - next_done
    #                     nextvalues = next_value
    #                 else:
    #                     nextnonterminal = 1.0 - dones[t + 1]
    #                     nextvalues = values[t + 1]
    #                 delta = (
    #                     rewards[t]
    #                     + self.gamma * nextvalues * nextnonterminal
    #                     - values[t]
    #                 )
    #                 advantages[t] = lastgaelam = (
    #                     delta
    #                     + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
    #                 )
    #             returns = advantages + values

    #         # flatten the batch
    #         b_obs = obs.reshape((-1,) + (self.observation_size))
    #         b_logprobs = logprobs.reshape(-1)
    #         b_actions = actions.reshape((-1,) + (self.action_size))
    #         b_advantages = advantages.reshape(-1)
    #         b_returns = returns.reshape(-1)
    #         b_values = values.reshape(-1)
