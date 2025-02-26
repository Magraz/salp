from deap import base
from deap import creator
from deap import tools
import torch
import torch.nn.functional as F

import random

from vmas.simulator.environment import Environment
from vmas.simulator.utils import save_video

from vmas_salp.learning.ccea.policies.mlp import MLP_Policy
from vmas_salp.learning.ccea.policies.gru import GRU_Policy
from vmas_salp.learning.ccea.policies.cnn import CNN_Policy

from vmas_salp.domain.create_env import create_env
from vmas_salp.learning.ccea.selection import (
    binarySelection,
    epsilonGreedySelection,
    softmaxSelection,
)
from vmas_salp.learning.ccea.types import (
    EvalInfo,
    PolicyEnum,
    SelectionEnum,
    FitnessShapingEnum,
    InitializationEnum,
    FitnessCalculationEnum,
)
from vmas_salp.learning.dataclasses import (
    CCEAConfig,
    PolicyConfig,
)

from vmas_salp.learning.types import (
    Team,
    JointTrajectory,
)

from vmas_salp.learning.ccea.utils import stack_and_pad_1d_tensors, pad_width

from copy import deepcopy
import numpy as np
import os
from pathlib import Path
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


class CooperativeCoevolutionaryAlgorithm:
    def __init__(
        self,
        batch_dir: str,
        trials_dir: str,
        trial_id: int,
        trial_name: str,
        video_name: str,
        device: str,
        ccea_config: CCEAConfig,
        policy_config: PolicyConfig,
        **kwargs,
    ):
        policy_config = PolicyConfig(**policy_config)
        ccea_config = CCEAConfig(**ccea_config)

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

        # Policy
        self.output_multiplier = policy_config.output_multiplier
        self.policy_hidden_layers = policy_config.hidden_layers
        self.policy_type = policy_config.type
        self.weight_initialization = policy_config.weight_initialization

        # CCEA
        self.n_gens = ccea_config.n_gens
        self.n_steps = ccea_config.n_steps
        self.subpop_size = ccea_config.subpopulation_size
        self.n_mutants = self.subpop_size // 2
        self.selection_method = ccea_config.selection
        self.fitness_shaping_method = ccea_config.fitness_shaping
        self.fitness_calculation = ccea_config.fitness_calculation
        self.max_std_dev = ccea_config.mutation["max_std_deviation"]
        self.min_std_dev = ccea_config.mutation["min_std_deviation"]
        self.mutation_mean = ccea_config.mutation["mean"]

        self.team_size = (
            kwargs.pop("team_size", 0) if self.use_teaming else self.n_agents
        )
        self.team_combinations = [
            list(combo) for combo in combinations(range(self.n_agents), self.team_size)
        ]

        self.std_dev_list = np.arange(
            start=self.max_std_dev,
            stop=self.min_std_dev,
            step=-((self.max_std_dev - self.min_std_dev) / (self.n_gens + 1)),
        )

        # Create the type of fitness we're optimizing
        creator.create("Individual", np.ndarray, fitness=0.0)

        # Now set up the toolbox
        self.toolbox = base.Toolbox()

        self.toolbox.register(
            "subpopulation",
            tools.initRepeat,
            list,
            self.createIndividual,
            n=self.subpop_size,
        )

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.subpopulation,
            n=self.n_agents,
        )

    def createIndividual(self):
        match (self.weight_initialization):
            case InitializationEnum.KAIMING:
                temp_model = self.generateTemplateNN()
                params = temp_model.get_params()
        return creator.Individual(params[:].cpu().numpy().astype(np.float32))

    def generateTemplateNN(self):
        match (self.policy_type):

            case PolicyEnum.GRU:
                agent_nn = GRU_Policy(
                    input_size=5,
                    hidden_size=self.policy_hidden_layers[0],
                    hidden_layers=len(self.policy_hidden_layers),
                    output_size=self.action_size,
                ).to(self.device)

            case PolicyEnum.CNN:
                agent_nn = CNN_Policy().to(self.device)

            case PolicyEnum.MLP:
                agent_nn = MLP_Policy(
                    input_size=16,
                    hidden_layers=len(self.policy_hidden_layers),
                    hidden_size=self.policy_hidden_layers[0],
                    output_size=self.action_size,
                ).to(self.device)

        return agent_nn

    def getBestAgents(self, population) -> list:
        best_agents = []

        # Get best agents
        for subpop in population:
            # Get the best N individuals
            best_ind = tools.selBest(subpop, 1)[0]
            best_agents.append(best_ind)

        return best_agents

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

        # Store joint states per environment for the first state
        agent_positions = torch.stack([agent.state.pos for agent in env.agents], dim=0)
        joint_states_per_env = [torch.empty((0, 2)).to(self.device) for _ in teams]

        tranposed_stacked_obs = (
            torch.stack(observations, -1).transpose(0, 1).transpose(0, -1)
        )
        joint_observations_per_env = [
            torch.empty((0, self.observation_size)).to(self.device) for _ in teams
        ]

        for i, (j_states, j_obs) in enumerate(
            zip(joint_states_per_env, joint_observations_per_env)
        ):
            joint_states_per_env[i] = torch.cat(
                (j_states, agent_positions[:, i, :]), dim=0
            )
            joint_observations_per_env[i] = torch.cat(
                (j_obs, tranposed_stacked_obs[:, i, :]), dim=0
            )

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
                    match (self.policy_type):
                        case PolicyEnum.MLP:
                            policy_output = policy.forward(observation[:, i])
                        case PolicyEnum.CNN:
                            left_right_lens = observation[:2, i]
                            target_data = observation[2:4, i]
                            left_neighbors_data = observation[
                                4 : int(left_right_lens[0]) + 4, i
                            ]
                            right_neighbors_data = observation[
                                int(left_right_lens[0])
                                + 4 : int(left_right_lens[1])
                                + int(left_right_lens[0])
                                + 4,
                                i,
                            ]
                            data = stack_and_pad_1d_tensors(
                                [target_data, left_neighbors_data, right_neighbors_data]
                            )
                            data = pad_width(data, 6)
                            policy_output = policy.forward(data)

                    actions[i] = torch.cat(
                        (
                            actions[i],
                            policy_output * self.output_multiplier,
                        ),
                        dim=0,
                    )

            observations, rewards, _, _ = env.step(actions)

            # Store joint states per environment
            agent_positions = torch.stack(
                [agent.state.pos for agent in env.agents], dim=0
            )

            for i, (j_states, j_obs) in enumerate(
                zip(joint_states_per_env, joint_observations_per_env)
            ):
                joint_states_per_env[i] = torch.cat(
                    (j_states, agent_positions[:, i, :]), dim=0
                )
                joint_observations_per_env[i] = torch.cat(
                    (j_obs, stacked_obs.transpose(0, 1).transpose(0, -1)[:, i, :]),
                    dim=0,
                )

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

        # Compute team fitness
        match (self.fitness_calculation):

            case FitnessCalculationEnum.AGG:
                g_per_env = torch.sum(torch.stack(G_list), dim=0).tolist()

            case FitnessCalculationEnum.LAST:
                g_per_env = G_list[-1].tolist()

        # Generate evaluation infos
        eval_infos = [
            EvalInfo(
                team=team,
                team_fitness=g_per_env[i],
                agent_fitnesses=g_per_env[i],
                joint_traj=JointTrajectory(
                    joint_state_traj=joint_states_per_env[i].reshape(
                        self.team_size, self.n_steps + 1, 2
                    ),
                    joint_obs_traj=joint_observations_per_env[i].reshape(
                        self.team_size, self.n_steps + 1, self.observation_size
                    ),
                ),
            )
            for i, team in enumerate(teams)
        ]

        return eval_infos

    def mutateIndividual(self, individual):

        individual += np.random.normal(
            loc=self.mutation_mean,
            scale=self.std_dev_list[self.gen],
            size=np.shape(individual),
        )

    def mutate(self, population):
        # Don't mutate the elites
        for n_individual in range(self.n_mutants):

            mutant_idx = n_individual + self.n_mutants

            for subpop in population:
                self.mutateIndividual(subpop[mutant_idx])
                subpop[mutant_idx].fitness = np.float32(0.0)

    def selectSubPopulation(self, subpopulation):

        match (self.selection_method):
            case SelectionEnum.BINARY:
                chosen_ones = binarySelection(subpopulation, tournsize=2)
            case SelectionEnum.EPSILON:
                chosen_ones = epsilonGreedySelection(
                    subpopulation, self.subpop_size // 2, epsilon=0.3
                )
            case SelectionEnum.SOFTMAX:
                chosen_ones = softmaxSelection(subpopulation, self.subpop_size // 2)
            case SelectionEnum.TOURNAMENT:
                chosen_ones = tools.selTournament(
                    subpopulation, self.subpop_size // 2, 2
                )

        offspring = chosen_ones + chosen_ones

        # Return a deepcopy so that modifying an individual that was selected does not modify every single individual
        # that came from the same selected individual
        return [deepcopy(individual) for individual in offspring]

    def select(self, population):
        # Perform a selection on that subpopulation and add it to the offspring population
        return [self.selectSubPopulation(subpop) for subpop in population]

    def shuffle(self, population):
        for subpop in population:
            random.shuffle(subpop)

    def assignFitnesses(
        self,
        eval_infos: list[EvalInfo],
    ):

        match (self.fitness_shaping_method):

            case FitnessShapingEnum.G:
                for eval_info in eval_infos:
                    for individual in eval_info.team.individuals:
                        individual.fitness = eval_info.team_fitness

            case FitnessShapingEnum.D:
                for eval_info in eval_infos:
                    for individual, combo_idx in zip(
                        eval_info.team.individuals,
                        eval_info.team.combination,
                    ):
                        individual.fitness = eval_info.agent_fitnesses[combo_idx]

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def createFitnessCSV(self, fitness_dir):
        header = ["gen", "avg_team_fitness", "best_team_fitness"]

        with open(fitness_dir, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([header])

    def writeFitnessCSV(self, fitness_dir, avg_fitness, best_fitness):

        # Now save it all to the csv
        with open(fitness_dir, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.gen, avg_fitness, best_fitness])

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
        if checkpoint_exists:

            pop, checkpoint_gen = self.load_checkpoint(
                checkpoint_name, fitness_dir, trial_dir
            )

        else:
            # Initialize the population
            pop = self.toolbox.population()

            # Create csv file for saving evaluation fitnesses
            self.createFitnessCSV(fitness_dir)

        for n_gen in range(self.n_gens + 1):

            # Create environment
            env = create_env(
                self.batch_dir,
                n_envs=self.subpop_size,
                n_agents=self.team_size,
                device=self.device,
            )

            # Set gen counter global var
            self.gen = n_gen

            # Get loading bar up to checkpoint
            if checkpoint_exists and n_gen <= checkpoint_gen:
                continue

            # Perform selection
            offspring = self.select(pop)

            # Perform mutation
            self.mutate(offspring)

            # Shuffle subpopulations in offspring
            # to make teams random
            self.shuffle(offspring)

            # Form teams for evaluation
            teams = self.formTeams(offspring, joint_policies=self.subpop_size)

            # Evaluate each team
            eval_infos = self.evaluateTeams(env, teams)

            # Now assign fitnesses to each individual
            self.assignFitnesses(eval_infos)

            # Evaluate best team of generation
            avg_team_fitness = (
                sum([eval_info.team_fitness for eval_info in eval_infos])
                / self.subpop_size
            )
            best_team_eval_info = max(eval_infos, key=lambda item: item.team_fitness)

            # Now populate the population with individuals from the offspring
            self.setPopulation(pop, offspring)

            # Save fitnesses
            self.writeFitnessCSV(
                fitness_dir, avg_team_fitness, best_team_eval_info.team_fitness
            )

            # Save trajectories and checkpoint
            if (n_gen > 0) and (n_gen % self.n_gens_between_save == 0):

                # Save checkpoint
                with open(os.path.join(trial_dir, "checkpoint.pickle"), "wb") as handle:
                    pickle.dump(
                        {
                            "best_team": best_team_eval_info.team,
                            "population": pop,
                            "gen": n_gen,
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
