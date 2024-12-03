#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import ValuesView
from vmas import render_interactively
from vmas.simulator.joints import Joint
from vmas.simulator.core import Agent, Landmark, Box, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils

from vmas_salp.domain.world import SalpWorld
from vmas_salp.domain.dynamics import SalpDynamics
from vmas_salp.domain.controller import SalpController
from vmas_salp.domain.sensors import SectorDensity
from vmas_salp.domain.utils import COLOR_MAP, sample_filtered_normal
import random
import math

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class SalpDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)

        self.viewer_zoom = kwargs.pop("viewer_zoom", 1.5)

        self.n_agents = kwargs.pop("n_agents", 2)
        self.agents_colors = kwargs.pop("agents_colors", ["BLUE"])
        self.n_targets = kwargs.pop("n_targets", 1)

        self.targets_positions = kwargs.pop("targets_positions", [[0.0, 3.0]])

        self.targets_colors = kwargs.pop("targets_colors", ["RED"])
        self.targets_values = torch.tensor(
            kwargs.pop("targets_values", [1.0]), device=device
        )
        self.agents_positions = kwargs.pop("agents_positions", [])
        self.agents_idx = [
            i for i, _ in enumerate(self.agents_positions[: self.n_agents])
        ]
        if kwargs.pop("shuffle_agents_positions", False):
            random.shuffle(self.agents_idx)

        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.1)
        self._lidar_range = kwargs.pop("lidar_range", 10.0)
        self._covering_range = kwargs.pop("covering_range", 0.25)

        self._agents_per_target = kwargs.pop("agents_per_target", 1)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.random_spawn = kwargs.pop("random_spawn", False)
        self.use_joints = kwargs.pop("use_joints", True)

        self.state_representation = "local_neighbors"

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.device = device

        self.global_rew = torch.zeros(batch_dim, device=device)
        self.covered_targets = torch.zeros((batch_dim, self.n_targets), device=device)

        # CONSTANTS
        self.agent_radius = 0.025
        self.target_radius = 0.11
        self.agent_dist = 0.05
        self.u_multiplier = 2.0

        # self.gravity_x_val = sample_filtered_normal(
        #     mean=0.0, std_dev=0.3, threshold=0.2
        # )

        gravity_x_vals = [0.4, -0.4]

        self.gravity_x_val = random.choice(gravity_x_vals)
        self.gravity_y_val = -0.3
        # Make world
        world = SalpWorld(
            batch_dim=batch_dim,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            device=device,
            substeps=15,
            collision_force=1500,
            joint_force=900,
            torque_constraint_force=1.5,
            # gravity=(
            #     self.gravity_x_val,
            #     self.gravity_y_val,
            # ),
        )

        # Set targets
        self._targets = []
        for i in range(self.n_targets):

            target = Landmark(
                name=f"target_{i}",
                collide=True,
                shape=Sphere(radius=self.target_radius),
                color=COLOR_MAP[self.targets_colors[i]],
            )

            target.value = self.targets_values[i]

            world.add_landmark(target)
            self._targets.append(target)

        # Add agents
        for i in range(self.n_agents):

            agent = Agent(
                name=f"agent_{i}",
                render_action=True,
                shape=Box(length=0.025, width=0.04),
                dynamics=SalpDynamics(),
                sensors=([SectorDensity(world, max_range=self._lidar_range)]),
                color=COLOR_MAP[self.agents_colors[i]],
                u_multiplier=self.u_multiplier,
            )
            agent.state.join = torch.zeros(batch_dim)
            agent.state.idx = self.agents_idx[i]
            world.add_agent(agent)

        # Add joints
        self.joint_list = []
        if self.use_joints:
            for i in range(self.n_agents - 1):
                joint = Joint(
                    world.agents[self.agents_idx[i]],
                    world.agents[self.agents_idx[i + 1]],
                    anchor_a=(0, 0),
                    anchor_b=(0, 0),
                    dist=self.agent_dist,
                    rotate_a=False,
                    rotate_b=False,
                    collidable=False,
                    width=0,
                )
                world.add_joint(joint)
                self.joint_list.append(joint)

        # Assign neighbors to agents
        for agent in world.agents:
            agent.state.local_neighbors = self.get_local_neighbors(agent, world.joints)
            agent.state.left_neighbors, agent.state.right_neighbors = self.get_all_neighbors(agent, world.agents)

        self.dist_rew = torch.zeros(batch_dim, device=device)
        self.rot_rew = self.dist_rew.clone()
        self.vel_reward = self.dist_rew.clone()
        self.pos_rew = self.dist_rew.clone()
        self.t = self.dist_rew.clone()

        world.zero_grad()

        return world

    def reset_world_at(self, env_index: int = None):

        if env_index is None:
            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets),
                False,
                device=self.world.device,
            )
        else:
            self.all_time_covered_targets[env_index] = False

        for idx in range(self.n_agents):
            pos = torch.ones(
                (self.world.batch_dim, self.world.dim_p), device=self.world.device
            ) * torch.tensor(
                self.agents_positions[idx],
                device=self.world.device,
            )
            self.world.agents[self.agents_idx[idx]].set_pos(
                pos,
                batch_index=env_index,
            )

        for idx, target in enumerate(self._targets):
            pos = torch.ones(
                (self.world.batch_dim, self.world.dim_p), device=self.world.device
            ) * torch.tensor(self.targets_positions[idx], device=self.world.device)
            target.set_pos(
                pos,
                batch_index=env_index,
            )

    def process_action(self, agent: Agent):

        # Single DOF movement
        x = (
            torch.cos(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
            * -agent.action.u[:, 0]
        )
        y = (
            torch.sin(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
            * -agent.action.u[:, 0]
        )

        agent.state.force = torch.stack((x, y), dim=-1)

        # Join action
        # if agent.state.join.any():
        #     self.world.detach_joint(self.joint_list[0])

    def calculate_global_reward(
        self, targets_pos: torch.Tensor, agent: Agent
    ) -> torch.Tensor:

        agents_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)

        agents_targets_dists = torch.cdist(agents_pos, targets_pos)
        agents_per_target = torch.sum(
            (agents_targets_dists < self._covering_range).type(torch.int),
            dim=1,
        )

        self.covered_targets = agents_per_target >= self._agents_per_target

        # After order has been taken into account continue
        agents_covering_targets_mask = agents_targets_dists < self._covering_range

        covered_targets_dists = agents_covering_targets_mask * agents_targets_dists

        masked_covered_targets_dists = torch.where(
            covered_targets_dists == 0, float("inf"), covered_targets_dists
        )

        clamped_covered_targets_dists = torch.clamp(
            masked_covered_targets_dists, min=1e-2
        )

        clamped_covered_targets_dists[torch.isinf(clamped_covered_targets_dists)] = 0

        global_reward_spread = torch.log10(
            self.covered_targets.squeeze(-1) / clamped_covered_targets_dists
        )

        global_reward_spread *= self.targets_values / (
            self.n_agents * self.world.batch_dim
        )

        global_reward_spread[torch.isnan(global_reward_spread)] = 0
        global_reward_spread[torch.isinf(global_reward_spread)] = 0

        return torch.sum(
            torch.sum(
                global_reward_spread,
                dim=-1,
            ),
            dim=-1,
        )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:

            targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)

            # Calculate G
            self.global_rew = self.calculate_global_reward(targets_pos, agent)

        if is_last:
            self.all_time_covered_targets += self.covered_targets
            self.current_order_per_env = torch.sum(
                self.all_time_covered_targets, dim=1
            ).unsqueeze(-1)

        covering_rew = torch.cat([self.global_rew])

        return covering_rew

    def get_local_neighbors(self, agent: Agent, joints: ValuesView):
        neighbors = []
        links = []

        # Get links
        for joint in joints:

            if agent == joint.entity_a:
                links.append(joint.entity_b)
            elif agent == joint.entity_b:
                links.append(joint.entity_a)

        # Get agents
        for joint in joints:

            if (joint.entity_a in links) and (joint.entity_b != agent):
                neighbors.append(joint.entity_b)
            elif (joint.entity_b in links) and (joint.entity_a != agent):
                neighbors.append(joint.entity_a)

        return neighbors

    def get_all_neighbors(self, agent, all_agents):
        l_neighbors = []
        r_neighbors = []

        for a in all_agents:
            if a != agent:
                if agent.state.idx < a.state.idx:
                    r_neighbors.append(a)
                else:
                    l_neighbors.append(a)

        return l_neighbors, r_neighbors

    def get_heading(self, agent: Agent):
        x = torch.cos(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
        y = torch.sin(agent.state.rot + 1.5 * torch.pi).squeeze(-1)

        return torch.stack((x, y), dim=-1)

    def neighbors_representation(self, agent: Agent):

        neighbor_states = []

        for neighbor in agent.state.local_neighbors:
            norm_force = torch.linalg.norm(
                F.normalize(neighbor.state.force), dim=-1
            ).unsqueeze(-1)
            neighbor_states.extend([norm_force, neighbor.state.rot])

        # Calculate heading and distance to target
        heading = self.get_heading(agent)

        dist_to_target = agent.state.pos - self._targets[0].state.pos
        norm_dist = torch.linalg.norm(dist_to_target, dim=-1).unsqueeze(-1)

        normalized_dist_to_target = F.normalize(dist_to_target)

        heading_difference = torch.sum(
            heading * normalized_dist_to_target, dim=1
        ).unsqueeze(-1)

        # Add zeros to observation to keep size consistent
        if len(agent.state.local_neighbors) < 2:
            pos_vel = torch.ones(
                (self.world.batch_dim, 1), device=self.world.device
            ) * torch.tensor([0.0], device=self.world.device)

            neighbor_states.extend([pos_vel, pos_vel])

        return torch.cat(
            [
                norm_dist,
                heading_difference,
                torch.linalg.norm(F.normalize(agent.state.force), dim=-1).unsqueeze(-1),
                agent.state.rot,
                *neighbor_states,
            ],
            dim=-1,
        )

    def aggregate_local_neighbors(self, agent: Agent):
        # Calculate heading and distance to target

        heading = self.get_heading(agent)

        dist_to_target = agent.state.pos - self._targets[0].state.pos
        norm_dist = torch.linalg.norm(dist_to_target, dim=-1).unsqueeze(-1)

        normalized_dist_to_target = F.normalize(dist_to_target)

        heading_difference = torch.sum(
            heading * normalized_dist_to_target, dim=1
        ).unsqueeze(-1)

        # Get all neighbors statesw
        left_neighbors_total_force = torch.zeros_like(agent.state.force)
        right_neighbors_total_force = torch.zeros_like(agent.state.force)

        for l_neighbor in agent.state.left_neighbors:
            left_neighbors_total_force += l_neighbor.state.force
        
        for r_neighbor in agent.state.right_neighbors:
            right_neighbors_total_force += r_neighbor.state.force

        left_norm_force = torch.linalg.norm(
                left_neighbors_total_force, dim=-1
            ).unsqueeze(-1)

        right_norm_force = torch.linalg.norm(
                right_neighbors_total_force, dim=-1
            ).unsqueeze(-1)

        return torch.cat(
            [
                norm_dist,
                heading_difference,
                agent.state.rot,
                left_norm_force,
                right_norm_force,
            ],
            dim=-1,
        )
    
    def all_agents_representation(self, agent: Agent):

        # Calculate heading and distance to target

        heading = self.get_heading(agent)

        dist_to_target = agent.state.pos - self._targets[0].state.pos
        norm_dist = torch.linalg.norm(dist_to_target, dim=-1).unsqueeze(-1)

        normalized_dist_to_target = F.normalize(dist_to_target)

        heading_difference = torch.sum(
            heading * normalized_dist_to_target, dim=1
        ).unsqueeze(-1)

        # Get all neighbors statesw
        all_neighbors_states = []

        for neighbor in agent.state.all_neighbors:
            norm_force = torch.linalg.norm(
                F.normalize(neighbor.state.force), dim=-1
            ).unsqueeze(-1)
            all_neighbors_states.extend(
                [
                    norm_force,
                    neighbor.state.rot,
                ]
            )

        return torch.cat(
            [
                norm_dist,
                heading_difference,
                torch.linalg.norm(F.normalize(agent.state.force), dim=-1).unsqueeze(-1),
                agent.state.rot,
                *all_neighbors_states,
            ],
            dim=-1,
        )

    def single_agent_representation(self, agent: Agent):

        dist_to_target = torch.abs(agent.state.pos - self._targets[0].state.pos)

        return torch.cat(
            [
                dist_to_target,
                agent.state.pos,
                agent.state.vel,
                agent.state.rot,
            ],
            dim=-1,
        )

    def observation(self, agent: Agent):

        match (self.state_representation):
            case "local_neighbors":
                return self.aggregate_local_neighbors(agent)
            case "all_neighbors":
                return self.all_agents_representation(agent)
            case "single_state":
                return self.single_agent_representation(agent)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "global_reward": (self.global_rew),
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        for i, target in enumerate(self._targets):
            range_circle = rendering.make_circle(self._covering_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*COLOR_MAP[self.targets_colors[i]].value)
            geoms.append(range_circle)

        return geoms

    def random_point_around_center(center_x, center_y, radius):
        """
        Generates a random (x, y) coordinate around a given circle center within the specified radius.
        
        Parameters:
            center_x (float): The x-coordinate of the circle center.
            center_y (float): The y-coordinate of the circle center.
            radius (float): The radius around the center where the point will be generated.
        
        Returns:
            tuple: A tuple (x, y) representing the random point.
        """
        # Generate a random angle in radians
        angle = random.uniform(0, 2 * math.pi)
        # Generate a random distance from the center, within the circle
        distance = random.uniform(0, radius)
        
        # Calculate the x and y coordinates
        random_x = center_x + distance * math.cos(angle)
        random_y = center_y + distance * math.sin(angle)
        
        return [random_x, random_y]


if __name__ == "__main__":
    render_interactively(__file__, joints=True, control_two_agents=True)
