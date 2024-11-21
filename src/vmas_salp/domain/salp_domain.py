#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, List

import torch
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
from vmas_salp.domain.utils import COLOR_MAP
import random

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
        self.agents_positions = kwargs.pop(
            "agents_positions", [[-0.1, 0.0], [0.0, 0.0]]
        )
        self.agents_idx = [
            i for i, _ in enumerate(self.agents_positions[: self.n_agents])
        ]
        if kwargs.pop("shuffle_agents_positions", False):
            random.shuffle(self.agents_idx)

        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.1)
        self._lidar_range = kwargs.pop("lidar_range", 0.35)
        self._covering_range = kwargs.pop("covering_range", 0.25)

        self._agents_per_target = kwargs.pop("agents_per_target", 1)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.random_spawn = kwargs.pop("random_spawn", False)
        self.use_joints = kwargs.pop("use_joints", True)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.device = device

        self.global_rew = torch.zeros(batch_dim, device=device)
        self.covered_targets = torch.zeros((batch_dim, self.n_targets), device=device)

        # CONSTANTS
        self.agent_radius = 0.025
        self.target_radius = 0.11
        self.agent_dist = 0.05
        self.u_multiplier = 2.0

        self.gravity_x_val = random.normalvariate(mu=0.0, sigma=0.5)
        self.gravity_y_val = -0.5
        # Make world
        world = SalpWorld(
            batch_dim=batch_dim,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            device=device,
            substeps=15,
            collision_force=1500,
            joint_force=900,
            torque_constraint_force=2.0,
            gravity=(
                self.gravity_x_val,
                self.gravity_y_val,
            ),
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
            agent.state.neighbors = self.get_neighbors(agent, world.joints)

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

        agent.action.u[:, :2] = torch.stack((x, y), dim=-1)

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

        min_covered_targets_dists, _ = torch.min(masked_covered_targets_dists, dim=1)

        min_covered_targets_dists = torch.clamp(min_covered_targets_dists, min=1e-2)

        min_covered_targets_dists[torch.isinf(min_covered_targets_dists)] = 0

        global_reward_spread = torch.log10(
            self.covered_targets / min_covered_targets_dists
        )

        global_reward_spread *= self.targets_values

        global_reward_spread[torch.isnan(global_reward_spread)] = 0
        global_reward_spread[torch.isinf(global_reward_spread)] = 0

        return torch.sum(
            global_reward_spread,
            dim=1,
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

    def get_neighbors(self, agent: Agent, joints: ValuesView):
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

    def observation(self, agent: Agent):

        # poi_sensors = agent.sensors[0].measure()[:, 4:]

        # neighbor_states = []
        # for neighbor in agent.state.neighbors:
        #     neighbor_states.extend([neighbor.state.pos, neighbor.state.vel])

        # # Add zeros to observation to keep size consistent
        # if len(agent.state.neighbors) < 2:
        #     pos_vel = torch.ones(
        #         (self.world.batch_dim, self.world.dim_p), device=self.world.device
        #     ) * torch.tensor([0.0, 0.0], device=self.world.device)

        #     neighbor_states.extend([pos_vel, pos_vel])

        # Return neighbor states
        # return torch.cat(
        #     [
        #         agent.state.pos,
        #         agent.state.vel,
        #         *neighbor_states,
        #         torch.abs(agent.state.pos - self._targets[0].state.pos),
        #     ],
        #     dim=-1,
        # )

        all_agents_states = []
        for idx in self.agents_idx:
            all_agents_states.extend(
                [self.world.agents[idx].state.pos, self.world.agents[idx].state.vel]
            )

        dist_to_target = agent.state.pos - self._targets[0].state.pos

        return torch.cat(
            [
                torch.abs(dist_to_target),
                torch.zeros_like(dist_to_target),
                *all_agents_states,
            ],
            dim=-1,
        )

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


if __name__ == "__main__":
    render_interactively(__file__, joints=True, control_two_agents=True)
