#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.joints import Joint
from vmas.simulator.core import Agent, Landmark, Box, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils

from vmas_salp.domain.custom_world import SalpWorld
from vmas_salp.domain.dynamics import SalpDynamics
from vmas_salp.domain.controller import SalpController
from vmas_salp.domain.utils import COLOR_MAP

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class SalpDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)

        self.viewer_zoom = kwargs.pop("viewer_zoom", 1)

        self.n_agents = kwargs.pop("n_agents", 5)
        self.agents_colors = kwargs.pop("agents_colors", [])
        self.n_targets = kwargs.pop("n_targets", 7)
        self.use_order = kwargs.pop("use_order", False)
        self.targets_positions = kwargs.pop("targets_positions", [])
        self.targets_colors = kwargs.pop("targets_colors", [])
        self.targets_values = torch.tensor(
            kwargs.pop("targets_values", []), device=device
        )
        self.agents_positions = kwargs.pop("agents_positions", [])

        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.2)
        self._lidar_range = kwargs.pop("lidar_range", 0.35)
        self._covering_range = kwargs.pop("covering_range", 0.25)

        self._agents_per_target = kwargs.pop("agents_per_target", 2)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.random_spawn = kwargs.pop("random_spawn", False)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.agent_radius = 0.05
        self.target_radius = self.agent_radius

        self.device = device

        self.global_rew = torch.zeros(batch_dim, device=device)
        self.covered_targets = torch.zeros((batch_dim, self.n_targets), device=device)

        # CONSTANTS
        self.agent_dist = 0.2
        self.v_range = kwargs.pop("v_range", 1.0)
        self.f_range = kwargs.pop("f_range", 1)
        self.u_range = self.f_range

        # Make world
        world = SalpWorld(
            batch_dim=batch_dim,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            device=device,
            substeps=5,
        )

        # Set targets
        self._targets = []
        for i in range(self.n_targets):

            target = Landmark(
                name=f"target_{i}",
                collide=False,
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
                shape=Sphere(radius=0.05),
                dynamics=SalpDynamics(),
                collide=True,
            )

            agent.state.join = torch.zeros(batch_dim)
            world.add_agent(agent)

        # Add joints
        self.joint_list = []
        for i in range(self.n_agents - 1):
            joint = Joint(
                world.agents[i],
                world.agents[i + 1],
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=self.agent_dist,
                rotate_a=True,
                rotate_b=True,
                collidable=True,
                width=0,
            )
            world.add_joint(joint)
            self.joint_list.append(joint)

        self.dist_rew = torch.zeros(batch_dim, device=device)
        self.rot_rew = self.dist_rew.clone()
        self.vel_reward = self.dist_rew.clone()
        self.pos_rew = self.dist_rew.clone()
        self.t = self.dist_rew.clone()

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

        for idx, agent in enumerate(self.world.agents):
            pos = torch.ones(
                (self.world.batch_dim, self.world.dim_p), device=self.world.device
            ) * torch.tensor(self.agents_positions[idx], device=self.world.device)
            agent.set_pos(
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

        # x = (
        #     torch.cos(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
        #     * -agent.action.u[:, 0]
        # )
        # y = (
        #     torch.sin(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
        #     * -agent.action.u[:, 0]
        # )

        # agent.action.u[:, :2] = torch.stack((x, y), dim=-1)

        if agent.state.join.any():
            self.world.detach_joint(self.joint_list[0])

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

    def observation(self, agent: Agent):

        # if self.step_counter == 600:
        #     joint = Joint(
        #         self.world.agents[0],
        #         self.world.agents[1],
        #         anchor_a=(0, 0),
        #         anchor_b=(0, 0),
        #         dist=self.agent_dist,
        #         rotate_a=False,
        #         rotate_b=False,
        #         collidable=True,
        #         width=0.01,
        #         mass=1,
        #     )
        #     self.world.add_joint(joint)

        # if self.step_counter == 400:
        #     self.world.attach_joint(self.joint_list[0])

        observations = []

        observations.append(agent.state.pos)

        observations.append(agent.state.vel)

        # for a in self.world.agents:
        #     if a != agent:
        #         observations.append(a.state.pos - agent.state.pos)

        # for a in self.world.agents:
        #     if a != agent:
        #         observations.append(a.state.vel - agent.state.vel)

        return torch.cat(
            observations,
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
    render_interactively(__file__, joints=True)
