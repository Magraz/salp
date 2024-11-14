#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.joints import Joint
from vmas.simulator.core import Agent, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils, X, Y
from domain.custom_world import SalpWorld
from domain.dynamics import SalpDynamics
from domain.controller import SalpController

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class SalpDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True
        self.viewer_zoom = 1.1

        self.step_counter = 0

        self.n_agents = kwargs.pop("n_agents", 3)
        self.with_joints = kwargs.pop("joints", True)

        # Reward
        self.observe_rel_pos = kwargs.pop("observe_rel_pos", False)
        self.observe_rel_vel = kwargs.pop("observe_rel_vel", False)
        self.observe_pos = kwargs.pop("observe_pos", True)

        # Controller
        self.use_controller = kwargs.pop("use_controller", False)

        self.v_range = kwargs.pop("v_range", 1.0)
        self.desired_vel = kwargs.pop("desired_vel", self.v_range)
        self.f_range = kwargs.pop("f_range", 1)

        controller_params = [1.5, 0.6, 0.002]

        self.u_range = self.v_range if self.use_controller else self.f_range

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.desired_distance = 1
        self.grid_spacing = self.desired_distance
        self.agent_dist = 0.2

        # Make world
        world = SalpWorld(
            batch_dim=batch_dim,
            device=device,
            drag=0.25,
            linear_friction=0.1,
            substeps=5,
        )

        self.desired_vel = torch.tensor(
            [0.0, self.desired_vel], device=device, dtype=torch.float32
        )
        self.desired_pos = 10.0

        # Add agents
        for i in range(self.n_agents):

            agent = Agent(
                name=f"agent_{i}",
                render_action=True,
                shape=Box(length=0.05, width=0.1),
                u_range=self.u_range,
                v_range=self.v_range,
                f_range=self.f_range,
                dynamics=SalpDynamics(),
                collide=False,
            )

            agent.state.join = torch.zeros(batch_dim)
            agent.controller = SalpController(
                agent, world, controller_params, "standard"
            )
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
                rotate_a=False,
                rotate_b=False,
                collidable=False,
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

        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        positions = [
            torch.tensor((0 + i * self.agent_dist, 0)) for i in range(self.n_agents)
        ]
        for i, agent in enumerate(agents):
            agent.controller.reset(env_index)

            for pos in positions:
                agent.set_pos(
                    pos,
                    batch_index=env_index,
                )

    def process_action(self, agent: Agent):

        x = (
            torch.cos(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
            * -agent.action.u[:, 0]
        )
        y = (
            torch.sin(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
            * -agent.action.u[:, 0]
        )

        agent.action.u[:, :2] = torch.stack((x, y), dim=-1)

        if agent.state.join.any():
            self.world.detach_joint(self.joint_list[0])

    def reward(self, agent: Agent):

        return torch.tensor([0])

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

        if self.observe_rel_pos:
            for a in self.world.agents:
                if a != agent:
                    observations.append(a.state.pos - agent.state.pos)

        if self.observe_rel_vel:
            for a in self.world.agents:
                if a != agent:
                    observations.append(a.state.vel - agent.state.vel)

        return torch.cat(
            observations,
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "dist_rew": self.dist_rew,
            "pos_rew": self.pos_rew,
        }


if __name__ == "__main__":
    render_interactively(__file__, joints=True)
