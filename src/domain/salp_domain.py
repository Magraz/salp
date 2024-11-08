#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.joints import Joint
from vmas.simulator.core import Agent, Sphere
from domain.custom_world import SalpWorld
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


def angle_to_vector(angle):
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)


def get_line_angle_0_90(rot: Tensor):
    angle = torch.abs(rot) % torch.pi
    other_angle = torch.pi - angle
    return torch.minimum(angle, other_angle)


def get_line_angle_0_180(rot):
    angle = rot % torch.pi
    return angle


def get_line_angle_dist_0_360(angle, goal):
    angle = angle_to_vector(angle)
    goal = angle_to_vector(goal)
    return -torch.einsum("bs,bs->b", angle, goal)


def get_line_angle_dist_0_180(angle, goal):
    angle = get_line_angle_0_180(angle)
    goal = get_line_angle_0_180(goal)
    return torch.minimum(
        (angle - goal).abs(),
        torch.minimum(
            (angle - (goal - torch.pi)).abs(),
            ((angle - torch.pi) - goal).abs(),
        ),
    ).squeeze(-1)


class SalpDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True
        self.viewer_zoom = 1.1

        self.step_counter = 0

        self.n_agents = kwargs.pop("n_agents", 3)
        self.with_joints = kwargs.pop("joints", True)

        # Reward
        self.vel_shaping_factor = kwargs.pop("vel_shaping_factor", 1)
        self.dist_shaping_factor = kwargs.pop("dist_shaping_factor", 1)
        self.wind_shaping_factor = kwargs.pop("wind_shaping_factor", 1)

        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 0)
        self.rot_shaping_factor = kwargs.pop("rot_shaping_factor", 0)
        self.energy_shaping_factor = kwargs.pop("energy_shaping_factor", 0)

        self.observe_rel_pos = kwargs.pop("observe_rel_pos", False)
        self.observe_rel_vel = kwargs.pop("observe_rel_vel", False)
        self.observe_pos = kwargs.pop("observe_pos", True)

        # Controller
        self.use_controller = kwargs.pop("use_controller", True)
        self.wind = torch.tensor(
            [0, -kwargs.pop("wind", 2)], device=device, dtype=torch.float32
        ).expand(batch_dim, 2)
        self.v_range = kwargs.pop("v_range", 0.5)
        self.desired_vel = kwargs.pop("desired_vel", self.v_range)
        self.f_range = kwargs.pop("f_range", 100)

        controller_params = [1.5, 0.6, 0.002]
        self.u_range = self.v_range if self.use_controller else self.f_range

        # Other
        self.cover_angle_tolerance = kwargs.pop("cover_angle_tolerance", 1)
        self.horizon = kwargs.pop("horizon", 200)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.desired_distance = 1
        self.grid_spacing = self.desired_distance
        self.agent_dist = 0.2

        # Make world
        world = SalpWorld(
            batch_dim=batch_dim, device=device, drag=0, linear_friction=0.1, substeps=5
        )

        self.desired_vel = torch.tensor(
            [0.0, self.desired_vel], device=device, dtype=torch.float32
        )
        self.max_pos = (self.horizon * world.dt) * self.desired_vel[Y]
        self.desired_pos = 10.0

        # Add agents
        for i in range(self.n_agents):

            agent = Agent(
                name=f"agent_{i}",
                render_action=True,
                shape=Sphere(radius=0.05),
                u_range=self.u_range,
                v_range=self.v_range,
                f_range=self.f_range,
                gravity=self.wind,
            )
            agent.controller = VelocityController(
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
                collidable=True,
                width=0.01,
                mass=1,
            )
            world.add_joint(joint)
            self.joint_list.append(joint)

        # world.attach_joint(joint_list[0])

        for agent in world.agents:
            agent.wind_rew = torch.zeros(batch_dim, device=device)
            agent.vel_rew = agent.wind_rew.clone()
            agent.energy_rew = agent.wind_rew.clone()

        self.dist_rew = torch.zeros(batch_dim, device=device)
        self.rot_rew = self.dist_rew.clone()
        self.vel_reward = self.dist_rew.clone()
        self.pos_rew = self.dist_rew.clone()
        self.t = self.dist_rew.clone()

        return world

    def set_wind(self, wind):
        self.wind = torch.tensor(
            [0, -wind], device=self.world.device, dtype=torch.float32
        ).expand(self.world.batch_dim, self.world.dim_p)

        for agent in self.world.agents:
            agent.gravity = self.wind

    def calc_agent_errors(
        self, pivot_agent: Agent, neighbor_agent: Agent
    ) -> tuple[float, float, float]:

        dist_error = (
            torch.linalg.vector_norm(
                pivot_agent.state.pos - neighbor_agent.state.pos,
                dim=-1,
            )
            - self.desired_distance
        ).abs()

        pos_error = (
            torch.maximum(
                pivot_agent.state.pos[:, Y],
                neighbor_agent.state.pos[:, Y],
            )
            - self.desired_pos
        ).abs()

        rot_error = get_line_angle_dist_0_180(
            self.get_agents_angle(pivot_agent, neighbor_agent), 0
        )

        return dist_error, pos_error, rot_error

    def calc_shaping_values(self) -> tuple[float, float, float]:
        rot_shaping = 0
        dist_shaping = 0
        pos_shaping = 0
        dist_error = 0
        pos_error = 0
        rot_error = 0

        for i, agent in enumerate(self.world.agents):
            # First agent only gets the distance error of one neighbor
            if i == 0:
                dist_error, pos_error, rot_error = self.calc_agent_errors(
                    agent, self.world.agents[i + 1]
                )
            # Last agent only gets the distance error of one neighbor
            elif i == self.n_agents - 1:
                dist_error, pos_error, rot_error = self.calc_agent_errors(
                    agent, self.world.agents[i - 1]
                )
            # Middle agents get error from 2 neighbor agents
            else:

                for offset in [-1, 1]:
                    temp_dist_error, temp_pos_error, temp_rot_error = (
                        self.calc_agent_errors(agent, self.world.agents[i + offset])
                    )

                    dist_error += temp_dist_error
                    pos_error += temp_pos_error
                    rot_error += temp_rot_error

            dist_shaping += (dist_error * self.dist_shaping_factor) / self.n_agents
            pos_shaping += (pos_error * self.pos_shaping_factor) / self.n_agents
            rot_shaping += (rot_error * self.rot_shaping_factor) / self.n_agents

        return dist_shaping, pos_shaping, rot_shaping

    def reset_world_at(self, env_index: int = None):
        start_angle = torch.zeros(
            (1, 1) if env_index is not None else (self.world.batch_dim, 1),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -torch.pi,
            torch.pi,
        )

        start_delta_x = (self.desired_distance / 2) * torch.cos(start_angle)
        start_delta_y = (self.desired_distance / 2) * torch.sin(start_angle)

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

            # if i == 0:
            #     agent.set_pos(
            #         -torch.cat([start_delta_x, start_delta_y], dim=1),
            #         batch_index=env_index,
            #     )
            # else:
            #     agent.set_pos(
            #         torch.cat([start_delta_x, start_delta_y], dim=1),
            #         batch_index=env_index,
            #     )

            if env_index is None:
                agent.vel_shaping = (
                    torch.linalg.vector_norm(agent.state.vel - self.desired_vel, dim=-1)
                    * self.vel_shaping_factor
                )
                agent.energy_shaping = torch.zeros(
                    self.world.batch_dim,
                    device=self.world.device,
                    dtype=torch.float32,
                )
                agent.wind_shaping = (
                    torch.linalg.vector_norm(agent.gravity, dim=-1)
                    * self.wind_shaping_factor
                )

            else:
                agent.vel_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.vel[env_index] - self.desired_vel
                    )
                    * self.vel_shaping_factor
                )
                agent.energy_shaping[env_index] = 0
                agent.wind_shaping[env_index] = (
                    torch.linalg.vector_norm(agent.gravity[env_index])
                    * self.wind_shaping_factor
                )

        if env_index is None:
            self.t = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.int
            )

            self.distance_shaping, self.pos_shaping, self.rot_shaping = (
                self.calc_shaping_values()
            )

        else:
            self.t[env_index] = 0
            self.distance_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.small_agent.state.pos[env_index]
                    - self.big_agent.state.pos[env_index]
                )
                - self.desired_distance
            ).abs() * self.dist_shaping_factor

            self.pos_shaping[env_index] = (
                (
                    torch.maximum(
                        self.big_agent.state.pos[env_index, Y],
                        self.small_agent.state.pos[env_index, Y],
                    )
                    - self.desired_pos
                ).abs()
            ) * self.pos_shaping_factor

            self.rot_shaping[env_index] = (
                get_line_angle_dist_0_180(self.get_agents_angle()[env_index], 0)
                * self.rot_shaping_factor
            )

    def process_action(self, agent: Agent):
        if self.use_controller:
            agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.t += 1
            self.set_friction()

            distance_shaping, pos_shaping, rot_shaping = self.calc_shaping_values()

            # Dist reward
            self.dist_rew = self.distance_shaping - distance_shaping
            self.distance_shaping = distance_shaping

            # Rot shaping
            self.rot_rew = self.rot_shaping - rot_shaping
            self.rot_shaping = rot_shaping

            # Pos reward
            self.pos_rew = self.pos_shaping - pos_shaping
            self.pos_shaping = pos_shaping

            # Vel reward
            for a in self.world.agents:
                vel_shaping = (
                    torch.linalg.vector_norm(a.state.vel - self.desired_vel, dim=-1)
                    * self.vel_shaping_factor
                )
                a.vel_rew = a.vel_shaping - vel_shaping
                a.vel_shaping = vel_shaping
            self.vel_reward = torch.stack(
                [a.vel_rew for a in self.world.agents],
                dim=1,
            ).mean(-1)

            # Energy reward
            for a in self.world.agents:
                energy_shaping = (
                    torch.linalg.vector_norm(a.action.u, dim=-1)
                    * self.energy_shaping_factor
                )
                a.energy_rew = a.energy_shaping - energy_shaping
                a.energy_rew[self.t < 10] = 0
                a.energy_shaping = energy_shaping

            self.energy_rew = torch.stack(
                [a.energy_rew for a in self.world.agents],
                dim=1,
            ).mean(-1)

            # Wind reward
            for a in self.world.agents:
                wind_shaping = (
                    torch.linalg.vector_norm(a.gravity, dim=-1)
                    * self.wind_shaping_factor
                )
                a.wind_rew = a.wind_shaping - wind_shaping
                a.wind_rew[self.t < 5] = 0
                a.wind_shaping = wind_shaping

            self.wind_rew = torch.stack(
                [a.wind_rew for a in self.world.agents],
                dim=1,
            ).mean(-1)

        return (
            self.dist_rew
            + self.vel_reward
            + self.rot_rew
            + self.energy_rew
            + self.wind_rew
            + self.pos_rew
        )

    def set_friction(self):
        dist_to_goal_angle = (
            get_line_angle_dist_0_360(
                self.get_agents_angle(self.world.agents[0], self.world.agents[-1]),
                torch.tensor([-torch.pi / 2], device=self.world.device).expand(
                    self.world.batch_dim, 1
                ),
            )
            + 1
        ).clamp(max=self.cover_angle_tolerance).unsqueeze(-1) + (
            1 - self.cover_angle_tolerance
        )  # Between 1 and 1 - tolerance
        dist_to_goal_angle = (dist_to_goal_angle - 1 + self.cover_angle_tolerance) / (
            self.cover_angle_tolerance
        )  # Between 1 and 0
        for agent in self.world.agents:
            agent.gravity = self.wind * dist_to_goal_angle

    def observation(self, agent: Agent):
        self.step_counter += 1

        if self.step_counter == 300:
            self.world.detach_joint(self.joint_list[1])

        if self.step_counter == 600:
            joint = Joint(
                self.world.agents[1],
                self.world.agents[2],
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=self.agent_dist,
                rotate_a=False,
                rotate_b=False,
                collidable=True,
                width=0.01,
                mass=1,
            )
            self.world.add_joint(joint)

        # if self.step_counter == 400:
        #     self.world.attach_joint(self.joint_list[0])

        observations = []
        if self.observe_pos:
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

    def get_agents_angle(self, pivot_agent, neighbor_agent):
        return torch.atan2(
            pivot_agent.state.pos[:, Y] - neighbor_agent.state.pos[:, Y],
            pivot_agent.state.pos[:, X] - neighbor_agent.state.pos[:, X],
        ).unsqueeze(-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "dist_rew": self.dist_rew,
            "rot_rew": self.rot_rew,
            "pos_rew": self.pos_rew,
            "agent_wind_rew": agent.wind_rew,
            "agent_vel_rew": agent.vel_rew,
            "agent_energy_rew": agent.energy_rew,
            "delta_vel_to_goal": torch.linalg.vector_norm(
                agent.state.vel - self.desired_vel, dim=-1
            ),
        }


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True, joints=True)
