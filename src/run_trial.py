#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import random
import time

import torch

from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.utils import save_video
from vmas.simulator.scenario import BaseScenario
from domain.salp_domain import SalpDomain
from testing.manual_control import manual_control
from pynput.keyboard import Listener


def use_vmas_env(
    name: str = "dummy",
    render: bool = False,
    save_render: bool = False,
    num_envs: int = 3,
    n_steps: int = 1000,
    random_action: bool = False,
    device: str = "cpu",
    scenario: BaseScenario = None,
    visualize_render: bool = True,
    **kwargs,
):
    """Example function to use a vmas environment

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (BaseScenario): Scenario Class
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action
        visualize_render (bool, optional): Whether to visualize the render. Defaults to ``True``.
        dict_spaces (bool, optional): Weather to return obs, rewards, and infos as dictionaries with agent names.
            By default, they are lists of len # of agents
        kwargs (dict, optional): Keyword arguments to pass to the scenario

    Returns:

    """
    assert not (save_render and not render), "To save the video you have to render it"

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        wrapper=None,
        seed=None,
        # Environment specific variables
        **kwargs,
    )
    mc = manual_control(kwargs.pop("n_agents", 0))

    frame_list = []  # For creating a gif
    init_time = time.time()

    with Listener(on_press=mc.on_press, on_release=mc.on_release) as listener:
        listener.join(timeout=1)
        for _ in range(n_steps):

            # VMAS actions can be either a list of tensors (one per agent)
            # or a dict of tensors (one entry per agent with its name as key)
            # Both action inputs can be used independently of what type of space its chosen

            actions = []
            for i, agent in enumerate(env.agents):

                if not random_action:
                    if i == mc.controlled_agent:
                        cmd_action = mc.cmd_vel[:] + mc.join[:]
                        action = torch.tensor(cmd_action).repeat(num_envs, 1)
                    else:
                        action = torch.tensor([0.0, 0.0, 0.0]).repeat(num_envs, 1)
                else:
                    action = env.get_random_action(agent)

                actions.append(action)

            obs, rews, dones, info = env.step(actions)

            if render:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=visualize_render,
                )
                if save_render:
                    frame_list.append(frame)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {name} scenario."
    )

    if render and save_render:
        save_video(name, frame_list, fps=1 / env.scenario.world.dt)


if __name__ == "__main__":
    scenario = SalpDomain()
    n_agents = 2

    use_vmas_env(
        name=f"SalpDomain_{n_agents}a",
        scenario=scenario,
        render=True,
        save_render=False,
        random_action=False,
        device="cpu",
        # Environment specific
        n_agents=n_agents,
        n_steps=10000,
        wind=0,
    )
