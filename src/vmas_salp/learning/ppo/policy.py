import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim

import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):
    def __init__(self, observation_size, action_size, lr):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.critic.to(dev)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_size), std=0.01),
        )
        self.actor_mean.to(dev)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size)).to(dev)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)

        # Storage setup
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(-1),
            probs.entropy().sum(-1),
            self.critic(x),
        )
