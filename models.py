import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.mu = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.net(obs)

        mu = self.mu(x)
        std = torch.exp(self.log_std).expand_as(mu)

        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)

        dist = Normal(mu, std)
        action = dist.rsample()
        action = torch.tanh(action) * 2.0

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def predict(self, obs):
        x = self.net(obs)
        mu = self.mu(x)

        return torch.tanh(mu) * 2.0


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.value = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.net(obs)
        v = self.value(x)
        return v
