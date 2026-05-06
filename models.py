import torch
import torch.nn as nn
from torch.distributions import Normal

from config import hidden_size


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)

        std = torch.exp(self.log_std)
        std = torch.clamp(std, 1e-3, 2.0)

        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)

        dist = Normal(mu, std)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate(self, obs, action):
        mu, std = self.forward(obs)

        dist = Normal(mu, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy

    def predict(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        return mu


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.value = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        return self.value(self.net(obs))
