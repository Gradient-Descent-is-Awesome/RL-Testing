import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        self.mu = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)

        std = torch.exp(self.log_std)
        std = torch.clamp(std, 1e-3, 2.0).expand_as(mu)

        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)

        dist = Normal(mu, std)
        raw_action = dist.rsample()

        tanh_action = torch.tanh(raw_action) * 2.0

        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        log_prob -= torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6).sum(dim=-1)

        return tanh_action, log_prob

    def evaluate(self, obs, action):
        mu, std = self.forward(obs)

        dist = Normal(mu, std)

        scaled_action = action / 2.0
        scaled_action = torch.clamp(scaled_action, -0.999, 0.999)

        raw_action = torch.atanh(scaled_action)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        log_prob -= torch.log(1 - torch.tanh(raw_action).pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return dist, log_prob, entropy

    def predict(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        return torch.tanh(mu) * 2.0


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        self.value = nn.Linear(64, 1)

    def forward(self, obs):
        return self.value(self.net(obs))
