from collections import deque

import gymnasium as gym
import numpy as np
import torch

from config import actor_lr, batch_size, buffer_size, critic_lr, n_epoch, total_steps
from models import Actor, Critic
from utils import RolloutBuffer, Trainer

env = gym.make("Pendulum-v1")

actor = Actor(3, 1)
critic = Critic(3)

actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

trainer = Trainer(
    actor, critic, actor_optim, critic_optim, batch_size=batch_size, n_epoch=n_epoch
)

buffer = RolloutBuffer()

current_steps = 0
current_episodes = 0
current_updates = 0

episode_rewards = []
last_100_rewards = deque(maxlen=100)

obs, _ = env.reset()

while current_steps < total_steps:
    buffer.clear()
    episode_reward = 0

    # 🔥 COLLECT FIXED ROLLOUT (NO EPISODE DEPENDENCY)
    while buffer.size() < buffer_size and current_steps < total_steps:
        state = torch.tensor(obs, dtype=torch.float32)

        action, log_prob = actor.sample(state)
        value = critic(state)

        next_obs, reward, term, trunc, _ = env.step(action.detach().numpy())
        done = term or trunc

        buffer.add(
            obs,
            action.detach().numpy(),
            reward,
            log_prob.detach(),
            value.detach(),
            done,
        )

        obs = next_obs
        episode_reward += reward
        current_steps += 1

        if done:
            obs, _ = env.reset()
            episode_rewards.append(episode_reward)
            last_100_rewards.append(episode_reward)
            episode_reward = 0
            current_episodes += 1

    actor_loss, critic_loss = trainer.update(buffer, obs)
    current_updates += 1

    print("\n==============================")
    print(f"UPDATE #{current_updates}")
    print(f"Steps: {current_steps}")
    print(f"Episodes: {current_episodes}")
    print(f"Last 100 reward mean: {np.mean(last_100_rewards):.2f}")
    print("Actor loss:", actor_loss)
    print("Critic loss:", critic_loss)
    print("Actor STD:", torch.exp(actor.log_std).item())
    print("==============================\n")

print("\n========== TRAINING DONE ==========")
print(f"Total updates: {current_updates}")
print(f"Total steps: {current_steps}")
print(f"Total episodes: {current_episodes}")
print(f"Final last 100 reward mean: {np.mean(last_100_rewards):.2f}")

env.close()
