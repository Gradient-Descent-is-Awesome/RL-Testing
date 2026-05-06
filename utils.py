import numpy as np
import torch
import torch.nn.functional as F


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.last_state = None

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def size(self):
        return len(self.states)

    def get_all(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.log_probs),
            np.array(self.values),
            np.array(self.dones),
        )


class GAE:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam

    def compute(self, rewards, values, dones, next_value):
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        values = np.append(values, next_value)

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return advantages, returns


class Trainer:
    def __init__(self, actor, critic, actor_optim, critic_optim, batch_size, n_epoch):
        self.actor = actor
        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.gae = GAE()

        self.batch_size = batch_size
        self.n_epoch = n_epoch

        self.clip_eps = 0.2
        self.entropy_coef = 0.0

    def update(self, buffer, next_state):
        states, actions, rewards, log_probs, values, dones = buffer.get_all()

        next_value = self.critic(torch.tensor(next_state, dtype=torch.float32)).item()

        advantages, returns = self.gae.compute(rewards, values, dones, next_value)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)

        dataset_size = states.shape[0]

        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        step_count = 0

        for _ in range(self.n_epoch):
            idx = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                new_log_probs, entropy = self.actor.evaluate(
                    batch_states, batch_actions
                )

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * batch_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= self.entropy_coef * entropy.mean()

                values_pred = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(values_pred, batch_returns)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                step_count += 1

        return actor_loss_sum / step_count, critic_loss_sum / step_count
