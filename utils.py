import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def batch(self, batch_size):
        idx = np.random.permutation(self.size())

        for start in range(0, self.size(), batch_size):
            batch_idx = idx[start : start + batch_size]

            yield (
                np.array(self.states)[batch_idx],
                np.array(self.actions)[batch_idx],
                np.array(self.rewards)[batch_idx],
                np.array(self.log_probs)[batch_idx],
                np.array(self.values)[batch_idx],
                np.array(self.dones)[batch_idx],
            )

    def size(self):
        return len(self.states)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
