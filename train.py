import gymnasium as gym

from config import batch_size, buffer_size
from models import Actor, Critic
from utils import RolloutBuffer

env = gym.make("Pendulum-v1")

# obs: Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
# act: Box(-2.0, 2.0, (1,), float32)

actor = Actor(3, 1)
critic = Critic(3)

buffer = RolloutBuffer()

obs, _ = env.reset()

finish = False
total_reward = 0.0

while not finish:
    act = env.action_space.sample()
    obs, rew, term, trun, _ = env.step(act)

    # obs: current state
    # rew: reward
    # term: env finish
    # trun: code interrupt

    total_reward += rew
    episode_over = term or trun

    buffer.add(obs, act, rew, [0.0], rew, episode_over)
    print(buffer.size())

    if buffer.size() >= buffer_size:
        print("Training model, buffer size reached 2048")
        for i, batch in enumerate(buffer.batch(batch_size)):
            states, actions, rewards, log_probs, values, dones = batch
            print(f"Batch {i}: {states.shape}, {actions.shape}")

        finish = True
        buffer.reset()


print(f"Total reward: {total_reward}")
print(f"Buffer size: {buffer.size()}")
env.close()
