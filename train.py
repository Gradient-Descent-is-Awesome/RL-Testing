import gymnasium as gym

from models import Actor

env = gym.make("Pendulum-v1", render_mode="human")

# obs: Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
# act: Box(-2.0, 2.0, (1,), float32)

actor = Actor(3, 1)

obs, _ = env.reset()

episode_over = False
total_reward = 0

while not episode_over:
    act = env.action_space.sample()
    obs, rew, term, trun, _ = env.step(act)

    # obs: current state
    # rew: reward
    # term: env finish
    # trun: code interrupt

    total_reward += rew
    episode_over = term or trun

print(f"Total reward: {total_reward}")
env.close()
