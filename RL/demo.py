import gym
import time
env = gym.make('CartPole-v0')
#env = env.unwrapped
env.reset()
for _ in range(1000):
    env.render(mode='rgb_array')
    env.step(env.action_space.sample())
    env.close()