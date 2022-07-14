from utils import *

def random(env, episodes, render=False):
    total_reward = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state, reward, done, _ = env.step(env.action_space.sample())
            episode_reward += reward
            if render and (episodes - episode) < 10:
                env.render()
        total_reward.append(episode_reward)
    env.close()
    return total_reward