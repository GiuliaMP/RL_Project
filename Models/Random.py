from utilities import *

def random(env, episodes, render=False):
    rew_list = []
    for _ in range(episodes):
        observation = env.reset()
        done = False
        tot_rew = 0
        while not done:
            observation, reward, done, info = env.step(env.action_space.sample())
            tot_rew += reward
            if render:
                env.render()
        rew_list.append(tot_rew)

    env.close()

    return rew_list