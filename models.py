import matplotlib.pyplot as plt
import numpy as np
from utilities import *
import collections
import gym
env = gym.make("LunarLander-v2")

def random_agent(episodes, render=False):
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
        rew_list.append(reward)

    env.close()

    return rew_list


def sarsa(alpha, gamma, episodes, render=False):
    tot_reward = []
    #q_table = init_Qtable(state_number, action_number)
    q_table = collections.defaultdict(float)
    for ep in range(episodes):
        eps = decay_function(ep)
        tot_ep_reward = 0
        done = False
        s = env.reset() # seed = 42
        s = discretize_states(s)
        #s = discretize(5, s)
        a = choose_action(q_table, s, eps)
        i = 0
        while not done:
            s_p, reward, done, _ = env.step(a)
            if render:
                env.render()
            s_p = discretize_states(s_p)
            #s_p = discretize(5, s_p)
            a_p = choose_action(q_table, s_p, eps)
            q_table[s,a] += alpha*(reward + gamma*q_table[s_p,a_p] - q_table[s,a])
            s, a = s_p, a_p
            tot_ep_reward += reward
            i+=1
        tot_reward.append(tot_ep_reward)
    env.close()
    return tot_reward


def q_learning(alpha, gamma, episodes, render=False):
    tot_reward = []
    #q_table = init_Qtable(state_number, action_number)
    q_table = collections.defaultdict(float)
    for ep in range(episodes):
        eps = decay_function(ep)
        tot_ep_reward = 0
        done = False
        s = env.reset() # seed = 42
        s = discretize_states(s)
        #s = discretize(5, s)
        i = 0
        while not done:
            a = choose_action(q_table, s, eps)
            s_p, reward, done, _ = env.step(a)
            if render:
                env.render()
            s_p = discretize_states(s_p)
            #s_p = discretize(5, s_p)
            a_p = choose_action(q_table, s_p, eps)
            q_action_s_p = [q_table[s_p,i] for i in actions] 
            q_table[s,a] += alpha*(reward + gamma*(np.max(q_action_s_p)) - q_table[s,a])
            s = s_p
            tot_ep_reward += reward
            i+=1
        tot_reward.append(tot_ep_reward)
    env.close()
    return tot_reward