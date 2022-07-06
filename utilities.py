
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import gym

env = gym.make("LunarLander-v2")

actions = [0,1,2,3]

def discretize_states(state):
    discrete_state = (min(5, max(-5, int((state[0]) / 0.05))), \
                        min(5, max(-5, int((state[1]) / 0.1))), \
                        min(2, max(-2, int((state[2]) / 0.1))), \
                        min(2, max(-2, int((state[3]) / 0.1))), \
                        min(2, max(-2, int((state[4]) / 0.1))), \
                        min(2, max(-2, int((state[5]) / 0.1))), \
                        int(state[6]), \
                        int(state[7]))

    return discrete_state

def choose_action(q_table, s, eps):
    if (np.random.random() <= eps):
        return env.action_space.sample() #Exploration
    else:
        q_action_s = [q_table[s,i] for i in actions] 
        return np.argmax(q_action_s) #Eplotation

def decay_function(ep):
    if (ep <= 100):
        return 0.5
    elif (ep > 100 and ep <= 300  ):
        return 0.2
    elif (ep > 300 and ep <= 500  ):
        return 0.1
    elif (ep > 500 and ep <= 700  ):
        return 0.01
    else:
        return 0.0

def discretize_space(min_lim, min_centre, max_centre, max_lim, n_discrete_values):
    space = np.zeros((n_discrete_values))
    space[1:n_discrete_values-1] = np.linspace(min_centre, max_centre, n_discrete_values-2)
    space[0] = min_lim
    space[-1] = max_lim
    return space

def plot_space(discrete_space):
    _ = [plt.plot([x, x], [0,5], color='r') for x in discrete_space]
    plt.show()

def plot_moving_average(reward):
    T = len(reward)
    avg = np.zeros(T)
    for t in range(T):
        avg[t] = np.mean(reward[max(0, t-10):(t+1)])
    plt.plot(avg)
    plt.show()