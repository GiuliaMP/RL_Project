
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import gym

env = gym.make("LunarLander-v2")

actions = [0,1,2,3]

def discretize_states(state):
    discrete_state = (min(2, max(-2, int((state[0]) / 0.05))), \
                        min(2, max(-2, int((state[1]) / 0.1))), \
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
    elif (ep > 100 and ep <= 200  ):
        return 0.2
    elif (ep > 200 and ep <= 300  ):
        return 0.1
    elif (ep > 300 and ep <= 400  ):
        return 0.01
    else:
        return 0.0