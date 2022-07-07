import matplotlib.pyplot as plt
import numpy as np
import math

# Discretization functions for lunar lander environment

def discretize_space_non_uniform(min_lim, min_centre, max_centre, max_lim, n_bins):
    space = np.zeros((n_bins-1))
    space[1:n_bins-2] = np.linspace(min_centre, max_centre, n_bins-3)
    space[0] = min_lim
    space[-1] = max_lim
    return space

def discretize_space_uniform(min_lim, max_lim, n_bins):
    space = np.linspace(min_lim, max_lim, n_bins-1)
    return space

def discretize_state(observation, discretization):
    sx, sy, vx, vy, theta, omega, bo1, bo2 = observation
    sx_d = int(np.digitize(sx, discretization[0]))
    sy_d = int(np.digitize(sy, discretization[1]))
    vx_d = int(np.digitize(vx, discretization[2]))
    vy_d = int(np.digitize(vy, discretization[3]))
    theta_d = int(np.digitize(theta, discretization[4]))
    omega_d = int(np.digitize(omega, discretization[5]))

    return (sx_d, sy_d, vx_d, vy_d, theta_d, omega_d, bo1, bo2)

# Other

def decay_function(episode, ep_min_decay):
    min_epsilon = 0.01
    max_epsilon = 1.0
    return max(min_epsilon, min(max_epsilon, 1.0 - 
                              math.log10((episode + 1) / (ep_min_decay*0.1))))

def choose_action_eps_greedy(env, q_table, s, eps):
    actions = [action for action in range(env.action_space.n)]
    if (np.random.random() <= eps):
        return env.action_space.sample() #Exploration
    else:
        q_action_s = [q_table[s,i] for i in actions] 
        return np.argmax(q_action_s) #Eplotation


# PLOT functions

def plot_space(discrete_space):
    _ = [plt.plot([x, x], [0,5], color='r') for x in discrete_space]
    plt.show()

def moving_avg(reward, window=10):
    T = len(reward)
    avg = np.zeros(T)
    for t in range(T):
        avg[t] = np.mean(reward[max(0, t-window):(t+1)])
    return avg