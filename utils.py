import matplotlib.pyplot as plt
import numpy as np
import math
import random
import gym
import torch

# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob
import base64, io
from matplotlib.colors import ListedColormap


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

    return (sx_d, sy_d, vx_d, vy_d, theta_d, omega_d, int(bo1), int(bo2))

def decay_function(episode, episode_min_decay):
    min_epsilon = 0.01
    max_epsilon = 1.0
    return max(min_epsilon, min(max_epsilon, 1.0 - 
                              math.log10((episode + 1) / (episode_min_decay*0.1))))

def choose_action_eps_greedy(env, table_s_a, s, eps):
    if (np.random.random() <= eps): #Exploration
        return env.action_space.sample() 
    else: # Exploitation
        actions = [action for action in range(env.action_space.n)]
        table_s = [table_s_a[s,a] for a in actions]
        table_s_argmax = np.argwhere(table_s == np.max(table_s))
        table_s_argmax = table_s_argmax.reshape(len(table_s_argmax))
        return np.random.choice(table_s_argmax)

def choose_action_eps_greedy_nn(env, q_values, eps):
    if random.random() > eps:
        return np.argmax(q_values)
    else:
        return env.action_space.sample()

def choose_action_epsilon_greedy_dqn(env, q_values,eps):
    if random.random() > eps:
        return np.argmax(q_values.cpu().data.numpy())
    else:
        return env.action_space.sample()

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
       
def save_video_of_model(qnet, env_name, checkpoint):
    env = gym.make(env_name)
    vid = video_recorder.VideoRecorder(env, path="./{}.mp4".format(env_name))
    qnet.load_state_dict(torch.load(checkpoint))
    state = env.reset()
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        vid.capture_frame()
        state = torch.from_numpy(state).float().unsqueeze(0)
        qnet.eval()
        with torch.no_grad():
            q_values = qnet(state)
        action = np.argmax(q_values.data.numpy())
        state, reward, done, _ = env.step(action)        
    env.close()

def policy_visualization_dqn(qnet, vx, vy, theta, omega, bol0, bol1, x_grid, y_grid):
    states = torch.empty(0, dtype=torch.float32)
    [X,Y] = np.meshgrid(x_grid,y_grid)
    for y in y_grid:
        for x in x_grid:
            states = torch.cat((states, torch.tensor([[x,y,vx,vy,theta,omega, bol0, bol1]])))
        states = states.type(torch.float32)
    with torch.no_grad():
        qnet.eval()
        q_values = qnet(states)
    q_values.shape
    x_policy = torch.argmax(q_values,1).cpu().numpy()
    x_policy = x_policy.reshape((len(y_grid),len(x_grid)))

    col_dict={0:'#d9459e', 1:'#e0d1f9', 2:'#9966ea', 3:'#ffdb63'}
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    labels = np.array(["Action 0: do nothing","Action 1: fire right (go left)","Action 2: fire main","Action 3: fire left (go right)"])
    ax, fig = plt.subplots(figsize=(12,8))
    plt.pcolor(X,Y,x_policy, cmap=cm)
    cbar = plt.colorbar()
    step = (cbar.vmax-cbar.vmin)/8
    cbar.set_ticks([cbar.vmin+step,cbar.vmin+step*3,cbar.vmin+step*5,cbar.vmin+step*7])
    cbar.set_ticklabels(labels)
    plt.title(f'vx={vx}, vy={vy}, theta={theta}, omega={omega}, bol0={bol0}, bol1={bol1}\n')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
