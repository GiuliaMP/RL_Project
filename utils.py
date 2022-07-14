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

def show_video(env_name):
    mp4list = glob.glob('./*.mp4')
    if len(mp4list) > 0:
        mp4 = './{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")
        
def show_video_of_model(qnet, env_name):
    env = gym.make(env_name)
    vid = video_recorder.VideoRecorder(env, path="./{}.mp4".format(env_name))
    qnet.load_state_dict(torch.load('checkpoint.pth'))
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