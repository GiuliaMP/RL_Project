from utils import *
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import deque
device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 

import torch.nn as nn
class Net(nn.Module):
    def __init__(self, input_size, output_size, seed):
        super(Net, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, output_size)
        
    def forward(self, state):
        x = self.linear1(state)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        out = self.linear3(x)
        return out

def train_model(model, inputs, y_targets, optimizer, loss=nn.functional.mse_loss):
        model.train()
        y_pred = model(inputs)
        loss = loss(y_pred, y_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        out = model(inputs)
    return out

# # Plot the loss
# step = np.linspace(0,1000,1000)
# plt.plot(step,np.array(loss_list))

def sarsa_va_ann(env, state_size, episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    total_reward = []                       
    reward_window = deque(maxlen=100)  # last 100 reward
    eps = eps_start    
    already_solved = False         

    action_size = env.action_space.n
    q_nets = [] # list of networks per action
    for a in range(action_size):
        net = Net(state_size,1,0)
        net.to(device)
        q_nets.append(net)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    for episode in range(episodes):
        episode_reward = 0
        done = False
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q_values = np.array([predict(q_nets[a],state).detach().cpu().numpy() for a in range(action_size)]).flatten()
        action = choose_action_eps_greedy_nn(env, q_values, eps)
        while not done:
            state_prime, reward, done, _ = env.step(action)
            state_prime = torch.from_numpy(state_prime).float().unsqueeze(0).to(device)
            q_values_prime = np.array([predict(q_nets[a],state_prime).detach().cpu().numpy() for a in range(action_size)]).flatten()
            action_prime = choose_action_eps_greedy_nn(env, q_values_prime, eps)
            q_values_target = np.array([reward + GAMMA * q_values_prime[action_prime] * (1 - done)])
            q_values_target = torch.from_numpy(q_values_target).float().unsqueeze(0).to(device)
            train_model(q_nets[action], state, q_values_target, optimizer)
            state, action = state_prime, action_prime
            episode_reward += reward
        
        eps = max(eps_end, eps_decay*eps)
                
        total_reward.append(episode_reward)    
        reward_window.append(episode_reward) 

        # Print utilities
        print('\rEpisode {}\tAverage Reward: {:.2f}'.format(episode, np.mean(reward_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Reward: {:.2f}'.format(episode, np.mean(reward_window)))
        if np.mean(reward_window)>=200.0:
            if not already_solved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(reward_window)))
                already_solved = True
   
    env.close()
    return total_reward