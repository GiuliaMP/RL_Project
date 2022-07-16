from utils import *
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import deque, namedtuple

device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

class QNet(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = self.linear1(state)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        out = self.linear3(x)
        return out

def masked_huber_loss(mask_value, clip_delta):
    def f(y_pred, y_true):
        error = y_true - y_pred
        cond  = torch.abs(error) < clip_delta
        mask_true = torch.not_equal(y_true, mask_value).type(torch.float32)
        masked_squared_error = 0.5 * torch.square(mask_true * (y_true - y_pred))
        linear_loss  = mask_true * (clip_delta * torch.abs(error) - 0.5 * (clip_delta ** 2))
        huber_loss = torch.where(cond, masked_squared_error, linear_loss)
        loss_value = torch.sum(huber_loss) / torch.sum(mask_true)
        if loss_value is None:
            return torch.tensor(-0.0, requires_grad=True).to(device)
        return loss_value
    f.__name__ = 'masked_huber_loss'
    return f

def soft_update(net, net_target, tau):
    for target_param, param in zip(net_target.parameters(), net.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

def train_model(model, inputs, y_targets, actions, optimizer, loss=nn.functional.mse_loss):
        model.train()
        y_pred = model(inputs).gather(1, actions) # selects y_pred correspondent to action of y_target
        loss = loss(y_pred, y_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        out = model(inputs)
    return out

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) # creates the memory queue
        self.batch_size = batch_size
        self.transition = namedtuple("Transition", field_names=["state", "action", "reward", "state_prime", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, state_prime, done):
        t = self.transition(state, action, reward, state_prime, done)
        self.memory.append(t)
    
    def sample(self):
        transitions = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([t.state.cpu() for t in transitions if t is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.action for t in transitions if t is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([t.reward for t in transitions if t is not None])).float().to(device)
        states_prime = torch.from_numpy(np.vstack([t.state_prime for t in transitions if t is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([t.done for t in transitions if t is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, states_prime, dones)

    def __len__(self):
        return len(self.memory)


def dqn(env, state_size, episodes=5000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    total_reward = []                       
    reward_window = deque(maxlen=100)  # last 100 reward
    eps = eps_start   
    already_solved = False        

    action_size = env.action_space.n 
    seed = 0
    q_net = QNet(state_size, action_size, seed).to(device)
    q_net_target = QNet(state_size, action_size, seed).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=LR)

    memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    step = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = predict(q_net, state)
            action = choose_action_epsilon_greedy_dqn(env, q_values,eps)
            state_prime, reward, done, _ = env.step(action)

            memory.add(state, action, reward, state_prime, done)
        
            step = (step + 1) % UPDATE_EVERY
            if step == 0:
                if len(memory) > BATCH_SIZE:
                    transitions = memory.sample()
                    states, actions, rewards, states_prime, dones = transitions
                    q_values_prime = predict(q_net_target,states_prime).detach().max(1)[0].unsqueeze(1) # return Qmax for each state_prime
                    q_values_max = rewards + GAMMA * q_values_prime * (1 - dones) # target q_values_max, 1 for each state
                    train_model(q_net, states, q_values_max, actions, optimizer, loss=masked_huber_loss(0.0,1.0))
                    soft_update(q_net, q_net_target, TAU)   

            state = state_prime
            episode_reward += reward

        eps = max(eps_end, eps_decay*eps)

        total_reward.append(episode_reward)   
        reward_window.append(episode_reward)  

        # Print utilities
        print('\rEpisode {}\tAverage Reward: {:.2f}'.format(episode, np.mean(reward_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Reward: {:.2f}'.format(episode, np.mean(reward_window)))
            # torch.save(qnet.state_dict(), f'checkpoint_{episode}.pth')
        if np.mean(reward_window)>=200.0:
            if not already_solved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(reward_window)))
                already_solved = True
            torch.save(q_net.state_dict(), 'Results/checkpoint_DQN.pth')
    env.close()
    return total_reward