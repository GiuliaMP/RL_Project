from utils import *
import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
class Net(nn.Module):
    def __init__(self,input,hidden_dim,output):
        super(Net,self).__init__()
        self.linear1=nn.Linear(input,128)
        self.linear2=nn.Linear(128, 256)
        self.linear3=nn.Linear(256,output)
         
    def forward(self,x):
        #x = nn.functional.normalize(x)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.tanh(x)  
        x = self.linear3(x)
        return x

    def train(self, epochs, input, target,  optimizer=torch.optim.Adam, lr=0.001, criterion=nn.functional.mse_loss):
        #input = input.clone().detach().requires_grad_(True)
        #target = torch.tensor(target, requires_grad=True)
        optimizer=optimizer(self.parameters(), lr=lr)
        loss_list = []
        for t in range(epochs):
            y_pred = self(input)
            loss = criterion(y_pred.type(torch.float32), target.type(torch.float32))
            loss_list.append(loss)
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in self.parameters():
                    param -= lr * param.grad

# # Plot the loss
# step = np.linspace(0,1000,1000)
# plt.plot(step,np.array(loss_list))

def sarsa_va_ann(env, n_var, ep_min_decay, alpha, gamma, episodes, render=False):
    action_size = env.action_space.n 
    q_nn_approx = []
    for a in range(action_size):
        net = Net(n_var,20,1)
        net.to(device)
        q_nn_approx.append(net)

    tot_reward = []
    for ep in range(episodes):
        if ep%100 == 0:
            print(ep)
        eps = decay_function(ep, ep_min_decay)
        tot_ep_reward = 0
        done = False
        s = env.reset() # seed = 42
        s = s.reshape(1,n_var)
        q_value_approx = np.array([q_nn_approx[a](torch.tensor(s).to(device)).detach().cpu().numpy() for a in range(action_size)]).flatten()
        a = choose_action_eps_greedy_nn(env, q_value_approx, eps)  
        while not done:
            s_p, reward, done, _ = env.step(a)
            if render:
                env.render()
            s_p = s_p.reshape(1, n_var)
            q_value_approx = np.array([q_nn_approx[a](torch.tensor(s).to(device)).detach().cpu().numpy() for a in range(action_size)]).flatten()
            q_value_approx_p = np.array([q_nn_approx[a](torch.tensor(s_p).to(device)).detach().cpu().numpy() for a in range(action_size)]).flatten()
            a_p = choose_action_eps_greedy_nn(env, q_value_approx_p, eps)
            target = (reward + gamma * q_value_approx_p[a_p]) # - q_value_approx[a])
            target = np.array(target).reshape(1,1)
            q_nn_approx[a].train(1, torch.tensor(s, requires_grad=True).to(device), torch.tensor(target, requires_grad=True).to(device))
            s,a = s_p, a_p
            tot_ep_reward += reward
        tot_reward.append(tot_ep_reward)
    env.close()
    return tot_reward