from utils import *
import collections

def sarsa_va_sg(env, discretization, n_bins, ep_min_decay, alpha, gamma, episodes, render=False):
    tot_reward = []
    # w = np.zeros((*n_bins, env.action_space.n))+0
    w = collections.defaultdict(float)
    for ep in range(episodes):
        eps = decay_function(ep, ep_min_decay)
        tot_ep_reward = 0
        done = False
        s = env.reset() # seed = 42
        x_indexes = discretize_state(s, discretization) 
        a = choose_action_eps_greedy(env, w, x_indexes, eps)  
        i = 0
        while not done:
            s_p, reward, done, _ = env.step(a)
            if render:
                env.render()
            #x = np.zeros((*n_bins,)) 
            x_indexes_p = discretize_state(s_p, discretization)    
            #x[(*x_indexes,)] = 1
            a_p = choose_action_eps_greedy(env, w, x_indexes_p, eps)
            delta_q = (reward + gamma * w[x_indexes_p, a_p] - w[x_indexes, a])
            w[x_indexes, a] += alpha * delta_q
            x_indexes, a = x_indexes_p, a_p
            tot_ep_reward += reward
            i+=1
        tot_reward.append(tot_ep_reward)
    env.close()
    return tot_reward
