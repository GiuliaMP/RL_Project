from utils import *
import collections

def sarsa(env, discretization, ep_min_decay, alpha, gamma, episodes, render=False):
    tot_reward = []
    #q_table = init_Qtable(state_number, action_number)
    q_table = collections.defaultdict(float)
    eps_list = []
    for ep in range(episodes):
        eps = decay_function(ep, ep_min_decay)
        eps_list.append(eps)
        tot_ep_reward = 0
        done = False
        s = env.reset() # seed = 42
        s = discretize_state(s, discretization)
        a = choose_action_eps_greedy(env, q_table, s, eps)
        i = 0
        while not done:
            s_p, reward, done, _ = env.step(a)
            if render:
                env.render()
            s_p = discretize_state(s_p, discretization)
            a_p = choose_action_eps_greedy(env, q_table, s_p, eps)
            q_table[s,a] += alpha*(reward + gamma*q_table[s_p,a_p] - q_table[s,a])
            s, a = s_p, a_p
            tot_ep_reward += reward
            i+=1
        tot_reward.append(tot_ep_reward)
    env.close()
    plt.plot(eps_list)
    return tot_reward