from utils import *
import collections

def q_learning(env, discretization, ep_min_decay, alpha, gamma, episodes, render=False):
    actions = [action for action in range(env.action_space.n)]
    tot_reward = []
    #q_table = init_Qtable(state_number, action_number)
    q_table = collections.defaultdict(float)
    for ep in range(episodes):
        eps = decay_function(ep, ep_min_decay)
        tot_ep_reward = 0
        done = False
        s = env.reset() # seed = 42
        s = discretize_state(s, discretization)
        #s = discretize(5, s)
        i = 0
        while not done:
            a = choose_action_eps_greedy(env, q_table, s, eps)
            s_p, reward, done, _ = env.step(a)
            if render:
                env.render()
            s_p = discretize_state(s_p, discretization)
            #s_p = discretize(5, s_p)
            a_p = choose_action_eps_greedy(env, q_table, s_p, eps)
            q_action_s_p = [q_table[s_p,i] for i in actions] 
            q_table[s,a] += alpha*(reward + gamma*(np.max(q_action_s_p)) - q_table[s,a])
            s = s_p
            tot_ep_reward += reward
            i+=1
        tot_reward.append(tot_ep_reward)
    env.close()
    return tot_reward