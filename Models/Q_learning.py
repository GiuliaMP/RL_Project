from utils import *
import collections

def q_learning(env, discretization, episode_min_decay, alpha, gamma, episodes, render=False):
    actions = [action for action in range(env.action_space.n)]
    total_reward = []
    q_table = collections.defaultdict(float)
    for episode in range(episodes):
        eps = decay_function(episode, episode_min_decay)
        episode_reward = 0
        done = False
        state = env.reset()
        state = discretize_state(state, discretization)
        while not done:
            action = choose_action_eps_greedy(env, q_table, state, eps)
            state_prime, reward, done, _ = env.step(action)
            if render and (episodes - episode) < 10:
                env.render()
            state_prime = discretize_state(state_prime, discretization)
            q_table_state_prime = [q_table[state_prime,i] for i in actions] 
            delta_q = (reward + gamma*(np.max(q_table_state_prime)) - q_table[state,action])
            q_table[state,action] += alpha*delta_q
            state = state_prime
            episode_reward += reward
        total_reward.append(episode_reward)
    env.close()
    return total_reward