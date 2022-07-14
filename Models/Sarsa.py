from utils import *
import collections

def sarsa(env, discretization, episode_min_decay, alpha, gamma, episodes, render=False):
    total_reward = []
    q_table = collections.defaultdict(float)
    for episode in range(episodes):
        eps = decay_function(episode, episode_min_decay)
        episode_reward = 0
        done = False
        state = env.reset()
        state = discretize_state(state, discretization)
        action = choose_action_eps_greedy(env, q_table, state, eps)
        while not done:
            state_prime, reward, done, _ = env.step(action)
            if render and (episodes - episode) < 10:
                env.render()
            state_prime = discretize_state(state_prime, discretization)
            action_prime = choose_action_eps_greedy(env, q_table, state_prime, eps)
            delta_q = (reward + gamma*q_table[state_prime,action_prime] - q_table[state,action])
            q_table[state,action] += alpha*delta_q
            state, action = state_prime, action_prime
            episode_reward += reward
        total_reward.append(episode_reward)
    env.close()
    return total_reward