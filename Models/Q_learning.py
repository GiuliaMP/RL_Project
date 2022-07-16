from utils import *
import collections
from collections import deque

def q_learning(env, discretization, alpha, gamma, episodes, render=False, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    reward_window = deque(maxlen=100)  # last 100 reward
    actions = [action for action in range(env.action_space.n)]
    total_reward = []
    q_table = collections.defaultdict(float)
    eps = eps_start  
    for episode in range(episodes):
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
        
        eps = max(eps_end, eps_decay*eps)
        
        total_reward.append(episode_reward)
        reward_window.append(episode_reward) 

        # Print utilities
        print('\rEpisode {}\tAverage Reward: {:.2f}'.format(episode, np.mean(reward_window)), end="")
        if np.mean(reward_window)>=200.0:
            if not already_solved:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(reward_window)))
                already_solved = True

    env.close()
    return total_reward