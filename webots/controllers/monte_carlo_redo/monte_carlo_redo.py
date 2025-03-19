import numpy as np
import random
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from webots_env import WebotsCarEnv
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns




STATE_BINS = [15, 8, 8, 6, 8, 15]  # 6 discrete dimensions
ACTION_BINS = [15, 6]

STATE_LIMITS = [
    (0, 150),     # speed
    (-100, 100),  # gps_x
    (-100, 100),  # gps_y
    (0, 100),     # lidar_dist
    (-90, 90),    # lidar_angle
    (0, 48)       # lane_deviation
]

ACTION_LIMITS = [
    (-0.4, 0.4), 
    (0.0, 150.0)
]


class MonteCarlo:
    def __init__(self, env: WebotsCarEnv, gamma: float = 1.0, epsilon: float = 0.1, Q0: float = 0.0, max_episode_size : int = 100000):
            self.env = env
            self.gamma = gamma
            self.epsilon = epsilon
            self.Q0 = Q0
            self.max_episode_size = max_episode_size
            
            self.n_actions = np.prod(ACTION_BINS)
            
            self.Q = defaultdict(lambda: np.full(self.n_actions, Q0)) # defaults to Q0
            self.C = defaultdict(lambda: np.zeros(self.n_actions)) # defaults to 0
            self.target_policy = defaultdict(lambda: np.ones(self.n_actions) / self.n_actions) # defaults to 1/number of actions
            self.behavior_policy = defaultdict(lambda: np.ones(self.n_actions) / self.n_actions) # defaults to 1/number of actions

    def discretize_state(self, state):
        discrete_state = []
        speed = state["speed"][0]
        gps_x, gps_y = state["gps"]
        lidar_dist = state["lidar_dist"][0]
        lidar_angle = state["lidar_angle"][0]
        lane_deviation = state["lane_deviation"][0]

        state_values = [speed, gps_x, gps_y, lidar_dist, lidar_angle, lane_deviation]

        for value, (low, high), bins in zip(state_values, STATE_LIMITS, STATE_BINS):
            value = np.clip(value, low, high)
            index = int((value - low) / (high - low) * (bins - 1))
            discrete_state.append(index)

        return tuple(discrete_state)

    def discretize_action(self, action):
        discrete_action = []
        for value, (low, high), bins in zip(action, ACTION_LIMITS, ACTION_BINS):
            value = np.clip(value, low, high)
            index = int((value - low) / (high - low) * (bins - 1))
            discrete_action.append(index)

        return tuple(discrete_action)

    def undigitize_action(self, discrete_action):
        continuous_action = []
        for index, (low, high), bins in zip(discrete_action, ACTION_LIMITS, ACTION_BINS):
            value = low + (index / (bins - 1)) * (high - low)
            continuous_action.append(value)

        return np.array(continuous_action)
    
    
    def create_target_greedy_policy(self):
        for state, Q_values in self.Q.items():
            best_action = np.argmax(Q_values)
            prob_vector = np.zeros(self.n_actions)
            prob_vector[best_action] = 1
            self.target_policy[state] = prob_vector
                
                
    def create_behavior_egreedy_policy(self):
        for state, target_prob_vector in self.target_policy.items():
            greedy_action_prob = 1 - self.epsilon + (self.epsilon / self.n_actions)
            behavior_prob_vector = np.full(self.n_actions, self.epsilon / self.n_actions)

            best_action = np.argmax(target_prob_vector)
            behavior_prob_vector[best_action] = greedy_action_prob
            self.behavior_policy[state] = behavior_prob_vector
            

    def egreedy_selection(self, state):
        state = str(state)
        behavior_prob_vector = self.behavior_policy[state]
        total_prob = sum(behavior_prob_vector)
        return random.choices(range(self.n_actions), weights=[prob / total_prob for prob in behavior_prob_vector])[0]


    def generate_egreedy_episode(self):
        episode = []
        state = self.env.reset()
        state = self.discretize_state(state)
        
        for _ in range(self.max_episode_size):
            action_index = self.egreedy_selection(state)
            discrete_action = np.unravel_index(action_index, ACTION_BINS)
            action = self.undigitize_action(discrete_action)
            
            next_state, reward, done = self.env.step(action)
            next_state = self.discretize_state(next_state)
            
            episode.append((state, action_index, reward))
            
            state = next_state
            
            if done:
                action_index = self.egreedy_selection(state)
                discrete_action = np.unravel_index(action_index, ACTION_BINS)
                action = self.undigitize_action(discrete_action)
                
                next_state, reward, _ = self.env.step(action)
                episode.append((state, action_index, reward))
                break
            
        return episode
    
    
    def generate_greedy_episode(self):
        episode = []
        state = self.env.reset()
        state = self.discretize_state(state)
        
        for _ in range(self.max_episode_size):
            best_action_index = np.argmax(self.target_policy[str(state)])
            discrete_action = np.unravel_index(best_action_index, ACTION_BINS)
            best_action = self.undigitize_action(discrete_action)
            
            next_state, reward, done = self.env.step(best_action)
            next_state = self.discretize_state(next_state)
            
            episode.append((state, best_action_index, reward))
            
            state = next_state
            
            if done:
                best_action_index = np.argmax(self.target_policy[str(state)])
                discrete_action = np.unravel_index(best_action_index, ACTION_BINS)
                action = self.undigitize_action(discrete_action)

                next_state, reward, _ = self.env.step(action)
                episode.append((state, best_action_index, reward))
                break
            
        return episode
    

    def update_offpolicy(self, episode):
        G = 0
        W = 1

        for state, action, reward in reversed(episode):
            state = str(state)
            G = self.gamma * G + reward
            self.C[state][action] += W
            
            self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])

            W *= self.target_policy[state][action] / self.behavior_policy[state][action]
            
            if W == 0:  
                break      
        
        self.create_target_greedy_policy()  
        self.create_behavior_egreedy_policy() 
        
        
    def update_onpolicy(self, episode):
        G = 0
        returns = {}

        for state, action, reward in reversed(episode): 
            state = str(state)
            if (state, action) not in returns:  
                returns[(state, action)] = True
                G = self.gamma * G + reward
                self.Q[state][action] += (G - self.Q[state][action]) / (self.C[state][action] + 1)
                self.C[state][action] += 1  

        for state, _, _ in reversed(episode):
            state = str(state)
            best_action = np.argmax(self.Q[state])
            self.target_policy[state] = np.full(self.n_actions, self.epsilon / self.n_actions)
            self.target_policy[state][best_action] += (1 - self.epsilon)  
        
        self.create_target_greedy_policy()  
        self.create_behavior_egreedy_policy()
        
        
    def train_offpolicy(self, num_episodes):
        for _ in range(num_episodes):
            episode = self.generate_egreedy_episode()
            self.update_offpolicy(episode)
            
            
    def get_greedy_policy(self):
        policy = {}
        for state, actions in self.Q.items():
            policy[state] = np.argmax(actions)
        return policy
    
    
def visualize_training(behavior_rewards, target_rewards):
    """Visualize the training progress."""
    df_behavior = pd.DataFrame(behavior_rewards)
    df_behavior.columns.name = 'episode'
    df_behavior.index.name = 'learner'
    df_behavior = df_behavior.stack().rename('returns').to_frame()

    df_target = pd.DataFrame(target_rewards)
    df_target.columns.name = 'episode'
    df_target.index.name = 'learner'
    df_target = df_target.stack().rename('returns').to_frame()

    plt.figure(figsize=(12, 6))

    sns.lineplot(data=df_behavior, x='episode', y='returns', label='Behavior Policy', color='blue')
    sns.lineplot(data=df_target, x='episode', y='returns', label='Target Policy', color='green')

    plt.title('Monte Carlo Training: Behavior vs. Target Policy')
    plt.xlabel('Episode')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    
    plt.savefig("monte_carlo_training_plot.png", dpi=300, bbox_inches='tight')

    
    
    
env = WebotsCarEnv()

steps = []
behavior_rewards = []
target_rewards = []

num_learners = 1
num_episodes = 200

for j in tqdm(range(num_learners)):
    MC = MonteCarlo(env, max_episode_size=1000000)
    this_steps = []
    this_rewards = []

    for k in range(num_episodes):
        episode = MC.generate_egreedy_episode()
        this_rewards.append(pd.DataFrame(episode).iloc[:, -1].sum())
        this_steps.append(len(episode))
        MC.update_offpolicy(episode)

    steps.append(this_steps)
    behavior_rewards.append(this_rewards)

    episode = MC.generate_egreedy_episode()
    target_rewards.append(pd.DataFrame(episode).iloc[:, -1].sum())

visualize_training(behavior_rewards, target_rewards)