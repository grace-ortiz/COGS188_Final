import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from webots_env import WebotsCarEnv
import matplotlib.pyplot as plt



STATE_BINS = [15, 10, 10, 6, 8, 15]  # 6 discrete dimensions
ACTION_BINS = [12, 6]

STATE_LIMITS = [
    (0, 150),     # speed
    (-100, 100),  # gps_x
    (-100, 100),  # gps_y
    (0, 100),     # lidar_dist
    (-90, 90),  # lidar_angle
    (0, 80)       # lane_deviation
]

ACTION_LIMITS = [
    (-0.5, 0.5), 
    (0.0, 150.0)
]

class SARSA:
    def __init__(self, env, alpha=0.05, gamma=0.98, epsilon=0.9, epsilon_min=0.05, epsilon_decay=0.999):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min  # minimum epsilon value
        self.epsilon_decay = epsilon_decay  # decay rate
        self.rewards_history = []
        self.epsilon_history = []
        
        # discretize state space
        self.q_table = np.zeros(tuple(STATE_BINS) + tuple(ACTION_BINS))
        
        
    def discretize_state(self, state):
        discrete_state = []
        
        speed = state["speed"][0]
        gps_x, gps_y = state["gps"]
        lidar_dist = state["lidar_dist"][0]
        lidar_angle = state["lidar_angle"][0]
        lane_deviation = state["lane_deviation"][0]
        
        state_values = [speed, gps_x, gps_y, lidar_dist, lidar_angle, lane_deviation]
        
        for i, (value, (low, high)) in enumerate(zip(state_values, STATE_LIMITS)):
            value = np.clip(value, low, high)
            index = int((value - low) / (high - low) * (STATE_BINS[i] - 1))
            discrete_state.append(index)
            
        return tuple(discrete_state)
    
    
    def discretize_action(self, action):
        discrete_action = []
        
        for i, (value, (low, high), bins) in enumerate(zip(action, ACTION_LIMITS, ACTION_BINS)):
            value = np.clip(value, low, high)
            index = int((value - low) / (high - low) * (bins - 1))
            discrete_action.append(index)
            
        return tuple(discrete_action)
    
    
    def choose_discrete_action(self, state):
        if np.random.rand() < self.epsilon:
            return (
                np.random.randint(0, ACTION_BINS[0]),  # random steering index
                np.random.randint(0, ACTION_BINS[1])   # random speed index
            )
        else:
            return np.unravel_index(np.argmax(self.q_table[state]), (ACTION_BINS[0], ACTION_BINS[1]))
    
    
    def undigitize_action(self, discrete_action):
        continuous_action = []
        for index, (low, high), bins in zip(discrete_action, ACTION_LIMITS, ACTION_BINS):
            value = low + (index / (bins - 1)) * (high - low)
            continuous_action.append(value)
    
        return np.array(continuous_action)
        
        
    def train(self, episodes=2500):
        for episode in range(episodes):
            state = self.env.reset()
            discrete_state = self.discretize_state(state)
            discrete_action = self.choose_discrete_action(discrete_state)
            
            done = False
            total_reward = 0
            
            while not done:
                next_state, reward, done = self.env.step(self.undigitize_action(discrete_action))
                discrete_next_state = self.discretize_state(next_state)
                next_discrete_action = self.choose_discrete_action(discrete_next_state)
                
                self.q_table[discrete_state][discrete_action] += self.alpha * (
                    reward + self.gamma * self.q_table[discrete_next_state][next_discrete_action] - self.q_table[discrete_state][discrete_action]
                )
                
                discrete_state, discrete_action = discrete_next_state, next_discrete_action
                total_reward += reward
             
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
               
            self.rewards_history.append(total_reward)
            self.epsilon_history.append(self.epsilon)
            
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
            
        self.plot_rewards()
            

    def plot_rewards(self, window_size=20):
            """Plots total rewards over episodes."""
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Plot total rewards
            color = 'tab:blue'
            ax1.set_xlabel("Episodes")
            ax1.set_ylabel("Total Reward", color=color)
            ax1.plot(self.rewards_history, label="Total Reward", color=color, alpha=0.4)  # Original rewards (faint)

            # ✅ Smooth the rewards
            if len(self.rewards_history) >= window_size:
                smoothed_rewards = np.convolve(self.rewards_history, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size - 1, len(self.rewards_history)), smoothed_rewards, color='green', label="Smoothed Reward", linewidth=2)
            
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Plot epsilon on the secondary y-axis
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel("Epsilon", color=color)
            ax2.plot(self.epsilon_history, label="Epsilon", color=color, linestyle='dashed')
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title(f"Rewards and Epsilon Decay\nAlpha: {self.alpha}, Gamma: {self.gamma}, State bins: {STATE_BINS}, Action bins: {ACTION_BINS}")
            fig.tight_layout()

            plt.grid()
            plt.savefig("rewards_epsilon_plot.png")
            print("Plot saved as rewards_epsilon_plot.png")
        
        
env = WebotsCarEnv()
sarsa_agent = SARSA(env)
sarsa_agent.train()
env.reset()

