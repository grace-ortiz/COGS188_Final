import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from webots_env import WebotsCarEnv
import matplotlib.pyplot as plt



STATE_BINS = [15, 8, 8, 6, 8, 15]  # 6 discrete dimensions
ACTION_BINS = [15, 6]

STATE_LIMITS = [
    (0, 150),     # speed
    (-100, 100),  # gps_x
    (-100, 100),  # gps_y
    (0, 100),     # lidar_dist
    (-90, 90),  # lidar_angle
    (0, 48)       # lane_deviation
]

ACTION_LIMITS = [
    (-0.4, 0.4), 
    (0.0, 150.0)
]

class SARSA:
    def __init__(self, env, alpha=0.08, gamma=0.99, epsilon=0.7, epsilon_min=0.1, epsilon_decay=0.995):
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

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return (
                np.random.randint(0, ACTION_BINS[0]),
                np.random.randint(0, ACTION_BINS[1])
            )
        else:
            return np.unravel_index(np.argmax(self.q_table[state]), (ACTION_BINS[0], ACTION_BINS[1]))

    def train(self, episodes=1000, save_interval=200):
        best_reward = float('-inf')

        for episode in range(episodes):
            state = self.env.reset()
            discrete_state = self.discretize_state(state)
            discrete_action = self.choose_action(discrete_state)

            total_reward = 0
            done = False

            while not done:
                # take action A, observe R and S'
                action = self.undigitize_action(discrete_action)
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # discretize next state and choose next action
                next_discrete_state = self.discretize_state(next_state)
                next_discrete_action = self.choose_action(next_discrete_state)

                # SARSA update
                q_current = self.q_table[discrete_state][discrete_action]
                q_next = self.q_table[next_discrete_state][next_discrete_action] if not done else 0

                # Q(S, A) ← Q(S, A) + α[R + γQ(S', A') − Q(S, A)]
                self.q_table[discrete_state][discrete_action] += self.alpha * (
                    reward + self.gamma * q_next - q_current
                )

                #move to the next state and action
                discrete_state = next_discrete_state
                discrete_action = next_discrete_action

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # save the best-performing Q-table
            if total_reward > best_reward:
                best_reward = total_reward
                with open("best_q_table.pkl", "wb") as f:
                    pickle.dump(self.q_table, f)
                print(f"Best Q-table saved at episode {episode + 1} with reward: {best_reward}")

            self.rewards_history.append(total_reward)
            self.epsilon_history.append(self.epsilon)

            # save Q-table periodically
            if (episode + 1) % save_interval == 0:
                filename = f"q_table_ep_{episode + 1}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(self.q_table, f)
                print(f"Q-table saved at episode {episode + 1}")

            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

        self.plot_rewards()



    def plot_rewards(self, window_size=20):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # plot rewards
        color = 'tab:blue'
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Total Reward", color=color)
        ax1.plot(self.rewards_history, label="Total Reward", color=color, alpha=0.4)

        # smoothed rewards
        if len(self.rewards_history) >= window_size:
            smoothed_rewards = np.convolve(self.rewards_history, np.ones(window_size) / window_size, mode='valid')
            ax1.plot(range(window_size - 1, len(self.rewards_history)), smoothed_rewards, color='green',
                     label="Smoothed Reward", linewidth=2)

        ax1.tick_params(axis='y', labelcolor=color)

        # Epsilon decay
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel("Epsilon", color=color)
        ax2.plot(self.epsilon_history, label="Epsilon", color=color, linestyle='dashed')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"SARSA Training Progress\nAlpha: {self.alpha}, Gamma: {self.gamma}")
        fig.tight_layout()

        plt.grid()
        plt.savefig("rewards_epsilon_plot.png")
        print("Plot saved as rewards_epsilon_plot.png")

            
    def run(self, q_table_path="best_q_table.pkl", episodes=10):
        with open(q_table_path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Loaded Q-table from {q_table_path}")

        total_rewards = []

        for episode in range(1, episodes + 1):
            state = self.env.reset()
            discrete_state = self.discretize_state(state)
            done = False
            total_reward = 0
            
            while not done:
                action = np.unravel_index(np.argmax(self.q_table[discrete_state]), (ACTION_BINS[0], ACTION_BINS[1]))
                continuous_action = self.undigitize_action(action)
                next_state, reward, done = self.env.step(continuous_action)
                discrete_state = self.discretize_state(next_state)

                total_reward += reward

            total_rewards.append(total_reward)

        avg_reward = np.mean(total_rewards)
        print(f"\n Average Reward over {episodes} episodes: {avg_reward:.2f}")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, episodes + 1), total_rewards, label="Total Reward per Episode", marker='o')
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Agent Performance Using Best Q-table")
        plt.grid(True)
        plt.legend()
        plt.show()
        
        
env = WebotsCarEnv()
sarsa_agent = SARSA(env)
sarsa_agent.train()
env.reset()

# sarsa_agent.run(q_table_path="q_table_ep_1000.pkl", episodes=10)
# env.reset()