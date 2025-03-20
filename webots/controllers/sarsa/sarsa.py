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
    def __init__(self, env, alpha=0.08, gamma=0.99, epsilon=0.7, epsilon_min=0.15, epsilon_decay=0.998):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min  # minimum epsilon value
        self.epsilon_decay = epsilon_decay  # decay rate
        self.rewards_history = []
        self.epsilon_history = []
        self.steps_history = []  # TRACK STEPS PER EPISODE
        
        # Discretized Q-table
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

    def train(self, episodes=500, save_interval=200):
        best_reward = float('-inf')

        for episode in range(episodes):
            state = self.env.reset()
            discrete_state = self.discretize_state(state)
            discrete_action = self.choose_action(discrete_state)

            total_reward = 0
            steps = 0  # <== Initialize step counter
            done = False

            while not done:
                action = self.undigitize_action(discrete_action)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1  # <== Increment step counter

                # SARSA update
                next_discrete_state = self.discretize_state(next_state)
                next_discrete_action = self.choose_action(next_discrete_state)
                q_current = self.q_table[discrete_state][discrete_action]
                q_next = self.q_table[next_discrete_state][next_discrete_action] if not done else 0

                self.q_table[discrete_state][discrete_action] += self.alpha * (
                    reward + self.gamma * q_next - q_current
                )

                # Move to the next state and action
                discrete_state = next_discrete_state
                discrete_action = next_discrete_action

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Save episode metrics
            self.rewards_history.append(total_reward)
            self.steps_history.append(steps)  # <== Store steps for this episode
            self.epsilon_history.append(self.epsilon)

            # Plot every 50 episodes
            if episode % 50 == 0:
                plot_rewards(self)

            # save the best-performing Q-table
            if total_reward > best_reward:
                best_reward = total_reward
                with open("best_q_table.pkl", "wb") as f:
                    pickle.dump(self.q_table, f)
                print(f"Best Q-table saved at episode {episode + 1} with reward: {best_reward}")

            # save Q-table periodically
            if (episode + 1) % save_interval == 0:
                filename = f"q_table_ep_{episode + 1}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(self.q_table, f)
                print(f"Q-table saved at episode {episode + 1}")

            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

        self.plot_rewards()

# Moving Average Function
def moving_average(data, window_size=20):
    if len(data) < window_size:
        return np.array(data)  
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def policy_efficiency(rewards, steps):
    """Calculate policy efficiency as avg reward per step, avoiding division by zero."""
    total_steps = np.sum(steps)
    if not rewards or not steps or total_steps == 0:
        return 0  # Avoid division by zero
    return np.sum(rewards) / max(total_steps, 1)  # Use max to prevent tiny values from causing spikes

def plot_rewards(self, window_size=20):
    """Plots total rewards, moving average, policy efficiency, and epsilon decay."""
    if len(self.rewards_history) < 20:
        print("Not enough data to plot (need at least 10 episodes).")
        return

    episodes = np.arange(len(self.rewards_history))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Total rewards per episode
    axes[0].plot(episodes, self.rewards_history, label="Total Reward", color="blue", alpha=0.6)
    axes[0].set_title("Total Rewards per Episode")
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Moving average of rewards (window size = 10)
    if len(self.rewards_history) >= window_size:
        smoothed_rewards = moving_average(self.rewards_history, window_size)
        axes[1].plot(range(len(smoothed_rewards)), smoothed_rewards, color="green", linestyle="dashed", linewidth=2, label="Smoothed Reward")
        axes[1].set_title(f"Moving Average of Rewards ({window_size}-Episode Window)")
        axes[1].set_xlabel("Episodes")
        axes[1].set_ylabel("Smoothed Reward")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

    # Policy efficiency (avg reward per step) - window size = 10
    if len(self.rewards_history) >= window_size and len(self.steps_history) >= window_size:
        avg_rewards_per_step = [
            policy_efficiency(self.rewards_history[max(0, i - window_size):i], self.steps_history[max(0, i - window_size):i])
            for i in range(window_size, len(self.rewards_history))
        ]
        axes[2].plot(range(len(avg_rewards_per_step)), avg_rewards_per_step, color="red", linestyle="solid", linewidth=1.5, label="Policy Efficiency")
        axes[2].set_title("Policy Efficiency (Avg Reward per Step)")
        axes[2].set_xlabel("Episodes")
        axes[2].set_ylabel("Efficiency")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig("sarsa_training_metrics.png", dpi=300)
    print("✅ Plots saved as sarsa_training_metrics.png")

        
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
    print(f"\n🔥 Average Reward over {episodes} episodes: {avg_reward:.2f}")

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


