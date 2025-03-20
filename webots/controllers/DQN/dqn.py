import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from webots_env import WebotsCarEnv

# Hyperparameters
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 64
MEMORY_SIZE = 100000
EPSILON_START = 0.85
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9995
TARGET_UPDATE = 5
EPISODE_LIMIT = 500

MAX_SPEED = 112.65  
STEERING_VALUES = np.linspace(-0.5, 0.5, 3)
SPEED_VALUES = np.linspace(0.0, 100.0, 3)
DISCRETE_ACTIONS = [(steer, speed) for steer in STEERING_VALUES for speed in SPEED_VALUES]

# Moving Average Function
def moving_average(data, window_size=20):
    if len(data) < window_size:
        return np.array(data)  
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Policy Efficiency Function
def policy_efficiency(rewards, steps):
    if not rewards or not steps or np.sum(steps) == 0:
        return 0
    return np.sum(rewards) / np.sum(steps)  # Avg reward per step

# Neural Network Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# DQN Agent
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = 6
        self.action_dim = len(DISCRETE_ACTIONS)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.epsilon_history = []
        self.rewards_history = []
        self.steps_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action_idx = int(torch.argmax(q_values, dim=1).item())
        return action_idx, np.array(DISCRETE_ACTIONS[action_idx])

    def store_experience(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, action_idxs, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(action_idxs, dtype=torch.long).to(self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor([float(d) for d in dones], dtype=torch.float32).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

def flatten_state(state):
    """Flattens the state dictionary into a numpy array."""
    return np.concatenate([
        np.array(state["speed"]).flatten(),
        np.array(state["gps"]).flatten(),
        np.array(state["lidar_dist"]).flatten(),
        np.array(state["lidar_angle"]).flatten(),
        np.array(state["lane_deviation"]).flatten()
    ])

# Training Function
def train_dqn(agent):
    for episode in range(EPISODE_LIMIT):
        state = flatten_state(agent.env.reset())
        total_reward, steps = 0, 0
        done = False

        while not done:
            action_idx, action = agent.select_action(state)
            next_state, reward, done = agent.env.step(action)
            next_state = flatten_state(next_state)

            agent.store_experience(state, action_idx, reward, next_state, done)
            agent.train()
            total_reward += reward
            state = next_state
            steps += 1

        agent.decay_epsilon()
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        agent.rewards_history.append(total_reward)
        agent.steps_history.append(steps)
        agent.epsilon_history.append(agent.epsilon)

        print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}")

        if episode % 50 == 0:
            plot_training_progress(agent)

def plot_training_progress(agent):
    if len(agent.rewards_history) < 20:
        print("Not enough data to plot (need at least 20 episodes).")
        return

    episodes = np.arange(len(agent.rewards_history))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot total reward per episode
    axes[0].plot(episodes, agent.rewards_history, label="Total Reward", color="blue", alpha=0.7)
    axes[0].set_title("Total Rewards per Episode")
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Plot moving average of rewards (only if enough data exists)
    if len(agent.rewards_history) >= 50:
        smoothed_rewards = moving_average(agent.rewards_history, window_size=20)
        axes[1].plot(range(len(smoothed_rewards)), smoothed_rewards, color="green", linestyle="dashed", linewidth=2, label="Smoothed Reward")
        axes[1].set_title("Moving Average of Rewards")
        axes[1].set_xlabel("Episodes")
        axes[1].set_ylabel("Smoothed Reward")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

    # Policy efficiency with rolling window
    window_size = 20  # Adjustable window size for smoother trends
    if len(agent.rewards_history) >= window_size:
        avg_rewards_per_step = [
            policy_efficiency(agent.rewards_history[max(0, i - window_size):i], agent.steps_history[max(0, i - window_size):i])
            for i in range(window_size, len(agent.rewards_history))
        ]
        axes[2].plot(range(len(avg_rewards_per_step)), avg_rewards_per_step, color="red", linestyle="solid", linewidth=1.5, label="Policy Efficiency (Rolling Avg)")
        axes[2].set_title("Policy Efficiency (Avg Reward per Step)")
        axes[2].set_xlabel("Episodes")
        axes[2].set_ylabel("Efficiency")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig("dqn_training_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()  # Optional: Show the plot for quick debugging
    print("✅ Plots saved as dqn_training_metrics.png")

# Run Training
if __name__ == "__main__":
    env = WebotsCarEnv()
    dqn_agent = DQNAgent(env)

    try:
        train_dqn(dqn_agent)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        plot_training_progress(dqn_agent)
        env.reset()
