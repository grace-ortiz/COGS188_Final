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
GAMMA = 0.99        # Discount factor
LR = 0.001          # Learning rate
BATCH_SIZE = 64     # Batch size for experience replay
MEMORY_SIZE = 10000 # Replay memory size
EPSILON_START = 1.0 # Initial exploration rate
EPSILON_MIN = 0.05  # Minimum exploration rate
EPSILON_DECAY = 0.995 # Decay rate
TARGET_UPDATE = 10  # Target network update frequency


class DQN(nn.Module):
    """Neural network model for Q-learning."""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values


class DQNAgent:
    """DQN agent with experience replay."""
    def __init__(self, env, alpha=LR):
        self.env = env
        self.state_dim = 6  # Speed (1), GPS (2), LiDAR dist (1), LiDAR angle (1), Lane deviation (1)
        self.action_dim = 2  # Steering angle, speed
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.epsilon_history = []
        self.rewards_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """Selects an action using ε-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.uniform([-0.5, 0.0], [0.5, 250.0])  # Random action in range
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]

    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Trains the DQN using experience replay."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        # Compute current Q-values
        current_q_values = self.policy_net(states)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values.squeeze(), target_q_values.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Updates target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate ε."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


def flatten_state(state):
    """Flattens dictionary state representation to a 1D array."""
    return np.concatenate([
        state["speed"], 
        state["gps"], 
        state["lidar_dist"], 
        state["lidar_angle"], 
        state["lane_deviation"]
    ]).astype(np.float32)


def train_dqn(agent, episodes=1000):
    """Main training loop for DQN."""
    for episode in range(episodes):
        state = flatten_state(agent.env.reset())
        total_reward = 0

        for t in range(500):
            action = agent.select_action(state)
            next_state, reward, done = agent.env.step(action)
            next_state = flatten_state(next_state)

            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            total_reward += reward

            state = next_state
            if done:
                break

        agent.decay_epsilon()

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        agent.rewards_history.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    plot_training_progress(agent)


def plot_training_progress(agent, window_size=20):
    """Plots total rewards and epsilon decay."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot total rewards
    color = 'tab:blue'
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Total Reward", color=color)
    ax1.plot(agent.rewards_history, label="Total Reward", color=color, alpha=0.4)

    # Smoothed rewards
    if len(agent.rewards_history) >= window_size:
        smoothed_rewards = np.convolve(agent.rewards_history, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size - 1, len(agent.rewards_history)), smoothed_rewards, color='green', label="Smoothed Reward", linewidth=2)

    ax1.tick_params(axis='y', labelcolor=color)

    # Plot epsilon on secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel("Epsilon", color=color)
    ax2.plot(agent.epsilon_history, label="Epsilon", color=color, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Rewards and Epsilon Decay\nGamma: {GAMMA}, LR: {LR}, State bins: 6, Action bins: 2")
    fig.tight_layout()

    plt.grid()
    plt.savefig("dqn_rewards_epsilon_plot.png")
    print("Plot saved as dqn_rewards_epsilon_plot.png")


# Initialize Webots Environment and Agent
env = WebotsCarEnv()
dqn_agent = DQNAgent(env)

# Train the DQN Agent
train_dqn(dqn_agent)
env.reset()