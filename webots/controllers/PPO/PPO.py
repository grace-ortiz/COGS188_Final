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
from webots.controllers.PPO.webots_env import WebotsCarEnv

# Hyperparameters
GAMMA = 0.98        # Discount factor (same as SARSA)
LR = 0.001          # Learning rate
BATCH_SIZE = 64     # Batch size for PPO updates
MEMORY_SIZE = 10000 # Replay memory size
CLIP_EPSILON = 0.2  # Clipping parameter for PPO
EPOCHS = 4          # Number of epochs for PPO updates
TARGET_UPDATE = 10  # Target network update frequency

# State and action limits (from SARSA)
STATE_LIMITS = [
    (0, 150),     # speed
    (-100, 100),  # gps_x
    (-100, 100),  # gps_y
    (0, 100),     # lidar_dist
    (-90, 90),    # lidar_angle
    (0, 80)       # lane_deviation
]

ACTION_LIMITS = [
    (-0.5, 0.5),  # steering angle
    (0.0, 150.0) # speed
]

class PolicyNetwork(nn.Module):
    """Neural network model for PPO policy."""
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

class ValueNetwork(nn.Module):
    """Neural network model for PPO value function."""
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent:
    """PPO agent with experience replay."""
    def __init__(self, env, alpha=LR):
        self.env = env
        self.state_dim = 6  # Speed (1), GPS (2), LiDAR dist (1), LiDAR angle (1), Lane deviation (1)
        self.action_dim = 2  # Steering angle, speed
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.value_net = ValueNetwork(self.state_dim).to(self.device)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=alpha)

    def select_action(self, state):
        """Selects an action using the policy network."""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy_net(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.cpu().numpy()[0]

    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Trains the PPO agent using experience replay."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        # Compute advantages
        with torch.no_grad():
            values = self.value_net(states)
            next_values = self.value_net(next_states)
            advantages = rewards + GAMMA * next_values * (1 - dones) - values

        # PPO updates
        for _ in range(EPOCHS):
            mean, std = self.policy_net(states)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions)
            old_log_probs = log_probs.detach()

            ratio = torch.exp(log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = nn.MSELoss()(values, rewards + GAMMA * next_values * (1 - dones))

            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()

def flatten_state(state):
    """Flattens dictionary state representation to a 1D array."""
    return np.concatenate([
        state["speed"], 
        state["gps"], 
        state["lidar_dist"], 
        state["lidar_angle"], 
        state["lane_deviation"]
    ]).astype(np.float32)

def train_ppo(agent, episodes=2500):
    """Main training loop for PPO."""
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

        agent.rewards_history.append(total_reward)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    plot_training_progress(agent)

def plot_training_progress(agent, window_size=20):
    """Plots total rewards over episodes."""
    plt.figure(figsize=(12, 6))
    plt.plot(agent.rewards_history, label="Total Reward", color='blue', alpha=0.4)

    # Smoothed rewards
    if len(agent.rewards_history) >= window_size:
        smoothed_rewards = np.convolve(agent.rewards_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(agent.rewards_history)), smoothed_rewards, color='green', label="Smoothed Reward", linewidth=2)

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"Rewards\nGamma: {GAMMA}, LR: {LR}, State bins: {STATE_BINS}, Action bins: {ACTION_BINS}")
    plt.grid()
    plt.legend()
    plt.savefig("ppo_rewards_plot.png")
    print("Plot saved as ppo_rewards_plot.png")

# Initialize Webots Environment and Agent
env = WebotsCarEnv()
ppo_agent = PPOAgent(env)

# Train the PPO Agent
train_ppo(ppo_agent)
env.reset()