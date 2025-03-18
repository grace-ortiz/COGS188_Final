import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
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
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
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


def train_dqn():
    """Main training loop for DQN."""
    env = WebotsCarEnv()
    state_dim = 6  # Speed (1), GPS (2), LiDAR dist (1), LiDAR angle (1), Lane deviation (1)
    action_dim = 2  # Steering angle, speed

    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = flatten_state(env.reset())
        total_reward = 0

        for t in range(500):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
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

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    env.close()


if __name__ == "__main__":
    train_dqn()