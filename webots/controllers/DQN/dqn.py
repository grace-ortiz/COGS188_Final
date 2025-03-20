#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
from collections import deque
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from webots_env import WebotsCarEnv

# Hyperparameters
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 64
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9995
TARGET_UPDATE = 5
EPISODE_LIMIT = 1000

MAX_SPEED = 112.65  
STEERING_VALUES = np.linspace(-0.5, 0.5, 3)
SPEED_VALUES = np.linspace(0.0, 100.0, 3)
DISCRETE_ACTIONS = [(steer, speed) for steer in STEERING_VALUES for speed in SPEED_VALUES]

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

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = 6
        self.action_dim = len(DISCRETE_ACTIONS)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.epsilon_history = []
        self.rewards_history = []
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
    return np.concatenate([
        state["speed"], state["gps"], state["lidar_dist"], state["lidar_angle"], state["lane_deviation"]
    ])

def plot_training_progress(agent):
    plt.figure(figsize=(12, 6))
    episodes = range(len(agent.rewards_history))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1 = ax1 = plt.gca()
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(agent.rewards_history, color=color, alpha=0.4, label='Total Reward')

    # Smooth rewards
    if len(agent.rewards_history) > 20:
        smoothed_rewards = np.convolve(agent.rewards_history, np.ones(20)/20, mode='valid')
        ax1.plot(smoothed_rewards, color=color, label='Smoothed Reward')

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = 'tab:orange'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(agent.epsilon_history, linestyle='--', color=color, label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    plt.title('DQN Training Progress')
    plt.grid(True)
    plt.legend()
    plt.savefig("dqn_rewards_epsilon_plot.png")
    plt.close()
    print("Plot saved as dqn_rewards_epsilon_plot.png")

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

            if steps > 10000:
                break

        agent.decay_epsilon()
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        # periodic epsilon reset to boost exploration occasionally
        if episode % 100 == 0 and agent.epsilon < 0.2:
            agent.epsilon = min(EPSILON_START, agent.epsilon + 0.3)
            print(f"Epsilon reset at episode {episode}: epsilon={agent.epsilon:.2f}")

        agent.rewards_history.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}")

        if episode % 50 == 0:
            plot_training_progress(agent)  # save intermediate plots

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
