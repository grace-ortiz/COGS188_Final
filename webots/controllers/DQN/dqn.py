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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from webots_env import WebotsCarEnv

# Hyperparameters
GAMMA = 0.99
LR = 0.0001          # Lower LR for stability
BATCH_SIZE = 64
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.997  # Decay faster so we exploit sooner
TARGET_UPDATE = 3    # Update target network more often
EPISODE_LIMIT = 1000   # 1000 episodes total

MAX_SPEED = 112.65  
STEERING_VALUES = np.linspace(-0.5, 0.5, 3)  # [-0.5, 0.0, 0.5]
SPEED_VALUES = np.linspace(0.0, 100.0, 3)   # [0.0, 50.0, 100.0]
DISCRETE_ACTIONS = [(steer, speed) for steer in STEERING_VALUES for speed in SPEED_VALUES]

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, env, alpha=LR):
        self.env = env
        self.state_dim = 6  # Updated to 6 to match flattened state (speed (1), gps (2), lidar_dist (1), lidar_angle (1), lane_deviation (1))
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

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        # Epsilon-greedy selection over discrete actions.
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action_idx = int(torch.argmax(q_values, dim=1).item())
        # Map the discrete action index to a continuous action.
        continuous_action = np.array(DISCRETE_ACTIONS[action_idx])
        return action_idx, continuous_action

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

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

def flatten_state(state):
    # Original state is a dictionary with keys:
    # "speed", "gps" (2-dim), "lidar_dist", "lidar_angle", "lane_deviation"
    return np.concatenate([
        state["speed"], 
        state["gps"], 
        state["lidar_dist"], 
        state["lidar_angle"], 
        state["lane_deviation"]
    ])

def train_dqn(agent):
    episode = 0

    try:
        while episode < EPISODE_LIMIT:
            state = flatten_state(agent.env.reset())
            total_reward = 0
            episode_steps = 0  # Counter for steps in the episode
            done = False

            # Run the simulation until the environment signals done.
            while not done:
                episode_steps += 1
                action_idx, action = agent.select_action(state)
                next_state, reward, done = agent.env.step(action)
                next_state = flatten_state(next_state)

                agent.store_experience(state, action_idx, reward, next_state, done)
                agent.train()
                total_reward += reward
                state = next_state

                # Optional: Add a safeguard for very long episodes
                if episode_steps >= 10000:  # or some max_steps limit
                    break

            agent.decay_epsilon()
            if episode % TARGET_UPDATE == 0:
                agent.update_target_network()

            agent.rewards_history.append(total_reward)
            agent.epsilon_history.append(agent.epsilon)

            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {episode_steps}, Epsilon: {agent.epsilon:.3f}")
            episode += 1

    except KeyboardInterrupt:
        print("⏹️ Training interrupted manually.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    
    finally:
        print("📊 Saving final plot...")
        plot_training_progress(agent)

def plot_training_progress(agent):
    if not agent.rewards_history:
        print("No rewards data to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.plot(agent.rewards_history, label="Total Reward", alpha=0.8, color='blue')
    plt.title("Rewards Progress")
    plt.legend()
    plt.grid()
    plt.savefig("dqn_rewards_plot.png")
    plt.close()
    print("Plot saved as dqn_rewards_plot.png")

if __name__ == "__main__":
    env = WebotsCarEnv()
    dqn_agent = DQNAgent(env)

    try:
        train_dqn(dqn_agent)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        plot_training_progress(dqn_agent)
        print("Final plot saved.")

    env.reset()