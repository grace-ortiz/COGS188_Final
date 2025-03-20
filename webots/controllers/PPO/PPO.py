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
GAMMA = 0.99         # Discount factor
LR = 0.0003          # Learning rate
BATCH_SIZE = 256     # Batch size for PPO updates
MEMORY_SIZE = 10000  # Replay memory size
CLIP_EPSILON = 0.2   # Clipping parameter for PPO
EPOCHS = 5          # Number of epochs for PPO updates

# State and action limits
STATE_DIM = 6         # Speed, GPS(x,y), LiDAR dist, LiDAR angle, lane deviation
ACTION_DIM = 2         # Steering angle, speed
MAX_EPISODES = 1000   # Number of training episodes
MAX_TIMESTEPS = 10000  # Max timesteps per episode
ACTION_LOW = np.array([-0.5, 0.0])    # Steering range: [-0.5, 0.5], Speed range: [0, 150]
ACTION_HIGH = np.array([0.5, 150.0])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
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
        std = torch.exp(self.log_std).clamp(0.01, 1.0)
        return mean, std


class ValueNetwork(nn.Module):
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
    def __init__(self, env):
        self.env = env
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Networks
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim).to(DEVICE)
        self.value_net = ValueNetwork(self.state_dim).to(DEVICE)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=LR
        )

        self.rewards_history = []

    def select_action(self, state):
        """Selects an action using the policy network with exploration noise."""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE).unsqueeze(0)
        mean, std = self.policy_net(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        
        # Sample raw action and calculate log probability on the raw action
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)  # sum over action dimensions
        
        # Apply tanh to squash the action to [-1, 1]
        squashed_action = torch.tanh(raw_action)
        mapped_action = map_action(squashed_action.cpu().numpy().squeeze())

        return mapped_action, log_prob.item()

    def store_experience(self, state, action, reward, next_state, done, log_prob):
        """Stores experience in memory."""
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def train(self):
        """Trains the PPO agent."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones, log_probs = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(DEVICE).view(-1, 1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(DEVICE).view(-1, 1)
        old_log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(DEVICE).view(-1, 1)

        with torch.no_grad():
            target_values = rewards + GAMMA * self.value_net(next_states) * (1 - dones)
            advantages = target_values - self.value_net(states)

        for _ in range(EPOCHS):
            mean, std = self.policy_net(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)

            # PPO loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped_ratios = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            
            advantage = advantages.detach()

            surrogate1 = ratios * advantage
            surrogate2 = clipped_ratios * advantage
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value loss
            value_loss = nn.MSELoss()(self.value_net(states), target_values)

            # Optimize
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()


def flatten_state(state):
    """Flattens dictionary state representation into a 1D array."""
    return np.concatenate([
        state["speed"], 
        state["gps"], 
        state["lidar_dist"], 
        state["lidar_angle"], 
        state["lane_deviation"]
    ]).astype(np.float32)
    
def map_action(action):
    """Maps action from [-1, 1] to the valid action range."""
    action = np.clip(action, -1.0, 1.0)  
    mapped_action = np.zeros_like(action)

    # Steering angle mapping
    mapped_action[0] = ACTION_LOW[0] + (0.5 * (action[0] + 1.0) * (ACTION_HIGH[0] - ACTION_LOW[0]))

    # Speed mapping
    mapped_action[1] = ACTION_LOW[1] + (0.5 * (action[1] + 1.0) * (ACTION_HIGH[1] - ACTION_LOW[1]))

    return mapped_action

def train_ppo(agent, episodes=MAX_EPISODES, plot_interval=50):
    """Main training loop for PPO with periodic plotting."""
    for episode in range(episodes):
        state = flatten_state(agent.env.reset())
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done = agent.env.step(action)
            next_state = flatten_state(next_state)

            agent.store_experience(state, action, reward, next_state, done, log_prob)
            agent.train()

            state = next_state
            total_reward += reward
            steps += 1

            if steps > 10000:
                break

        agent.rewards_history.append(total_reward)

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {steps}")

        # Save plot periodically
        if episode % plot_interval == 0:
            plot_training_progress(agent, filename=f"ppo_rewards_plot_ep{episode}.png")

    # Save final plot
    plot_training_progress(agent, filename="ppo_rewards_plot_final.png")


def plot_training_progress(agent, window_size=20, filename="ppo_rewards_plot.png"):
    """Plots total rewards over episodes and saves the figure."""
    plt.figure(figsize=(12, 6))
    plt.plot(agent.rewards_history, label="Total Reward", color='blue', alpha=0.4)

    if len(agent.rewards_history) >= window_size:
        smoothed_rewards = np.convolve(
            agent.rewards_history, np.ones(window_size) / window_size, mode='valid'
        )
        plt.plot(range(window_size - 1, len(agent.rewards_history)), smoothed_rewards, 
                 color='green', label="Smoothed Reward", linewidth=2)

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("PPO Rewards Over Episodes")
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    plt.close()  # close the figure to free memory
    print(f"Plot saved as {filename}")


# Initialize environment and agent, then start training
env = WebotsCarEnv()
ppo_agent = PPOAgent(env)

train_ppo(ppo_agent)
env.reset()