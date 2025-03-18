import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
from webots_env import WebotsCarEnv
from torch.distributions import Normal

# Hyperparameters
GAMMA = 0.99        # Discount factor
LR = 0.0003         # Learning rate
CLIP_EPSILON = 0.2  # Clipping parameter for PPO
EPOCHS = 4          # Number of epochs for training
BATCH_SIZE = 64     # Batch size for training
MEMORY_SIZE = 10000 # Replay memory size


class PolicyNetwork(nn.Module):
    """Neural network model for PPO policy."""
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, output_dim)  # Mean of the action distribution
        self.log_std = nn.Parameter(torch.zeros(output_dim))  # Log standard deviation

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))  # Output mean in the range [-1, 1]
        std = torch.exp(self.log_std)  # Standard deviation
        return mu, std


class ValueNetwork(nn.Module):
    """Neural network model for PPO value function."""
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Outputs a single value (state value)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PPOAgent:
    """PPO agent with experience replay."""
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=LR
        )
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """Selects an action using the policy network."""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        mu, std = self.policy_net(state_tensor)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.cpu().numpy()[0], log_prob.cpu().item()

    def store_experience(self, state, action, log_prob, reward, done):
        """Stores experience in memory."""
        self.memory.append((state, action, log_prob, reward, done))

    def compute_returns(self, rewards, dones):
        """Computes discounted returns."""
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + GAMMA * discounted_reward
            returns.insert(0, discounted_reward)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def train(self):
        """Trains the PPO agent."""
        if len(self.memory) < BATCH_SIZE:
            return

        # Unpack memory
        states, actions, log_probs_old, rewards, dones = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32).to(self.device)
        returns = self.compute_returns(rewards, dones)
        values = self.value_net(states).squeeze()

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute advantages
        advantages = returns - values.detach()

        # Train for multiple epochs
        for _ in range(EPOCHS):
            # Shuffle indices for mini-batch updates
            indices = np.arange(len(self.memory))
            np.random.shuffle(indices)

            for start in range(0, len(self.memory), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]

                # Sample mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Compute new log probabilities and values
                mu, std = self.policy_net(batch_states)
                dist = Normal(mu, std)
                log_probs_new = dist.log_prob(batch_actions).sum(-1)
                values_new = self.value_net(batch_states).squeeze()

                # Compute policy loss (clipped objective)
                ratio = torch.exp(log_probs_new - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss (MSE)
                value_loss = self.loss_fn(values_new, batch_returns)

                # Compute entropy (for exploration)
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear memory after training
        self.memory = []


def flatten_state(state):
    """Flattens dictionary state representation to a 1D array."""
    return np.concatenate([
        state["speed"], 
        state["gps"], 
        state["lidar_dist"], 
        state["lidar_angle"], 
        state["lane_deviation"]
    ]).astype(np.float32)


def train_ppo():
    """Main training loop for PPO."""
    env = WebotsCarEnv()
    state_dim = 6  # Speed (1), GPS (2), LiDAR dist (1), LiDAR angle (1), Lane deviation (1)
    action_dim = 2  # Steering angle, speed

    agent = PPOAgent(state_dim, action_dim)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = flatten_state(env.reset())
        total_reward = 0

        for t in range(500):
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            next_state = flatten_state(next_state)

            agent.store_experience(state, action, log_prob, reward, done)
            total_reward += reward

            state = next_state
            if done:
                break

        # Train after each episode
        agent.train()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    train_ppo()