import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, chaotic_feature_dim, portfolio_dim, hidden_size, num_layers, num_stocks):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(state_dim + chaotic_feature_dim, hidden_size, num_layers, batch_first=True)
        self.fc_portfolio = nn.Linear(hidden_size + portfolio_dim, hidden_size)
        self.fc_selection = nn.Linear(hidden_size, num_stocks)  # Stock selection
        self.fc_allocation = nn.Linear(hidden_size, num_stocks)  # Cash allocation
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state, chaotic_features, portfolio_state, hidden=None):
        # Combine stock state and chaotic features
        augmented_state = torch.cat([state, chaotic_features], dim=2)
        lstm_out, hidden = self.lstm(augmented_state, hidden)
        lstm_out = lstm_out[:, -1, :]  # Take the last timestep

        # Combine LSTM output with portfolio state
        portfolio_combined = torch.cat([lstm_out, portfolio_state], dim=1)
        fc_out = torch.relu(self.fc_portfolio(portfolio_combined))

        # Stock selection 
        stock_selection = self.tanh(self.fc_selection(fc_out))

        # Cash allocation (softmax for proportions)
        allocation = self.softmax(self.fc_allocation(fc_out))

        # Mask allocation with stock selection and re-normalize
        allocation = allocation * stock_selection
        allocation = allocation / allocation.sum(dim=1, keepdim=True)

        return stock_selection, allocation, hidden

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, chaotic_feature_dim, portfolio_dim, action_dim, hidden_size, num_layers):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(state_dim + chaotic_feature_dim, hidden_size, num_layers, batch_first=True)
        self.fc_concat = nn.Linear(hidden_size + portfolio_dim + action_dim, 256)
        self.fc_q1 = nn.Linear(256, 256)
        self.fc_q2 = nn.Linear(256, 256)
        self.fc_q1_out = nn.Linear(256, 1)
        self.fc_q2_out = nn.Linear(256, 1)

    def forward(self, state, chaotic_features, portfolio_state, action, hidden=None):
        # Combine stock state and chaotic features
        augmented_state = torch.cat([state, chaotic_features], dim=2)
        lstm_out, hidden = self.lstm(augmented_state, hidden)
        lstm_out = lstm_out[:, -1, :]  # Take the last timestep

        # Concatenate LSTM output, portfolio state, and action
        combined = torch.cat([lstm_out, portfolio_state, action], dim=1)
        concat_out = torch.relu(self.fc_concat(combined))

        # Q-value estimation
        q1 = torch.relu(self.fc_q1(concat_out))
        q1 = self.fc_q1_out(q1)
        q2 = torch.relu(self.fc_q2(concat_out))
        q2 = self.fc_q2_out(q2)

        return q1, q2

    def Q1(self, state, chaotic_features, portfolio_state, action, hidden=None):
        augmented_state = torch.cat([state, chaotic_features], dim=2)
        lstm_out, hidden = self.lstm(augmented_state, hidden)
        lstm_out = lstm_out[:, -1, :]
        combined = torch.cat([lstm_out, portfolio_state, action], dim=1)
        concat_out = torch.relu(self.fc_concat(combined))
        q1 = torch.relu(self.fc_q1(concat_out))
        return self.fc_q1_out(q1)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, chaotic_features, portfolio_states, actions, rewards, next_states, next_chaotic_features, next_portfolio_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(chaotic_features, dtype=torch.float32),
            torch.tensor(portfolio_states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(next_chaotic_features, dtype=torch.float32),
            torch.tensor(next_portfolio_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )
#TD3 algo
class TD3:
    def __init__(self, state_dim, chaotic_feature_dim, portfolio_dim, action_dim, hidden_size, num_layers, num_stocks, max_action, env_action_space_high, env_action_space_low):
        self.exploration_phase = 365  # Number of episodes for chaotic exploration
        self.actor = Actor(state_dim, chaotic_feature_dim, portfolio_dim, hidden_size, num_layers, num_stocks)
        self.actor_target = Actor(state_dim, chaotic_feature_dim, portfolio_dim, hidden_size, num_layers, num_stocks)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, chaotic_feature_dim, portfolio_dim, action_dim, hidden_size, num_layers)
        self.critic_target = Critic(state_dim, chaotic_feature_dim, portfolio_dim, action_dim, hidden_size, num_layers)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(size=1000000)

        self.max_action = max_action
        self.env_action_space_high = env_action_space_high
        self.env_action_space_low = env_action_space_low

        self.policy_noise = 0.2  # Noise standard deviation
        self.noise_clip = 0.5  # Noise clipping
        self.policy_delay = 2  # Delayed policy updates
        self.total_it = 0

        # Chaotic noise parameters
        self.chaotic_map_state = np.random.rand()  # Initial state for the chaotic map

    def chaotic_noise(self):
        """Logistic map to generate chaotic noise."""
        r = 3.99  # Chaos parameter
        self.chaotic_map_state = r * self.chaotic_map_state * (1 - self.chaotic_map_state)
        return self.chaotic_map_state

    def select_action(self, state, chaotic_features, portfolio_state, current_episode=0, hidden=None):
        """Select action with or without chaotic noise during exploration."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        chaotic_features = torch.tensor(chaotic_features, dtype=torch.float32).unsqueeze(0)
        portfolio_state = torch.tensor(portfolio_state, dtype=torch.float32).unsqueeze(0)
        
        # Generate actions using the actor network
        stock_selection, allocation, _ = self.actor(state, chaotic_features, portfolio_state, hidden)
        actions = torch.cat([stock_selection, allocation], dim=1).detach().numpy().flatten()

        # Add chaotic noise during exploration phase
        if current_episode < self.exploration_phase:
            noise = np.array([self.chaotic_noise() for _ in range(actions.shape[0])])
            actions += noise

        # Clip actions to valid range
        return np.clip(actions, self.env_action_space_low, self.env_action_space_high)

    def train(self, batch_size=100, discount=0.99, tau=0.005):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        # Sample from the replay buffer
        states, chaotic_features, portfolio_states, actions, rewards, next_states, next_chaotic_features, next_portfolio_states, dones = self.replay_buffer.sample(batch_size)

        # Compute target Q-values
        with torch.no_grad():
            noise = torch.clamp(
                torch.normal(0, self.policy_noise, size=actions.shape), -self.noise_clip, self.noise_clip
            ).to(actions.device)

            # Generate next actions using the target actor
            stock_selection, allocation, _ = self.actor_target(next_states, next_chaotic_features, next_portfolio_states)
            next_actions = torch.cat([stock_selection, allocation], dim=1) + noise
            next_actions = next_actions.clamp(self.env_action_space_low, self.env_action_space_high)

            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_chaotic_features, next_portfolio_states, next_actions)
            target_q = rewards + discount * (1 - dones) * torch.min(target_q1, target_q2)

        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, chaotic_features, portfolio_states, actions)

        # Compute critic loss
        critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + torch.nn.functional.mse_loss(current_q2, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            stock_selection, allocation, _ = self.actor(states, chaotic_features, portfolio_states)
            actions = torch.cat([stock_selection, allocation], dim=1)
            actor_loss = -self.critic.Q1(states, chaotic_features, portfolio_states, actions).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.total_it += 1
