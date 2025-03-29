import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter

# Load hyperparameters from JSON file
with open("best_params.json", "r") as f:
    params = json.load(f)

# Load trained agent
agent = torch.load('td3_agent.pth')
agent.eval()  # Set to evaluation mode
agent.exploration_phase = 0

# Load test data
data = np.load("data/test_data.npy")  

# Set test parameters
num_episodes = 200  # Small number for quick evaluation
num_stocks = data.shape[1]
initial_cash = 100_000
max_steps = data.shape[0]

# Apply Kalman filter using tuned parameters
data = apply_kalman_filter(data, 
                           observation_covariance=params["observation_covariance"], 
                           transition_covariance=params["transition_covariance"])

# Chaotic Feature Extractor setup
chaotic_extractor = ChaoticFeatureExtractor()
all_chaotic_features = chaotic_extractor.extract_features(data)
chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks

# Environment setup
env = StockEnv(num_stocks=num_stocks, data=data, initial_cash=initial_cash)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

reward_history = []
avg_reward_history = []
critic_loss_history = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    episode_critic_loss = []


    for step in range(max_steps):
        # Select action
        chaotic_features = all_chaotic_features[:, step, :].flatten() # (N*F*C)
        
        action = agent.select_action(
            state=state,
            chaotic_features=chaotic_features,
            current_episode=episode
        )

        print(f"Actions at step {step}: {action}")

        # Step in environment
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        next_chaotic_features = all_chaotic_features[:, step + 1, :].flatten() if step + 1 < max_steps else chaotic_features  # Handle last step case

        # Add transition to replay buffer
        agent.replay_buffer.add(
            (state, chaotic_features, action, reward, next_state,next_chaotic_features, done)
        )

        # Train the agent
        critic_loss,_ = agent.train(batch_size=params['batch_size'], discount=params['discount'], tau=params['tau'])
        episode_critic_loss.append(critic_loss)

        state = next_state

        if done:
            print(f"🚨 Episode {episode} ended early at step {step} due to termination condition.")

            break

    # Logging
    reward_history.append(total_reward)
    avg_reward = np.mean(reward_history[-10:])  # Moving average over last 10 episodes
    avg_reward_history.append(avg_reward)
    avg_critic_loss = np.mean(episode_critic_loss)
    critic_loss_history.append(avg_critic_loss)

    print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {total_reward:.2f}, Avg Reward = {avg_reward:.2f}, Avg Critic Loss = {avg_critic_loss:.4f}")

# Plot rewards
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label="Total Reward")
plt.plot(avg_reward_history, label="Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("TD3 Training Rewards")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(critic_loss_history, label="Critic Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("TD3 Critic Loss Over Time")
plt.legend()
plt.show()