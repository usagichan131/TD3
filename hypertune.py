import optuna
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from tqdm import tqdm   # ✅ NEW

from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter


# =========================
# VALIDATION FUNCTION
# =========================
def evaluate_agent(agent, env, chaotic_features, num_stocks, lookback_window):
    state = env.reset()
    total_reward = 0
    max_steps = env.data.shape[0]

    with torch.no_grad():  # ✅ tránh leak memory
        for step in range(max_steps - lookback_window + 1):
            current_state_sequence = state[:num_stocks * lookback_window * (env.data.shape[-1])].reshape(
                lookback_window, num_stocks * (env.data.shape[-1])
            )

            chaotic_features_sequence = chaotic_features[:, step:step + lookback_window, :].reshape(
                lookback_window, -1
            )

            action = agent.select_action(
                state=current_state_sequence,
                chaotic_features=chaotic_features_sequence,
                current_episode=9999
            )

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                break

    return total_reward


# =========================
# OBJECTIVE FUNCTION
# =========================
def objective(trial):
    # Load data
    train_data = np.load("data/train_data.npy")
    val_data = np.load("data/val_data.npy")

    num_stocks = train_data.shape[1]
    initial_cash = 100_000
    lookback_window = 15

    # Hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 384, 512])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    discount = trial.suggest_float("discount", 0.95, 0.995, step=0.005)
    tau = trial.suggest_float("tau", 5e-4, 5e-3, log=True)
    exploration_phase = trial.suggest_int("exploration_phase", 100, 300, step=20)

    observation_covariance = trial.suggest_float("observation_covariance", 0.1, 3.0, log=True)
    transition_covariance = trial.suggest_float("transition_covariance", 0.01, 0.5, log=True)

    num_episodes = 400

    # =========================
    # Data processing
    # =========================
    filtered_train = apply_kalman_filter(train_data, observation_covariance, transition_covariance)
    filtered_val = apply_kalman_filter(val_data, observation_covariance, transition_covariance)

    chaotic_extractor = ChaoticFeatureExtractor()
    chaotic_train = chaotic_extractor.extract_features(filtered_train)
    chaotic_val = chaotic_extractor.extract_features(filtered_val)

    chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks

    # =========================
    # Environments
    # =========================
    train_env = StockEnv(num_stocks=num_stocks, data=filtered_train, initial_cash=initial_cash)
    val_env = StockEnv(num_stocks=num_stocks, data=filtered_val, initial_cash=initial_cash)

    action_dim = train_env.action_space.shape[0]

    # =========================
    # Agent
    # =========================
    agent = TD3(
        state_dim=num_stocks * (filtered_train.shape[-1]),
        chaotic_feature_dim=chaotic_feature_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_stocks=num_stocks,
        max_action=1.0,
        env_action_space_high=1.0,
        env_action_space_low=0.0
    )

    agent.exploration_phase = exploration_phase
    max_steps = filtered_train.shape[0]

    reward_history = []

    # =========================
    # TRAIN LOOP (WITH TQDM)
    # =========================
    episode_bar = tqdm(range(num_episodes), desc=f"Trial {trial.number}", leave=False)

    for episode in episode_bar:
        state = train_env.reset()
        total_reward = 0

        for step in range(max_steps - lookback_window + 1):
            current_state_sequence = state[:num_stocks * lookback_window * (filtered_train.shape[-1])].reshape(
                lookback_window, num_stocks * (filtered_train.shape[-1])
            )

            chaotic_features_sequence = chaotic_train[:, step:step + lookback_window, :].reshape(
                lookback_window, -1
            )

            action = agent.select_action(
                state=current_state_sequence,
                chaotic_features=chaotic_features_sequence,
                current_episode=episode
            )

            next_state, reward, done, _ = train_env.step(action)
            total_reward += reward

            next_state_sequence = next_state[:num_stocks * lookback_window * (filtered_train.shape[-1])].reshape(
                lookback_window, num_stocks * (filtered_train.shape[-1])
            )

            next_chaotic_features_sequence = chaotic_train[:, step + 1:step + lookback_window + 1, :].reshape(
                lookback_window, -1
            ) if step + 1 < max_steps - lookback_window + 1 else chaotic_features_sequence

            agent.replay_buffer.add(
                (current_state_sequence, chaotic_features_sequence, action, reward,
                 next_state_sequence, next_chaotic_features_sequence, done)
            )

            agent.train(batch_size=batch_size, discount=discount, tau=tau)

            state = next_state

            if done:
                break

        reward_history.append(total_reward)

        # Update tqdm bar
        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            episode_bar.set_postfix({
                "reward": f"{total_reward:.2f}",
                "avg10": f"{avg_reward:.2f}"
            })

    # =========================
    # VALIDATION
    # =========================
    val_reward = evaluate_agent(agent, val_env, chaotic_val, num_stocks, lookback_window)

    print(f"\n[Trial {trial.number}] VALIDATION REWARD: {val_reward:.2f}")

    return val_reward


# =========================
# RUN OPTUNA
# =========================
def run_optimization(n_trials=10):
    study = optuna.create_study(direction="maximize")

    print(f"Starting optimization with {n_trials} trials...\n")

    study.optimize(objective, n_trials=n_trials)

    print("\nBest trial:")
    print(f"Value: {study.best_trial.value}")
    print("Params:", study.best_trial.params)

    return study.best_trial.params


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    best_params = run_optimization(n_trials=5)
    print("Done.")