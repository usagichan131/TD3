import optuna
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter

def objective(trial):
    # Load data
    data = np.load("data/full_data_ver5_4stocks.npy")
    
    # Hyperparameters to optimize
    hidden_size = trial.suggest_int("hidden_size", 64, 512, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_int("batch_size", 32, 256, log=True)
    discount = trial.suggest_float("discount", 0.9, 0.99)
    tau = trial.suggest_float("tau", 1e-4, 1e-2, log=True)
    exploration_phase = trial.suggest_int("exploration_phase", 200, 1000)
    
    # Kalman filter hyperparameters
    observation_covariance = trial.suggest_float("observation_covariance", 0.1, 10.0, log=True)
    transition_covariance = trial.suggest_float("transition_covariance", 0.01, 1.0, log=True)
    
    # Fixed parameters
    num_stocks = data.shape[1]
    initial_cash = 100_000
    num_episodes = 200  # Reduced for faster optimization
    max_steps = data.shape[0]
    
    # Chaotic Feature Extractor setup
    chaotic_extractor = ChaoticFeatureExtractor()
    all_chaotic_features = chaotic_extractor.extract_features(data)
    chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks
    
    # Kalman filter application
    filtered_data = apply_kalman_filter(
        data,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    
    # Environment setup
    env = StockEnv(num_stocks=num_stocks, data=filtered_data, initial_cash=initial_cash)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # TD3 Agent setup
    max_action = 1.0
    agent = TD3(
        state_dim=state_dim,
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
    
    # Logging
    reward_history = []
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select action
            chaotic_features = all_chaotic_features[:, step, :].flatten()
            
            action = agent.select_action(
                state=state,
                chaotic_features=chaotic_features,
                current_episode=episode
            )
            
            # Step in environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            next_chaotic_features = all_chaotic_features[:, step + 1, :].flatten() if step + 1 < max_steps else chaotic_features
            
            # Add transition to replay buffer
            agent.replay_buffer.add(
                (state, chaotic_features, action, reward, next_state, next_chaotic_features, done)
            )
            
            # Train the agent
            critic_loss, _ = agent.train(batch_size=batch_size, discount=discount, tau=tau)
            
            state = next_state
            
            if done:
                break
        
        reward_history.append(total_reward)
        
        # Report intermediate results to Optuna
        if episode % 10 == 0 and episode > 0:
            trial.report(np.mean(reward_history[-10:]), episode)
            
            # Pruning (early stopping of unpromising trials)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
            print(f"Trial {trial.number}, Episode {episode}: Avg Reward = {avg_reward:.2f}")
    
    # Return average reward over the last 50 episodes (or all if fewer than 50)
    final_reward = np.mean(reward_history[-50:]) if len(reward_history) >= 50 else np.mean(reward_history)
    return final_reward

def run_optimization(n_trials=50, study_name=None, storage=None, load_if_exists=True):
    # Create study name if not provided
    if study_name is None:
        study_name = f"td3_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=100)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Print and save results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters to file
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save best parameters
    best_params_file = os.path.join(results_dir, "best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump(trial.params, f, indent=2)
    
    # Generate optimization visualization plots
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(results_dir, f"{study_name}_history.png"))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(results_dir, f"{study_name}_importance.png"))
    
    fig = optuna.visualization.plot_contour(study)
    fig.write_image(os.path.join(results_dir, f"{study_name}_contour.png"))
    
    return study.best_params

def train_with_best_params(best_params):
    """Train a model with the best parameters found by Optuna"""
    # Load your original training code here but use best_params
    # This can be imported from your train.py if refactored
    print("Training with best parameters:", best_params)
    
    # Here you would call your training function with the best parameters
    # Example: train_model(**best_params)

if __name__ == "__main__":
    # SQLite storage for persistence (optional)
    # storage = "sqlite:///optuna_studies.db"
    
    # For simple testing without database:
    storage = None
    
    # Run the optimization
    best_params = run_optimization(
        n_trials=50,  # Adjust based on your compute resources and time constraints
        storage=storage
    )
    
    # Optionally train a final model with the best parameters
    # train_with_best_params(best_params)
