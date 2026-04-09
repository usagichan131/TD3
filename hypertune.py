import torch
import numpy as np
import os
import optuna
from collections import deque
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
from typing import Dict, Any

# Import your custom modules
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TD3HyperparameterTuner:
    def __init__(self, data_path: str, initial_cash: float = 100_000):
        """
        Initialize the TD3 hyperparameter tuner.
        
        Args:
            data_path: Path to the training data
            initial_cash: Initial cash for trading
        """
        self.data = np.load(data_path)
        self.initial_cash = initial_cash
        self.num_stocks = self.data.shape[1]
        self.max_steps = self.data.shape[0]
        
        # Setup chaotic feature extractor
        self.chaotic_extractor = ChaoticFeatureExtractor()
        self.all_chaotic_features = self.chaotic_extractor.extract_features(self.data)
        self.chaotic_feature_dim = self.chaotic_extractor.output_dim * self.num_stocks
        
        # Store original data - Kalman filter will be applied with hypertuned parameters in each trial
        self.original_data = self.data.copy()
        
        # Create results directory
        os.makedirs('./hypertuning_results', exist_ok=True)
        
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Number of stocks: {self.num_stocks}")
        logger.info(f"Chaotic feature dimension: {self.chaotic_feature_dim}")
        
        print(f"🎯 TD3 Hyperparameter Tuner Initialized")
        print(f"   📊 Data shape: {self.data.shape}")
        print(f"   🏭 Number of stocks: {self.num_stocks}")
        print(f"   🌀 Chaotic feature dimension: {self.chaotic_feature_dim}")
        print(f"   💰 Initial cash: ${self.initial_cash:,}")
        print(f"   📁 Results directory: ./hypertuning_results/")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Average reward over last 20% of episodes
        """
        try:
            print(f"\n🔬 Starting Trial {trial.number}")
            print("=" * 50)
            
            # Sample hyperparameters
            print(f"🎯 Sampling hyperparameters...")
            params = self.sample_hyperparameters(trial)
            
            print(f"📋 Trial {trial.number} Parameters:")
            for param, value in params.items():
                print(f"   {param}: {value}")
            
            # Apply Kalman filter with trial parameters
            print(f"🔧 Applying Kalman filter with trial parameters...")
            print(f"   Observation covariance: {params['observation_covariance']:.4f}")
            print(f"   Transition covariance: {params['transition_covariance']:.4f}")
            filtered_data = apply_kalman_filter(
                self.original_data, 
                params['observation_covariance'], 
                params['transition_covariance']
            )
            
            # Setup environment
            print(f"🏢 Setting up environment...")
            env = StockEnv(
                num_stocks=self.num_stocks, 
                data=filtered_data,  # Use filtered data
                initial_cash=self.initial_cash
            )
            
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            print(f"   State dim: {state_dim}, Action dim: {action_dim}")
            
            # Initialize TD3 agent with sampled parameters
            print(f"🤖 Initializing TD3 agent...")
            agent = TD3(
                state_dim=self.num_stocks * (filtered_data.shape[-1]),  # Use filtered data
                chaotic_feature_dim=self.chaotic_feature_dim,
                action_dim=action_dim,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                num_stocks=self.num_stocks,
                max_action=1.0,
                env_action_space_high=1.0,
                env_action_space_low=0.0
            )
            
            agent.exploration_phase = params['exploration_phase']
            print(f"   Agent initialized with {params['hidden_size']} hidden units, {params['num_layers']} layers")
            
            # Training loop
            print(f"🚀 Starting training for {params['num_episodes']} episodes...")
            reward_history = []
            
            for episode in range(params['num_episodes']):
                if episode % 10 == 0 or episode < 5:
                    print(f"   📍 Episode {episode + 1}/{params['num_episodes']}")
                
                state = env.reset()
                total_reward = 0
                steps_completed = 0
                
                for step in range(self.max_steps - params['lookback_window'] + 1):
                    # Progress indicator for longer episodes
                    if step % 200 == 0 and step > 0:
                        progress = (step / (self.max_steps - params['lookback_window'])) * 100
                        print(f"      Step {step}/{self.max_steps - params['lookback_window']} ({progress:.1f}%)")
                    
                    # Extract state sequences - FIXED: Use filtered_data
                    current_state_sequence = state[:self.num_stocks * params['lookback_window'] * (filtered_data.shape[-1])].reshape(
                        params['lookback_window'], self.num_stocks * (filtered_data.shape[-1])
                    )
                    
                    chaotic_features_sequence = self.all_chaotic_features[:, step:step + params['lookback_window'], :].reshape(
                        params['lookback_window'], -1
                    )
                    
                    if chaotic_features_sequence.dtype == np.dtype('O'):
                        chaotic_features_sequence = np.array(chaotic_features_sequence, dtype=np.float32)
                    
                    # Select action
                    action = agent.select_action(
                        state=current_state_sequence,
                        chaotic_features=chaotic_features_sequence,
                        current_episode=episode
                    )
                    
                    # Step in environment
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    
                    # Extract next state sequences - FIXED: Use filtered_data
                    next_state_sequence = next_state[:self.num_stocks * params['lookback_window'] * (filtered_data.shape[-1])].reshape(
                        params['lookback_window'], self.num_stocks * (filtered_data.shape[-1])
                    )
                    next_chaotic_features_sequence = (
                        self.all_chaotic_features[:, step + 1:step + params['lookback_window'] + 1, :].reshape(params['lookback_window'], -1)
                        if step + 1 < self.max_steps - params['lookback_window'] + 1
                        else chaotic_features_sequence
                    )
                    
                    # Add to replay buffer
                    agent.replay_buffer.add((
                        current_state_sequence, chaotic_features_sequence, action, reward,
                        next_state_sequence, next_chaotic_features_sequence, done
                    ))
                    
                    # Train agent
                    agent.train(
                        batch_size=params['batch_size'],
                        discount=params['discount'],
                        tau=params['tau']
                    )
                    
                    state = next_state
                    steps_completed = step + 1
                    
                    if done:
                        print(f"      Episode {episode + 1} ended early at step {step + 1}")
                        break
                
                reward_history.append(total_reward)
                
                # Print episode results
                if episode % 10 == 0 or episode < 5 or episode == params['num_episodes'] - 1:
                    avg_reward = np.mean(reward_history[-min(10, len(reward_history)):])
                    print(f"      Episode {episode + 1} reward: {total_reward:.2f}, Avg last 10: {avg_reward:.2f}")
                
                # Early stopping based on performance
                if episode > 20 and episode % 10 == 0:
                    recent_avg = np.mean(reward_history[-10:])
                    print(f"      Recent average (last 10 episodes): {recent_avg:.2f}")
                    if recent_avg < -50000:  # Stop if performance is too poor
                        print(f"🚨 Trial {trial.number}: Early stopping due to poor performance (avg: {recent_avg:.2f})")
                        return recent_avg
            
            # Return average reward over last 20% of episodes for stability
            eval_episodes = max(1, int(0.2 * len(reward_history)))
            final_performance = np.mean(reward_history[-eval_episodes:])
            
            print(f"✅ Trial {trial.number} completed!")
            print(f"   Total episodes: {len(reward_history)}")
            print(f"   Final performance (avg last {eval_episodes} episodes): {final_performance:.2f}")
            print(f"   Best episode reward: {max(reward_history):.2f}")
            print(f"   Worst episode reward: {min(reward_history):.2f}")
            
            return final_performance
            
        except Exception as e:
            print(f"❌ Trial {trial.number} failed with error: {str(e)}")
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return -100000  # Return very poor performance for failed trials

    def sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters for the trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        return {
            # Network architecture
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512, 1024]),
            'num_layers': trial.suggest_int('num_layers', 2, 5),
            
            # Training parameters
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'discount': trial.suggest_float('discount', 0.9, 0.99, step=0.01),
            'tau': trial.suggest_float('tau', 1e-4, 1e-2, log=True),
            'exploration_phase': trial.suggest_int('exploration_phase', 20, 120),
            
            # Environment parameters
            'lookback_window': 15,  # Fixed as per original training
            'num_episodes': 120,
            'observation_covariance': trial.suggest_float("observation_covariance", 0.1, 3.0, log=True),
            'transition_covariance': trial.suggest_float("transition_covariance", 0.01, 0.5, log=True)
        }
               

    def optimize(self, n_trials: int = 100, study_name: str = None) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials to run
            study_name: Name of the study
            
        Returns:
            Optuna study object
        """
        if study_name is None:
            study_name = f"TD3_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🎯 STARTING HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        print(f"📊 Study name: {study_name}")
        print(f"🔢 Total trials: {n_trials}")
        print(f"📈 Data shape: {self.data.shape}")
        print(f"💰 Initial cash: ${self.initial_cash:,}")
        print("=" * 60)
        
        # Create study
        print(f"🔬 Creating Optuna study...")
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20,
                interval_steps=10
            )
        )
        print(f"   ✅ Study created with TPE sampler and Median pruner")
        
        logger.info(f"Starting optimization with {n_trials} trials")
        
        # Optimize
        print(f"\n🚀 Starting optimization process...")
        study.optimize(self.objective, n_trials=n_trials, timeout=None)
        
        print(f"\n🎉 OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print(f"🏆 Best trial number: {study.best_trial.number}")
        print(f"📈 Best objective value: {study.best_value:.4f}")
        print(f"✅ Total trials completed: {len(study.trials)}")
        print(f"❌ Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        
        # Save results
        print(f"\n💾 Saving results...")
        self.save_results(study, study_name)
        
        return study

    def save_results(self, study: optuna.Study, study_name: str):
        """
        Save optimization results.
        
        Args:
            study: Optuna study object
            study_name: Name of the study
        """
        print(f"📁 Creating results directory...")
        
        # Save best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            'study_name': study_name,
            'best_value': best_value,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON
        results_file = f'./hypertuning_results/{study_name}_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✅ Results saved to {results_file}")
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Best value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save study object
        study_file = f'./hypertuning_results/{study_name}_study.pkl'
        optuna.study.save_study(study, study_file)
        print(f"   ✅ Study object saved to {study_file}")
        
        # Plot optimization history
        print(f"📊 Generating optimization plots...")
        self.plot_optimization_history(study, study_name)

    def plot_optimization_history(self, study: optuna.Study, study_name: str):
        """
        Plot optimization history and parameter importance.
        
        Args:
            study: Optuna study object
            study_name: Name of the study
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Optimization history
            axes[0, 0].plot([trial.value for trial in study.trials])
            axes[0, 0].set_title('Optimization History')
            axes[0, 0].set_xlabel('Trial')
            axes[0, 0].set_ylabel('Objective Value')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Parameter importance
            if len(study.trials) > 10:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())[:10]  # Top 10 parameters
                values = [importance[p] for p in params]
                
                axes[0, 1].barh(params, values)
                axes[0, 1].set_title('Parameter Importance (Top 10)')
                axes[0, 1].set_xlabel('Importance')
            
            # Best value over time
            best_values = []
            best_so_far = float('-inf')
            for trial in study.trials:
                if trial.value is not None and trial.value > best_so_far:
                    best_so_far = trial.value
                best_values.append(best_so_far)
            
            axes[1, 0].plot(best_values)
            axes[1, 0].set_title('Best Value Over Time')
            axes[1, 0].set_xlabel('Trial')
            axes[1, 0].set_ylabel('Best Objective Value')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Trial distribution
            values = [trial.value for trial in study.trials if trial.value is not None]
            axes[1, 1].hist(values, bins=20, alpha=0.7)
            axes[1, 1].set_title('Distribution of Objective Values')
            axes[1, 1].set_xlabel('Objective Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'./hypertuning_results/{study_name}_optimization_plots.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting optimization history: {str(e)}")

def main():
    """
    Main function to run hyperparameter optimization.
    """
    print(f"\n🚀 TD3 HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = "/home/trhang/Documents/TD3/data/train_processed_data2.npy"
    INITIAL_CASH = 100_000
    N_TRIALS = 40   # Adjust based on computational budget ok roi dung di
    
    print(f"📋 Configuration:")
    print(f"   📂 Data path: {DATA_PATH}")
    print(f"   💰 Initial cash: ${INITIAL_CASH:,}")
    print(f"   🔢 Number of trials: {N_TRIALS}")
    
    # Create tuner
    print(f"\n🔧 Initializing tuner...")
    tuner = TD3HyperparameterTuner(
        data_path=DATA_PATH,
        initial_cash=INITIAL_CASH
    )
    
    # Run optimization
    print(f"\n🎯 Running optimization...")
    study = tuner.optimize(n_trials=N_TRIALS)
    
    # Print results
    print("\n" + "="*60)
    print("🎉 HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("="*60)
    print(f"🏆 Best objective value: {study.best_value:.4f}")
    print(f"🔧 Best parameters:")
    for param, value in study.best_params.items():
        print(f"   {param}: {value}")
    print(f"🔄 Total trials completed: {len(study.trials)}")
    print(f"✅ Successful trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"❌ Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"✂️ Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # Train final model with best parameters
    print(f"\n💡 Next Steps:")
    print(f"   1. Check results in: ./hypertuning_results/")
    print(f"   2. Use best parameters in your training script")
    print(f"   3. Train final model with optimized hyperparameters")
    
    print(f"\n🎯 Best hyperparameters to use:")
    print(f"   Copy these values to your train.py:")
    for param, value in study.best_params.items():
        print(f"   {param} = {value}")
    
    return study

if __name__ == "__main__":
    main()