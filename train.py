import torch
import numpy as np
import os
import json
from collections import deque
import matplotlib.pyplot as plt
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter
from datetime import datetime
import traceback

def load_best_hyperparameters(config_path="./hypertuning_results"):
    """
    Load best hyperparameters from hyperparameter search.
    Tries multiple possible file locations.
    """
    possible_files = [
        os.path.join(config_path, "best_params.json"),
        os.path.join(config_path, "*_best_params.json"),
        os.path.join(config_path, "*_FIXED_results.json")
    ]
    
    # Try to find the most recent results file
    try:
        result_files = []
        for pattern in possible_files:
            if "*" in pattern:
                import glob
                result_files.extend(glob.glob(pattern))
            else:
                if os.path.exists(pattern):
                    result_files.append(pattern)
        
        if not result_files:
            raise FileNotFoundError("No hyperparameter files found")
        
        # Use the most recent file
        latest_file = max(result_files, key=os.path.getmtime)
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Handle different file formats
        if 'best_params' in data:
            best_params = data['best_params']
            best_value = data.get('best_validation_performance', data.get('best_value', 'Unknown'))
            print(f"✅ Loaded hyperparameters from: {latest_file}")
            print(f"📊 Best validation performance was: {best_value}")
        else:
            best_params = data
            print(f"✅ Loaded hyperparameters from: {latest_file}")
        
        return best_params
        
    except Exception as e:
        print(f"❌ Error loading hyperparameters: {e}")
        print(f"💡 Please run the fixed hyperparameter optimization first!")
        return None

def get_actual_dimensions(filtered_data, lookback_window, num_stocks, initial_cash):
    """
    Calculate actual dimensions from environment - CRITICAL for consistency.
    """
    temp_env = StockEnv(
        num_stocks=num_stocks, 
        data=filtered_data, 
        initial_cash=initial_cash
    )
    
    sample_state = temp_env.reset()
    state_sequence_dim = num_stocks * lookback_window * filtered_data.shape[-1]
    actual_state_sequence = sample_state[:state_sequence_dim]
    features_per_timestep = len(actual_state_sequence) // lookback_window
    
    print(f"📐 Dimension Analysis:")
    print(f"   Filtered data shape: {filtered_data.shape}")
    print(f"   Sample state shape: {sample_state.shape}")
    print(f"   Features per timestep: {features_per_timestep}")
    print(f"   Action dimension: {temp_env.action_space.shape[0]}")
    
    return {
        'features_per_timestep': features_per_timestep,
        'total_state_dim': len(sample_state),
        'action_dim': temp_env.action_space.shape[0],
        'state_sequence_dim': state_sequence_dim
    }

def combine_train_val_data(train_path, val_path):
    """
    Combine training and validation data for final model training.
    This is correct practice AFTER hyperparameter optimization is complete.
    """
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    
    # Combine along time dimension (maintaining temporal order)
    combined_data = np.concatenate([train_data, val_data], axis=0)
    
    print(f"📊 Data Combination (Post-Hyperparameter Optimization):")
    print(f"   Training data: {train_data.shape}")
    print(f"   Validation data: {val_data.shape}")
    print(f"   Combined data: {combined_data.shape}")
    print(f"   ✅ This is correct - hyperparameters already optimized on separate data")
    
    return combined_data

def main():
    print("🚀 FINAL MODEL TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 60)
    print("✅ Using hyperparameters optimized on separate validation data")
    print("✅ Training on combined train+val data (standard practice)")
    print("✅ Test data remains untouched for final evaluation")
    print("=" * 60)
    
    # STEP 1: Load best hyperparameters from hyperparameter search
    print("🔍 Loading optimized hyperparameters...")
    best_params = load_best_hyperparameters()
    if best_params is None:
        print("❌ Cannot proceed without hyperparameters.")
        print("💡 Please run the fixed hyperparameter optimization first!")
        return
    
    print("🎯 Optimized Hyperparameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"   {param:25}: {value:.6f}")
        else:
            print(f"   {param:25}: {value}")
    
    # STEP 2: Load and combine train + validation data for final training
    print(f"\n📊 Loading training data...")
    try:
        combined_data = combine_train_val_data(
            "/home/trhang/Documents/TD3/data/train_data.npy",
            "/home/trhang/Documents/TD3/data/val_data.npy"
        )
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("💡 Please ensure your train_data.npy and val_data.npy exist")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # STEP 3: Extract hyperparameters with defaults
    num_stocks = combined_data.shape[1]
    initial_cash = 100_000
    max_steps = combined_data.shape[0]
    
    # Extract hyperparameters (with fallback defaults)
    batch_size = best_params.get('batch_size', 32)
    discount = best_params.get('discount', 0.97)
    tau = best_params.get('tau', 0.001)
    exploration_phase = best_params.get('exploration_phase', 50)
    hidden_size = best_params.get('hidden_size', 256)
    num_layers = best_params.get('num_layers', 3)
    num_episodes = best_params.get('num_episodes', 100)  # Might want to increase for final training
    lookback_window = best_params.get('lookback_window', 15)
    observation_covariance = best_params.get('observation_covariance', 1.0)
    transition_covariance = best_params.get('transition_covariance', 0.05)
    
    # Generate unique model ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iteration = f"final_optimized_{timestamp}"
    
    print(f"\n📋 Final Training Configuration:")
    print(f"   Combined data shape: {combined_data.shape}")
    print(f"   Number of stocks: {num_stocks}")
    print(f"   Training episodes: {num_episodes}")
    print(f"   Batch size: {batch_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Number of layers: {num_layers}")
    print(f"   Lookback window: {lookback_window}")
    print(f"   Model ID: {iteration}")
    
    # STEP 4: Apply Kalman filter with optimized parameters
    print(f"\n🔧 Applying Kalman filter with optimized parameters...")
    print(f"   Observation covariance: {observation_covariance}")
    print(f"   Transition covariance: {transition_covariance}")
    
    try:
        filtered_data = apply_kalman_filter(
            combined_data, 
            observation_covariance=observation_covariance, 
            transition_covariance=transition_covariance
        )
        print(f"   ✅ Kalman filter applied successfully")
        print(f"   Filtered data shape: {filtered_data.shape}")
    except Exception as e:
        print(f"❌ Error applying Kalman filter: {e}")
        return
    
    # STEP 5: Extract chaotic features from filtered data
    print(f"\n🌀 Extracting chaotic features from filtered data...")
    try:
        chaotic_extractor = ChaoticFeatureExtractor()
        chaotic_features = chaotic_extractor.extract_features(filtered_data)
        chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks
        print(f"   Chaotic features shape: {chaotic_features.shape}")
        print(f"   Chaotic feature dimension: {chaotic_feature_dim}")
    except Exception as e:
        print(f"❌ Error extracting chaotic features: {e}")
        return
    
    # STEP 6: Calculate actual dimensions (CRITICAL!)
    print(f"\n📐 Calculating actual dimensions...")
    try:
        dims = get_actual_dimensions(filtered_data, lookback_window, num_stocks, initial_cash)
    except Exception as e:
        print(f"❌ Error calculating dimensions: {e}")
        return
    
    # STEP 7: Environment setup
    print(f"\n🏢 Setting up trading environment...")
    try:
        env = StockEnv(
            num_stocks=num_stocks, 
            data=filtered_data, 
            initial_cash=initial_cash
        )
        print(f"   Environment created successfully")
        print(f"   State dimension: {dims['total_state_dim']}")
        print(f"   Action dimension: {dims['action_dim']}")
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")
        return
    
    # STEP 8: TD3 Agent setup with optimized hyperparameters
    print(f"\n🤖 Initializing TD3 agent with optimized architecture...")
    try:
        agent = TD3(
            state_dim=dims['features_per_timestep'],  # Use calculated dimension
            chaotic_feature_dim=chaotic_feature_dim,
            action_dim=dims['action_dim'],
            hidden_size=hidden_size,      # From hyperparameter optimization
            num_layers=num_layers,        # From hyperparameter optimization
            num_stocks=num_stocks,
            max_action=1.0,
            env_action_space_high=1.0,
            env_action_space_low=0.0
        )
        
        agent.exploration_phase = exploration_phase
        print(f"   ✅ TD3 agent initialized with optimized hyperparameters")
        print(f"      Network: {dims['features_per_timestep']} -> {hidden_size} x {num_layers} -> {dims['action_dim']}")
    except Exception as e:
        print(f"❌ Error initializing TD3 agent: {e}")
        return
    
    # STEP 9: Create model directory
    os.makedirs('./model', exist_ok=True)
    
    # STEP 10: Training loop with enhanced progress tracking
    print(f"\n🎯 Starting Final Training with Optimized Hyperparameters...")
    print("=" * 60)
    
    reward_history = []
    avg_reward_history = []
    critic_loss_history = []
    actor_loss_history = []
    best_reward = float('-inf')
    patience_counter = 0
    patience_limit = 20  # Early stopping if no improvement
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            episode_critic_losses = []
            episode_actor_losses = []
            
            # Progress indicator
            if episode % 10 == 0 or episode < 5:
                progress = (episode / num_episodes) * 100
                print(f"📍 Episode {episode + 1}/{num_episodes} ({progress:.1f}%)")
            
            # Episode loop
            for step in range(max_steps - lookback_window + 1):
                # Extract state sequence with correct dimensions
                state_sequence_length = dims['state_sequence_dim']
                current_state_sequence = state[:state_sequence_length].reshape(
                    lookback_window, dims['features_per_timestep']
                )
                
                # Extract chaotic features sequence
                chaotic_features_sequence = chaotic_features[:, step:step + lookback_window, :].reshape(
                    lookback_window, -1
                )
                
                # Handle object arrays
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
                
                # Extract next state sequence
                next_state_sequence = next_state[:state_sequence_length].reshape(
                    lookback_window, dims['features_per_timestep']
                )
                
                # Next chaotic features
                next_chaotic_features_sequence = (
                    chaotic_features[:, step + 1:step + lookback_window + 1, :].reshape(lookback_window, -1)
                    if step + 1 < max_steps - lookback_window + 1
                    else chaotic_features_sequence
                )
                
                # Add to replay buffer
                agent.replay_buffer.add((
                    current_state_sequence, chaotic_features_sequence, action, reward,
                    next_state_sequence, next_chaotic_features_sequence, done
                ))
                
                # Train with optimized hyperparameters
                critic_loss, actor_loss = agent.train(
                    batch_size=batch_size,    # From optimization
                    discount=discount,        # From optimization
                    tau=tau                  # From optimization
                )
                
                if critic_loss is not None:
                    episode_critic_losses.append(critic_loss)
                if actor_loss is not None:
                    episode_actor_losses.append(actor_loss)
                
                state = next_state
                
                if done:
                    if episode % 10 == 0:
                        print(f"   Episode {episode + 1} ended early at step {step + lookback_window}")
                    break
            
            # Episode tracking
            reward_history.append(total_reward)
            avg_reward = np.mean(reward_history[-20:])  # Last 20 episodes
            avg_reward_history.append(avg_reward)
            
            avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0
            avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
            critic_loss_history.append(avg_critic_loss)
            actor_loss_history.append(avg_actor_loss)
            
            # Track best model and early stopping
            if total_reward > best_reward:
                best_reward = total_reward
                patience_counter = 0
                # Save best model
                torch.save(agent, f'./model/best_{iteration}.pth')
                if episode % 10 == 0:
                    print(f"   🏆 New best reward: {best_reward:.4f} (saved)")
            else:
                patience_counter += 1
            
            # Detailed logging every 10 episodes
            if episode % 10 == 0 or episode == num_episodes - 1:
                print(f"   Episode {episode + 1:3d}: Reward = {total_reward:8.2f}, "
                      f"Avg20 = {avg_reward:8.2f}, "
                      f"C_Loss = {avg_critic_loss:.4f}, "
                      f"A_Loss = {avg_actor_loss:.4f}")
            
            # Early stopping check
            if patience_counter >= patience_limit and episode > 50:
                print(f"   🛑 Early stopping: No improvement for {patience_limit} episodes")
                break
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return
    
    print(f"\n🎉 Training Completed!")
    print("=" * 60)
    
    # STEP 11: Save final model with complete configuration
    print(f"💾 Saving final model and configuration...")
    try:
        # Save final model
        torch.save(agent, f'./model/final_{iteration}.pth')
        
        # Prepare comprehensive model configuration
        model_config = {
            'model_metadata': {
                'model_id': iteration,
                'training_date': datetime.now().isoformat(),
                'model_type': 'TD3_Final_Optimized',
                'hyperparameter_optimized': True
            },
            'best_hyperparameters': best_params,
            'model_architecture': {
                'state_dim': dims['features_per_timestep'],
                'chaotic_feature_dim': chaotic_feature_dim,
                'action_dim': dims['action_dim'],
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'num_stocks': num_stocks,
                'lookback_window': lookback_window
            },
            'training_configuration': {
                'combined_data_shape': list(combined_data.shape),
                'filtered_data_shape': list(filtered_data.shape),
                'batch_size': batch_size,
                'discount': discount,
                'tau': tau,
                'num_episodes': len(reward_history),
                'initial_cash': initial_cash,
                'observation_covariance': observation_covariance,
                'transition_covariance': transition_covariance,
                'exploration_phase': exploration_phase
            },
            'training_results': {
                'reward_history': reward_history,
                'avg_reward_history': avg_reward_history,
                'critic_loss_history': critic_loss_history,
                'actor_loss_history': actor_loss_history,
                'best_reward': best_reward,
                'final_avg_reward': avg_reward_history[-1] if avg_reward_history else 0,
                'total_episodes_completed': len(reward_history)
            },
            'dimension_info': dims
        }
        
        # Save configuration
        with open(f'./model/config_{iteration}.json', 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✅ Models and configuration saved:")
        print(f"   - Best model: ./model/best_{iteration}.pth")
        print(f"   - Final model: ./model/final_{iteration}.pth")
        print(f"   - Configuration: ./model/config_{iteration}.json")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    # STEP 12: Generate training visualization
    print(f"\n📊 Generating training plots...")
    try:
        plt.figure(figsize=(18, 10))
        
        # Reward history
        plt.subplot(2, 3, 1)
        plt.plot(reward_history, 'b-', alpha=0.6, linewidth=1, label="Episode Reward")
        plt.plot(avg_reward_history, 'r-', linewidth=2, label="20-Episode Average")
        plt.axhline(y=best_reward, color='green', linestyle='--', alpha=0.8, label=f'Best: {best_reward:.2f}')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards (Optimized Hyperparameters)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Critic loss
        plt.subplot(2, 3, 2)
        if critic_loss_history:
            plt.plot(critic_loss_history, 'g-', linewidth=1.5, label="Critic Loss")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.title("Critic Loss Evolution")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Actor loss
        plt.subplot(2, 3, 3)
        if actor_loss_history:
            plt.plot(actor_loss_history, 'm-', linewidth=1.5, label="Actor Loss")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.title("Actor Loss Evolution")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Reward distribution
        plt.subplot(2, 3, 4)
        if reward_history:
            plt.hist(reward_history, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=np.mean(reward_history), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(reward_history):.2f}')
            plt.axvline(x=best_reward, color='green', linestyle='--', linewidth=2, label=f'Best: {best_reward:.2f}')
            plt.xlabel("Episode Reward")
            plt.ylabel("Frequency")
            plt.title("Reward Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Hyperparameters display
        plt.subplot(2, 3, 5)
        param_text = "Optimized Hyperparameters:\n\n"
        for param, value in best_params.items():
            if isinstance(value, float):
                param_text += f"{param}: {value:.4f}\n"
            else:
                param_text += f"{param}: {value}\n"
        
        plt.text(0.1, 0.95, param_text, fontsize=9, verticalalignment='top', 
                transform=plt.gca().transAxes, family='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title("Configuration Used")
        
        # Performance summary
        plt.subplot(2, 3, 6)
        summary_text = f"""Training Summary:

Episodes Completed: {len(reward_history)}
Best Reward: {best_reward:.2f}
Final Avg (20 ep): {avg_reward_history[-1]:.2f}

Model ID: {iteration}
Hyperparameter Optimized: ✅
Combined Train+Val Data: ✅
Test Data Preserved: ✅
        """
        
        plt.text(0.1, 0.95, summary_text, fontsize=10, verticalalignment='top',
                transform=plt.gca().transAxes, family='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title("Training Summary")
        
        plt.suptitle(f"Final TD3 Training Results - {iteration}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_file = f'./model/training_results_{iteration}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"✅ Training plots saved to: {plot_file}")
        
    except Exception as e:
        print(f"⚠️ Error generating plots: {e}")
    
    # STEP 13: Final summary and next steps
    print(f"\n📋 FINAL TRAINING SUMMARY")
    print("=" * 60)
    print(f"🏆 Best Episode Reward: {best_reward:.4f}")
    print(f"📈 Final 20-Episode Average: {avg_reward_history[-1]:.4f}")
    print(f"🔄 Episodes Completed: {len(reward_history)}")
    print(f"📊 Used Hyperparameters: Optimized via validation")
    print(f"💾 Model ID: {iteration}")
    print(f"✅ Training Data: Combined train+val (post-optimization)")
    print(f"🚫 Test Data: Untouched and ready for final evaluation")
    
    if len(reward_history) > 0:
        print(f"\n📊 Performance Statistics:")
        print(f"   Mean reward: {np.mean(reward_history):.4f}")
        print(f"   Std deviation: {np.std(reward_history):.4f}")
        print(f"   Improvement: {reward_history[-1] - reward_history[0]:.4f}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Evaluate model on test data using test.py")
    print(f"   2. Load best model: torch.load('./model/best_{iteration}.pth')")
    print(f"   3. Compare test performance with validation performance")
    print(f"   4. Deploy model for live trading (if test results are satisfactory)")
    
    print(f"\n✅ TRAINING COMPLETE - Model ready for testing!")
    return iteration

if __name__ == "__main__":
    model_id = main()
    if model_id:
        print(f"\nModel successfully trained with ID: {model_id}")
    else:
        print(f"\nTraining failed. Please check error messages above.")