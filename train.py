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
    possible_files = [
        os.path.join(config_path, "best_params.json"),
        os.path.join(config_path, "*_best_params.json"),
        os.path.join(config_path, "*_FIXED_results.json")
    ]
 
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
 
        latest_file = max(result_files, key=os.path.getmtime)
 
        with open(latest_file, 'r') as f:
            data = json.load(f)
 
        if 'best_params' in data:
            best_params = data['best_params']
            best_value = data.get('best_validation_performance', data.get('best_value', 'Unknown'))
            print(f"✅ Loaded hyperparameters from: {latest_file}")
            print(f"📊 Best validation performance: {best_value}")
        else:
            best_params = data
            print(f"✅ Loaded hyperparameters from: {latest_file}")
 
        return best_params
 
    except Exception as e:
        print(f"❌ Error loading hyperparameters: {e}")
        return None
 
 
def get_actual_dimensions(filtered_data, lookback_window, num_stocks, initial_cash):
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
    print(f"   Filtered data shape    : {filtered_data.shape}")
    print(f"   Sample state shape     : {sample_state.shape}")
    print(f"   Features per timestep  : {features_per_timestep}")
    print(f"   Action dimension       : {temp_env.action_space.shape[0]}")
 
    return {
        'features_per_timestep': features_per_timestep,
        'total_state_dim': len(sample_state),
        'action_dim': temp_env.action_space.shape[0],
        'state_sequence_dim': state_sequence_dim
    }
 
 
def combine_train_val_data(train_path, val_path):
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    combined_data = np.concatenate([train_data, val_data], axis=0)
 
    print(f"📊 Data Combination:")
    print(f"   Training data : {train_data.shape}")
    print(f"   Validation    : {val_data.shape}")
    print(f"   Combined      : {combined_data.shape}")
 
    return combined_data
 
 
def main():
    print("🚀 FINAL MODEL TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 60)
 
    # STEP 1: Load hyperparameters
    print("🔍 Loading optimized hyperparameters...")
    best_params = load_best_hyperparameters()
    if best_params is None:
        print("❌ Cannot proceed without hyperparameters.")
        return
 
    print("🎯 Optimized Hyperparameters:")
    for param, value in best_params.items():
        print(f"   {param:25}: {value:.6f}" if isinstance(value, float) else f"   {param:25}: {value}")
 
    # STEP 2: Load data
    print(f"\n📊 Loading training data...")
    try:
        combined_data = combine_train_val_data(
            "/home/trhang/Documents/TD3/data/train_data.npy",
            "/home/trhang/Documents/TD3/data/val_data.npy"
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
 
    # STEP 3: Extract hyperparameters
    num_stocks = combined_data.shape[1]
    initial_cash = 100_000
    max_steps = combined_data.shape[0]
 
    batch_size           = best_params.get('batch_size', 32)
    discount             = best_params.get('discount', 0.97)
    tau                  = best_params.get('tau', 0.001)
    exploration_phase    = best_params.get('exploration_phase', 50)
    hidden_size          = best_params.get('hidden_size', 256)
    num_layers           = best_params.get('num_layers', 3)
    num_episodes         = best_params.get('num_episodes', 150)
    lookback_window      = best_params.get('lookback_window', 15)
    observation_covariance = best_params.get('observation_covariance', 1.0)
    transition_covariance  = best_params.get('transition_covariance', 0.05)
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iteration = f"final_optimized_{timestamp}"
 
    print(f"\n📋 Final Training Configuration:")
    print(f"   Combined data shape : {combined_data.shape}")
    print(f"   Number of stocks    : {num_stocks}")
    print(f"   Training episodes   : {num_episodes}")
    print(f"   Batch size          : {batch_size}")
    print(f"   Hidden size         : {hidden_size}")
    print(f"   Lookback window     : {lookback_window}")
    print(f"   Model ID            : {iteration}")
 
    # STEP 4: Kalman filter
    print(f"\n🔧 Applying Kalman filter...")
    try:
        filtered_data = apply_kalman_filter(
            combined_data,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance
        )
        print(f"   ✅ Filtered data shape: {filtered_data.shape}")
    except Exception as e:
        print(f"❌ Kalman filter error: {e}")
        return
 
    # STEP 5: Chaotic features
    print(f"\n🌀 Extracting chaotic features...")
    try:
        chaotic_extractor = ChaoticFeatureExtractor()
        chaotic_features = chaotic_extractor.extract_features(filtered_data)
        chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks
        print(f"   Chaotic features shape : {chaotic_features.shape}")
        print(f"   Chaotic feature dim    : {chaotic_feature_dim}")
 
        # ✅ FIX Bug 5: Kiểm tra NaN trong chaos features
        nan_count = np.isnan(chaotic_features).sum()
        inf_count = np.isinf(chaotic_features).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️  Found {nan_count} NaN and {inf_count} Inf in chaotic features — replacing with 0")
            chaotic_features = np.nan_to_num(chaotic_features, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            print(f"   ✅ Chaotic features: no NaN/Inf detected")
 
    except Exception as e:
        print(f"❌ Chaotic feature error: {e}")
        return
 
    # STEP 6: Dimensions
    print(f"\n📐 Calculating actual dimensions...")
    try:
        dims = get_actual_dimensions(filtered_data, lookback_window, num_stocks, initial_cash)
    except Exception as e:
        print(f"❌ Dimension error: {e}")
        return
 
    # STEP 7: Environment
    print(f"\n🏢 Setting up trading environment...")
    try:
        env = StockEnv(
            num_stocks=num_stocks,
            data=filtered_data,
            initial_cash=initial_cash
        )
        print(f"   ✅ Environment created")
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return
 
    # STEP 8: TD3 Agent
    print(f"\n🤖 Initializing TD3 agent...")
    try:
        agent = TD3(
            state_dim=dims['features_per_timestep'],
            chaotic_feature_dim=chaotic_feature_dim,
            action_dim=dims['action_dim'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_stocks=num_stocks,
            max_action=1.0,
            env_action_space_high=1.0,
            env_action_space_low=0.0
        )
        agent.exploration_phase = exploration_phase
        print(f"   ✅ TD3 agent initialized")
    except Exception as e:
        print(f"❌ TD3 init error: {e}")
        return
 
    os.makedirs('./model', exist_ok=True)
 
    # STEP 9: Training loop
    print(f"\n🎯 Starting Training...")
    print("=" * 60)
 
    reward_history = []
    avg_reward_history = []
    critic_loss_history = []
    actor_loss_history = []
    best_reward = float('-inf')
    patience_counter = 0
    patience_limit = 40  # ✅ FIX: tăng từ 20 lên 40
 
    try:
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            episode_critic_losses = []
            episode_actor_losses = []
 
            if episode % 10 == 0 or episode < 5:
                progress = (episode / num_episodes) * 100
                print(f"📍 Episode {episode + 1}/{num_episodes} ({progress:.1f}%)")
 
            for step in range(max_steps - lookback_window + 1):
                state_sequence_length = dims['state_sequence_dim']
                current_state_sequence = state[:state_sequence_length].reshape(
                    lookback_window, dims['features_per_timestep']
                )
 
                chaotic_features_sequence = chaotic_features[:, step:step + lookback_window, :].reshape(
                    lookback_window, -1
                )
 
                if chaotic_features_sequence.dtype == np.dtype('O'):
                    chaotic_features_sequence = np.array(chaotic_features_sequence, dtype=np.float32)
 
                action = agent.select_action(
                    state=current_state_sequence,
                    chaotic_features=chaotic_features_sequence,
                    current_episode=episode
                )
 
                next_state, reward, done, info = env.step(action)
 
                # ✅ Log reward gốc cho chart
                total_reward += reward
 
                # ✅ KHÔNG normalize — reward đã ở scale nhỏ ~[-0.01, 0.01]
                # Đưa thẳng vào buffer
                next_state_sequence = next_state[:state_sequence_length].reshape(
                    lookback_window, dims['features_per_timestep']
                )
 
                next_chaotic_features_sequence = (
                    chaotic_features[:, step + 1:step + lookback_window + 1, :].reshape(lookback_window, -1)
                    if step + 1 < max_steps - lookback_window + 1
                    else chaotic_features_sequence
                )
 
                # ✅ FIX: chỉ 7 items trong buffer (không có reward gốc lẫn normalized)
                agent.replay_buffer.add((
                    current_state_sequence,
                    chaotic_features_sequence,
                    action,
                    reward,                          # reward gốc ~[-0.01, 0.01]
                    next_state_sequence,
                    next_chaotic_features_sequence,
                    done
                ))
 
                critic_loss, actor_loss = agent.train(
                    batch_size=batch_size,
                    discount=discount,
                    tau=tau
                )
 
                if critic_loss is not None:
                    episode_critic_losses.append(critic_loss)
                if actor_loss is not None:
                    episode_actor_losses.append(actor_loss)
 
                state = next_state
 
                if done:
                    if episode % 10 == 0:
                        print(f"   Done at step {step + lookback_window} | "
                              f"Portfolio: {env.portfolio_value:.2f} | "
                              f"Drawdown: {env._calculate_max_drawdown():.3f}")
                    break
 
            reward_history.append(total_reward)
            avg_reward = np.mean(reward_history[-20:])
            avg_reward_history.append(avg_reward)
 
            avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0
            avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
            critic_loss_history.append(avg_critic_loss)
            actor_loss_history.append(avg_actor_loss)
 
            if total_reward > best_reward:
                best_reward = total_reward
                patience_counter = 0
                torch.save(agent, f'./model/best_{iteration}.pth')
                if episode % 10 == 0:
                    print(f"   🏆 New best reward: {best_reward:.6f} (saved)")
            else:
                patience_counter += 1
 
            if episode % 10 == 0 or episode == num_episodes - 1:
                print(f"   Episode {episode + 1:3d}: "
                      f"Reward={total_reward:.6f}, "
                      f"Avg20={avg_reward:.6f}, "
                      f"C_Loss={avg_critic_loss:.6f}, "
                      f"A_Loss={avg_actor_loss:.6f}")
 
            if patience_counter >= patience_limit and episode > 50:
                print(f"   🛑 Early stopping at episode {episode + 1}")
                break
 
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print(traceback.format_exc())
        return
 
    print(f"\n🎉 Training Completed!")
    print("=" * 60)
 
    # STEP 10: Save model
    print(f"💾 Saving model...")
    try:
        torch.save(agent, f'./model/final_{iteration}.pth')
 
        model_config = {
            'model_metadata': {
                'model_id': iteration,
                'training_date': datetime.now().isoformat(),
                'model_type': 'TD3_Chaos_Fixed',
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
            'training_results': {
                'reward_history': reward_history,
                'avg_reward_history': avg_reward_history,
                'critic_loss_history': critic_loss_history,
                'actor_loss_history': actor_loss_history,
                'best_reward': best_reward,
                'total_episodes': len(reward_history)
            }
        }
 
        with open(f'./model/config_{iteration}.json', 'w') as f:
            json.dump(model_config, f, indent=2)
 
        print(f"   ✅ Best model  : ./model/best_{iteration}.pth")
        print(f"   ✅ Final model : ./model/final_{iteration}.pth")
        print(f"   ✅ Config      : ./model/config_{iteration}.json")
 
    except Exception as e:
        print(f"❌ Save error: {e}")
 
    # STEP 11: Plot
    print(f"\n📊 Generating training plots...")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
 
        # Reward
        axes[0, 0].plot(reward_history, 'b-', alpha=0.6, linewidth=1, label="Episode Reward")
        axes[0, 0].plot(avg_reward_history, 'r-', linewidth=2, label="20-Episode Avg")
        if best_reward != float('-inf'):
            axes[0, 0].axhline(y=best_reward, color='green', linestyle='--',
                               label=f'Best: {best_reward:.4f}')
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward (% return per episode)")
        axes[0, 0].set_title("Training Rewards")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
 
        # Critic loss
        axes[0, 1].plot(critic_loss_history, 'g-', linewidth=1.5, label="Critic Loss")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Critic Loss Evolution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
 
        # Actor loss
        axes[0, 2].plot(actor_loss_history, 'm-', linewidth=1.5, label="Actor Loss")
        axes[0, 2].set_xlabel("Episode")
        axes[0, 2].set_ylabel("Loss")
        axes[0, 2].set_title("Actor Loss Evolution")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
 
        # Reward distribution
        axes[1, 0].hist(reward_history, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=np.mean(reward_history), color='red', linestyle='--',
                           label=f'Mean: {np.mean(reward_history):.4f}')
        axes[1, 0].axvline(x=best_reward, color='green', linestyle='--',
                           label=f'Best: {best_reward:.4f}')
        axes[1, 0].set_xlabel("Episode Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Reward Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
 
        # Hyperparameters
        param_text = "Optimized Hyperparameters:\n\n"
        for k, v in best_params.items():
            param_text += f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n"
        axes[1, 1].text(0.05, 0.95, param_text, fontsize=9, va='top',
                        transform=axes[1, 1].transAxes, family='monospace',
                        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title("Configuration")
 
        # Summary
        summary = (
            f"Training Summary:\n\n"
            f"Episodes     : {len(reward_history)}\n"
            f"Best Reward  : {best_reward:.6f}\n"
            f"Final Avg    : {avg_reward_history[-1]:.6f}\n"
            f"Mean Reward  : {np.mean(reward_history):.6f}\n"
            f"Std Reward   : {np.std(reward_history):.6f}\n\n"
            f"Reward scale : ~[-0.01, 0.01]\n"
            f"= % return per step\n"
        )
        axes[1, 2].text(0.05, 0.95, summary, fontsize=10, va='top',
                        transform=axes[1, 2].transAxes, family='monospace',
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
        axes[1, 2].axis('off')
        axes[1, 2].set_title("Training Summary")
 
        plt.suptitle(f"TD3 + Chaos Training — {iteration}", fontsize=14, fontweight='bold')
        plt.tight_layout()
 
        plot_file = f'./model/training_results_{iteration}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"✅ Plot saved: {plot_file}")
 
    except Exception as e:
        print(f"⚠️ Plot error: {e}")
        print(traceback.format_exc())
 
    # STEP 12: Summary
    print(f"\n📋 FINAL SUMMARY")
    print("=" * 60)
    print(f"🏆 Best Reward       : {best_reward:.6f}")
    print(f"📈 Final Avg (20 ep) : {avg_reward_history[-1]:.6f}")
    print(f"🔄 Episodes Done     : {len(reward_history)}")
    print(f"💾 Model ID          : {iteration}")
    print(f"\n📌 Reward Interpretation:")
    print(f"   Reward per step ≈ % return (thập phân)")
    print(f"   total_reward    ≈ tổng % return qua {len(reward_history)} episodes")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Chạy test.py với best model")
    print(f"   2. So sánh với Buy & Hold benchmark")
    print(f"   3. Tính Sharpe, MDD, Calmar ratio")
 
    return iteration
 
 
if __name__ == "__main__":
    model_id = main()
    if model_id:
        print(f"\n✅ Model ID: {model_id}")
    else:
        print(f"\n❌ Training failed.")