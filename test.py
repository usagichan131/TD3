import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from stockEnv import StockEnv
from TD3 import TD3
from OptiPhaseSpace import ChaoticFeatureExtractor
from kalmanfilter import apply_kalman_filter

# Load the processed test data
data = np.load("/home/trhang/Documents/TD3/data/test_processed_data2.npy")

# Test configuration - matching training parameters
num_stocks = data.shape[1]
initial_cash = 100_000
lookback_window = 15  # Same as training
max_steps = data.shape[0]

# Chaotic Feature Extractor setup - same as training
chaotic_extractor = ChaoticFeatureExtractor()
all_chaotic_features = chaotic_extractor.extract_features(data)
chaotic_feature_dim = chaotic_extractor.output_dim * num_stocks

# Kalman filter setup - same parameters as training
data = apply_kalman_filter(data, observation_covariance=2.3327797486414723, transition_covariance=0.013047010546831547)

# Environment setup - same as training
test_env = StockEnv(num_stocks=num_stocks, data=data, initial_cash=initial_cash)
state_dim = test_env.observation_space.shape[0]
action_dim = test_env.action_space.shape[0]

# Load the trained agent with error handling
def load_agent_safely():
    # Try different loading methods and paths
    loading_attempts = [
        # Method 1: Load full agent (original)
        {
            'paths': ['./model/td3_5.4stocks_params150947.pth', '/home/trhang/Documents/TD3/model/td3.pth'],
            'method': 'full_agent'
        },
        # Method 2: Load from checkpoint
        {
            'paths': ['./model/td3_checkpoint.pth', '/home/trhang/Documents/TD3/model/td3_checkpoint.pth'],
            'method': 'checkpoint'
        }
    ]
    
    for attempt in loading_attempts:
        for path in attempt['paths']:
            try:
                print(f"🔄 Trying to load agent from: {path} (method: {attempt['method']})")
                
                if attempt['method'] == 'full_agent':
                    # Load full agent
                    agent = torch.load(path, weights_only=False, map_location='cpu')
                    print(f"✅ Full agent loaded successfully from {path}")
                    return agent
                    
                elif attempt['method'] == 'checkpoint':
                    # Load from checkpoint
                    checkpoint = torch.load(path, weights_only=False, map_location='cpu')
                    config = checkpoint['model_config']
                    
                    # Create new agent with saved config
                    agent = TD3(
                        state_dim=config['state_dim'],
                        chaotic_feature_dim=config['chaotic_feature_dim'],
                        action_dim=config['action_dim'],
                        hidden_size=config['hidden_size'],
                        num_layers=config['num_layers'],
                        num_stocks=config['num_stocks'],
                        max_action=config['max_action'],
                        env_action_space_high=config['env_action_space_high'],
                        env_action_space_low=config['env_action_space_low']
                    )
                    
                    # Load trained weights
                    agent.load_state_dict(checkpoint['agent_state_dict'])
                    agent.exploration_phase = config['exploration_phase']
                    
                    print(f"✅ Agent loaded from checkpoint: {path}")
                    print(f"📋 Training history available: {len(checkpoint.get('training_history', {}).get('reward_history', []))} episodes")
                    return agent
                    
            except FileNotFoundError:
                print(f"❌ File not found: {path}")
                continue
                
            except RuntimeError as e:
                if "PytorchStreamReader failed" in str(e):
                    print(f"❌ Corrupted model file: {path}")
                    continue
                else:
                    print(f"❌ Runtime error loading {path}: {e}")
                    continue
                    
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                continue
    
    # If all attempts failed, create new agent
    print("⚠️ Could not load any trained model. Creating new agent for architecture testing...")
    
    state_dim_calc = num_stocks * (data.shape[-1])
    action_dim_calc = num_stocks
    
    new_agent = TD3(
        state_dim=state_dim_calc,
        chaotic_feature_dim=chaotic_feature_dim,
        action_dim=action_dim_calc,
        hidden_size=128,
        num_layers=2,
        num_stocks=num_stocks,
        max_action=1.0,
        env_action_space_high=1.0,
        env_action_space_low=0.0
    )
    
    print("⚠️ WARNING: Using untrained agent. Results will not be meaningful!")
    print("💡 Please retrain your model with the updated training script.")
    return new_agent

try:
    agent = load_agent_safely()
except Exception as e:
    print(f"❌ Critical error: {e}")
    exit(1)

# Set to testing mode
agent.exploration_phase = 0  # No exploration during testing

# Testing configuration
num_test_episodes = 1
test_rewards = []
portfolio_values = []
actions_taken = []
daily_returns = []

print(f"🧪 Starting testing with {max_steps - lookback_window + 1} steps...")
print(f"📊 Test data shape: {data.shape}")
print(f"💰 Initial cash: ${initial_cash:,}")

# Testing loop
for episode in range(num_test_episodes):
    print(f"\n🔄 Running test episode {episode + 1}/{num_test_episodes}")
    
    state = test_env.reset()
    total_reward = 0
    episode_portfolio_values = [test_env.portfolio_value]
    episode_actions = []
    
    # Progress tracking
    progress_interval = max(1, (max_steps - lookback_window + 1) // 10)
    
    for step in range(max_steps - lookback_window + 1):
        # Progress indicator
        if step % progress_interval == 0:
            progress = (step / (max_steps - lookback_window)) * 100
            print(f"📈 Progress: {progress:.1f}% - Portfolio Value: ${test_env.portfolio_value:,.2f}")
        
        # Extract state sequence and portfolio state
        current_state_sequence = state[:num_stocks * lookback_window * (data.shape[-1])].reshape(
            lookback_window, num_stocks * (data.shape[-1])
        )
        portfolio_state = state[num_stocks * lookback_window * (data.shape[-1]):]
        
        # Get sequence of chaotic features
        chaotic_features_sequence = all_chaotic_features[:, step:step + lookback_window, :].reshape(
            lookback_window, -1
        )
        
        # Convert object array to float32 if needed
        if chaotic_features_sequence.dtype == np.dtype('O'):
            chaotic_features_sequence = np.array(chaotic_features_sequence, dtype=np.float32)
        
        # Select action using the actor network (no exploration)
        action = agent.select_action(
            state=current_state_sequence,
            chaotic_features=chaotic_features_sequence,
            current_episode=episode + agent.exploration_phase + 1000  # Ensure no exploration
        )
        
        # Step in environment
        next_state, reward, done, info = test_env.step(action)
        total_reward += reward
        
        # Store action and portfolio value
        episode_actions.append(action.copy())
        episode_portfolio_values.append(test_env.portfolio_value)
        
        # Update state
        state = next_state
        
        if done:
            print(f"⚠️ Test Episode {episode + 1} ended early at step {step + lookback_window}")
            break
    
    # Calculate daily returns
    episode_returns = np.diff(episode_portfolio_values) / episode_portfolio_values[:-1]
    daily_returns.append(episode_returns)
    
    # Record results
    test_rewards.append(total_reward)
    portfolio_values.append(episode_portfolio_values)
    actions_taken.append(episode_actions)
    
    # Episode summary
    final_value = episode_portfolio_values[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    print(f"\n📋 Episode {episode + 1} Summary:")
    print(f"   💰 Final Portfolio Value: ${final_value:,.2f}")
    print(f"   📈 Total Return: {total_return:.2f}%")
    print(f"   🎯 Total Reward: {total_reward:.7f}")
    print(f"   📊 Steps Completed: {len(episode_portfolio_values) - 1}")

# Performance metrics calculation
def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown"""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    return abs(max_drawdown)

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio

def calculate_volatility(returns):
    """Calculate annualized volatility"""
    return np.std(returns) * np.sqrt(252)  # Assuming daily returns, 252 trading days

# Calculate metrics for each episode
print(f"\n📊 Performance Metrics:")
print("=" * 60)

max_drawdowns = []
sharpe_ratios = []
volatilities = []
total_returns = []

for i, (values, returns) in enumerate(zip(portfolio_values, daily_returns)):
    max_drawdown = calculate_max_drawdown(values)
    sharpe_ratio = calculate_sharpe_ratio(returns)
    volatility = calculate_volatility(returns)
    total_return = (values[-1] - initial_cash) / initial_cash * 100
    
    max_drawdowns.append(max_drawdown)
    sharpe_ratios.append(sharpe_ratio)
    volatilities.append(volatility)
    total_returns.append(total_return)
    
    print(f"Episode {i + 1}:")
    print(f"   📉 Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
    print(f"   📊 Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"   📈 Volatility: {volatility:.4f}")
    print(f"   💹 Total Return: {total_return:.2f}%")

# Overall statistics
if len(test_rewards) > 1:
    print(f"\n🏆 Overall Statistics:")
    print("=" * 60)
    print(f"Average Test Reward: {np.mean(test_rewards):.7f}")
    print(f"Average Max Drawdown: {np.mean(max_drawdowns):.4f} ({np.mean(max_drawdowns)*100:.2f}%)")
    print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.4f}")
    print(f"Average Volatility: {np.mean(volatilities):.4f}")
    print(f"Average Total Return: {np.mean(total_returns):.2f}%")

# Plotting results
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('TD3 Trading Agent - Test Performance', fontsize=16, fontweight='bold')

# Plot 1: Portfolio Value Over Time
best_episode = np.argmax(test_rewards) if len(test_rewards) > 1 else 0
axes[0, 0].plot(portfolio_values[best_episode], 'b-', linewidth=2, label=f'Episode {best_episode + 1}')
axes[0, 0].axhline(y=initial_cash, color='r', linestyle='--', alpha=0.7, label='Initial Cash')
axes[0, 0].set_xlabel('Trading Step')
axes[0, 0].set_ylabel('Portfolio Value ($)')
axes[0, 0].set_title('Portfolio Value Over Time (Best Episode)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Daily Returns
if len(daily_returns[best_episode]) > 0:
    axes[0, 1].plot(daily_returns[best_episode], 'g-', alpha=0.7)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Trading Step')
    axes[0, 1].set_ylabel('Daily Return')
    axes[0, 1].set_title('Daily Returns (Best Episode)')
    axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Action Distribution (for first stock as example)
if len(actions_taken[best_episode]) > 0:
    first_stock_actions = [action[0] for action in actions_taken[best_episode]]
    axes[1, 0].plot(first_stock_actions, 'm-', alpha=0.7)
    axes[1, 0].set_xlabel('Trading Step')
    axes[1, 0].set_ylabel('Action Value')
    axes[1, 0].set_title('Actions for First Stock (Best Episode)')
    axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Drawdown
portfolio_vals = portfolio_values[best_episode]
peak = np.maximum.accumulate(portfolio_vals)
drawdown = (portfolio_vals - peak) / peak
axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
axes[1, 1].plot(drawdown, 'r-', linewidth=1)
axes[1, 1].set_xlabel('Trading Step')
axes[1, 1].set_ylabel('Drawdown')
axes[1, 1].set_title('Drawdown Over Time (Best Episode)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary table
print(f"\n📋 Test Summary Table:")
print("=" * 80)
print(f"{'Metric':<25} {'Value':<15} {'Description'}")
print("-" * 80)
print(f"{'Initial Cash':<25} ${initial_cash:,:<14} {'Starting portfolio value'}")
print(f"{'Final Value':<25} ${portfolio_values[0][-1]:,:<14.2f} {'Final portfolio value'}")
print(f"{'Total Return':<25} {total_returns[0]:<14.2f}% {'Overall return percentage'}")
print(f"{'Max Drawdown':<25} {max_drawdowns[0]*100:<14.2f}% {'Maximum portfolio decline'}")
print(f"{'Sharpe Ratio':<25} {sharpe_ratios[0]:<14.4f} {'Risk-adjusted return'}")
print(f"{'Volatility':<25} {volatilities[0]:<14.4f} {'Annualized volatility'}")
print(f"{'Total Steps':<25} {len(portfolio_values[0])-1:<14} {'Number of trading steps'}")

print(f"\n✅ Testing completed successfully!")