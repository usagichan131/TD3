import numpy as np
import gym
from gym import spaces

class StockEnv(gym.Env):
    def __init__(
        self,
        num_stocks,
        data,
        initial_cash=100_000,
        transaction_cost=0.001,
        tax_rate=0.001,
        penalty_weight=0.001,
    ):
        super(StockEnv, self).__init__()
        
        # Initialization parameters
        self.num_stocks = num_stocks
        self.data = data
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.tax_rate = tax_rate
        self.penalty_weight = penalty_weight

        #Store historical portfolio values for MDD calculation
        self.portfolio_history = []
        
        # Initialize portfolio state
        self.reset()
        
        # Action space: [stock_selection (binary), cash_allocation (proportions)]
        # [stock_action (-1 for sell, 0 for hold, 1 for buy), cash_allocation (proportions)]
        self.action_space = spaces.Box(
            low=np.array([-1] * num_stocks + [0] * num_stocks),
            high=np.array([1] * num_stocks + [1] * num_stocks),
            dtype=np.float32,
        )
        
        # Observation space: price data  + indicators + chaotic features + portfolio state
        feature_dim = self.data.shape[-1]  # Exclude the timestamp column
        obs_dim = feature_dim * num_stocks + 2 + num_stocks  # Add cash, portfolio value, and shares held
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )


    def reset(self):
        # Reset portfolio state
        self.cash_balance = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.shares_held = np.zeros(self.num_stocks, dtype=np.float32)
        self.current_step = 0
        self.portfolio_history = [self.initial_cash]  # Track portfolio value history
        
        observation = self._get_observation()
        return observation

       

    def step(self, action):
        # Parse action
        stock_selection = action[:self.num_stocks] #> 0.5  
        cash_allocation = action[self.num_stocks:]  # Allocation proportions
        cash_allocation /= np.sum(cash_allocation)  # Normalize to ensure sum = 1
        
        # Get current prices
        current_prices = self.data[self.current_step, :, 0]
        
        # Execute trades and calculate reward
        reward, transaction_costs, taxes = self._execute_trade(
            current_prices, stock_selection, cash_allocation
        )
        
        # Update timestep and check if done
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1 or self.portfolio_value <= 0
        
        # Get next observation
        next_observation = self._get_observation()
        
        return next_observation, reward, done, {
            "transaction_costs": transaction_costs,
            "taxes": taxes,
                    }

    def _get_observation(self):
        # Get current prices and features
        features = self.data[self.current_step, :, :]

        features_flat = features.flatten()
        
        # Construct observation: features + portfolio state
        obs = np.concatenate(
            [features_flat, [self.cash_balance, self.portfolio_value], self.shares_held]
        )
        return obs

    def _execute_trade(self, current_prices, stock_selection, cash_allocation):
        """
        Executes trades based on the action, updates the portfolio state, 
        and calculates the reward components.
        """
        # Portfolio value before trades
        old_portfolio_value = self.portfolio_value
        
        # Track transaction costs and taxes
        transaction_costs = 0
        taxes = 0
        
        # Execute trades
        total_trade_volume = 0
        for i in range(self.num_stocks):
            action = stock_selection[i]
            allocation = cash_allocation[i]

            if action > 0.5:
                # Allocate cash to this stock
                trade_value = self.cash_balance * allocation
                num_shares = trade_value // current_prices[i]
                trade_volume = num_shares * current_prices[i]
                
                # Calculate costs
                transaction_costs += trade_volume * self.transaction_cost
                taxes += trade_volume * self.tax_rate
                
                # Update portfolio
                self.shares_held[i] += num_shares
                self.cash_balance -= trade_volume
                
                # Accumulate trade volume
                total_trade_volume += trade_volume

            elif action < -0.5:
                sell_value = self.shares_held[i] * current_prices[i]
                trade_volume = sell_value
                
                # Calculate costs
                transaction_costs += trade_volume * self.transaction_cost
                taxes += trade_volume * self.tax_rate
                
                # Update portfolio
                self.cash_balance += sell_value
                self.shares_held[i] = 0  # Sell all for simplicity
        

        # Portfolio value after trades
        new_portfolio_value = (
            self.cash_balance + np.sum(self.shares_held * current_prices)
        )
        self.portfolio_history.append(new_portfolio_value)  # Store portfolio value history
        max_drawdown = self._calculate_max_drawdown()

        # Portfolio return
        portfolio_return = new_portfolio_value - old_portfolio_value
        
        # # Opportunity cost for unallocated cash
        # opportunity_cost = self.cash_balance * self.penalty_weight
        
        # Final reward
        reward = portfolio_return - self.penalty_weight*(transaction_costs + taxes +max_drawdown) #- opportunity_cost
        
        # Update portfolio value
        self.portfolio_value = new_portfolio_value
        
        return reward, transaction_costs, taxes #, opportunity_cost



    def _calculate_max_drawdown(self):
            """
            Computes maximum drawdown from portfolio history.
            """
            if len(self.portfolio_history) < 2:
                return 0  # No drawdown at the start

            peak = np.maximum.accumulate(self.portfolio_history)
            drawdown = (self.portfolio_history - peak) / peak
            max_drawdown = np.min(drawdown)  # Max drawdown is the worst drop

            return abs(max_drawdown) 

    def render(self, mode="human"):
        """
        Render the environment's state for visualization.
        """
        print(f"Step: {self.current_step}")
        print(f"Cash Balance: {self.cash_balance:.2f}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print(f"Shares Held: {self.shares_held}")
