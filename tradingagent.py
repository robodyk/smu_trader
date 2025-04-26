from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress

# Import the Simple PPO implementation
from ppo import SimpleNN, SimplePPO


class TradingAgent:
    def __init__(self):
        """
        Simple trading agent for cryptocurrency using PPO.
        """
        self.agent = None
        self.trained = False

    def reward_function(self, history):
        return np.log(
            history["portfolio_valuation", -1] / history["portfolio_valuation", -2],
        )

    # def reward_function(self, history):
    #     """
    #     Pure Alpha Reward Function that strongly incentivizes generating
    #     returns uncorrelated with market movements.
    #
    #     This reward function:
    #     1. Rewards ONLY the component of returns not explained by market movement
    #     2. Provides zero reward for market-following behavior
    #     3. Penalizes strategies that amplify market volatility
    #
    #     Parameters
    #     ----------
    #     - history: Environment history containing portfolio and market information
    #
    #     Returns
    #     -------
    #     - reward: Alpha-based reward that ignores market-correlated returns
    #
    #     """
    #     # Minimum history needed for meaningful calculation
    #     MIN_HISTORY = 20
    #
    #     # If we don't have enough history, use a simpler reward until we build up history
    #     if len(history["portfolio_valuation"]) < 3:
    #         # Simple return for first few steps
    #         portfolio_return = (
    #             history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    #         ) - 1
    #         # Apply trading cost
    #         if len(history["position"]) >= 2:
    #             position_change = abs(history["position", -1] - history["position", -2])
    #             trading_cost = 0.0005 * position_change
    #             return portfolio_return - trading_cost
    #         return portfolio_return
    #
    #     # Calculate the current single-period returns
    #     current_portfolio_return = (
    #         history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    #     ) - 1
    #     current_market_return = (
    #         history["data_close", -1] / history["data_close", -2]
    #     ) - 1
    #
    #     # Trading cost calculation
    #     trading_cost = 0
    #     if len(history["position"]) >= 2:
    #         position_change = abs(history["position", -1] - history["position", -2])
    #         trading_cost = 0.0005 * position_change
    #
    #     # If we have enough history, calculate alpha using regression
    #     if len(history["portfolio_valuation"]) >= MIN_HISTORY:
    #         # Extract history for alpha calculation
    #         window = min(MIN_HISTORY, len(history["portfolio_valuation"]) - 1)
    #
    #         # Calculate return series
    #         portfolio_values = history["portfolio_valuation", -window - 1 :]
    #         market_values = history["data_close", -window - 1 :]
    #
    #         portfolio_returns = [
    #             (portfolio_values[i] / portfolio_values[i - 1]) - 1
    #             for i in range(1, len(portfolio_values))
    #         ]
    #         market_returns = [
    #             (market_values[i] / market_values[i - 1]) - 1
    #             for i in range(1, len(market_values))
    #         ]
    #
    #         # Calculate beta using linear regression
    #         try:
    #             slope, intercept, _, _, _ = linregress(
    #                 market_returns,
    #                 portfolio_returns,
    #             )
    #             beta = slope
    #             historical_alpha = intercept
    #         except:
    #             # Fallback if regression fails
    #             beta = 1.0 if sum(market_returns) != 0 else 0
    #             historical_alpha = np.mean(portfolio_returns) - beta * np.mean(
    #                 market_returns,
    #             )
    #
    #         # Calculate expected return based on CAPM
    #         expected_market_return = beta * current_market_return
    #
    #         # Alpha is the excess return over what would be expected given the beta
    #         single_period_alpha = current_portfolio_return - expected_market_return
    #
    #         # Calculate information ratio (alpha / tracking error)
    #         residuals = [
    #             portfolio_returns[i] - (beta * market_returns[i] + historical_alpha)
    #             for i in range(len(portfolio_returns))
    #         ]
    #         tracking_error = np.std(residuals) + 1e-8  # Avoid division by zero
    #         information_ratio = historical_alpha / tracking_error
    #
    #         # Reward is heavily weighted toward alpha generation
    #         alpha_reward = single_period_alpha
    #
    #         # Apply non-linear scaling to emphasize significant alpha
    #         if alpha_reward > 0:
    #             alpha_reward = np.power(1 + alpha_reward, 3) - 1
    #
    #         # Add small component for historical information ratio
    #         ir_component = 0.1 * information_ratio
    #
    #         # Penalty for high beta (market correlation)
    #         # This strongly discourages simply amplifying market movements
    #         beta_penalty = 0.2 * abs(beta)
    #
    #         # Final reward combines alpha with IR bonus and beta penalty
    #         reward = alpha_reward + ir_component - beta_penalty - trading_cost
    #
    #         return reward
    #
    #     # If we don't have enough history for regression but more than a few points
    #     # Use a simpler approximation of alpha
    #     portfolio_return = current_portfolio_return
    #     market_return = current_market_return
    #
    #     # Estimate alpha as the difference between returns, applying a penalty for market correlation
    #     if (portfolio_return > 0 and market_return > 0) or (
    #         portfolio_return < 0 and market_return < 0
    #     ):
    #         # If returns move in same direction, reduce reward for the correlated portion
    #         sign_correlation = np.sign(portfolio_return) == np.sign(market_return)
    #         if sign_correlation:
    #             approximate_alpha = portfolio_return - (0.5 * market_return)
    #         else:
    #             approximate_alpha = portfolio_return
    #     else:
    #         # If returns move in opposite directions, that's pure alpha
    #         approximate_alpha = portfolio_return
    #
    #     # Apply trading cost
    #     reward = approximate_alpha - trading_cost
    #
    #     return reward

    def make_features(self, df):
        """
        Comprehensive feature engineering for OHLCV data with multiple window sizes.
        All features are normalized or relative to ensure they're comparable across different cryptocurrencies.

        Parameters
        ----------
        - df: DataFrame with OHLCV data (must contain open, high, low, close, volume columns)

        Returns
        -------
        - df: DataFrame with added features

        """
        import numpy as np

        # Define window sizes for different time horizons
        short_window = 12  # 12 hours
        medium_window = 24  # 1 day
        long_window = 72  # 3 days

        # =====================
        # 1. Basic price features (all relative)
        # =====================

        # Price changes (returns)
        df["ret_close_1h"] = df["close"].pct_change(1).fillna(0)
        df["ret_close_4h"] = df["close"].pct_change(4).fillna(0)
        df["ret_close_24h"] = df["close"].pct_change(24).fillna(0)

        # Price ratios (all scale-invariant)
        df["ratio_open_close"] = (df["close"] / df["open"]).fillna(1)
        df["ratio_high_close"] = (df["high"] / df["close"]).fillna(1)
        df["ratio_low_close"] = (df["low"] / df["close"]).fillna(1)
        df["ratio_high_low"] = (df["high"] / df["low"]).fillna(1)

        # Log returns for better statistical properties
        df["log_ret_1h"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)

        # =====================
        # 2. Volume features (all normalized)
        # =====================

        # Normalize volume relative to its own history
        for window in [short_window, medium_window, long_window]:
            vol_max = df["volume"].rolling(window).max()
            df[f"vol_norm_{window}h"] = (df["volume"] / vol_max).fillna(0.5)

        # Volume changes (percentage)
        df["vol_change_1h"] = df["volume"].pct_change(1).fillna(0)
        df["vol_change_4h"] = df["volume"].pct_change(4).fillna(0)

        # Volume relative to its moving average
        for window in [short_window, medium_window, long_window]:
            vol_ma = df["volume"].rolling(window).mean().fillna(df["volume"].mean())
            df[f"vol_rel_ma_{window}h"] = (df["volume"] / vol_ma).fillna(1)

        # =====================
        # 3. Relative Price Levels
        # =====================

        # Price relative to moving averages (scale-invariant)
        for window in [short_window, medium_window, long_window]:
            # Simple Moving Average
            sma = df["close"].rolling(window).mean().fillna(df["close"])
            df[f"price_rel_sma_{window}h"] = (df["close"] / sma).fillna(1)

            # Exponential Moving Average
            ema = df["close"].ewm(span=window, adjust=False).mean().fillna(df["close"])
            df[f"price_rel_ema_{window}h"] = (df["close"] / ema).fillna(1)

        # MACD (normalized by price to make it scale-invariant)
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        # Normalize MACD by the current price level
        df["macd_norm"] = (macd / df["close"]).fillna(0)
        signal_line = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist_norm"] = ((macd - signal_line) / df["close"]).fillna(0)

        # =====================
        # 4. Volatility Indicators (all normalized)
        # =====================

        # True Range (as percentage of price)
        high_low = (df["high"] - df["low"]) / df["close"]
        high_close_prev = (
            (df["high"] - df["close"].shift(1)).abs() / df["close"]
        ).fillna(0)
        low_close_prev = (
            (df["low"] - df["close"].shift(1)).abs() / df["close"]
        ).fillna(0)

        # True Range as percentage of close price
        df["tr_pct"] = pd.concat(
            [high_low, high_close_prev, low_close_prev],
            axis=1,
        ).max(axis=1)

        # ATR as percentage of price (scale-invariant)
        for window in [short_window, medium_window, long_window]:
            df[f"atr_pct_{window}h"] = (
                df["tr_pct"].rolling(window).mean().fillna(df["tr_pct"].mean())
            )

        # Historical volatility (already scale-invariant)
        for window in [short_window, medium_window, long_window]:
            df[f"volatility_{window}h"] = (
                df["ret_close_1h"].rolling(window).std().fillna(0)
            )

        # =====================
        # 5. Momentum Indicators (all normalized)
        # =====================

        # RSI (already scale-invariant by design)
        for window in [short_window, medium_window, long_window]:
            delta = df["close"].diff().fillna(0)
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.rolling(window).mean().fillna(0)
            avg_loss = loss.rolling(window).mean().fillna(0)

            # Handle division by zero
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            df[f"rsi_{window}h"] = (100 - (100 / (1 + rs))).fillna(
                50,
            ) / 100  # Normalize to 0-1

        # Stochastic Oscillator (already scale-invariant by design)
        for window in [short_window, medium_window, long_window]:
            low_min = df["low"].rolling(window).min().fillna(df["low"])
            high_max = df["high"].rolling(window).max().fillna(df["high"])

            # Handle cases where high_max equals low_min
            denom = high_max - low_min
            denom = denom.replace(0, 1e-10)

            df[f"stoch_{window}h"] = ((df["close"] - low_min) / denom).fillna(0.5)

        # Normalized price momentum (percentage change over windows)
        for window in [short_window, medium_window, long_window]:
            df[f"momentum_{window}h"] = (
                df["close"] / df["close"].shift(window) - 1
            ).fillna(0)

        # =====================
        # 6. Bollinger Bands (normalized)
        # =====================

        for window in [short_window, medium_window, long_window]:
            # Calculate relative width of Bollinger Bands (volatility indicator)
            mid = df["close"].rolling(window).mean()
            std = df["close"].rolling(window).std()

            upper = mid + 2 * std
            lower = mid - 2 * std

            # Position within the bands (0 = at lower band, 1 = at upper band)
            band_range = upper - lower
            band_range = band_range.replace(0, 1e-10)  # Avoid division by zero
            df[f"bb_pos_{window}h"] = ((df["close"] - lower) / band_range).fillna(0.5)

            # Width of bands relative to price (volatility measure)
            df[f"bb_width_{window}h"] = ((upper - lower) / mid).fillna(0)

        return df

    def get_position_list(self):
        """
        Define the valid position values for the trading agent.

        Returns:
        - List of position values from -1.0 to 2.0 in increments of 0.1

        """
        return [x / 10.0 for x in range(-10, 21)]

    def train(self, env, num_episodes=100):
        """
        Train the agent using SimplePPO.

        Parameters
        ----------
        - env: Trading environment
        - num_episodes: Number of episodes to train for

        """
        self.agent = SimplePPO(
            env=env,
        )

        print("Starting training...")
        rewards = []
        steps = []

        for episode in range(num_episodes):
            # Train for one episode
            reward, num_steps, actor_loss, critic_loss = self.agent.train_episode()

            # Store metrics
            rewards.append(reward)
            steps.append(num_steps)

            # Log progress
            if (episode + 1) % 5 == 0:
                avg_reward = np.mean(rewards[-5:])
                avg_steps = np.mean(steps[-5:])
                print(
                    f"Episode {episode + 1}/{num_episodes} - Avg Reward: {avg_reward:.4f}, Avg Steps: {avg_steps:.1f}, Actor loss: {actor_loss:.3f}, critic_loss: {critic_loss:.3f}",
                )

        self.trained = True
        print("Training completed!")

        return rewards

    def get_test_position(self, observation):
        """
        Determine the position to take during testing.

        Parameters
        ----------
        - observation: Current state observation

        Returns
        -------
        - action: Action index to take

        """
        if not self.trained or self.agent is None:
            # Default strategy if agent not trained
            return 10  # Default to neutral position (0.0)

        # Use trained policy (deterministic for testing)
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float)
            action_probs = self.agent.actor(obs_tensor)
            action = torch.argmax(action_probs).item()

        return action

    def test(self, env, n_epochs):
        # DO NOT CHANGE - all changes will be ignored after upload to BRUTE!
        for _ in range(n_epochs):
            done, truncated = False, False
            observation, info = env.reset()
            while not done and not truncated:
                new_position = self.get_test_position(observation)
                observation, reward, done, truncated, info = env.step(new_position)
