import random
from collections import Counter, deque

import numpy as np
import pandas as pd
import torch
from torch import nn, optim


class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            # nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            # nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class TradingAgent:
    def __init__(self):
        # Define hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.lr_decay_gamma = 0.9
        self.batch_size = 128
        self.n_batches = 1
        self.memory_size = 100000
        self.update_target_frequency = 1000

        # Initialize variables to be set in train method
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.scheduler = None
        self.memory = None
        self.input_dim = None
        self.output_dim = None
        self.hidden_dim = 256
        self.steps = 0
        self.device = torch.device("cpu")

    def reward_function(self, history):
        # Log return as reward

        # hist_pos_idx = history["position_index"]
        # current_pos_idx = hist_pos_idx[-1]
        # last_pos_idx = hist_pos_idx[-2]
        # if current_pos_idx == last_pos_idx:
        #     return 0
        # for i in range(2, len(hist_pos_idx) + 1):
        #     if hist_pos_idx[-i] != last_pos_idx:
        #         break
        return np.log(
            history["portfolio_valuation", -1] / history["portfolio_valuation", -2],
        )

    def make_features(self, df):
        # Create basic features
        # Price changes
        df["feature_close"] = df["close"].pct_change()
        df["feature_open"] = df["close"] / df["open"]
        df["feature_high"] = df["high"] / df["close"]
        df["feature_low"] = df["low"] / df["close"]

        # Volume indicator
        df["feature_volume"] = df["volume"] / df["volume"].rolling(24).max()

        # 1. SMA - Simple Moving Average (already implemented as ma7 and ma25)
        df["feature_sma7"] = df["close"].rolling(window=7).mean() / df["close"]
        df["feature_sma20"] = df["close"].rolling(window=20).mean() / df["close"]
        df["feature_sma50"] = df["close"].rolling(window=50).mean() / df["close"]

        # 2. OBV - On Balance Volume
        obv = np.zeros(len(df))
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv[i] = obv[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv[i] = obv[i - 1] - df["volume"].iloc[i]
            else:
                obv[i] = obv[i - 1]
        df["feature_obv"] = obv
        # Normalize OBV
        df["feature_obv"] = (
            df["feature_obv"] / df["feature_obv"].rolling(window=20).max()
        )

        # 3. Momentum
        df["feature_momentum"] = df["close"] / df["close"].shift(5) - 1

        # 4. Stochastic Oscillator
        n = 14  # Standard period
        df["feature_stoch_k"] = 100 * (
            (df["close"] - df["low"].rolling(window=n).min())
            / (df["high"].rolling(window=n).max() - df["low"].rolling(window=n).min())
        )
        df["feature_stoch_d"] = df["feature_stoch_k"].rolling(window=3).mean()

        # 5. MACD - Moving Average Convergence Divergence
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["feature_macd"] = ema12 - ema26
        df["feature_macd_signal"] = df["feature_macd"].ewm(span=9, adjust=False).mean()
        df["feature_macd_hist"] = df["feature_macd"] - df["feature_macd_signal"]

        # 6. CCI - Commodity Channel Index
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        ma_tp = typical_price.rolling(window=20).mean()
        mean_deviation = np.zeros(len(df))
        for i in range(20, len(df)):
            mean_deviation[i] = np.mean(
                np.abs(typical_price.iloc[i - 20 : i] - ma_tp.iloc[i]),
            )
        df["feature_cci"] = (typical_price - ma_tp) / (0.015 * mean_deviation)

        # 7. ADX - Average Directional Index
        # True Range
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        # Plus Directional Movement (+DM)
        plus_dm = df["high"].diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -df["low"].diff()), 0)

        # Minus Directional Movement (-DM)
        minus_dm = -df["low"].diff()
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > df["high"].diff()), 0)

        # Smoothed +DM and -DM
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)

        # Directional Index (DX)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

        # Average Directional Index (ADX)
        df["feature_adx"] = dx.rolling(window=14).mean()
        df["feature_plus_di"] = plus_di
        df["feature_minus_di"] = minus_di

        # 8. TRIX - Triple Exponential Average
        ema1 = df["close"].ewm(span=15, adjust=False).mean()
        ema2 = ema1.ewm(span=15, adjust=False).mean()
        ema3 = ema2.ewm(span=15, adjust=False).mean()
        df["feature_trix"] = 100 * (ema3 / ema3.shift(1) - 1)

        # 9. ROC - Rate of Change
        df["feature_roc"] = 100 * (df["close"] / df["close"].shift(10) - 1)

        # 10. SAR - Parabolic Stop and Reverse
        # Simple implementation
        df["feature_sar"] = df["close"].shift(1)  # Using lagged price as a simple proxy

        # 11. TEMA - Triple Exponential Moving Average
        ema1 = df["close"].ewm(span=20, adjust=False).mean()
        ema2 = ema1.ewm(span=20, adjust=False).mean()
        ema3 = ema2.ewm(span=20, adjust=False).mean()
        df["feature_tema"] = 3 * ema1 - 3 * ema2 + ema3
        df["feature_tema"] = df["feature_tema"] / df["close"]  # Normalize

        # 12. TRIMA - Triangular Moving Average
        sma = df["close"].rolling(window=10).mean()
        df["feature_trima"] = sma.rolling(window=10).mean()
        df["feature_trima"] = df["feature_trima"] / df["close"]  # Normalize

        # 13. WMA - Weighted Moving Average
        weights = np.arange(1, 11)
        df["feature_wma"] = (
            df["close"]
            .rolling(window=10)
            .apply(
                lambda x: np.sum(weights * x) / weights.sum(),
                raw=True,
            )
        )
        df["feature_wma"] = df["feature_wma"] / df["close"]  # Normalize

        # 14. DEMA - Double Exponential Moving Average
        ema = df["close"].ewm(span=20, adjust=False).mean()
        ema_of_ema = ema.ewm(span=20, adjust=False).mean()
        df["feature_dema"] = 2 * ema - ema_of_ema
        df["feature_dema"] = df["feature_dema"] / df["close"]  # Normalize

        # 15. MFI - Money Flow Index
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        # Get positive and negative money flow
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

        # Calculate money flow ratio
        positive_mf = pd.Series(positive_flow).rolling(window=14).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=14).sum()
        money_flow_ratio = positive_mf / negative_mf

        # Calculate MFI
        df["feature_mfi"] = 100 - (100 / (1 + money_flow_ratio))

        # 16. CMO - Chande Momentum Oscillator
        close_change = df["close"].diff()
        up_sum = close_change.where(close_change > 0, 0).rolling(window=14).sum()
        down_sum = -close_change.where(close_change < 0, 0).rolling(window=14).sum()
        df["feature_cmo"] = 100 * ((up_sum - down_sum) / (up_sum + down_sum))

        # 17. STOCHRSI - Stochastic RSI
        # Calculate RSI first
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate StochRSI
        df["feature_stochrsi"] = 100 * (
            (rsi - rsi.rolling(window=14).min())
            / (rsi.rolling(window=14).max() - rsi.rolling(window=14).min())
        )

        # 18. UO - Ultimate Oscillator
        bp = df["close"] - pd.DataFrame([df["low"], df["close"].shift(1)]).min()
        tr = pd.DataFrame(
            [
                df["high"] - df["low"],
                np.abs(df["high"] - df["close"].shift(1)),
                np.abs(df["low"] - df["close"].shift(1)),
            ],
        ).max()

        avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
        avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
        avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()

        df["feature_uo"] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7

        # 19. BOP - Balance Of Power
        df["feature_bop"] = (df["close"] - df["open"]) / (df["high"] - df["low"])

        # 20. ATR - Average True Range
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["feature_atr"] = tr.rolling(window=14).mean()
        df["feature_atr"] = df["feature_atr"] / df["close"]  # Normalize

        # Relative Strength Index (RSI)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["feature_rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        rolling_mean = df["close"].rolling(window=20).mean()
        rolling_std = df["close"].rolling(window=20).std()
        df["feature_bb_upper"] = (rolling_mean + (rolling_std * 2)) / df["close"]
        df["feature_bb_lower"] = (rolling_mean - (rolling_std * 2)) / df["close"]

        # Normalize all absolute value features using the logarithmic normalization formula
        # s'ij = log(sij/s00) × 10

        # First, create a copy of the original dataframe to get the base value s00 for each row
        df_base = df.copy()

        # Define a function to apply the normalization
        def normalize_feature(feature_series):
            # Check if this is a feature that should be normalized (all features except pct_change types)
            if not (
                feature_series.name.startswith("feature_")
                and "_" in feature_series.name
            ):
                return feature_series

            # Skip features that are already normalized as ratios
            if feature_series.name in [
                "feature_close",
                "feature_open",
                "feature_high",
                "feature_low",
                "feature_volume",
                "feature_sma7",
                "feature_sma20",
                "feature_sma50",
                "feature_tema",
                "feature_trima",
                "feature_wma",
                "feature_dema",
                "feature_bb_upper",
                "feature_bb_lower",
                "feature_atr",
            ]:
                return feature_series

            # For absolute-value based features, apply the normalization formula
            # Get the base value (first value in the series, or a small positive number if zero)
            s00 = max(abs(feature_series.iloc[0]), 1e-6)

            # Apply formula: s'ij = log(sij/s00) × 10
            # Handle positive and negative values appropriately
            normalized = np.zeros_like(feature_series.values)
            for i in range(len(feature_series)):
                sij = feature_series.iloc[i]
                # Avoid division by zero or log of zero/negative
                if sij == 0:
                    normalized[i] = 0
                else:
                    sign = np.sign(sij)
                    normalized[i] = sign * np.log(abs(sij) / s00) * 10

            return pd.Series(
                normalized,
                index=feature_series.index,
                name=feature_series.name,
            )

        # Apply normalization to all feature columns
        for col in df.columns:
            if col.startswith("feature_"):
                df[col] = normalize_feature(df[col])

        # Don't drop NaN values as required
        # Instead, fill them with a reasonable value
        df = df.fillna(0)

        return df

    def get_position_list(self):
        # Specify the possible positions/actions
        # From -1.0 to 2.0 with step 0.1
        # return [-1, 0, 1, 2]
        return [x / 2.0 for x in range(-2, 5)]

    def _initialize_models(self, input_dim, output_dim):
        # Initialize DQN model and target model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = DQNetwork(input_dim, self.hidden_dim, output_dim).to(self.device)
        self.target_model = DQNetwork(input_dim, self.hidden_dim, output_dim).to(
            self.device,
        )
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, self.lr_decay_gamma
        )
        self.memory = ReplayBuffer(self.memory_size)

    def _get_action(self, state, is_training=True):
        # Epsilon-greedy action selection
        if is_training and np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.output_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def _update_model(self):
        if len(self.memory) < self.batch_size * self.n_batches * 2:
            return float("inf")

        losses = []
        for _ in range(self.n_batches):
            # Sample batch from replay buffer
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(np.array(dones)).to(self.device)

            # Get current Q values
            current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Get next Q values (using target network)
            with torch.no_grad():
                next_q = self.target_model(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q

            # Compute loss and optimize
            loss = nn.functional.mse_loss(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update target network periodically
            if self.steps % self.update_target_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            losses.append(loss.item())
        return np.mean(losses)

    def train(self, env):
        # Run multiple episodes for training
        num_episodes = 30  # At least 15 episodes as mentioned in the assignment
        max_episode_steps = 100000

        # For the first run, we need to determine the input and output dimensions
        observation, _ = env.reset()

        # Input dimension is the length of the observation
        input_dim = len(observation)

        # Output dimension is the number of possible positions/actions
        output_dim = len(self.get_position_list())

        # Initialize the models
        self._initialize_models(input_dim, output_dim)

        # Training loop
        for episode in range(num_episodes):
            done, truncated = False, False
            observation, _ = env.reset()
            total_reward = 0
            episode_steps = 0

            losses = []
            actions = []
            while not done and not truncated and episode_steps < max_episode_steps:
                episode_steps += 1

                # Select action
                action = self._get_action(observation)
                actions.append(action)

                # Take action
                next_observation, reward, done, truncated, _ = env.step(action)

                # Store transition in replay buffer
                self.memory.push(observation, action, reward, next_observation, done)

                # Update model
                losses.append(self._update_model())

                if episode_steps % 1000 == 0:
                    print(f"loss: {np.mean(losses)} reward:{total_reward}")
                    print(" ".join(f"{a}: {c}" for a, c in Counter(actions).items()))
                    losses = []

                # Update current state and step counter
                observation = next_observation
                total_reward += reward
                self.steps += 1

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    def get_test_position(self, observation):
        # Use trained model to predict the best action
        if self.model is None:
            # If model wasn't trained, return a default action
            return 10  # Maps to position 0.0

        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
            q_values = self.model(state_tensor)
            action = q_values.argmax().item()
            return action

    def test(self, env, n_epochs):
        # DO NOT CHANGE - all changes will be ignored after upload to BRUTE!
        for _ in range(n_epochs):
            done, truncated = False, False
            observation, info = env.reset()
            while not done and not truncated:
                new_position = self.get_test_position(observation)
                observation, reward, done, truncated, info = env.step(new_position)


if __name__ == "__main__":
    # Sanity check using CartPole environment to verify DQN implementation
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Create a modified version of TradingAgent specifically for CartPole testing
    class CartPoleTester(TradingAgent):
        def __init__(self):
            super().__init__()
            # Adjust hyperparameters for CartPole
            self.gamma = 0.99
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.learning_rate = 0.0001
            self.batch_size = 64
            self.memory_size = 10000
            self.update_target_frequency = 100
            self.hidden_dim = 64

        def test_cartpole(self, num_episodes=100, max_steps=50000):
            # Create CartPole environment
            env = gym.make("CartPole-v1")

            # Initialize models
            input_dim = env.observation_space.shape[0]  # 4 for CartPole
            output_dim = env.action_space.n  # 2 for CartPole
            self._initialize_models(input_dim, output_dim)

            # Training metrics
            rewards_history = []
            rewards = []
            avg_rewards = []
            episodes_length = []
            losses = []

            # Training loop
            for episode in range(num_episodes):
                observation, _ = env.reset()
                episode_reward = 0
                done = False
                truncated = False
                steps = 0

                while not done and not truncated and steps < max_steps:
                    # Select action
                    action = self._get_action(observation)

                    # Take action
                    next_observation, reward, done, truncated, _ = env.step(action)

                    # Modify reward to make learning faster
                    if done and steps < max_steps - 1:
                        reward = -10

                    # Store transition in replay buffer
                    self.memory.push(
                        observation,
                        action,
                        reward,
                        next_observation,
                        done or truncated,
                    )

                    # Update model
                    loss = self._update_model()
                    losses.append(loss)

                    # Update current state and metrics
                    observation = next_observation
                    episode_reward += reward
                    steps += 1

                # Collect metrics
                rewards_history.append(episode_reward)
                rewards.append(episode_reward)
                episodes_length.append(steps)
                avg_reward = np.mean(
                    rewards_history[-100:],
                )  # Moving average of last 100 episodes
                avg_rewards.append(avg_reward)

                # Print progress
                if episode % 10 == 0:
                    print(
                        f"Episode: {episode}, Reward: {np.mean(rewards)}, Steps: {steps}, Epsilon: {self.epsilon:.4f}, Loss: {np.mean(losses)}",
                    )
                    rewards = []
                    losses = []

                # Check if solved
                if avg_reward >= 195.0 and episode >= 100:
                    print(f"Environment solved in {episode} episodes!")
                    break

            # Plot training progress
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(rewards_history)
            plt.plot(avg_rewards, "r")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Rewards per Episode")
            plt.legend(["Reward", "Avg Reward"])

            plt.subplot(1, 2, 2)
            plt.plot(episodes_length)
            plt.xlabel("Episode")
            plt.ylabel("Steps")
            plt.title("Steps per Episode")

            plt.tight_layout()
            plt.savefig("cartpole_dqn_results.png")
            plt.show()

            # Test the trained agent
            self.test_trained_agent(env)

            return rewards_history, episodes_length

        def test_trained_agent(self, env, episodes=5):
            """Test the trained agent without exploration"""
            for episode in range(episodes):
                observation, _ = env.reset()
                total_reward = 0
                done = False
                truncated = False
                steps = 0

                while not done and not truncated:
                    # Select best action (no exploration)
                    with torch.no_grad():
                        state_tensor = (
                            torch.FloatTensor(observation).to(self.device).unsqueeze(0)
                        )
                        q_values = self.model(state_tensor)
                        action = q_values.argmax().item()

                    # Take action
                    observation, reward, done, truncated, _ = env.step(action)
                    total_reward += reward
                    steps += 1

                print(
                    f"Test Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}",
                )

    print("Running CartPole sanity check to verify DQN implementation...")
    tester = CartPoleTester()
    rewards, steps = tester.test_cartpole(num_episodes=200)

    # Analyze final performance
    recent_rewards = rewards[-100:]
    avg_reward = np.mean(recent_rewards)
    print("\nFinal performance metrics:")
    print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
    print(f"Max reward: {max(recent_rewards)}")

    # Check if it solved CartPole
    if avg_reward >= 195.0:
        print("\nSANITY CHECK PASSED: DQN implementation successfully solved CartPole!")
        print(
            "The DQN implementation is working correctly and can be applied to the trading environment.",
        )
    else:
        print("\nSANITY CHECK WARNING: DQN didn't fully solve CartPole.")
        print(
            "The implementation may need improvements or just more training episodes.",
        )
        if avg_reward >= 150.0:
            print(
                "However, performance is reasonable, suggesting the core algorithm is functioning.",
            )
