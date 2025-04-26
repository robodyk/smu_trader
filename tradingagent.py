import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam


class FeedForwardNN(nn.Module):
    """
    A standard Feed Forward Neural Network with the architecture from the reference implementation.
    """

    def __init__(self, in_dim, out_dim):
        """
        Initialize the network and set up the layers.

        Parameters
        ----------
        - in_dim (int): Input dimensions.
        - out_dim (int): Output dimensions.

        """
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

        # Initialize weights with low values
        nn.init.normal_(self.layer1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.layer2.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.layer3.weight, mean=0.0, std=0.1)

    def forward(self, obs):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        - obs (Tensor or ndarray): Input observation.

        Returns
        -------
        - Tensor: Output of the forward pass.

        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.ln1(self.layer1(obs)))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm implementation.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.0003,
        gamma=0.99,
        clip=0.2,
        ent_coef=0.0,
        critic_factor=0.5,
        boundary_factor=0.1,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        n_updates=5,
    ):
        """
        Initialize the PPO agent.
        """
        self.lr = lr
        self.gamma = gamma
        self.clip = clip
        self.ent_coef = ent_coef
        self.critic_factor = critic_factor
        self.boundary_factor = boundary_factor
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.n_updates = n_updates

        self.s_dim = state_dim
        self.a_dim = action_dim

        self.actor = FeedForwardNN(self.s_dim, self.a_dim)
        self.critic = FeedForwardNN(self.s_dim, 1)

        # Initialize action variance (for exploration)
        self.cov_var = nn.Parameter(torch.full(size=(self.a_dim,), fill_value=1.0))
        self.cov_mat = torch.diag(self.cov_var)

        # Use a single optimizer for both networks
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + [self.cov_var],
            lr=lr,
        )

        # Memory for storing trajectory during an episode
        self.reset_memory()

    def reset_memory(self):
        """Reset memory at the start of each episode."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.terminals = []

    def select_action(self, state):
        """
        Select an action using the current policy.
        Returns continuous action and its log probability.
        """
        with torch.no_grad():
            mean = self.actor(state)
            dist = MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.numpy(), log_prob.item()

    def evaluate_actions(self, states, actions):
        """
        Evaluate the policy and value function.
        """
        # Convert to tensor if needed
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float)

        # Get state values
        values = self.critic(states).squeeze(-1)

        # Get action log probs
        means = self.actor(states)
        dist = MultivariateNormal(means, self.cov_mat)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return values, log_probs, entropy

    def store_transition(self, state, action, reward, log_prob, done):
        """Store a transition in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.terminals.append(done)

    def compute_returns_and_advantages(self):
        """Compute returns and advantages using GAE."""
        # Convert lists to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float)
        rewards = np.array(self.rewards)
        terminals = np.array(self.terminals)
        old_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float)

        # Compute values
        with torch.no_grad():
            values = self.critic(states).squeeze(-1).numpy()

        # Compute GAE advantages and returns
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 0.0
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - terminals[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            advantages[t] = last_gae_lam

        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (
            torch.tensor(returns, dtype=torch.float),
            torch.tensor(advantages, dtype=torch.float),
            old_log_probs,
        )

    def update(self, action_lb=-1, action_ub=2.0):
        """Update policy and value networks using PPO."""
        # Process stored trajectory
        states = torch.tensor(np.array(self.states), dtype=torch.float)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float)
        returns, advantages, old_log_probs = self.compute_returns_and_advantages()

        # Perform multiple update epochs
        for _ in range(self.n_updates):
            # Get new log probs and values
            values, log_probs, entropy = self.evaluate_actions(states, actions)

            # Calculate PPO ratio
            ratios = torch.exp(log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            # Calculate final losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()

            action_boundaries = torch.tensor(
                [[action_lb], [action_ub]],
                dtype=torch.float,
            )
            boundary_violations = torch.maximum(
                torch.zeros_like(actions),
                torch.abs(actions) - action_boundaries[1],
            )
            boundary_penalty = torch.mean(boundary_violations**2)
            actor_loss_with_penalty = (
                actor_loss + self.boundary_factor * boundary_penalty
            )

            # Combined loss
            loss = (
                actor_loss_with_penalty
                + self.critic_factor * critic_loss
                + self.ent_coef * entropy_loss
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters())
                + list(self.critic.parameters())
                + [self.cov_var],
                self.max_grad_norm,
            )

            self.optimizer.step()

        # Reset memory after update
        self.reset_memory()


class TradingAgent:
    def __init__(self):
        """Initialize the trading agent."""
        self.agent = None
        self.state_dim = None  # Will be set during first observation
        self.action_dim = 1  # For continuous position adjustment
        self.trained = False
        self.episode_rewards = []
        self.position_range = [-1.0, 2.0]  # Min and max positions

    def reward_function(self, history):
        """Modified reward with risk penalty"""
        # Base reward is log return
        log_return = np.log(
            history["portfolio_valuation", -1] / history["portfolio_valuation", -2],
        )

        # Add penalty for excessive risk (large position changes)
        position_change = abs(history["position", -1] - history["position", -2])
        risk_penalty = 0.1 * position_change  # Penalize large position changes

        return log_return - risk_penalty

    def make_features(self, df):
        """Prepare features for the agent."""
        # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
        df["feature_close"] = df["close"].pct_change()
        # Create the feature : close[t] / open[t]
        df["feature_open"] = df["close"] / df["open"]
        # Create the feature : high[t] / close[t]
        df["feature_high"] = df["high"] / df["close"]
        # Create the feature : low[t] / close[t]
        df["feature_low"] = df["low"] / df["close"]
        # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
        df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()

        # Add more technical indicators that might be useful for crypto trading
        # 1. RSI - Relative Strength Index
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["feature_rsi"] = 100 - (100 / (1 + rs))

        # 2. MACD - Moving Average Convergence Divergence
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["feature_macd"] = ema12 - ema26
        df["feature_macd_signal"] = df["feature_macd"].ewm(span=9).mean()

        # 3. Bollinger Bands
        rolling_mean = df["close"].rolling(window=20).mean()
        rolling_std = df["close"].rolling(window=20).std()
        df["feature_bb_upper"] = rolling_mean + (rolling_std * 2)
        df["feature_bb_lower"] = rolling_mean - (rolling_std * 2)
        df["feature_bb_position"] = (df["close"] - df["feature_bb_lower"]) / (
            df["feature_bb_upper"] - df["feature_bb_lower"]
        )

        # Fill NaN values with neutral defaults
        default_values = {
            "feature_close": 0.0,  # No change
            "feature_open": 1.0,  # Close = Open
            "feature_high": 1.0,  # High = Close
            "feature_low": 1.0,  # Low = Close
            "feature_volume": 0.1,  # Low but not zero volume
            "feature_rsi": 50.0,  # Neutral RSI
            "feature_macd": 0.0,  # No momentum
            "feature_macd_signal": 0.0,  # No momentum
            "feature_bb_position": 0.5,  # Middle of the band
            "feature_bb_upper": 1.02,  # 2% above close price
            "feature_bb_lower": 0.98,  # 2% below close price
        }

        # Apply default values
        for feature, default_value in default_values.items():
            if feature in df.columns:
                df[feature] = df[feature].fillna(default_value)

        return df

    def get_position_list(self):
        """Get the list of allowable positions."""
        return [x / 1000.0 for x in range(-1000, 2001)]

    def continuous_to_discrete_action(self, continuous_action):
        """Complete rework of the action mapping"""
        # Always default to neutral position (0.0) when in doubt
        return np.argmin(np.abs(self.get_position_list() - continuous_action.squeeze()))

    def train(self, env):
        """Train the agent using PPO."""
        print("Starting PPO training...")

        # Initialize the agent if not already initialized
        observation, _ = env.reset()
        if self.state_dim is None:
            self.state_dim = len(observation)
            self.agent = PPO(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                lr=0.0002,  # Learning rate
                gamma=0.99,  # Discount factor
                clip=0.2,  # PPO clip parameter
                ent_coef=0.0,  # Entropy coefficient (increased for exploration)
                critic_factor=0.5,  # Value loss coefficient
                max_grad_norm=np.inf,  # Max gradient norm
                gae_lambda=0.0,  # GAE lambda
                n_updates=5,  # Number of PPO updates per batch
            )

        # Training loop (multiple episodes)
        total_episodes = 100  # You can adjust this

        for episode in range(total_episodes):
            observation, _ = env.reset()
            episode_reward = 0
            step_count = 0
            done, truncated = False, False
            discrete_actions = []

            while not (done or truncated):
                # Convert observation to tensor for the agent
                obs_tensor = torch.tensor(observation, dtype=torch.float).unsqueeze(0)

                # Select continuous action
                continuous_action, log_prob = self.agent.select_action(obs_tensor)

                # Convert to discrete action
                discrete_action = self.continuous_to_discrete_action(continuous_action)
                discrete_actions.append(discrete_action)

                # Perform action in the environment
                new_observation, reward, done, truncated, info = env.step(
                    discrete_action,
                )

                # Store transition
                self.agent.store_transition(
                    observation,
                    continuous_action,
                    reward,
                    log_prob,
                    done or truncated,
                )

                # Update for next step
                observation = new_observation
                episode_reward += reward
                step_count += 1

                # If enough steps, perform a PPO update
                if len(self.agent.states) >= 24 or done or truncated:
                    self.agent.update()

            print("mean action: " + str(np.mean(discrete_actions)))
            # Track rewards
            self.episode_rewards.append(episode_reward)

            # Print progress
            if (episode + 1) % 5 == 0:
                avg_reward = np.mean(self.episode_rewards[-5:])
                print(
                    f"Episode {episode + 1}/{total_episodes}, Avg Reward: {avg_reward:.4f}, Steps: {step_count}",
                )

        print("Training complete!")
        self.trained = True

        # Save model if needed
        self.save_model("ppo_bitcoin_agent.pkl")

    def get_test_position(self, observation):
        """Return position for testing based on the trained policy."""
        if not self.trained or self.agent is None:
            # Default fallback if not trained
            return 10  # Position 0.0 (all in USD)

        # Convert observation to tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float).unsqueeze(0)

        # Get continuous action from the policy
        with torch.no_grad():
            continuous_action, _ = self.agent.select_action(obs_tensor)

        # Map to discrete action
        discrete_action = self.continuous_to_discrete_action(continuous_action)

        return discrete_action

    def test(self, env, n_epochs):
        """Test the agent for a number of epochs."""
        print(f"Testing agent for {n_epochs} epochs...")

        test_rewards = []

        for epoch in range(n_epochs):
            observation, info = env.reset()
            epoch_reward = 0
            done, truncated = False, False

            while not (done or truncated):
                # Get position from policy
                action = self.get_test_position(observation)

                # Step in environment
                observation, reward, done, truncated, info = env.step(action)
                epoch_reward += reward

            test_rewards.append(epoch_reward)
            print(f"Test Epoch {epoch + 1}/{n_epochs}, Reward: {epoch_reward:.4f}")

        avg_reward = np.mean(test_rewards)
        print(f"Average Test Reward: {avg_reward:.4f}")

        return avg_reward

    def save_model(self, filepath):
        """Save the trained model to a file."""
        if self.agent is not None:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "actor_state": self.agent.actor.state_dict(),
                        "critic_state": self.agent.critic.state_dict(),
                        "cov_var": self.agent.cov_var.detach().numpy(),
                        "state_dim": self.state_dim,
                        "action_dim": self.action_dim,
                    },
                    f,
                )
            print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model from a file."""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            # Initialize agent if needed
            if self.agent is None:
                self.state_dim = data["state_dim"]
                self.action_dim = data["action_dim"]
                self.agent = PPO(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                )

            # Load state dictionaries
            self.agent.actor.load_state_dict(data["actor_state"])
            self.agent.critic.load_state_dict(data["critic_state"])
            self.agent.cov_var = nn.Parameter(
                torch.tensor(data["cov_var"], dtype=torch.float),
            )
            self.agent.cov_mat = torch.diag(self.agent.cov_var)

            self.trained = True
            print(f"Model loaded from {filepath}")
            return True
        return False
