import random
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch import nn, optim


class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        self.gamma = 0.97  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0003
        self.batch_size = 24
        self.n_batches = 5
        self.memory_size = 10000
        self.update_target_frequency = 100

        # Initialize variables to be set in train method
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.memory = None
        self.input_dim = None
        self.output_dim = None
        self.hidden_dim = 64
        self.steps = 0
        self.device = torch.device("cpu")

    def reward_function(self, history):
        # Log return as reward
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

        # Technical indicators (simple ones)
        # Moving averages
        df["feature_ma7"] = df["close"].rolling(window=7).mean() / df["close"]
        df["feature_ma25"] = df["close"].rolling(window=25).mean() / df["close"]

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

        # Don't drop NaN values as required
        # Instead, fill them with a reasonable value
        df = df.fillna(0)

        return df

    def get_position_list(self):
        # Specify the possible positions/actions
        # From -1.0 to 2.0 with step 0.1
        return [x / 10.0 for x in range(-10, 21)]

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
        max_episode_steps = 10000

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
            while not done and not truncated and episode_steps < max_episode_steps:
                episode_steps += 1

                # Select action
                action = self._get_action(observation)

                # Take action
                next_observation, reward, done, truncated, _ = env.step(action)

                # Store transition in replay buffer
                self.memory.push(observation, action, reward, next_observation, done)

                # Update model
                losses.append(self._update_model())

                if self.steps % 1000 == 0:
                    print(f"loss: {np.mean(losses)} reward:{total_reward}")
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
