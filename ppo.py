import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


class SimpleNN(nn.Module):
    """
    Simple feedforward neural network for PPO.
    """

    def __init__(self, in_dim, out_dim):
        super(SimpleNN, self).__init__()

        # Simple architecture with just two layers
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        x = F.relu(self.fc1(x))

        # Apply softmax for actor, nothing for critic
        if self.fc2.out_features > 1:  # Actor
            x = F.softmax(self.fc2(x), dim=-1)
        else:  # Critic
            x = self.fc2(x)

        return x


class BiggerNN(nn.Module):
    """
    slightly bigger feedforward neural network for PPO, critic network wasnt learning well on the simple one.
    """

    def __init__(self, in_dim, out_dim):
        super(BiggerNN, self).__init__()

        # Simple architecture with just two layers
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Apply softmax for actor, nothing for critic
        if self.fc2.out_features > 1:  # Actor
            x = F.softmax(self.fc3(x), dim=-1)
        else:  # Critic
            x = self.fc3(x)

        return x


class LSTMCritic(nn.Module):
    """
    LSTM-based critic network for value function approximation.
    Captures temporal patterns in cryptocurrency price data.
    """

    def __init__(self, input_dim, hidden_size=64, num_layers=2, dropout=0.1):
        super(LSTMCritic, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layers for value output
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

        # Initialize weights
        self._initialize_weights()

        # For maintaining state between batches if needed
        self.hidden = None

    def _initialize_weights(self):
        """Initialize network weights with Xavier/Glorot initialization"""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def init_hidden(self, batch_size=1, device=torch.device("cpu")):
        """Initialize hidden state"""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )

    def detach_hidden(self):
        """Detach hidden states from the computation graph"""
        if self.hidden is not None:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def forward(self, x, hidden=None, sequence_length=None):
        """
        Forward pass through the network.

        Parameters
        ----------
        - x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        - hidden: Initial hidden state (optional)
        - sequence_length: Number of time steps to process (optional)

        Returns
        -------
        - Value estimate tensor of shape (batch_size, 1)

        """
        # Handle different input shapes
        if len(x.shape) == 2:
            # If input is (batch_size, features), unsqueeze to add sequence dimension
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        # Use provided hidden state or initialize a new one
        if hidden is None:
            if self.hidden is None or self.hidden[0].size(1) != batch_size:
                self.hidden = self.init_hidden(batch_size, x.device)
            hidden = self.hidden

        # Process sequence through LSTM
        lstm_out, self.hidden = self.lstm(x, hidden)

        # Use only the output from the last time step
        last_output = lstm_out[:, -1]

        # Process through fully connected layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        value = self.fc2(x)

        return value

    def reset_hidden(self):
        """Reset the hidden state"""
        self.hidden = None


class SimplePPO:
    """
    Simple Proximal Policy Optimization (PPO) implementation for discrete action spaces.
    Minimalist version without GAE and with simplified architecture.
    """

    def __init__(
        self,
        env,
        actor_nn=SimpleNN,
        critic_nn=BiggerNN,
        lr=0.0003,
        gamma=0.99,
        clip=0.2,
        n_updates=3,
    ):
        self.lr = lr
        self.gamma = gamma
        self.clip = clip
        self.n_updates = n_updates

        self.env = env
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = len(env.unwrapped.positions)  # Number of discrete actions

        # Simple policy and value networks
        self.actor = actor_nn(self.s_dim, self.a_dim)
        self.critic = critic_nn(self.s_dim, 1)

        # Separate optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def reset(self):
        if isinstance(self.critic, LSTMCritic):
            self.critic.reset_hidden()

    def select_action(self, s):
        """
        Select action based on the current state.
        """
        # Ensure state is a tensor
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float)

        # Get action probabilities from actor
        action_probs = self.actor(s)

        # Create categorical distribution
        dist = Categorical(action_probs)

        # Sample action
        action = dist.sample()

        # Get log probability
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach()

    def compute_returns(self, rewards, terminals, values=None, n_steps=None):
        """
        Compute n-step returns with bootstrapping from critic values.

        Parameters
        ----------
        - rewards: List of rewards from the episode
        - terminals: List of terminal flags
        - values: Value estimates from the critic (optional)
        - n_steps: Number of steps to look ahead (None for full returns)

        Returns
        -------
        - returns: Tensor of computed returns

        """
        # Use full episode if n_steps is None
        if n_steps is None:
            n_steps = len(rewards)

        episode_length = len(rewards)
        returns = np.zeros_like(rewards, dtype=float)

        for i in range(episode_length):
            # For each time step i, look ahead n steps (or until episode end)
            lookahead = min(n_steps, episode_length - i)

            # Initialize return for time step i
            R = 0

            # Compute discounted sum of rewards for n steps
            for j in range(lookahead):
                # If we hit a terminal state, stop accumulating
                if i + j < episode_length and terminals[i + j] and j > 0:
                    break

                # Add discounted reward
                R += (self.gamma**j) * rewards[i + j]

            # Add bootstrapped value if we didn't reach the end of episode and didn't hit terminal
            if (
                i + lookahead < episode_length
                and not terminals[i + lookahead - 1]
                and values is not None
            ):
                bootstrap_index = i + lookahead
                if isinstance(values, torch.Tensor):
                    bootstrap_value = values[bootstrap_index].item()
                else:
                    bootstrap_value = values[bootstrap_index]

                # Add bootstrapped value with appropriate discount
                R += (self.gamma**lookahead) * bootstrap_value

            returns[i] = R

        # Convert to tensor and normalize
        returns = torch.tensor(returns, dtype=torch.float)

        # Normalize returns
        if len(returns) > 1:  # Only normalize if we have more than one return
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self, states, actions, old_log_probs, rewards, terminals):
        """
        Perform PPO update step with simplified logic.
        """
        # Convert to tensors if they're not already
        if len(states) < 10:
            return 0, 0
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long)
        if not isinstance(old_log_probs, torch.Tensor):
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float)

        # Perform multiple PPO updates
        values = self.critic(states).squeeze()
        returns = self.compute_returns(rewards, terminals, values, n_steps=6)
        if not isinstance(returns, torch.Tensor):
            returns = torch.tensor(returns, dtype=torch.float)
        for _ in range(self.n_updates):
            # Get current values
            values = self.critic(states).squeeze()

            # Get current action probabilities
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            curr_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Compute advantages
            advantages = returns - values.detach()

            # Compute PPO ratio
            ratios = torch.exp(curr_log_probs - old_log_probs)

            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss using PyTorch's MSE
            critic_loss = nn.MSELoss()(values, returns)

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        return actor_loss, critic_loss

    def train_episode(
        self,
        runs=100,
        batch_size=48,
        largest_window=72,
        neutral_action=10,
    ):
        """
        Train for one episode.
        """
        # Reset environment
        state, _ = self.env.reset()

        # Run episode
        done = False
        truncated = False
        total_reward = 0

        for _ in range(largest_window):
            state, _, _, _, _ = self.env.step(neutral_action)

        finished = False
        for _ in range(runs):
            states, actions, rewards, log_probs, terminals = [], [], [], [], []
            for _ in range(batch_size):
                # Select action
                action, log_prob = self.select_action(state)

                # Take action in environment
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                terminals.append(done or truncated)

                # Update state and reward
                state = next_state
                total_reward += reward

                # if reward <= -100:
                #     done = True

                # If episode ended, break
                if done or truncated:
                    finished = True
                    break

            # Compute returns

            # Update policy
            actor_loss, critic_loss = self.update(
                states,
                actions,
                log_probs,
                rewards,
                terminals,
            )
            self.reset()

            if finished:
                break

        return total_reward, len(rewards), actor_loss, critic_loss
