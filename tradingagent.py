import numpy as np


class TradingAgent:
    def reward_function(self, history):
        # TODO feel free to change the reward function ...
        #  This is the default one used in the gym-trading-env library, however, there might be better ones
        #  @see https://gym-trading-env.readthedocs.io/en/latest/customization.html#custom-reward-function
        return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

    def make_features(self, df):
        # TODO feel free to include your own features - for example, Bollinger bounds
        #  IMPORTANT - do not create features that look ahead in time.
        #  Doing so will result in 0 points from the homework.
        #  @see https://gym-trading-env.readthedocs.io/en/latest/features.html

        # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
        # this is percentual change in time
        df["feature_close"] = df["close"].pct_change()
        # Create the feature : close[t] / open[t]
        df["feature_open"] = df["close"] / df["open"]
        # Create the feature : high[t] / close[t]
        df["feature_high"] = df["high"] / df["close"]
        # Create the feature : low[t] / close[t]
        df["feature_low"] = df["low"] / df["close"]
        # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
        df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
        df.dropna(inplace=True)

        # the library automatically adds two features - your position and

        return df

    def get_position_list(self):
        # TODO feel free to specify different set of actions
        #  here, the acceptable actions are positions -1.0, -0.9, ..., 2.0
        #  corresponding actions are integers 0, 1, ..., 30
        #  @see https://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#action-space
        return [x / 10.0 for x in range(-10, 21)];

    def train(self, env):
        # TODO implement your version of the train method
        #  you can run several epizodes ...
        #  no strict bounds set, but use Reinforcement learning techniques,
        #  the choice of algorithm is Your responsibility

        # Run an episode until it ends
        #TODO - do not forget to repeat for multiple epizodes - you have 15 training datasets available, meaning
        # that 15 episodes is probably minimum ;)
        done, truncated = False, False
        observation, info = env.reset()
        while not done and not truncated:
            # At every timestep, pick a random position index from your position, i.e., a number between -1 and 2
            new_position = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(new_position)

    def get_test_position(self, observation):
        # TODO implement the method that will return position for testing
        #  In other words, this method will contain policy used for testing, not training.
        return 20 if observation[1] > 1.05 else 10  # all in USD ... maps to position 0.0

    def test(self, env, n_epochs):
        # DO NOT CHANGE - all changes will be ignored after upload to BRUTE!
        for _ in range(n_epochs):
            done, truncated = False, False
            observation, info = env.reset()
            while not done and not truncated:
                new_position = self.get_test_position(observation)
                observation, reward, done, truncated, info = env.step(new_position)
