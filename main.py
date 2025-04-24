from pathlib import Path

import gymnasium as gym
from tradingagent import TradingAgent
import pandas as pd
import numpy as np

# a bugfix for an error, I got with the MultiDatasetTradingEnv
from gym_trading_env.environments import MultiDatasetTradingEnv
_orig_reset = MultiDatasetTradingEnv.reset

def _new_reset(self, seed=None, options=None):
    return _orig_reset(self, seed=seed)

# this is to fix another bug in the environment that is installed from pip, on github it is corrected
# to make sure that everything works nice, patch is included here ...
def _next_dataset(self):
    self._episodes_on_this_dataset = 0
    # Find the indexes of the less explored dataset
    potential_dataset_pathes = np.where(self.dataset_nb_uses == self.dataset_nb_uses.min())[0]
    # Pick one of them
    random_int = np.random.randint(potential_dataset_pathes.size)
    dataset_idx = potential_dataset_pathes[ random_int ]
    dataset_path = self.dataset_pathes[dataset_idx]
    self.dataset_nb_uses[dataset_idx] += 1 # Update nb use counts

    self.name = Path(dataset_path).name
    return self.preprocess(pd.read_pickle(dataset_path))


MultiDatasetTradingEnv.next_dataset = _next_dataset
MultiDatasetTradingEnv.reset = _new_reset

# use this class for debugging and training the method, however, the changes to the code will NOT be usable in BRUTE,
# only the tradingagent.py is important

pd.options.mode.chained_assignment = None

agent = TradingAgent()

env = gym.make("MultiDatasetTradingEnv",
               name="TrainingEnvironment",
               dataset_dir='./data/train*.pkl',
               preprocess=agent.make_features,
               reward_function=agent.reward_function,
               trading_fees=0.01 / 100,
               borrow_interest_rate=0.0003 / 100,
               portfolio_initial_value=1000,
               initial_position=0.0,
               max_episode_duration="max",
               positions=agent.get_position_list(),
               )

print("going to train")
agent.train(env)

env = gym.make("MultiDatasetTradingEnv",
               name="TrainingEnvironment",
               dataset_dir='./data/test*.pkl',
               preprocess=agent.make_features,
               reward_function=agent.reward_function,
               trading_fees=0.01 / 100,
               borrow_interest_rate=0.0003 / 100,
               portfolio_initial_value=1000,
               initial_position=0.0,
               max_episode_duration="max",
               positions=agent.get_position_list(),
               )


print("going to test")
agent.test(env, 15)
# this will return the market return and portfolio return values
print(env.unwrapped.get_metrics())

# If you want to visualize the results for better debugging, this guide might be usefult
# https://gym-trading-env.readthedocs.io/en/latest/render.html
