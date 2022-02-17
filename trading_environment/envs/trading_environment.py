import gym
from gym import spaces
from gym.utils import seeding

from trading_environment.envs.data_source import DataSource
from trading_environment.envs.trading_simulator import TradingSimulator


class TradingEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning.

    - 0: HOLD
    - 1: LONG
    - 2: SHORT
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 data,
                 steps=500,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4,
                 take_profit_percentage=None,
                 stop_loss_percentage=None,
                 normalize=True):
        self.steps = steps
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps

        self.data_source = DataSource(data=data, steps=self.steps, normalize=normalize)

        self.simulator = TradingSimulator(steps=self.steps,
                                          trading_cost_bps=self.trading_cost_bps,
                                          time_cost_bps=self.time_cost_bps,
                                          take_profit_percentage=take_profit_percentage,
                                          stop_loss_percentage=stop_loss_percentage
                                          )

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(self.data_source.min_values,
                                            self.data_source.max_values)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Returns state observation, reward, done and info"""
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        observation, done = self.data_source.take_step()
        reward, info = self.simulator.take_step(action=action,
                                                observation=observation)
        return observation, reward, done, info

    def reset(self):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]

    # TODO
    def render(self, mode='human'):
        """Not implemented"""
        pass
