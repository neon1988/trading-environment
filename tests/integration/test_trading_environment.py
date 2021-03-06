import os
import unittest

import numpy as np
import pandas as pd

from trading_environment.envs import TradingEnvironment


class TestTraigingEnvironment(unittest.TestCase):

    def get_data(self):
        dirname = os.path.dirname(__file__)
        file = f'{dirname}/../assets/10set_usdt_1800.csv'

        df = pd.read_csv(file,
                         usecols=[0, 1, 2, 3, 4, 5],
                         names=['timestamp', 'open', 'close', 'low', 'high', 'volume'],
                         decimal='.', keep_default_na=False,
                         dtype={'timestamp': np.int32, 'open': np.float32, 'close': np.float32, 'low': np.float32,
                                'high': np.float32},
                         encoding="UTF-8", index_col='timestamp').sort_values(by='timestamp')

        return df

    def test_step_returns(self):
        env = TradingEnvironment(data=self.get_data(), steps=3, trading_cost_bps=0.01, time_cost_bps=0.01)

        step = env.step(0)

        self.assertEqual(4, len(step))
        self.assertEqual(3, len(step[0]))
        self.assertEqual(-0.01, step[1])
        self.assertFalse(step[2])
        self.assertEqual(10, len(step[3]))

    def test_last_step_not_empty(self):
        steps = 3

        env = TradingEnvironment(data=self.get_data(), steps=steps, trading_cost_bps=0.01, time_cost_bps=0.01)
        env.reset()

        for episode_step in range(steps):

            next_state, reward, done, _ = env.step(1)

            if done:
                break

        result = env.simulator.result()

        final = result.iloc[-1]

        self.assertEqual(steps, len(result))
        self.assertEqual(1, final['action'])

