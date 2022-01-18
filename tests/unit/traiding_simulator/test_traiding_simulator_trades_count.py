import unittest

import numpy as np

from trading_environment.envs.trading_simulator import TradingSimulator


class TestTraidingSimulatorTradesCount(unittest.TestCase):
    def test_long_trades_increment(self):
        simulator = TradingSimulator()

        observation = np.array([100, 101])

        step = simulator.take_step(1, observation)

        self.assertEqual(1, step[1]['long_trades_count'])
        self.assertEqual(1, step[1]['trades_count'])

        step = simulator.take_step(0, observation)

        self.assertEqual(1, step[1]['long_trades_count'])
        self.assertEqual(1, step[1]['trades_count'])

        step = simulator.take_step(1, observation)

        self.assertEqual(2, step[1]['long_trades_count'])
        self.assertEqual(2, step[1]['trades_count'])

        step = simulator.take_step(0, observation)

        self.assertEqual(2, step[1]['long_trades_count'])
        self.assertEqual(2, step[1]['trades_count'])

    def test_short_trades_increment(self):
        simulator = TradingSimulator()

        observation = np.array([100, 101])

        step = simulator.take_step(2, observation)

        self.assertEqual(1, step[1]['short_trades_count'])
        self.assertEqual(1, step[1]['trades_count'])

        step = simulator.take_step(0, observation)

        self.assertEqual(1, step[1]['short_trades_count'])
        self.assertEqual(1, step[1]['trades_count'])

        step = simulator.take_step(2, observation)

        self.assertEqual(2, step[1]['short_trades_count'])
        self.assertEqual(2, step[1]['trades_count'])

        step = simulator.take_step(0, observation)

        self.assertEqual(2, step[1]['short_trades_count'])
        self.assertEqual(2, step[1]['trades_count'])
