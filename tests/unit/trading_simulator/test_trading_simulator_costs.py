import unittest

import numpy as np

from trading_environment.envs.trading_simulator import TradingSimulator


class TestTraidingSimulatorCosts(unittest.TestCase):
    def test_trade_cost(self):
        simulator = TradingSimulator(trading_cost_bps=0.1, time_cost_bps=0)

        self.assertEqual(0.1, simulator.trading_cost_bps)

        observation = np.array([0, 42])

        self.assertEqual(0, simulator.take_step(0, observation)[1]['costs'])
        self.assertEqual(0.1, simulator.take_step(1, observation)[1]['costs'])
        self.assertEqual(0.2, simulator.take_step(2, observation)[1]['costs'])

    def test_time_cost(self):
        simulator = TradingSimulator(trading_cost_bps=0, time_cost_bps=0.2)

        self.assertEqual(0.2, simulator.time_cost_bps)

        observation = np.array([0, 42])

        self.assertEqual(0.2, simulator.take_step(0, observation)[1]['costs'])
        self.assertEqual(0, simulator.take_step(1, observation)[1]['costs'])
        self.assertEqual(0, simulator.take_step(2, observation)[1]['costs'])
        self.assertEqual(0, simulator.take_step(2, observation)[1]['costs'])
