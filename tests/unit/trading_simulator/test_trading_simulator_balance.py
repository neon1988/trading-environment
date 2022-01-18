import unittest

import numpy as np

from trading_environment.envs.trading_simulator import TradingSimulator


class TestTraidingSimulatorBalance(unittest.TestCase):
    def test_increment(self):
        simulator = TradingSimulator(steps=100,
                                     trading_cost_bps=0,
                                     time_cost_bps=0)

        step = simulator.take_step(1, np.array([0, 32000]))

        self.assertEqual(0, step[1]['balance'])

        step = simulator.take_step(1, np.array([0, 32000]))

        self.assertEqual(0, step[1]['balance'])

        step = simulator.take_step(0, np.array([0, 34000]))

        self.assertEqual(2000, step[1]['balance'])

        step = simulator.take_step(1, np.array([0, 34000]))

        self.assertEqual(2000, step[1]['balance'])

        step = simulator.take_step(0, np.array([0, 36000]))

        self.assertEqual(4000, step[1]['balance'])

    def test_balance_present_in_last_step(self):
        simulator = TradingSimulator(steps=3,
                                     trading_cost_bps=0,
                                     time_cost_bps=0)

        step1 = simulator.take_step(1, np.array([0, 32000]))
        step2 = simulator.take_step(0, np.array([0, 35000]))
        step3 = simulator.take_step(0, np.array([0, 36000]))

        self.assertEqual(0, step1[1]['balance'])
        self.assertEqual(3000, step2[1]['balance'])
        self.assertEqual(3000, step3[1]['balance'])

    def test_balance_change_only_when_order_closed(self):

        simulator = TradingSimulator(steps=4,
                                     trading_cost_bps=0,
                                     time_cost_bps=0)

        step1 = simulator.take_step(2, np.array([0, 36000]))
        step2 = simulator.take_step(2, np.array([0, 35000]))
        step3 = simulator.take_step(0, np.array([0, 32000]))

        self.assertEqual(0, step1[1]['balance'])
        self.assertEqual(0, step2[1]['balance'])
        self.assertEqual(4000, step3[1]['balance'])
