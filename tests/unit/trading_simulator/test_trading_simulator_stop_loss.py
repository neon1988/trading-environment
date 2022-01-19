import unittest

import numpy as np

from trading_environment.envs.trading_simulator import TradingSimulator


class TestTraidingStopLoss(unittest.TestCase):
    def test_long_position(self):
        stop_loss_percentage = 3

        simulator = TradingSimulator(stop_loss_percentage=stop_loss_percentage)

        step1 = simulator.take_step(1, np.array([0, 32000]))
        step2 = simulator.take_step(1, np.array([0, 31500]))
        step3 = simulator.take_step(1, np.array([0, 31800]))
        step4 = simulator.take_step(1, np.array([0, 31000]))

        self.assertEqual(1, step1[1]['opened_position_type'])
        self.assertEqual(1, step2[1]['opened_position_type'])
        self.assertEqual(1, step3[1]['opened_position_type'])
        self.assertEqual(0, step4[1]['opened_position_type'])

    def test_short_position(self):

        stop_loss_percentage = 3

        simulator = TradingSimulator(stop_loss_percentage=stop_loss_percentage)

        step1 = simulator.take_step(2, np.array([0, 32000]))
        step2 = simulator.take_step(2, np.array([0, 32500]))
        step3 = simulator.take_step(2, np.array([0, 32800]))
        step4 = simulator.take_step(2, np.array([0, 33000]))

        self.assertEqual(-1, step1[1]['opened_position_type'])
        self.assertEqual(-1, step2[1]['opened_position_type'])
        self.assertEqual(-1, step3[1]['opened_position_type'])
        self.assertEqual(0, step4[1]['opened_position_type'])
