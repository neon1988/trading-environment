import unittest

import numpy as np

from trading_environment.envs.trading_simulator import TradingSimulator


class TestTraidingSimulatorGetTrades(unittest.TestCase):

    def test_empty(self):
        simulator = TradingSimulator()

        self.assertEqual(0, len(simulator.get_trades()))

    def test_dont_append_not_closed_order(self):
        simulator = TradingSimulator()

        simulator.take_step(1, np.array([0, 42]))

        self.assertEqual(0, len(simulator.get_trades()))

    def test_append_long_trade(self):
        simulator = TradingSimulator()

        simulator.take_step(1, np.array([0, 42]))
        simulator.take_step(0, np.array([0, 48]))

        self.assertEqual(1, len(simulator.get_trades()))

        trade = simulator.get_trades().iloc[0]

        self.assertEqual(1, trade['trade'])
        self.assertEqual(42, trade['open_price'])
        self.assertEqual(48, trade['close_price'])
        self.assertEqual(0, trade['open_index'])
        self.assertEqual(1, trade['close_index'])

    def test_append_short_trade(self):
        simulator = TradingSimulator()

        simulator.take_step(2, np.array([0, 42]))
        simulator.take_step(0, np.array([0, 48]))

        self.assertEqual(1, len(simulator.get_trades()))

        trade = simulator.get_trades().iloc[0]

        self.assertEqual(-1, trade['trade'])
        self.assertEqual(42, trade['open_price'])
        self.assertEqual(48, trade['close_price'])
        self.assertEqual(0, trade['open_index'])
        self.assertEqual(1, trade['close_index'])

    def test_append_long_and_short_trade(self):
        simulator = TradingSimulator()

        simulator.take_step(1, np.array([0, 42]))
        simulator.take_step(2, np.array([0, 48]))
        simulator.take_step(0, np.array([0, 48]))

        self.assertEqual(2, len(simulator.get_trades()))

    def test_append_short_and_long_trade(self):
        simulator = TradingSimulator()

        simulator.take_step(2, np.array([0, 42]))
        simulator.take_step(1, np.array([0, 48]))
        simulator.take_step(0, np.array([0, 48]))

        self.assertEqual(2, len(simulator.get_trades()))
