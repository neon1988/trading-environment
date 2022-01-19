import unittest

import numpy as np

from trading_environment.envs.trading_simulator import TradingSimulator


class TestTraidingGetOpenPositionProfitPercentage(unittest.TestCase):
    def test_profit_if_long_position(self):
        simulator = TradingSimulator(trading_cost_bps=0.3, time_cost_bps=0.1)
        simulator.step = 1
        simulator.opened_position_price[1] = 42000
        simulator.prices[1] = 43000
        simulator.positions[1] = 1

        self.assertAlmostEqual(1.78095238, simulator.get_open_position_profit_percentage())

    def test_loss_if_long_position(self):
        simulator = TradingSimulator(trading_cost_bps=0.3, time_cost_bps=0.1)
        simulator.step = 1
        simulator.opened_position_price[1] = 42000
        simulator.prices[1] = 41000
        simulator.positions[1] = 1

        self.assertAlmostEqual(-2.980952380, simulator.get_open_position_profit_percentage())

    def test_loss_if_short_position(self):
        simulator = TradingSimulator(trading_cost_bps=0.3, time_cost_bps=0.1)
        simulator.step = 1
        simulator.opened_position_price[1] = 42000
        simulator.prices[1] = 43000
        simulator.positions[1] = -1

        self.assertAlmostEqual(-2.98095238095, simulator.get_open_position_profit_percentage())

    def test_profit_if_short_position(self):
        simulator = TradingSimulator(trading_cost_bps=0.3, time_cost_bps=0.1)
        simulator.step = 1
        simulator.opened_position_price[1] = 42000
        simulator.prices[1] = 41000
        simulator.positions[1] = -1

        self.assertAlmostEqual(1.78095238095, simulator.get_open_position_profit_percentage())
