import unittest

import numpy as np

from trading_environment.envs.trading_simulator import TradingSimulator


class TestTraidingSimulatorTakeStep(unittest.TestCase):
    def test_step_increment(self):
        simulator = TradingSimulator()

        observation = np.array([100, 101])

        self.assertEqual(0, simulator.step)

        simulator.take_step(0, observation)

        self.assertEqual(1, simulator.step)

    def test_first_step_open_long_position(self):
        simulator = TradingSimulator()

        observation = np.array([100, 42000])

        simulator.take_step(1, observation)

        self.assertEqual(1, simulator.trades[0])
        self.assertEqual(42000, simulator.prices[0])
        self.assertEqual(1, simulator.actions[0])
        self.assertEqual(42000, simulator.opened_position_price[0])
        self.assertEqual(0, simulator.balance[0])

    def test_first_step_open_short_position(self):
        simulator = TradingSimulator()

        observation = np.array([100, 32000])

        simulator.take_step(2, observation)

        self.assertEqual(-1, simulator.trades[0])
        self.assertEqual(32000, simulator.prices[0])
        self.assertEqual(2, simulator.actions[0])
        self.assertEqual(32000, simulator.opened_position_price[0])
        self.assertEqual(0, simulator.balance[0])

    def test_open_and_close_profitable_long_position(self):
        simulator = TradingSimulator(trading_cost_bps=0, time_cost_bps=0)

        observation = np.array([0, 32000])

        step = simulator.take_step(1, observation)

        self.assertTrue(step[1]['opened_position_price'] > 0)
        self.assertTrue(step[1]['opened_position_type'] > 0)

        observation = np.array([0, 34000])

        step = simulator.take_step(0, observation)

        self.assertFalse(step[1]['opened_position_price'] > 0)
        self.assertFalse(step[1]['opened_position_type'] > 0)
        self.assertEqual(2000, step[1]['balance'])
        self.assertEqual(2000, step[1]['pnl'])
        self.assertEqual(6.25, step[1]['reward'])

    def test_open_and_close_unprofitable_long_position(self):
        simulator = TradingSimulator(trading_cost_bps=0, time_cost_bps=0)

        observation = np.array([0, 32000])

        simulator.take_step(1, observation)

        observation = np.array([0, 30000])

        step = simulator.take_step(0, observation)

        self.assertEqual(-2000, step[1]['balance'])
        self.assertEqual(-2000, step[1]['pnl'])
        self.assertEqual(-6.25, step[1]['reward'])

    def test_open_and_close_profitable_short_position(self):
        simulator = TradingSimulator(trading_cost_bps=0, time_cost_bps=0)

        observation = np.array([0, 32000])

        step = simulator.take_step(2, observation)

        self.assertEqual(32000, step[1]['opened_position_price'])
        self.assertEqual(-1, step[1]['opened_position_type'])

        observation = np.array([0, 30000])

        step = simulator.take_step(0, observation)

        self.assertEqual(0, step[1]['opened_position_price'])
        self.assertEqual(0, step[1]['opened_position_type'])
        self.assertEqual(2000, step[1]['balance'])
        self.assertEqual(2000, step[1]['pnl'])
        self.assertEqual(6.25, step[1]['reward'])

    def test_open_and_close_unprofitable_short_position(self):
        simulator = TradingSimulator(trading_cost_bps=0, time_cost_bps=0)

        observation = np.array([0, 32000])

        step = simulator.take_step(2, observation)

        self.assertEqual(32000, step[1]['opened_position_price'])
        self.assertEqual(-1, step[1]['opened_position_type'])

        observation = np.array([0, 34000])

        step = simulator.take_step(0, observation)

        self.assertEqual(0, step[1]['opened_position_price'])
        self.assertEqual(0, step[1]['opened_position_type'])
        self.assertEqual(-2000, step[1]['balance'])
        self.assertEqual(-2000, step[1]['pnl'])
        self.assertEqual(-6.25, step[1]['reward'])

    def test_order_auto_close_on_last_step(self):
        simulator = TradingSimulator(steps=2, trading_cost_bps=0, time_cost_bps=0)

        observation = np.array([0, 32000])

        step = simulator.take_step(1, observation)

        self.assertEqual(1, step[1]['opened_position_type'])

        step = simulator.take_step(1, observation)

        self.assertEqual(0, step[1]['opened_position_type'])

    def test_open_long_order_and_open_short_order(self):
        simulator = TradingSimulator(steps=100, trading_cost_bps=0, time_cost_bps=0)

        observation = np.array([0, 32000])

        step = simulator.take_step(1, observation)

        self.assertEqual(1, step[1]['opened_position_type'])

        step = simulator.take_step(2, observation)

        self.assertEqual(-1, step[1]['opened_position_type'])
