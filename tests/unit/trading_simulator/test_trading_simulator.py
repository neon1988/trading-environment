import unittest

from trading_environment.envs.trading_simulator import TradingSimulator


class TestTraidingSimulator(unittest.TestCase):
    def test_init(self):
        steps = 10

        simulator = TradingSimulator(steps=steps,
                                     trading_cost_bps=0.01,
                                     time_cost_bps=0.01)

        self.assertIsInstance(simulator, TradingSimulator)
        self.assertEqual(simulator.step, 0)
        self.assertEqual(steps, len(simulator.actions))
        self.assertEqual(steps, len(simulator.balance))
        self.assertEqual(steps, len(simulator.pnl))
        self.assertEqual(steps, len(simulator.rewards))
        self.assertEqual(steps, len(simulator.positions))
        self.assertEqual(steps, len(simulator.costs))
        self.assertEqual(steps, len(simulator.trades))
        self.assertEqual(steps, len(simulator.opened_position_price))
        self.assertEqual(steps, len(simulator.prices))

    def test_reset(self):
        steps = 10

        simulator = TradingSimulator(steps=steps,
                                     trading_cost_bps=0.01,
                                     time_cost_bps=0.01)

        simulator.reset()

        self.assertEqual(0, simulator.step)
