import os
import unittest

import numpy as np
import pandas as pd

from trading_environment.envs.data_source import DataSource


class TestDataSource(unittest.TestCase):
    def test_init(self):
        data_source = DataSource(data=self.get_data(), normalize=False)

        self.assertIsInstance(data_source, DataSource)

        self.assertAlmostEqual(-0.06020266, data_source.min_values['returns'], 5)
        self.assertAlmostEqual(0.141000, data_source.min_values['close'], 5)
        self.assertAlmostEqual(0.027000, data_source.min_values['open'], 5)

        self.assertAlmostEqual(27.382977, data_source.max_values['returns'], 5)
        self.assertAlmostEqual(5.288000, data_source.max_values['close'], 5)
        self.assertAlmostEqual(5.288000, data_source.max_values['open'], 5)

    def test_normalize(self):
        data_source = DataSource(data=self.get_data(), normalize=True)

        self.assertAlmostEqual(-0.06020266, data_source.min_values['returns'], 5)
        self.assertAlmostEqual(0.141000, data_source.min_values['close'], 5)
        self.assertAlmostEqual(-6.251915, data_source.min_values['open'], 5)

        self.assertAlmostEqual(27.382977, data_source.max_values['returns'], 5)
        self.assertAlmostEqual(5.288000, data_source.max_values['close'], 5)
        self.assertAlmostEqual(0.917148, data_source.max_values['open'], 5)

    def test_reset(self):
        data_source = DataSource(steps=30, data=self.get_data(), normalize=True)

        data_source.offset = 42
        data_source.step = 24

        data_source.reset()

        self.assertEqual(0, data_source.step)

    def get_data(self):
        dirname = os.path.dirname(__file__)
        file = f'{dirname}/../../assets/10set_usdt_1800.csv'

        df = pd.read_csv(file,
                         usecols=[0, 1, 2, 3, 4, 5],
                         names=['timestamp', 'open', 'close', 'low', 'high', 'volume'],
                         decimal='.', keep_default_na=False,
                         dtype={'timestamp': np.int32, 'open': np.float32, 'close': np.float32, 'low': np.float32,
                                'high': np.float32},
                         encoding="UTF-8", index_col='timestamp').sort_values(by='timestamp')

        return df

    def test_take_step(self):
        data_source = DataSource(steps=3, data=self.get_data(), normalize=True)

        data_source.reset()
        data_source.offset = 3

        step1 = data_source.take_step()

        self.assertAlmostEqual(0.05805516, step1[0][0], 5)
        self.assertAlmostEqual(4.374, step1[0][1], 5)
        self.assertAlmostEqual(-0.6581107, step1[0][2], 5)

    def test_done_after_equal_number_of_steps(self):
        data_source = DataSource(steps=3, data=self.get_data(), normalize=True)

        data_source.reset()
        data_source.offset = 3

        step1 = data_source.take_step()
        step2 = data_source.take_step()
        step3 = data_source.take_step()
        step4 = data_source.take_step()
        step5 = data_source.take_step()

        self.assertFalse(step1[1])
        self.assertFalse(step2[1])
        self.assertFalse(step3[1])
        self.assertTrue(step4[1])
        self.assertTrue(step5[1])

    def test_get_high(self):
        data_source = DataSource(steps=3, data=self.get_data(), normalize=False)
        self.assertTrue(type(data_source.get_high()) == int)

    def test_set_offset(self):
        data_source = DataSource(steps=3, data=self.get_data(), normalize=False)

        data_source.set_offset(3)

        self.assertEqual(3, data_source.offset)
