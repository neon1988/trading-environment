import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


class DataSource:
    """
    Data source for TradingEnvironment
    """

    def __init__(self, data, steps=500, normalize=True):
        self.steps = steps
        self.normalize = normalize
        self.data = data
        self.preprocess_data()
        self.min_values = self.data.min()
        self.max_values = self.data.max()
        self.step = 0
        self.offset = None

    def preprocess_data(self):
        """calculate returns and percentiles, then removes missing values"""

        self.data['returns'] = self.data.close.pct_change()
        # self.data['ret_2'] = self.data.close.pct_change(2)
        # self.data['ret_5'] = self.data.close.pct_change(5)
        # self.data['ret_10'] = self.data.close.pct_change(10)
        # self.data['ret_21'] = self.data.close.pct_change(21)

        self.data = self.data.fillna(0)

        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['high', 'low', 'volume'], axis=1)
                     .dropna())

        r = self.data.returns.copy()
        close = self.data.close.copy()

        if self.normalize:
            self.data = pd.DataFrame(scale(self.data),
                                     columns=self.data.columns,
                                     index=self.data.index)

        features = self.data.columns.drop(['returns', 'close'])

        self.data['returns'] = r  # don't scale returns
        self.data['close'] = close  # don't scale close

        self.data = self.data.loc[:, ['returns', 'close'] + list(features)]

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.data.index) - self.steps
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step].values
        self.step += 1
        done = self.step > self.steps
        return obs, done
