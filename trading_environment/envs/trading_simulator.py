import numpy as np
import pandas as pd


class TradingSimulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps=500, trading_cost_bps=0.01, time_cost_bps=0.01):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.balance = np.zeros(self.steps)
        self.pnl = np.zeros(self.steps)
        self.rewards = np.zeros(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.opened_position_price = np.zeros(self.steps)
        self.prices = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.balance.fill(0)
        self.rewards.fill(0)
        self.pnl.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.balance.fill(0)
        self.opened_position_price.fill(0)
        self.prices.fill(0)

    # Actions:
    # - 0: HOLD
    # - 1: LONG
    # - 2: SHORT

    def take_step(self, action, observation):

        [0, 1, 2].index(action)

        start_balance = self.balance[max(0, self.step - 1)]
        start_opened_position_price = self.opened_position_price[max(0, self.step - 1)]
        start_position = self.positions[max(0, self.step - 1)]

        self.prices[self.step] = observation[1]
        self.actions[self.step] = action
        self.opened_position_price[self.step] = start_opened_position_price
        self.balance[self.step] = start_balance
        self.positions[self.step] = start_position

        if self.step < 1:
            if action == 1:
                self.open_long_order()
            if action == 2:
                self.open_short_order()
        elif self.step + 1 >= self.steps:
            if self.is_order_opened():
                self.close_order()
        else:
            if action == 0:
                if self.is_order_opened():
                    self.close_order()
            elif action == 1:
                if start_position < 0:
                    self.close_order()
                if start_position < 1:
                    self.open_long_order()
            elif action == 2:
                if start_position > 0:
                    self.close_order()
                if start_position > -1:
                    self.open_short_order()

        time_cost = 0 if self.positions[self.step] else self.time_cost_bps

        self.costs[self.step] = self.costs[self.step] + time_cost
        self.rewards[self.step] -= self.costs[self.step]

        if self.is_opened_order_long():
            self.positions[self.step] = 1
        elif self.is_opened_order_short():
            self.positions[self.step] = -1

        reward = self.rewards[self.step]

        info = {
            'reward': reward,
            'pnl': self.pnl[self.step],
            'balance': self.balance[self.step],
            'costs': self.costs[self.step],
            'opened_position_type': self.positions[self.step],
            'opened_position_price': self.opened_position_price[self.step],
            'trades_count': self.trades_count(),
            'long_trades_count': self.long_trades_count(),
            'short_trades_count': self.short_trades_count()
        }

        self.step += 1
        return reward, info

    def open_long_order(self) -> None:
        if self.is_order_opened():
            raise Exception('Order already opened')

        self.opened_position_price[self.step] = self.get_current_price()
        self.trades[self.step] = 1
        self.positions[self.step] = 1
        self.costs[self.step] += self.trading_cost_bps

    def open_short_order(self) -> None:
        if self.is_order_opened():
            raise Exception('Order already opened')

        self.opened_position_price[self.step] = self.get_current_price()
        self.trades[self.step] = -1
        self.positions[self.step] = -1
        self.costs[self.step] += self.trading_cost_bps

    def close_order(self) -> None:
        if not self.is_order_opened():
            raise Exception('Order is not opened')

        open_order_cost = self.trading_cost_bps * self.opened_position_price[self.step] / 100
        close_order_cost = self.trading_cost_bps * self.prices[self.step] / 100

        if self.is_opened_order_long():
            self.pnl[self.step] = self.get_current_price() - self.opened_position_price[
                self.step] - open_order_cost - close_order_cost

            self.rewards[self.step] += self.get_current_price() * 100 / self.opened_position_price[self.step] - 100

        elif self.is_opened_order_short():
            self.pnl[self.step] = self.opened_position_price[
                                      self.step] - self.get_current_price() - open_order_cost - close_order_cost

            self.rewards[self.step] += 100 - self.get_current_price() * 100 / self.opened_position_price[self.step]

        self.balance[self.step] += self.pnl[self.step]
        self.opened_position_price[self.step] = 0
        self.positions[self.step] = 0
        self.costs[self.step] += self.trading_cost_bps

    def get_opened_position_price(self) -> int:
        return self.opened_position_price[max(0, self.step - 1)]

    def is_order_opened(self) -> bool:
        return self.opened_position_price[self.step] != 0

    def is_opened_order_long(self) -> bool:
        return self.positions[self.step] > 0

    def is_opened_order_short(self) -> bool:
        return self.positions[self.step] < 0

    def get_previous_action(self) -> int:
        return self.actions[max(0, self.step - 1)]

    def get_current_price(self) -> float:
        return self.prices[self.step]

    def result(self) -> pd.DataFrame:
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action': self.actions,
                             'reward': self.rewards,
                             'cost': self.costs,
                             'trades': self.trades,
                             'positions': self.positions,
                             'balance': self.balance,
                             'opened_position_price': self.opened_position_price,
                             'prices': self.prices})

    def long_trades_count(self) -> int:
        return len(self.trades[self.trades > 0])

    def short_trades_count(self) -> int:
        return len(self.trades[self.trades < 0])

    def trades_count(self) -> int:
        return len(self.trades[self.trades != 0])

    def get_trades(self) -> pd.DataFrame:

        start_position = None
        start_price = None
        trade = 0
        open_index = None
        close_price = None

        trades = pd.DataFrame(columns=['trade', 'open_price', 'close_price', 'open_index', 'close_index'])

        for index, row in self.result().iterrows():
            if row['positions'] != start_position:

                if row['positions'] == 1:
                    if start_position == -1:
                        close_price = row['prices']

                        trades = trades.append(
                            {'trade': trade, 'open_price': start_price, 'close_price': close_price,
                             'open_index': open_index,
                             'close_index': index}, ignore_index=True)

                    start_price = row['prices']
                    trade = row['trades']
                    open_index = index

                if row['positions'] == -1:
                    if start_position == 1:
                        close_price = row['prices']

                        trades = trades.append(
                            {'trade': trade, 'open_price': start_price, 'close_price': close_price,
                             'open_index': open_index,
                             'close_index': index}, ignore_index=True)

                    start_price = row['prices']
                    trade = row['trades']
                    open_index = index

                if row['positions'] == 0:
                    close_price = row['prices']

                    if close_price > 0:
                        trades = trades.append(
                            {'trade': trade, 'open_price': start_price, 'close_price': close_price,
                             'open_index': open_index,
                             'close_index': index}, ignore_index=True)

                        start_price = 0
                        trade = 0
                        open_index = None

            start_position = row['positions']

        return trades
