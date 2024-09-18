import numpy as np
import pandas as pd

class Backtest:
    def __init__(self, initial_balance):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_curve = []
        self.trade_log = []
        self.current_drawdown = 0
        self.max_drawdown = 0

    def execute_trade(self, entry_price, exit_price, amount, is_long=True):
        if is_long:
            profit = (exit_price - entry_price) * amount
        else:
            profit = (entry_price - exit_price) * amount

        self.balance += profit
        self.equity_curve.append(self.balance)
        self.trade_log.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'balance': self.balance
        })

        self.calculate_drawdown()

    def calculate_drawdown(self):
        peak = max(self.equity_curve, default=self.initial_balance)
        drawdown = peak - self.balance
        self.current_drawdown = drawdown
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def calculate_metrics(self):
        total_profit = self.balance - self.initial_balance
        total_trades = len(self.trade_log)
        win_trades = len([t for t in self.trade_log if t['profit'] > 0])
        loss_trades = total_trades - win_trades
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t['profit'] for t in self.trade_log if t['profit'] > 0], default=0)
        avg_loss = np.mean([t['profit'] for t in self.trade_log if t['profit'] < 0], default=0)

        # 盈亏比
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        return {
            'total_profit': total_profit,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': self.max_drawdown,
            'balance': self.balance
        }

# 示例如何使用 Backtest 类：
# backtest = Backtest(initial_balance=10000)
# backtest.execute_trade(100, 110, 1, is_long=True)
# metrics = backtest.calculate_metrics()
# print(metrics)
