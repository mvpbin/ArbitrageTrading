import pandas as pd

def backtest_strategy(strategy, historical_data, initial_capital):
    capital = initial_capital
    for i in range(1, len(historical_data)):
        entry_price = historical_data['close'][i-1]
        exit_price = historical_data['close'][i]
        profit = strategy(entry_price, exit_price)
        capital += profit
    return capital