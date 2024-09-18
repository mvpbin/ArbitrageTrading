from abc import ABC, abstractmethod
from strategy_optimizer import StrategyOptimizer  # 引入策略优化器

class Strategy(ABC):
    @abstractmethod
    def execute(self, market_data, conn, symbol, amount, predicted_price):
        pass

class GridTradingStrategy(Strategy):
    def __init__(self):
        self.optimizer = StrategyOptimizer(self)  # 引入优化器

    def execute(self, market_data, conn, symbol, amount, predicted_price):
        # 使用优化器进行参数优化
        optimal_params = self.optimizer.optimize(market_data)
        grid_levels = [market_data['close'].iloc[-1] * (1 + 0.01 * i) for i in range(-5, 6)]
        for level in grid_levels:
            if market_data['close'].iloc[-1] <= level < predicted_price:
                profit = (predicted_price - market_data['close'].iloc[-1]) * amount
                log_trade(conn, symbol, amount, market_data['close'].iloc[-1], predicted_price, profit, 'Grid Trading')

class MeanReversionStrategy(Strategy):
    def __init__(self):
        self.optimizer = StrategyOptimizer(self)

    def execute(self, market_data, conn, symbol, amount, predicted_price):
        optimal_params = self.optimizer.optimize(market_data)
        if market_data['close'].iloc[-1] > optimal_params['threshold']:
            profit = (predicted_price - market_data['close'].iloc[-1]) * amount
            log_trade(conn, symbol, amount, market_data['close'].iloc[-1], predicted_price, profit, 'Mean Reversion')

class TrendFollowingStrategy(Strategy):
    def __init__(self):
        self.optimizer = StrategyOptimizer(self)

    def execute(self, market_data, conn, symbol, amount, predicted_price):
        optimal_params = self.optimizer.optimize(market_data)
        if market_data['close'].iloc[-1] > market_data['close'].mean():
            profit = (predicted_price - market_data['close'].iloc[-1]) * amount
            log_trade(conn, symbol, amount, market_data['close'].iloc[-1], predicted_price, profit, 'Trend Following')
