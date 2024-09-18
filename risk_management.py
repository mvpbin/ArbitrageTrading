import logging

class RiskManagement:
    def __init__(self, initial_capital, max_drawdown, stop_loss_percentage, trailing_stop_percentage):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown = max_drawdown
        self.stop_loss_percentage = stop_loss_percentage
        self.trailing_stop_percentage = trailing_stop_percentage
        self.peak_capital = initial_capital
        self.stop_loss_level = initial_capital * (1 - stop_loss_percentage)
        self.trailing_stop_level = None

    def update_capital(self, new_capital):
        self.current_capital = new_capital

        # 更新最大资本峰值
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # 更新动态止损线
        self.stop_loss_level = self.peak_capital * (1 - self.stop_loss_percentage)

        # 更新动态盈利保护（跟踪止损线）
        if self.trailing_stop_percentage is not None:
            self.trailing_stop_level = self.peak_capital * (1 - self.trailing_stop_percentage)

    def check_risk(self):
        # 计算当前回撤
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        # 检查是否超过最大回撤
        if drawdown > self.max_drawdown:
            logging.warning(f"警告：超过最大回撤限制{self.max_drawdown * 100}%，当前回撤：{drawdown * 100}%")
            return False

        # 检查是否触发止损线
        if self.current_capital < self.stop_loss_level:
            logging.warning(f"警告：当前资本低于止损线，当前资本：{self.current_capital}，止损线：{self.stop_loss_level}")
            return False

        # 检查是否触发跟踪止损线
        if self.trailing_stop_level and self.current_capital < self.trailing_stop_level:
            logging.warning(f"警告：触发跟踪止损，当前资本：{self.current_capital}，跟踪止损线：{self.trailing_stop_level}")
            return False

        return True

    def log_status(self):
        logging.info(f"当前资本：{self.current_capital}，最高资本：{self.peak_capital}")
        logging.info(f"当前止损线：{self.stop_loss_level}，当前跟踪止损线：{self.trailing_stop_level if self.trailing_stop_level else '未设置'}")
