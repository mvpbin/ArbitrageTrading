import logging

def risk_management(current_capital, max_drawdown, initial_capital):
    drawdown = (initial_capital - current_capital) / initial_capital
    if drawdown > max_drawdown:
        logging.warning(f"警告：超过最大回撤限制{max_drawdown*100}%，当前回撤：{drawdown*100}%")
        return False
    return True