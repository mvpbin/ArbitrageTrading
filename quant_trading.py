import ccxt
import json
import sys
import logging
import sqlite3
from ml_model import train_predict_model, fetch_market_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_name='config.json'):
    with open(file_name, 'r') as file:
        return json.load(file)

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        logging.error(f"数据库连接失败: {e}")
        return None

def create_table(conn):
    try:
        sql_create_trades_table = """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            amount REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            profit REAL NOT NULL,
            strategy TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        ); """
        c = conn.cursor()
        c.execute(sql_create_trades_table)
    except Exception as e:
        logging.error(f"创建表失败: {e}")

def fetch_prices(binance_exchange, okx_exchange, symbol):
    try:
        binance_price = binance_exchange.fetch_ticker(symbol)['last']
        okx_price = okx_exchange.fetch_ticker(symbol)['last']
        return binance_price, okx_price
    except Exception as e:
        logging.error(f"获取价格失败: {e}")
        return None, None

def check_price_change(binance_price, okx_price, threshold=0.01):
    price_change = abs(binance_price - okx_price) / min(binance_price, okx_price)
    return price_change >= threshold

def execute_trades(exchange, conn, symbol, amount, initial_capital, max_drawdown):
    binance_price, okx_price = fetch_prices(exchange['binance'], exchange['okx'], symbol)

    if binance_price is None or okx_price is None:
        logging.error("无法获取价格，跳过此次交易尝试。")
        return

    volatility = fetch_market_data(exchange['binance'], symbol)['close'].std()
    dynamic_threshold = 0.01 * (1 + volatility)
    
    if not check_price_change(binance_price, okx_price, dynamic_threshold):
        logging.info(f"{symbol}价格变动低于动态阈值 {dynamic_threshold:.2%}，跳过交易。")
        return

    market_data = fetch_market_data(exchange['binance'], symbol)
    if market_data.empty:
        logging.error("无法获取有效的市场数据，跳过此次模型预测。")
        return
    
    predicted_price = train_predict_model(market_data)
    if predicted_price is None:
        logging.info("模型未能提供有效预测，跳过此次交易。")
        return

    logging.info(f"{symbol}的预测未来价格为：{predicted_price}")

    adjusted_amount = amount / (1 + volatility)

    # 新增网格交易策略
    grid_levels = [binance_price * (1 + 0.01 * i) for i in range(-5, 6)]
    for level in grid_levels:
        if okx_price <= level < predicted_price:
            execute_grid_trade(conn, symbol, adjusted_amount, okx_price, predicted_price, level)

    # 新增均值回复策略
    if binance_price > predicted_price:
        execute_mean_reversion(conn, symbol, adjusted_amount, okx_price, predicted_price)
    
    # 趋势跟踪策略
    if market_data['close'].iloc[-1] > market_data['close'].mean():
        execute_trend_following(conn, symbol, adjusted_amount, okx_price, predicted_price)

def execute_grid_trade(conn, symbol, amount, entry_price, exit_price, level):
    profit = (exit_price - entry_price) * amount
    log_trade(conn, symbol, amount, entry_price, exit_price, profit, 'Grid Trading')

def execute_mean_reversion(conn, symbol, amount, entry_price, predicted_price):
    profit = (predicted_price - entry_price) * amount
    log_trade(conn, symbol, amount, entry_price, predicted_price, profit, 'Mean Reversion')

def execute_trend_following(conn, symbol, amount, entry_price, predicted_price):
    profit = (predicted_price - entry_price) * amount
    log_trade(conn, symbol, amount, entry_price, predicted_price, profit, 'Trend Following')

def risk_management(current_capital, max_drawdown, initial_capital):
    drawdown = (initial_capital - current_capital) / initial_capital
    if drawdown > max_drawdown:
        logging.warning(f"警告：超过最大回撤限制{max_drawdown*100}%，当前回撤：{drawdown*100}%")
        return False
    return True

def log_trade(conn, symbol, amount, entry_price, exit_price, profit, strategy):
    sql = '''INSERT INTO trades(symbol, amount, entry_price, exit_price, profit, strategy)
              VALUES(?,?,?,?,?,?)'''
    try:
        cur = conn.cursor()
        cur.execute(sql, (symbol, amount, entry_price, exit_price, profit, strategy))
        conn.commit()
        logging.info(f"交易记录已保存: {symbol}, {amount}, {entry_price}, {exit_price}, {profit:.2f} USDT, 策略: {strategy}")
    except sqlite3.Error as e:
        logging.error(f"记录交易到数据库失败: {e}")
    finally:
        if cur: cur.close()

if __name__ == "__main__":
    config = load_config()
    exchange = {
        'binance': ccxt.binance(config['binance']),
        'okx': ccxt.okx(config['okx']),
    }
    conn = create_connection("trading.db")
    create_table(conn)
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    while True:
        for symbol in symbols:
            execute_trades(exchange, conn, symbol, 0.001, 1000.0, 0.1)
