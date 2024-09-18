import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ta  # 引入ta库以计算技术指标

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_path = "C:/Users/mvpbi/Desktop/ArbitrageTrading/saved_model.keras"

# 获取市场数据
def fetch_market_data(exchange, symbol, timeframe='1d', limit=3650):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logging.error(f"获取市场数据失败: {e}")
        return pd.DataFrame()

# 数据预处理，加入技术指标特征
def preprocess_data(data):
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))

    # 计算RSI, SMA, EMA等技术指标
    data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
    data['sma'] = ta.trend.SMAIndicator(close=data['close'], window=20).sma_indicator()
    data['ema'] = ta.trend.EMAIndicator(close=data['close'], window=20).ema_indicator()

    # 特征归一化
    scaler = MinMaxScaler()
    features = ['close', 'log_return', 'volume', 'rsi', 'sma', 'ema']
    data[features] = scaler.fit_transform(data[features])
    data.dropna(inplace=True)
    return data

# 构建模型，增加LSTM层
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(None, input_shape)),
        tf.keras.layers.LSTM(64, activation='relu'),  # 添加第二层LSTM
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # 添加MeanAbsoluteError和MeanSquaredError作为指标
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError(), MeanSquaredError()])
    return model

# 训练模型并进行预测
def train_predict_model(data, retrain=False):
    if data.empty:
        logging.error("没有可用数据来训练模型或进行预测")
        return None

    data = preprocess_data(data)
    X = data.drop(columns=['close'])
    y = data['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.path.exists(model_path) and not retrain:
        model = tf.keras.models.load_model(model_path)
    else:
        model = build_model(X_train.shape[1])

    # 提前停止和学习率调度
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=2, 
              callbacks=[early_stopping, lr_scheduler])

    model.save(model_path)

    # 模型预测
    predict_x = X_test.iloc[-1:].values
    prediction = model.predict(predict_x)[0][0]

    # 评估模型性能
    evaluation = model.evaluate(X_test, y_test, verbose=2)
    logging.info(f"模型评估结果: {evaluation}")
    logging.info(f"Mean Absolute Error (MAE): {evaluation[1]}, Mean Squared Error (MSE): {evaluation[2]}")

    return prediction
