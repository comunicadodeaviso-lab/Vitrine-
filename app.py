"""
crypto_signal_rf.py
Script de exemplo para prever sinais de alta/baixa em cripto usando ccxt + indicadores técnicos + RandomForest.
Use com responsabilidade. Teste em dados históricos e paper-trade antes de operar com dinheiro real.
"""

import ccxt
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
from datetime import datetime

# ---------- CONFIG ----------
EXCHANGE_ID = "binance"      # ou 'kraken', 'coinbasepro', etc (via ccxt)
SYMBOL = "BTC/USDT"          # símbolo comum
TIMEFRAME = "15m"            # timeframe dos candles (1m,5m,15m,1h,4h,1d...)
LIMIT = 1000                 # quantos candles históricos buscar (ajuste)
MODEL_PATH = "rf_crypto_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
# ----------------------------

def fetch_ohlcv(exchange, symbol, timeframe, limit=500):
    """
    Retorna DataFrame com colunas: timestamp, open, high, low, close, volume
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def add_indicators(df):
    df = df.copy()
    # SMA and EMA
    df['sma_10'] = SMAIndicator(df['close'], window=10).sma_indicator()
    df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['ema_10'] = EMAIndicator(df['close'], window=10).ema_indicator()
    df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    # RSI
    df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()
    # MACD
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    # Bollinger bands
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    # Price returns and other helpers
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_3'] = df['close'].pct_change(3)
    df['vol_1'] = df['volume'].pct_change(1)
    # Drop NA
    df = df.dropna()
    return df

def create_features_labels(df):
    df = df.copy()
    # Label: next candle close > current close -> 1 else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()  # drop last row where target is NA
    feature_cols = [
        'sma_10','sma_50','ema_10','ema_50',
        'rsi_14','macd','macd_signal',
        'bb_h','bb_l','returns_1','returns_3','vol_1'
    ]
    X = df[feature_cols]
    y = df['target']
    return X, y, df

def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=TEST_SIZE, shuffle=False)
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds))
    return model, scaler, X_train, X_test, y_train, y_test

def generate_signals(model, scaler, X):
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:,1]  # prob of label=1 (up)
    preds = model.predict(Xs)
    signals = pd.DataFrame({'prob_up': probs, 'pred': preds}, index=X.index)
    # Decision rule example: buy if prob_up > 0.6 and pred==1; sell if prob_up<0.4 and pred==0
    signals['signal'] = 0
    signals.loc[signals['prob_up'] > 0.6, 'signal'] = 1   # BUY
    signals.loc[signals['prob_up'] < 0.4, 'signal'] = -1  # SELL
    return signals

def simple_backtest(df, signals):
    # Assume entering at next candle open and exiting after 1 candle (example)
    df_bt = df.copy().join(signals, how='inner')
    df_bt['next_open'] = df_bt['close'].shift(-1)  # approximation: use next close as next price if open not available
    df_bt = df_bt.dropna()
    # strategy returns: for buy signal, (next_open / close) -1 ; for sell signal inverse (short) - simplified
    df_bt['strat_ret'] = 0.0
    buy_mask = df_bt['signal'] == 1
    sell_mask = df_bt['signal'] == -1
    df_bt.loc[buy_mask, 'strat_ret'] = df_bt.loc[buy_mask, 'next_open'] / df_bt.loc[buy_mask, 'close'] - 1
    df_bt.loc[sell_mask, 'strat_ret'] = - (df_bt.loc[sell_mask, 'next_open'] / df_bt.loc[sell_mask, 'close'] - 1)
    df_bt['cum_ret'] = (1 + df_bt['strat_ret']).cumprod()
    total_return = df_bt['cum_ret'].iloc[-1] - 1 if len(df_bt)>0 else 0
    print(f"Simple strategy total return (multiplicative): {total_return:.4f}")
    return df_bt

def main():
    # init exchange
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({
        # 'apiKey': 'YOUR_KEY', 'secret': 'YOUR_SECRET',   # só necessário para endpoints privados
        'enableRateLimit': True,
    })

    print(f"Fetching {LIMIT} candles for {SYMBOL} {TIMEFRAME} from {EXCHANGE_ID}...")
    df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, limit=LIMIT)
    print("Candles fetched:", len(df))
    df_ind = add_indicators(df)
    X, y, df_full = create_features_labels(df_ind)

    model, scaler, X_train, X_test, y_train, y_test = train_model(X, y)
    # salvar modelo e scaler
    joblib.dump({'model': model, 'scaler': scaler}, MODEL_PATH)
    print("Modelo salvo em", MODEL_PATH)

    # gerar sinais sobre todo o dataset (ou usar apenas últimos N)
    signals = generate_signals(model, scaler, X)
    df_bt = simple_backtest(df_full, signals)

    # exemplo: mostrar últimos sinais
    last = df_full.join(signals).tail(10)[['close','prob_up','pred','signal']]
    print("Últimos sinais:\n", last)

if __name__ == "__main__":
    main()

