"""
train_model.py — GitHub Actions Version (ROOT)
================================================
Purpose : Runs automatically on GitHub every hour (scheduled via .github/workflows/train.yml)
          Downloads latest market data, trains Bidirectional LSTM, pushes market_predictor.tflite to repo
Output  : market_predictor.tflite  (same directory — GitHub Actions picks this up for commit)
Note    : Keep this file LEAN — GitHub Actions free tier has 2-core CPU, 7 GB RAM, 6-hour limit
"""

import numpy as np
import tensorflow as tf
import os
import warnings
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─── Technical Indicator Helpers ──────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return (100 - (100 / (1 + rs))) / 100.0  # → [0, 1]

def compute_macd(series):
    return series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# ─── Data Loading ─────────────────────────────────────────────────────────────

def get_training_data(time_steps=60):
    tickers = ["^TWII", "^KS11", "^HSI", "^BSESN", "^GDAXI", "^DJI"]
    X_all, y_all = [], []

    print("Fetching 10y daily data for all 6 markets...")
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  [{ticker}] skip: {e}"); continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.dropna(subset=["Close", "Volume", "High", "Low"])
        if len(df) < time_steps + 30:
            continue

        close = df["Close"]
        # Features: [NormClose, NormVol, RSI14, NormMACD, NormEMA10, NormEMA30, Volatility, DayOfWeek]
        s = lambda arr: MinMaxScaler().fit_transform(arr.values.reshape(-1,1)).flatten()
        combined = np.column_stack([
            s(close),
            s(df["Volume"]),
            compute_rsi(close).values,
            s(compute_macd(close)),
            s(compute_ema(close, 10)),
            s(compute_ema(close, 30)),
            ((df["High"] - df["Low"]) / close.clip(lower=1.0)).values,
            (pd.Series(df.index.dayofweek, index=df.index) / 4.0).values,
        ])
        norm_close = s(close)
        valid = ~np.isnan(combined).any(axis=1)
        combined, norm_close = combined[valid], norm_close[valid]
        if len(combined) < time_steps + 1: continue

        for i in range(len(combined) - time_steps):
            X_all.append(combined[i:i+time_steps])
            y_all.append(norm_close[i+time_steps])
        print(f"  [{ticker}] {len(combined)-time_steps} samples")

    if not X_all:
        raise RuntimeError("No data loaded — check internet.")
    print(f"Total: {len(X_all)} training samples\n")
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)

# ─── Model Build & Export ─────────────────────────────────────────────────────

def build_and_export():
    X, y = get_training_data()

    inp = tf.keras.Input(shape=(60, 8))
    x   = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inp)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False))(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    x   = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")

    model.fit(X, y, epochs=30, batch_size=64, validation_split=0.1,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
              ],
              shuffle=True, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("market_predictor.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"Saved market_predictor.tflite ({os.path.getsize('market_predictor.tflite')//1024} KB)")

if __name__ == "__main__":
    build_and_export()
