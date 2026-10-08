"""
train_model.py — GitHub Actions Version
Runs hourly via .github/workflows/train.yml
Output: market_predictor.tflite (committed back to repo for OTA delivery)
"""

import numpy as np
import tensorflow as tf
import os
import warnings
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Technical Indicator Helpers ───────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return (100 - (100 / (1 + rs))) / 100.0   # → [0, 1]

def compute_macd(series):
    return series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def minmax_norm(arr):
    scaler = MinMaxScaler()
    return scaler.fit_transform(arr.reshape(-1, 1)).flatten()

# ── Data Loading ──────────────────────────────────────────────────────────────

def get_training_data(time_steps=60):
    tickers = ["^TWII", "^KS11", "^HSI", "^BSESN", "^GDAXI", "^DJI"]
    X_all, y_all = [], []

    print("Fetching 10y daily data for all 6 markets...")

    # Use a plain requests session — avoids Chrome impersonation entirely
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                period="10y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                session=session        # bypass Chrome impersonation
            )
        except Exception as e:
            print(f"  [{ticker}] skip: {e}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.dropna(subset=["Close", "Volume", "High", "Low"])
        if len(df) < time_steps + 30:
            print(f"  [{ticker}] only {len(df)} rows — skipping")
            continue

        close = df["Close"]

        # 8 Features matching FeatureExtractor.java exactly
        rsi14     = compute_rsi(close)
        macd_line = compute_macd(close)
        ema10     = compute_ema(close, 10)
        ema30     = compute_ema(close, 30)
        volatility = (df["High"] - df["Low"]) / close.clip(lower=1.0)
        dow       = pd.Series(df.index.dayofweek, index=df.index) / 4.0

        norm_close = minmax_norm(close.values)
        combined   = np.column_stack([
            norm_close,                          # F0
            minmax_norm(df["Volume"].values),    # F1
            rsi14.values,                        # F2
            minmax_norm(macd_line.values),       # F3
            minmax_norm(ema10.values),           # F4
            minmax_norm(ema30.values),           # F5
            minmax_norm(volatility.values),      # F6
            dow.values,                          # F7
        ])

        valid      = ~np.isnan(combined).any(axis=1)
        combined   = combined[valid]
        norm_close = norm_close[valid]

        if len(combined) < time_steps + 1:
            continue

        for i in range(len(combined) - time_steps):
            X_all.append(combined[i : i + time_steps])
            y_all.append(norm_close[i + time_steps])

        print(f"  [{ticker}] +{len(combined) - time_steps} samples")

    if not X_all:
        raise RuntimeError("No data loaded — all tickers failed. Check internet/Yahoo API.")

    print(f"Total: {len(X_all)} training samples")
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)

# ── Model ─────────────────────────────────────────────────────────────────────

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

    model.fit(
        X, y,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ],
        shuffle=True,
        verbose=1
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("market_predictor.tflite", "wb") as f:
        f.write(tflite_model)

    print(f"Saved market_predictor.tflite ({os.path.getsize('market_predictor.tflite') // 1024} KB)")

if __name__ == "__main__":
    build_and_export()
