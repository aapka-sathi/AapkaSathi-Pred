"""
train_model.py — GitHub Actions Version
Uses Stooq via pandas-datareader (Yahoo Finance blocks GitHub IPs)
Output: market_predictor.tflite
"""
import numpy as np
import tensorflow as tf
import os, warnings
import pandas as pd
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Stooq symbols for our 6 markets (lowercase, no ^ prefix)
STOOQ_TICKERS = {
    "^TWII":  "^twii",
    "^KS11":  "^ks11",
    "^HSI":   "^hsi",
    "^BSESN": "^bsesn",
    "^GDAXI": "^gdaxi",
    "^DJI":   "^dji",
}

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(com=period-1, adjust=False).mean()
    al = loss.ewm(com=period-1, adjust=False).mean()
    return (100 - (100 / (1 + ag/(al+1e-9)))) / 100.0

def compute_macd(s):
    return s.ewm(span=12,adjust=False).mean() - s.ewm(span=26,adjust=False).mean()

def minmax(arr):
    return MinMaxScaler().fit_transform(arr.reshape(-1,1)).flatten()

def get_training_data(time_steps=60):
    X_all, y_all = [], []
    end   = datetime.today()
    start = end - timedelta(days=365*10)

    print("Fetching 10y daily data via Stooq...")
    for ticker, stooq_sym in STOOQ_TICKERS.items():
        try:
            df = web.DataReader(stooq_sym, "stooq", start, end)
            df = df.sort_index()  # Stooq returns newest-first
        except Exception as e:
            print(f"  [{ticker}] skip: {e}"); continue

        df = df.dropna()
        if len(df) < time_steps + 30:
            print(f"  [{ticker}] only {len(df)} rows — skipping"); continue

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)

        rsi14 = compute_rsi(close)
        macd  = compute_macd(close)
        ema10 = close.ewm(span=10,adjust=False).mean()
        ema30 = close.ewm(span=30,adjust=False).mean()
        vola  = (high - low) / close.clip(lower=1.0)
        dow   = pd.Series(df.index.dayofweek, index=df.index) / 4.0

        nc = minmax(close.values)
        combined = np.column_stack([
            nc,
            minmax(vol.values),
            rsi14.values,
            minmax(macd.values),
            minmax(ema10.values),
            minmax(ema30.values),
            minmax(vola.values),
            dow.values,
        ])

        valid    = ~np.isnan(combined).any(axis=1)
        combined = combined[valid]
        nc       = nc[valid]
        if len(combined) < time_steps + 1: continue

        for i in range(len(combined) - time_steps):
            X_all.append(combined[i:i+time_steps])
            y_all.append(nc[i+time_steps])

        print(f"  [{ticker}] +{len(combined)-time_steps} samples")

    if not X_all:
        raise RuntimeError("No data loaded from Stooq.")
    print(f"Total: {len(X_all)} samples")
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)

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
        ], shuffle=True, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open("market_predictor.tflite", "wb") as f:
        f.write(converter.convert())
    print(f"Saved {os.path.getsize('market_predictor.tflite')//1024} KB")

if __name__ == "__main__":
    build_and_export()
