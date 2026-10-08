"""
train_model.py — GitHub Actions Version
Uses synthetic GBM data (Yahoo/Stooq both block GitHub Actions IPs)
GBM = Geometric Brownian Motion — industry-standard market simulation
"""
import numpy as np
import tensorflow as tf
import os, warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def minmax(arr):
    return MinMaxScaler().fit_transform(arr.reshape(-1,1)).flatten()

def compute_rsi(prices, period=14):
    delta = np.diff(prices, prepend=prices[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).ewm(com=period-1, adjust=False).mean().values
    al = pd.Series(loss).ewm(com=period-1, adjust=False).mean().values
    return (100 - 100/(1 + ag/(al+1e-9))) / 100.0

def compute_macd(prices):
    s = pd.Series(prices)
    return minmax((s.ewm(span=12,adjust=False).mean() - s.ewm(span=26,adjust=False).mean()).values)

def generate_market(n_days=2520, start_price=10000.0, annual_vol=0.18, annual_drift=0.07, seed=42):
    """Simulate realistic market OHLCV using Geometric Brownian Motion."""
    np.random.seed(seed)
    dt   = 1/252
    mu   = annual_drift * dt
    sig  = annual_vol * np.sqrt(dt)
    r    = np.random.normal(mu, sig, n_days)
    close = start_price * np.cumprod(1 + r)
    high  = close * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    vol   = np.random.lognormal(mean=15, sigma=1, size=n_days)
    return close, high, low, vol

# Realistic params for each of 6 markets
MARKETS = [
    (10000, 0.15, 0.06, 1),   # TWII
    (2500,  0.18, 0.05, 2),   # KS11
    (20000, 0.20, 0.03, 3),   # HSI
    (60000, 0.20, 0.10, 4),   # BSESN
    (15000, 0.18, 0.07, 5),   # GDAXI
    (35000, 0.15, 0.08, 6),   # DJI
]

def get_training_data(time_steps=60):
    X_all, y_all = [], []
    print("Generating synthetic market data (GBM simulation)...")

    for i, (start_p, vol, drift, seed) in enumerate(MARKETS):
        close, high, low, volume = generate_market(
            n_days=2520, start_price=start_p,
            annual_vol=vol, annual_drift=drift, seed=seed
        )
        days = np.arange(len(close))

        rsi14  = compute_rsi(close)
        macd   = compute_macd(close)
        ema10  = pd.Series(close).ewm(span=10,adjust=False).mean().values
        ema30  = pd.Series(close).ewm(span=30,adjust=False).mean().values
        vola   = (high - low) / np.clip(close, 1, None)
        dow    = (days % 5) / 4.0   # 0=Mon…4=Fri

        nc = minmax(close)
        combined = np.column_stack([
            nc,
            minmax(volume),
            rsi14,
            macd,
            minmax(ema10),
            minmax(ema30),
            minmax(vola),
            dow,
        ])

        valid    = ~np.isnan(combined).any(axis=1)
        combined = combined[valid]
        nc       = nc[valid]

        for j in range(len(combined) - time_steps):
            X_all.append(combined[j:j+time_steps])
            y_all.append(nc[j+time_steps])

        print(f"  Market {i+1}/6: +{len(combined)-time_steps} samples")

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

    model.fit(X, y, epochs=20, batch_size=64, validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ], shuffle=True, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open("market_predictor.tflite", "wb") as f:
        f.write(converter.convert())
    print(f"Done: {os.path.getsize('market_predictor.tflite')//1024} KB")

if __name__ == "__main__":
    build_and_export()
