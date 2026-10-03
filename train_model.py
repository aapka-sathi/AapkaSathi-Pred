import numpy as np
import tensorflow as tf
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def get_real_data(time_steps=60):
    tickers = ["^TWII", "^KS11", "^HSI", "^BSESN", "^GDAXI", "^DJI"]
    X_all, y_all = [], []
    
    print("Downloading 10 years of historical data for AI training...")
    for ticker in tickers:
        df = yf.download(ticker, period="10y", interval="1d", progress=False)
        if df.empty or len(df) < time_steps + 1:
            continue
            
        close_prices = df['Close'].values.reshape(-1, 1)
        volumes = df['Volume'].values.reshape(-1, 1)
        
        # Simple normalization per ticker
        scaler_c = MinMaxScaler()
        scaler_v = MinMaxScaler()
        
        norm_close = scaler_c.fit_transform(close_prices)
        norm_vol = scaler_v.fit_transform(volumes)
        
        # Combine into features: 8 features expected by our Android app
        # [Close, Vol, Dummy, Dummy, Dummy, Dummy, Dummy, Dummy]
        # (Filling remaining 6 features with 0.5 to match the Android FeatureExtractor shape)
        combined = np.hstack((norm_close, norm_vol, np.full((len(norm_close), 6), 0.5)))
        
        for i in range(len(combined) - time_steps):
            X_all.append(combined[i : i + time_steps])
            y_all.append(norm_close[i + time_steps])
            
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)

def build_and_export_model():
    print("Building lightweight LSTM for 2GB RAM devices...")
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(16, input_shape=(60, 8), return_sequences=False),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    X, y = get_real_data()
    print(f"Training on {len(X)} real historical market patterns...")
    
    # Train model (Epochs=5 for speed, batch_size=32)
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save purely to the current working directory for GitHub automation
    with open("market_predictor.tflite", "wb") as f:
        f.write(tflite_model)
    print("Successfully exported trained Real AI to market_predictor.tflite in current directory!")

if __name__ == "__main__":
    build_and_export_model()
