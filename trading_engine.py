import pandas as pd
import numpy as np

def calculate_indicators(df, ema_period=200, rsi_period=14, atr_period=10):
    if df.empty:
        return df

    # Exponential Moving Average (EMA)
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD 4C (Histogram)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(atr_period).mean()

    # Bollinger Bands
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['upper_band'] = df['sma20'] + (df['std20'] * 2)
    df['lower_band'] = df['sma20'] - (df['std20'] * 2)

    # Volume Weighted Average Price (VWAP)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)

    # Williams Fractals
    df['fractal_high'] = np.nan
    df['fractal_low'] = np.nan
    
    for i in range(2, len(df) - 2):
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
            df['high'].iloc[i] > df['high'].iloc[i-2] and 
            df['high'].iloc[i] > df['high'].iloc[i+1] and 
            df['high'].iloc[i] > df['high'].iloc[i+2]):
            df['fractal_high'].iloc[i] = df['high'].iloc[i]
            
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
            df['low'].iloc[i] < df['low'].iloc[i-2] and 
            df['low'].iloc[i] < df['low'].iloc[i+1] and 
            df['low'].iloc[i] < df['low'].iloc[i+2]):
            df['fractal_low'].iloc[i] = df['low'].iloc[i]

    return df

def calculate_position_size(account_balance, risk_percent, entry_price, stop_loss_price):
    risk_amount = account_balance * (risk_percent / 100.0)
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0.0
    return risk_amount / risk_per_unit
