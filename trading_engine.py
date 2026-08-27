import pandas as pd
import numpy as np
import hmac
import hashlib
import time
import requests
import urllib.parse
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

DB_NAME = "quantclaw_journal.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            size REAL,
            pnl REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_trade_to_db(symbol, side, entry_price, size, status="Active"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO trades (timestamp, symbol, side, entry_price, size, pnl, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, side, entry_price, size, 0.0, status))
    conn.commit()
    conn.close()

def get_trades_from_db():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    return df

def clear_trades_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM trades")
    conn.commit()
    conn.close()

# --- Advanced Quantitative Feature Engineering & Multi-AI Models ---
def calculate_indicators(df, ema_period=200, rsi_period=14, atr_period=10):
    if df.empty or len(df) < max(ema_period, rsi_period, atr_period, 50):
        df['ai_score'] = 50.0
        df['lstm_score'] = 50.0
        df['sentiment_score'] = 50.0
        df['rl_action'] = "HOLD"
        df['model_accuracy'] = 50.0
        return df

    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()

    # RSI & Momentum
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR & Volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(atr_period).mean()

    # Bollinger Bands & %B Feature
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['upper_band'] = df['sma20'] + (df['std20'] * 2)
    df['lower_band'] = df['sma20'] - (df['std20'] * 2)
    df['bb_percent'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'] + 1e-9)

    # Rate of Change (ROC)
    df['roc'] = df['close'].pct_change(periods=5) * 100

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)

    # 1. Enhanced Random Forest ML Classifier with Extended Features
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    ml_features = ['rsi', 'macd_hist', 'atr', 'bb_percent', 'roc']
    clean_ml = df.dropna(subset=ml_features + ['target'])

    model_accuracy = 52.0
    if len(clean_ml) > 40:
        X = clean_ml[ml_features]
        y = clean_ml['target']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X_scaled, y)
        
        # Calculate historical accuracy score mock metric
        preds = model.predict(X_scaled)
        model_accuracy = float(np.mean(preds == y) * 100)

        all_X_scaled = scaler.transform(df[ml_features].fillna(0))
        df['ai_score'] = model.predict_proba(all_X_scaled)[:, 1] * 100
    else:
        df['ai_score'] = 50.0

    df['model_accuracy'] = model_accuracy

    # 2. Enhanced LSTM Sequential Pattern Simulation
    df['lstm_score'] = np.clip(df['ai_score'] * 0.4 + (df['close'] / df['ema'] - 1) * 100 + 50, 10, 95)

    # 3. Market Sentiment Analyzer (FinBERT Simulated Core)
    df['sentiment_score'] = np.clip(50 + (df['rsi'] - 50) * 0.7 + df['roc'] * 1.5 + np.random.normal(0, 3, len(df)), 10, 95)

    # 4. Multi-AI Consensus Policy (Reinforcement Learning Decision Matrix)
    consensus_score = (df['ai_score'] + df['lstm_score'] + df['sentiment_score']) / 3
    conditions = [
        consensus_score > 58,
        consensus_score < 42
    ]
    choices = ["BUY", "SELL"]
    df['rl_action'] = np.select(conditions, choices, default="HOLD")

    return df

def calculate_position_size(account_balance, risk_percent, entry_price, stop_loss_price):
    risk_amount = account_balance * (risk_percent / 100.0)
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0.0
    return risk_amount / risk_per_unit

def execute_binance_order(api_key, api_secret, symbol, side, quantity, testnet=True):
    base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
    endpoint = "/fapi/v1/order"
    
    clean_symbol = symbol.replace("-USD", "").replace("/", "")
    if "BTC" in clean_symbol and not clean_symbol.endswith("USDT"):
        clean_symbol = "BTCUSDT"
    elif "ETH" in clean_symbol and not clean_symbol.endswith("USDT"):
        clean_symbol = "ETHUSDT"

    params = {
        "symbol": clean_symbol,
        "side": side.upper(),
        "type": "MARKET",
        "quantity": f"{quantity:.3f}",
        "timestamp": int(time.time() * 1000)
    }
    
    query_string = urllib.parse.urlencode(params)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params["signature"] = signature
    headers = {"X-MBX-APIKEY": api_key}
    
    try:
        response = requests.post(base_url + endpoint, headers=headers, params=params, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def poll_telegram_commands(token):
    if not token:
        return []
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("ok"):
            return res.get("result", [])
    except Exception:
        pass
    return []
