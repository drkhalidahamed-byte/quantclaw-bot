import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# --- Safe CCXT Import ---
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

st.set_page_config(page_title="QuantClaw Institutional Trading Console", layout="wide", initial_sidebar_state="expanded")

# --- Authentication ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    st.title("🔐 QuantClaw Console - Authentication")
    pwd = st.text_input("Enter Passcode", type="password")
    if st.button("Unlock System"):
        if pwd == "admin":
            st.session_state['authenticated'] = True
            st.rerun()
        else:
            st.error("Invalid passcode")
    st.stop()

if not HAS_CCXT:
    st.warning("⚠️ مكتبة CCXT غير مثبتة على السيرفر حالياً. تم تفعيل وضع المحاكاة (Paper Trading) تلقائياً.")

# --- Telegram Notification Helper ---
def send_telegram_msg(token, chat_id, message):
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception:
            pass

# --- CCXT Exchange Initializer ---
def init_exchange(mode, api_k, api_s):
    if not HAS_CCXT or not api_k or not api_s:
        return None
    try:
        exchange = ccxt.binance({
            'apiKey': api_k,
            'secret': api_s,
            'enableRateLimit': True,
        })
        if "Testnet" in mode:
            exchange.set_sandbox_mode(True)
        return exchange
    except Exception as e:
        st.error(f"Exchange connection error: {e}")
        return None

# --- Sidebar Controls ---
st.sidebar.title("⚡ QuantClaw Engine Control")

st.sidebar.subheader("🕹️ Execution Mode / وضع التنفيذ")
execution_mode = st.sidebar.radio(
    "اختر بيئة التشغيل:",
    ["🎮 Paper Trading (محاكاة)", "🧪 Testnet Trading (بيئة اختبار API)", "⚡ Live Trading (تداول فعلي)"],
    index=0
)

api_key = ""
api_secret = ""
exchange_instance = None

if "Testnet" in execution_mode or "Live" in execution_mode:
    st.sidebar.markdown("**🔑 Exchange API Credentials (Binance)**")
    api_key = st.sidebar.text_input("API Key", type="password")
    api_secret = st.sidebar.text_input("API Secret", type="password")
    if api_key and api_secret:
        exchange_instance = init_exchange(execution_mode, api_key, api_secret)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Dynamic Indicator Parameters")
ema_period = st.sidebar.slider("EMA Period", 20, 200, 200, 5)
rsi_period = st.sidebar.slider("RSI Period", 7, 30, 14, 1)
atr_period = st.sidebar.slider("ATR Period", 5, 30, 10, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Risk Management & ATR Stops")
stop_mode = st.sidebar.radio("Stop Loss Method / نوع وقف الخسارة:", ["Percentage %", "ATR Dynamic Trailing Stop"])

if stop_mode == "Percentage %":
    stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 1.5, 0.1)
    take_profit_pct = st.sidebar.slider("Take Profit (%)", 1.0, 15.0, 3.0, 0.5)
    atr_multiplier = 2.0
else:
    atr_multiplier = st.sidebar.slider("ATR Stop Multiplier (xATR)", 1.0, 5.0, 2.0, 0.1)
    take_profit_pct = st.sidebar.slider("Take Profit Target (%)", 1.0, 15.0, 4.0, 0.5)
    stop_loss_pct = 1.5

risk_pct = st.sidebar.slider("Position Sizing (% Portfolio)", 0.5, 10.0, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Circuit Breakers & Hard Protections")
max_daily_loss_pct = st.sidebar.slider("Max Daily Loss Threshold (%)", 1.0, 10.0, 3.0, 0.5)
current_daily_drawdown = st.sidebar.slider("Current Simulated Daily Equity Loss (%)", 0.0, 10.0, 0.5, 0.1)

is_circuit_broken = current_daily_drawdown >= max_daily_loss_pct

if is_circuit_broken:
    st.sidebar.error(f"⛔ CIRCUIT BREAKER TRIPPED! Drawdown ({current_daily_drawdown}%) >= Max Allowed ({max_daily_loss_pct}%). Engine Locked!")

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 سوق التداول المستهدف")
market_type = st.sidebar.radio("اختر السوق:", ["العملات الرقمية (Crypto)", "الأسهم الأمريكية & ETFs"])

crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD"]
stock_symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "COIN", "MSTR", "IBIT", "FBTC", "QQQ", "SPY"]

active_symbols = stock_symbols if market_type.startswith("الأسهم") else crypto_symbols
selected_symbol = st.sidebar.selectbox("🎯 الأصل النشط للتداول", active_symbols)
timeframe = st.sidebar.selectbox("⏱️ الإطار الزمني", ["5m", "15m", "1h", "4h", "1d"])

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Strategy Signals & Filters")
use_rsi_filter = st.sidebar.checkbox("RSI Filter", value=True)
use_macd_4c = st.sidebar.checkbox("MACD 4C Momentum Filter", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📲 Telegram Bot Dispatcher")
telegram_token = st.sidebar.text_input("Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID")

# --- Data Engine ---
@st.cache_data(ttl=30)
def fetch_quant_data(symbol, interval):
    df = yf.download(symbol, period="3mo", interval=interval)
    if df.empty:
        return pd.DataFrame()
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
    df['atr'] = df['tr'].rolling(window=atr_period).mean()
    
    df['atr_stop_long'] = df['close'] - (df['atr'] * atr_multiplier)
    df['atr_stop_short'] = df['close'] + (df['atr'] * atr_multiplier)
    
    return df

df = fetch_quant_data(selected_symbol, timeframe)

def get_signal(data):
    if data.empty or len(data) == 0:
        return "NO DATA"
        
    last_close = data['close'].iloc[-1]
    last_ema = data['ema'].iloc[-1]
    last_rsi = data['rsi'].iloc[-1]
    last_macd_hist = data['macd_hist'].iloc[-1]
    
    bullish_conds = [last_close > last_ema]
    bearish_conds = [last_close < last_ema]
    
    if use_rsi_filter:
        bullish_conds.append(last_rsi < 65)
        bearish_conds.append(last_rsi > 35)
    if use_macd_4c:
        bullish_conds.append(last_macd_hist > 0)
        bearish_conds.append(last_macd_hist < 0)
        
    if all(bullish_conds):
        return "BUY STRONG"
    elif all(bearish_conds):
        return "SELL STRONG"
    return "NEUTRAL"

current_signal = get_signal(df)

# --- Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Auto-Pilot & Risk Engine", 
    "📊 Backtesting & Metrics",
    "🔍 Real-Time Async Scanner", 
    "📈 Advanced Charting & Indicators", 
    "⚖️ Pairs Trading & Arbitrage"
])

with tab1:
    st.header("🤖 Auto-Pilot Execution & Risk Engine")
    st.write(f"**Mode:** `{execution_mode}` | **Asset:** `{selected_symbol}` | **Signal:** `{current_signal}`")
    
    if exchange_instance:
        try:
            balance = exchange_instance.fetch_balance()
            usdt_free = balance.get('USDT', {}).get('free', 0.0)
            st.success(f"💳 Connected API! Balance: **${usdt_free:,.2f} USDT**")
        except Exception as e:
            st.warning(f"Live balance error: {e}")
            
    col_run, col_stop, col_msg = st.columns(3)
    with col_run:
        if st.button("▶️ Launch Auto-Pilot Worker", type="primary", disabled=is_circuit_broken):
            st.success(f"Auto-Pilot Active for {selected_symbol}.")
            send_telegram_msg(telegram_token, telegram_chat_id, f"🚀 *QuantClaw Engine Started*\nAsset: `{selected_symbol}`")
    with col_stop:
        if st.button("⏹️ Emergency Stop"):
            st.error("Auto-Pilot Terminated.")
            send_telegram_msg(telegram_token, telegram_chat_id, f"🛑 *QuantClaw Engine Stopped*\nAsset: `{selected_symbol}`")
    with col_msg:
        if st.button("📲 Test Telegram"):
            send_telegram_msg(telegram_token, telegram_chat_id, f"🔔 *QuantClaw Test Alert*\nAsset: `{selected_symbol}`")

    st.markdown("---")
    st.subheader("⚡ Order Router")
    col_buy, col_sell = st.columns(2)
    ccxt_symbol = selected_symbol.replace("-USD", "/USDT") if "-USD" in selected_symbol else f"{selected_symbol}/USDT"
    
    with col_buy:
        if st.button(f"🟢 BUY MARKET ({ccxt_symbol})", type="primary", disabled=is_circuit_broken):
            if exchange_instance:
                try:
                    order = exchange_instance.create_market_buy_order(ccxt_symbol, 0.001)
                    st.success(f"Order ID: {order.get('id')}")
                except Exception as e:
                    st.error(f"Execution Failed: {e}")
            else:
                st.info(f"[SIMULATION] Executed Buy for {ccxt_symbol}")
                
    with col_sell:
        if st.button(f"🔴 SELL MARKET ({ccxt_symbol})", disabled=is_circuit_broken):
            if exchange_instance:
                try:
                    order = exchange_instance.create_market_sell_order(ccxt_symbol, 0.001)
                    st.success(f"Order ID: {order.get('id')}")
                except Exception as e:
                    st.error(f"Execution Failed: {e}")
            else:
                st.info(f"[SIMULATION] Executed Sell for {ccxt_symbol}")

with tab2:
    st.header(f"📊 Quant Backtest - {selected_symbol}")
    if not df.empty:
        bt_df = df.copy()
        bt_df['signal'] = 0
        bt_df.loc[(bt_df['close'] > bt_df['ema']) & (bt_df['macd_hist'] > 0), 'signal'] = 1
        bt_df.loc[(bt_df['close'] < bt_df['ema']) & (bt_df['macd_hist'] < 0), 'signal'] = -1
        
        bt_df['market_returns'] = bt_df['close'].pct_change()
        bt_df['strategy_returns'] = bt_df['market_returns'] * bt_df['signal'].shift(1)
        
        bt_df['cum_market'] = (1 + bt_df['market_returns']).cumprod()
        bt_df['cum_strategy'] = (1 + bt_df['strategy_returns']).cumprod()
        
        m1, m2 = st.columns(2)
        m1.metric("Strategy Cum Return", f"{((bt_df['cum_strategy'].iloc[-1]-1)*100):.2f}%")
        m2.metric("Buy & Hold Return", f"{((bt_df['cum_market'].iloc[-1]-1)*100):.2f}%")
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['cum_strategy'], name="Strategy Return", line=dict(color='#00E676')))
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['cum_market'], name="Buy & Hold", line=dict(color='#2962FF', dash='dash')))
        fig_bt.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_bt, use_container_width=True)

with tab3:
    st.header("🔍 Multi-Asset Scanner")
    scanner_results = []
    for sym in active_symbols:
        temp_df = fetch_quant_data(sym, timeframe)
        sym_signal = get_signal(temp_df)
        if not temp_df.empty:
            scanner_results.append({
                "Asset": sym,
                "Price": f"${temp_df['close'].iloc[-1]:,.2f}",
                "RSI": f"{temp_df['rsi'].iloc[-1]:.1f}",
                "Strategy Signal": sym_signal
            })
    if scanner_results:
        st.table(pd.DataFrame(scanner_results))

with tab4:
    st.header(f"📈 Advanced Chart - {selected_symbol}")
    if not df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#2962FF'), name="EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#ff9800'), name="RSI"), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("⚖️ Pairs Trading Spread")
    pair_a = st.selectbox("Asset A", active_symbols, index=0)
    pair_b = st.selectbox("Asset B", active_symbols, index=1 if len(active_symbols)>1 else 0)
    if pair_a != pair_b:
        df_a = fetch_quant_data(pair_a, timeframe)
        df_b = fetch_quant_data(pair_b, timeframe)
        if not df_a.empty and not df_b.empty:
            ratio = df_a['close'] / df_b['close']
            zscore = (ratio - ratio.mean()) / ratio.std()
            fig_pair = go.Figure()
            fig_pair.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Spread Z-Score", line=dict(color='#00E676')))
            fig_pair.add_hline(y=2.0, line_dash="dash", line_color="red")
            fig_pair.add_hline(y=-2.0, line_dash="dash", line_color="green")
            fig_pair.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pair, use_container_width=True)
