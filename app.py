import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from trading_engine import calculate_indicators, calculate_position_size

st.set_page_config(page_title="QuantClaw Institutional Trading Console", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar Controls ---
st.sidebar.title("⚡ QuantClaw Engine Control")
market_type = st.sidebar.radio("السوق:", ["العملات الرقمية (Crypto)", "الأسهم الأمريكية & ETFs"])

crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD"]
stock_symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "COIN", "MSTR", "QQQ", "SPY"]

active_symbols = stock_symbols if market_type.startswith("الأسهم") else crypto_symbols
selected_symbol = st.sidebar.selectbox("🎯 الأصل النشط للتداول", active_symbols)
timeframe = st.sidebar.selectbox("⏱️ الإطار الزمني", ["5m", "15m", "1h", "4h", "1d"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ إعدادات المؤشرات والمخاطر")
ema_period = st.sidebar.slider("EMA Period", 20, 200, 200, 5)
rsi_period = st.sidebar.slider("RSI Period", 7, 30, 14, 1)
atr_period = st.sidebar.slider("ATR Period", 5, 30, 10, 1)
atr_multiplier = st.sidebar.slider("ATR Stop Multiplier (xATR)", 1.0, 5.0, 2.0, 0.1)
trading_fee_pct = st.sidebar.number_input("عمولة التداول لكل صفقة (%)", value=0.075, step=0.005, format="%.3f")

# --- Telegram Bot Helper ---
st.sidebar.markdown("---")
st.sidebar.subheader("📲 التنبيهات (Telegram Bot)")
telegram_token = st.sidebar.text_input("Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID")

def send_telegram_alert(message):
    if telegram_token and telegram_chat_id:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=5)
            st.sidebar.success("تم إرسال التنبيه!")
        except Exception as e:
            st.sidebar.error(f"خطأ في الإرسال: {e}")

# --- Data Fetching Helper ---
@st.cache_data(ttl=30)
def fetch_data(symbol, interval):
    df = yf.download(symbol, period="3mo", interval=interval)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return calculate_indicators(df, ema_period, rsi_period, atr_period)

df = fetch_data(selected_symbol, timeframe)

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Backtesting Engine",
    "🔍 Real-Time Async Scanner", 
    "📈 Advanced Charting", 
    "⚖️ Pairs Trading & Arbitrage",
    "🛡️ Risk & Position Calculator"
])

# TAB 1: BACKTESTING
with tab1:
    st.header(f"📊 Quantitative Backtest Engine - {selected_symbol}")
    if not df.empty:
        bt_df = df.copy()
        bt_df['signal'] = 0
        bt_df.loc[(bt_df['close'] > bt_df['ema']) & (bt_df['macd_hist'] > 0), 'signal'] = 1
        bt_df.loc[(bt_df['close'] < bt_df['ema']) & (bt_df['macd_hist'] < 0), 'signal'] = -1
        
        bt_df['trades'] = bt_df['signal'].diff().fillna(0) != 0
        bt_df['market_returns'] = bt_df['close'].pct_change()
        fee_cost = (trading_fee_pct / 100.0) * bt_df['trades']
        bt_df['strategy_returns'] = (bt_df['market_returns'] * bt_df['signal'].shift(1)) - fee_cost
        
        bt_df['cum_market'] = (1 + bt_df['market_returns']).cumprod()
        bt_df['cum_strategy'] = (1 + bt_df['strategy_returns'].fillna(0)).cumprod()
        
        total_trades = int(bt_df['trades'].sum())
        winning_trades = int((bt_df['strategy_returns'] > 0).sum())
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        sharpe_ratio = (bt_df['strategy_returns'].mean() / (bt_df['strategy_returns'].std() + 1e-9)) * np.sqrt(365)
        cum_max = bt_df['cum_strategy'].cummax()
        max_drawdown = ((bt_df['cum_strategy'] - cum_max) / cum_max).min() * 100
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Win Rate %", f"{win_rate:.1f}%")
        m2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        m3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        m4.metric("Total Trades (Net Fees)", f"{total_trades}")
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['cum_strategy'], name="Quant Strategy (Net)", line=dict(color='#00E676', width=2)))
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['cum_market'], name="Buy & Hold", line=dict(color='#2962FF', width=1.5, dash='dash')))
        fig_bt.update_layout(template="plotly_dark", height=450, title="Net Equity Curve (After Fees)")
        st.plotly_chart(fig_bt, use_container_width=True)

# TAB 2: REAL-TIME SCANNER
with tab2:
    st.header("🔍 Multi-Asset Real-Time Scanner")
    scanner_data = []
    signals_to_alert = []
    
    for sym in active_symbols:
        temp_df = fetch_data(sym, timeframe)
        if not temp_df.empty:
            last_close = temp_df['close'].iloc[-1]
            last_ema = temp_df['ema'].iloc[-1]
            last_rsi = temp_df['rsi'].iloc[-1]
            last_macd = temp_df['macd_hist'].iloc[-1]
            
            signal = "NEUTRAL"
            if last_close > last_ema and last_macd > 0 and last_rsi < 65:
                signal = "🟢 BUY STRONG"
                signals_to_alert.append(f"🟢 BUY Signal detected on {sym} at ${last_close:,.2f}")
            elif last_close < last_ema and last_macd < 0 and last_rsi > 35:
                signal = "🔴 SELL STRONG"
                signals_to_alert.append(f"🔴 SELL Signal detected on {sym} at ${last_close:,.2f}")
                
            scanner_data.append({
                "Asset": sym,
                "Price": f"${last_close:,.2f}",
                "RSI (14)": f"{last_rsi:.1f}",
                "EMA Status": "ABOVE" if last_close > last_ema else "BELOW",
                "MACD Hist": f"{last_macd:.3f}",
                "Signal": signal
            })
            
    if scanner_data:
        st.dataframe(pd.DataFrame(scanner_data), use_container_width=True)
        
    if st.button("📲 إرسال التنبيهات النشطة إلى Telegram"):
        if signals_to_alert:
            msg = "\n".join(signals_to_alert)
            send_telegram_alert(f"🚨 **QuantClaw Market Alert** 🚨\n{msg}")
        else:
            st.info("لا توجد إشارات جديدة حالياً للإرسال.")

# TAB 3: CHARTING
with tab3:
    st.header(f"📈 Advanced Charting & ATR Stop Bands - {selected_symbol}")
    if not df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#2962FF', width=1.5), name=f"EMA {ema_period}"), row=1, col=1)
        atr_stop_long = df['close'] - (df['atr'] * atr_multiplier)
        fig.add_trace(go.Scatter(x=df.index, y=atr_stop_long, line=dict(color='#FF5252', width=1, dash='dot'), name="ATR Stop (Long)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#FF9800', width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: PAIRS TRADING
with tab4:
    st.header("⚖️ Statistical Arbitrage & Pair Z-Score")
    col_a, col_b = st.columns(2)
    with col_a:
        pair_a = st.selectbox("Asset A", active_symbols, index=0)
    with col_b:
        pair_b = st.selectbox("Asset B", active_symbols, index=1 if len(active_symbols) > 1 else 0)
        
    if pair_a != pair_b:
        df_a = fetch_data(pair_a, timeframe)
        df_b = fetch_data(pair_b, timeframe)
        if not df_a.empty and not df_b.empty:
            ratio = df_a['close'] / df_b['close']
            zscore = (ratio - ratio.mean()) / (ratio.std() + 1e-9)
            fig_pair = go.Figure()
            fig_pair.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Spread Z-Score", line=dict(color='#00E676', width=2)))
            fig_pair.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Overbought (+2σ)")
            fig_pair.add_hline(y=-2.0, line_dash="dash", line_color="green", annotation_text="Oversold (-2σ)")
            fig_pair.update_layout(template="plotly_dark", height=450, title=f"Statistical Spread Z-Score: {pair_a} / {pair_b}")
            st.plotly_chart(fig_pair, use_container_width=True)

# TAB 5: RISK CALCULATOR
with tab5:
    st.header("🛡️ Dynamic Position Sizing & Risk Calculator")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        acc_balance = st.number_input("رأس مال الحساب ($)", value=10000.0, step=500.0)
        risk_pct = st.slider("نسبة المخاطرة في الصفقة (%)", 0.5, 5.0, 1.0, 0.1)
    with col_r2:
        if not df.empty:
            current_p = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            entry_price = st.number_input("سعر الدخول ($)", value=float(current_p))
            stop_loss = st.number_input("سعر وقف الخسارة ($)", value=float(current_p - (current_atr * atr_multiplier)))
            pos_units = calculate_position_size(acc_balance, risk_pct, entry_price, stop_loss)
            position_value = pos_units * entry_price
            
            st.subheader("📊 نتائج الحساب:")
            st.success(f"حجم الكمية الموصى بها: **{pos_units:.4f}**")
            st.info(f"إجمالي قيمة الصفقة (Position Value): **${position_value:,.2f}**")
            st.warning(f"أقصى مبلغ مخاطرة (Max Loss): **${(acc_balance * risk_pct / 100):,.2f}**")
