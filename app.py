import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from trading_engine import calculate_indicators, calculate_position_size

st.set_page_config(page_title="QuantClaw Trading Console", layout="wide", initial_sidebar_state="expanded")

# Initialize Session State for Active Simulated Positions
if "active_trades" not in st.session_state:
    st.session_state.active_trades = []

# --- Sidebar Controls ---
st.sidebar.title("⚡ QuantClaw Control")
market_type = st.sidebar.radio("السوق:", ["العملات الرقمية (Crypto)", "الأسهم الأمريكية & ETFs"])

crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
stock_symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "COIN", "MSTR"]

active_symbols = stock_symbols if market_type.startswith("الأسهم") else crypto_symbols
selected_symbol = st.sidebar.selectbox("🎯 الأصل النشط", active_symbols)
timeframe = st.sidebar.selectbox("⏱️ الإطار الزمني", ["5m", "15m", "1h", "4h", "1d"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ إعدادات المؤشرات")
ema_period = st.sidebar.slider("EMA Period", 20, 200, 200, 5)
rsi_period = st.sidebar.slider("RSI Period", 7, 30, 14, 1)
atr_period = st.sidebar.slider("ATR Period", 5, 30, 10, 1)
atr_multiplier = st.sidebar.slider("ATR Stop Multiplier", 1.0, 5.0, 2.0, 0.1)
risk_reward_ratio = st.sidebar.slider("Risk/Reward Ratio (TP Multiplier)", 1.0, 5.0, 2.0, 0.5)

# Telegram Setup
st.sidebar.markdown("---")
st.sidebar.subheader("📲 Telegram Bot")
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
            st.sidebar.error(f"خطأ Telegram: {e}")

# Safe Data Fetching Helper with Error Handlers
@st.cache_data(ttl=30)
def fetch_data(symbol, interval):
    try:
        if interval in ["1m", "2m", "5m"]:
            period = "7d"
        elif interval in ["15m", "30m"]:
            period = "1mo"
        elif interval in ["1h", "60m"]:
            period = "3mo"
        else:
            period = "1y"
            
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [str(c).lower() for c in df.columns]
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        df = calculate_indicators(df, ema_period, rsi_period, atr_period)
        return df.dropna(subset=['close'])
    except Exception as e:
        return pd.DataFrame()

df = fetch_data(selected_symbol, timeframe)

# Tabs Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 الشارت التفاعلي (Advanced Chart)",
    "🤖 التشغيل الآلي والصفقات (Live Auto-Trader)",
    "🔍 الماسح اللحظي (Async Scanner)", 
    "📊 محاكي الاستراتيجية (Backtest)",
    "🛡️ إدارة المخاطر (Risk Engine)"
])

# --- TAB 1: ADVANCED CHART ---
with tab1:
    st.header(f"📈 الشارت التفاعلي والمؤشرات - {selected_symbol}")
    if not df.empty and len(df) > 5:
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.04, 
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"OHLC, EMA, VWAP & SL/TP Levels ({selected_symbol})", "MACD Histogram", "RSI Momentum")
        )
        
        # Candlesticks
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
        
        # EMA & VWAP
        if 'ema' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#2962FF', width=1.5), name=f"EMA {ema_period}"), row=1, col=1)
        if 'vwap' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], line=dict(color='#E91E63', width=1.5, dash='dash'), name="VWAP"), row=1, col=1)
            
        # Bollinger Bands
        if 'upper_band' in df.columns and 'lower_band' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="Upper BB"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="Lower BB"), row=1, col=1)
        
        # ATR Stop Loss Line on Chart
        if 'atr' in df.columns:
            current_close = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            atr_stop_val = current_close - (current_atr * atr_multiplier)
            take_profit_val = current_close + ((current_close - atr_stop_val) * risk_reward_ratio)
            
            fig.add_hline(y=atr_stop_val, line_dash="dot", line_color="#FF5252", annotation_text=f"Dynamic SL: ${atr_stop_val:,.2f}", row=1, col=1)
            fig.add_hline(y=take_profit_val, line_dash="dot", line_color="#00E676", annotation_text=f"Dynamic TP: ${take_profit_val:,.2f}", row=1, col=1)

        # MACD
        if 'macd_hist' in df.columns:
            colors = ['#00E676' if val >= 0 else '#FF5252' for val in df['macd_hist'].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], marker_color=colors, name="MACD Hist"), row=2, col=1)
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#FF9800', width=1.5), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ لم يتم العثور على بيانات كافية لرسم الشارت.")

# --- TAB 2: AUTOMATED EXECUTION & ACTIVE POSITIONS ---
with tab2:
    st.header("🤖 محرك التشغيل الآلي ومتابعة الصفقات النشطة")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("🔑 إعدادات التنفيذ (API)")
        api_key = st.text_input("Binance API Key", type="password")
        api_secret = st.text_input("Binance API Secret", type="password")
        execution_mode = st.radio("النمط:", ["محاكاة ورقية (Paper Trading)", "حقيقي / Testnet"])
        
    with col_e2:
        st.subheader("⚡ أوامر التداول السريعة")
        auto_trade_toggle = st.checkbox("تفعيل البوت الآلي للمراقبة")
        trade_amount_usd = st.number_input("مبلغ الصفقة ($)", value=100.0, step=10.0)
        
        if not df.empty:
            last_p = df['close'].iloc[-1]
            last_ema = df['ema'].iloc[-1] if 'ema' in df.columns else last_p
            last_macd = df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else 0
            has_vol_spike = df['vol_spike'].iloc[-1] if 'vol_spike' in df.columns else False
            
            if has_vol_spike:
                st.warning("⚠️ **تنبيه هام:** تم رصد ارتفاع مفاجئ في أحجام التداول (Volume Spike) على الشمعة الأخيرة!")

            if st.button("🛒 فتح صفقة شراء سريعة (Market Buy)"):
                st.session_state.active_trades.append({
                    "Symbol": selected_symbol,
                    "Type": "BUY",
                    "Entry": last_p,
                    "Size": trade_amount_usd,
                    "Status": "Active"
                })
                st.success(f"تم فتح صفقة شراء على {selected_symbol} بسعر ${last_p:,.2f}")
                send_telegram_alert(f"🛒 فتح صفقة شراء على {selected_symbol} بسعر ${last_p:,.2f}")

    st.markdown("---")
    st.subheader("📋 الصفقات النشطة حالياً (Active Simulated Positions)")
    if st.session_state.active_trades:
        active_df = pd.DataFrame(st.session_state.active_trades)
        st.dataframe(active_df, use_container_width=True)
        if st.button("🗑️ إغلاق جميع الصفقات النشطة"):
            st.session_state.active_trades = []
            st.experimental_rerun()
    else:
        st.info("لا توجد صفقات مفتوحة حالياً.")

# --- TAB 3: SCANNER ---
with tab3:
    st.header("🔍 الماسح اللحظي للأسواق (مع فحص Volume Spikes)")
    scanner_data = []
    for sym in active_symbols:
        temp = fetch_data(sym, timeframe)
        if not temp.empty:
            lp = temp['close'].iloc[-1]
            lr = temp['rsi'].iloc[-1] if 'rsi' in temp.columns else 0
            lem = temp['ema'].iloc[-1] if 'ema' in temp.columns else lp
            lm = temp['macd_hist'].iloc[-1] if 'macd_hist' in temp.columns else 0
            v_spike = temp['vol_spike'].iloc[-1] if 'vol_spike' in temp.columns else False
            sig = "🟢 BUY" if lp > lem and lm > 0 else ("🔴 SELL" if lp < lem and lm < 0 else "⚪ NEUTRAL")
            if v_spike:
                sig += " + ⚡ Vol Spike"
            scanner_data.append({"Symbol": sym, "Price": f"${lp:,.2f}", "RSI": f"{lr:.1f}", "Signal": sig})
    
    if scanner_data:
        scan_df = pd.DataFrame(scanner_data)
        st.dataframe(scan_df, use_container_width=True)
        st.download_button("📥 تصدير نتائج الماسح إلى CSV", data=scan_df.to_csv(index=False), file_name="scanner_results.csv", mime="text/csv")

# --- TAB 4: BACKTEST ---
with tab4:
    st.header("📊 محاكي الاستراتيجيات (Backtest Engine)")
    if not df.empty and 'ema' in df.columns and 'macd_hist' in df.columns:
        bt_df = df.copy()
        bt_df['signal'] = np.where((bt_df['close'] > bt_df['ema']) & (bt_df['macd_hist'] > 0), 1, -1)
        bt_df['returns'] = bt_df['close'].pct_change() * bt_df['signal'].shift(1)
        bt_df['cum_returns'] = (1 + bt_df['returns'].fillna(0)).cumprod()
        
        st.line_chart(bt_df['cum_returns'])
        st.download_button("📥 تصدير نتائج الباك تيست إلى CSV", data=bt_df.to_csv(), file_name=f"backtest_{selected_symbol}.csv", mime="text/csv")

# --- TAB 5: RISK ENGINE ---
with tab5:
    st.header("🛡️ حاسبة المخاطر وأهداف الربح المتقدمة")
    acc = st.number_input("رأس المال الإجمالي ($)", value=10000.0)
    risk = st.slider("المخاطرة للصفقة الواحدة (%)", 0.5, 5.0, 1.0)
    if not df.empty and 'atr' in df.columns:
        cp = df['close'].iloc[-1]
        sl = cp - (df['atr'].iloc[-1] * atr_multiplier)
        tp = cp + ((cp - sl) * risk_reward_ratio)
        units = calculate_position_size(acc, risk, cp, sl)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("سعر الدخول المقترح", f"${cp:,.2f}")
        col_r2.metric("وقف الخسارة (SL)", f"${sl:,.2f}")
        col_r3.metric("هدف الربح (TP)", f"${tp:,.2f}")
        st.metric("حجم الكمية الموصى بها (Units)", f"{units:.4f}")
