import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from trading_engine import calculate_indicators, calculate_position_size, execute_binance_order

st.set_page_config(page_title="QuantClaw Institutional Console", layout="wide", initial_sidebar_state="expanded")

if "active_trades" not in st.session_state:
    st.session_state.active_trades = []

# --- Sidebar Controls ---
st.sidebar.title("⚡ QuantClaw Institutional")
market_type = st.sidebar.radio("السوق:", ["العملات الرقمية (Crypto)", "الأسهم الأمريكية & ETFs"])

crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
stock_symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "COIN"]

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
        except Exception:
            pass

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
    except Exception:
        return pd.DataFrame()

df = fetch_data(selected_symbol, timeframe)

# Tabs Navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 الشارت التفاعلي والذكاء الاصطناعي",
    "🤖 التنفيذ الآلي والحقيقي (API)",
    "🗺️ خريطة الحرارة والماسح الشامل", 
    "📊 تقاطع الأطر الزمنية المتعددة",
    "📊 محاكي الاستراتيجية (Backtest)",
    "🛡️ إدارة المخاطر المؤسسية"
])

# --- TAB 1: ADVANCED CHART & AI SCORE ---
with tab1:
    st.header(f"📈 الشارت التفاعلي وتحليل الذكاء الاصطناعي - {selected_symbol}")
    if not df.empty and len(df) > 5:
        ai_prob = df['ai_score'].iloc[-1]
        col_ai1, col_ai2, col_ai3 = st.columns(3)
        col_ai1.metric("مؤشر ثقة الذكاء الاصطناعي (AI Confidence)", f"{ai_prob:.1f}%", "Bullish Bias" if ai_prob > 50 else "Bearish Bias")
        col_ai2.metric("السعر الحالي", f"${df['close'].iloc[-1]:,.2f}")
        col_ai3.metric("مستوى الـ RSI", f"{df['rsi'].iloc[-1]:.1f}")

        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.04, 
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"OHLC, EMA, VWAP & AI Overlay ({selected_symbol})", "MACD Histogram", "RSI Momentum")
        )
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
        if 'ema' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#2962FF', width=1.5), name=f"EMA {ema_period}"), row=1, col=1)
        if 'vwap' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], line=dict(color='#E91E63', width=1.5, dash='dash'), name="VWAP"), row=1, col=1)
            
        if 'upper_band' in df.columns and 'lower_band' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="Upper BB"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], line=dict(color='rgba(255,255,255,0.2)', width=1), name="Lower BB"), row=1, col=1)
        
        if 'macd_hist' in df.columns:
            colors = ['#00E676' if val >= 0 else '#FF5252' for val in df['macd_hist'].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], marker_color=colors, name="MACD Hist"), row=2, col=1)
        
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#FF9800', width=1.5), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: LIVE & TESTNET API EXECUTION ---
with tab2:
    st.header("🤖 محرك التشغيل الآلي والتنفيذ الفعلي عبر API")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("🔑 إعدادات ربط الحساب (Binance API)")
        api_key_input = st.text_input("API Key", type="password")
        api_secret_input = st.text_input("API Secret", type="password")
        execution_target = st.radio("بيئة التنفيذ:", ["محاكاة ورقية محلية (Paper Trading)", "تداول حقيقي تجريبي (Binance Testnet)"])
        
    with col_e2:
        st.subheader("⚡ لوحة تنفيذ الصفقات الفورية")
        trade_amount_usd = st.number_input("قيمة الصفقة ($)", value=100.0, step=10.0)
        
        if not df.empty:
            last_p = df['close'].iloc[-1]
            if st.button("🚀 إرسال أمر شراء فوري (Market Buy)"):
                if "Testnet" in execution_target and api_key_input and api_secret_input:
                    qty = trade_amount_usd / last_p
                    res = execute_binance_order(api_key_input, api_secret_input, selected_symbol, "BUY", qty, testnet=True)
                    st.json(res)
                else:
                    st.session_state.active_trades.append({
                        "Symbol": selected_symbol,
                        "Type": "BUY",
                        "Entry": last_p,
                        "Size": trade_amount_usd,
                        "Status": "Active Paper"
                    })
                    st.success(f"تم تنفيذ صفقة ورقية على {selected_symbol} بسعر ${last_p:,.2f}")
                    send_telegram_alert(f"🛒 صفقة جديدة: BUY {selected_symbol} @ ${last_p:,.2f}")

    st.markdown("---")
    st.subheader("📋 الصفقات النشطة المسجلة")
    if st.session_state.active_trades:
        st.dataframe(pd.DataFrame(st.session_state.active_trades), use_container_width=True)
        if st.button("🗑️ مسح السجل"):
            st.session_state.active_trades = []
            st.rerun()
    else:
        st.info("لا توجد صفقات نشطة مسجلة.")

# --- TAB 3: HEATMAP & SCANNER ---
with tab3:
    st.header("🗺️ خريطة حرارة السوق والماسح الشامل")
    all_market_symbols = crypto_symbols + stock_symbols
    heat_data = []
    for sym in all_market_symbols:
        t_df = fetch_data(sym, "1h")
        if not t_df.empty and len(t_df) > 2:
            p_now = t_df['close'].iloc[-1]
            p_prev = t_df['close'].iloc[-2]
            pct_change = ((p_now - p_prev) / p_prev) * 100
            rsi_val = t_df['rsi'].iloc[-1] if 'rsi' in t_df.columns else 50
            ai_s = t_df['ai_score'].iloc[-1] if 'ai_score' in t_df.columns else 50
            heat_data.append({"Symbol": sym, "Price": f"${p_now:,.2f}", "Change %": f"{pct_change:+.2f}%", "RSI": f"{rsi_val:.1f}", "AI Score": f"{ai_s:.1f}%"})
    
    if heat_data:
        h_df = pd.DataFrame(heat_data)
        st.dataframe(h_df, use_container_width=True)
        st.download_button("📥 تصدير تقرير السوق", data=h_df.to_csv(index=False), file_name="market_heatmap.csv", mime="text/csv")

# --- TAB 4: MULTI-TIMEFRAME CONFLUENCE ---
with tab4:
    st.header("📊 تقاطع المؤشرات عبر الأطر الزمنية المتعددة (Multi-Timeframe)")
    tf_list = ["15m", "1h", "4h"]
    confluence_results = []
    for tf in tf_list:
        sub_df = fetch_data(selected_symbol, tf)
        if not sub_df.empty:
            lp = sub_df['close'].iloc[-1]
            lem = sub_df['ema'].iloc[-1] if 'ema' in sub_df.columns else lp
            lrsi = sub_df['rsi'].iloc[-1] if 'rsi' in sub_df.columns else 50
            trend = "🟢 Bullish" if lp > lem and lrsi > 50 else "🔴 Bearish"
            confluence_results.append({"Timeframe": tf, "Price": f"${lp:,.2f}", "RSI": f"{lrsi:.1f}", "Trend State": trend})
            
    if confluence_results:
        st.dataframe(pd.DataFrame(confluence_results), use_container_width=True)

# --- TAB 5: BACKTEST ---
with tab5:
    st.header("📊 محاكي الاستراتيجيات المتقدم (Backtest)")
    if not df.empty and 'ema' in df.columns and 'macd_hist' in df.columns:
        bt_df = df.copy()
        bt_df['signal'] = np.where((bt_df['close'] > bt_df['ema']) & (bt_df['macd_hist'] > 0), 1, -1)
        bt_df['returns'] = bt_df['close'].pct_change() * bt_df['signal'].shift(1)
        bt_df['cum_returns'] = (1 + bt_df['returns'].fillna(0)).cumprod()
        st.line_chart(bt_df['cum_returns'])

# --- TAB 6: RISK ENGINE ---
with tab6:
    st.header("🛡️ حاسبة المخاطر المؤسسية وأهداف الأرباح")
    acc = st.number_input("رأس المال ($)", value=10000.0)
    risk = st.slider("نسبة المخاطرة لكل صفقة (%)", 0.5, 5.0, 1.0)
    if not df.empty and 'atr' in df.columns:
        cp = df['close'].iloc[-1]
        sl = cp - (df['atr'].iloc[-1] * atr_multiplier)
        tp = cp + ((cp - sl) * risk_reward_ratio)
        units = calculate_position_size(acc, risk, cp, sl)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("سعر الدخول", f"${cp:,.2f}")
        c2.metric("وقف الخسارة (SL)", f"${sl:,.2f}")
        c3.metric("هدف الربح (TP)", f"${tp:,.2f}")
        st.metric("الكمية الموصى بها (Units)", f"{units:.4f}")
