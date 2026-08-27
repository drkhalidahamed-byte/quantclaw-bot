import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from trading_engine import (
    calculate_indicators, calculate_position_size, execute_binance_order,
    log_trade_to_db, get_trades_from_db, clear_trades_db
)

st.set_page_config(page_title="QuantClaw Enterprise Console", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar Controls ---
st.sidebar.title("⚡ QuantClaw Enterprise")
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 الشارت التفاعلي",
    "🤖 مساعد الذكاء الاصطناعي (AI Chat)",
    "🚀 التنفيذ الآلي والـ API",
    "🗺️ خريطة الحرارة والماسح", 
    "📊 تقاطع الأطر الزمنية",
    "📒 سجل الأداء والمحافظ (SQLite)",
    "🛡️ إدارة المخاطر المؤسسية"
])

# --- TAB 1: ADVANCED CHART ---
with tab1:
    st.header(f"📈 التحليل الفني المتقدم - {selected_symbol}")
    if not df.empty and len(df) > 5:
        ai_prob = df['ai_score'].iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("مؤشر الثقة الذكي (AI Score)", f"{ai_prob:.1f}%")
        c2.metric("السعر الحالي", f"${df['close'].iloc[-1]:,.2f}")
        c3.metric("RSI Momentum", f"{df['rsi'].iloc[-1]:.1f}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
        if 'ema' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#2962FF', width=1.5), name=f"EMA {ema_period}"), row=1, col=1)
        if 'vwap' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], line=dict(color='#E91E63', width=1.5, dash='dash'), name="VWAP"), row=1, col=1)
        if 'macd_hist' in df.columns:
            colors = ['#00E676' if val >= 0 else '#FF5252' for val in df['macd_hist'].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], marker_color=colors, name="MACD Hist"), row=2, col=1)
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#FF9800', width=1.5), name="RSI"), row=3, col=1)
        
        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: AI TRADING ASSISTANT ---
with tab2:
    st.header("🤖 مساعد التداول الذكي (AI Analyst)")
    st.write("اسأل مساعد الذكاء الاصطناعي عن وضع الأصل الحالي أو اطلب تقييماً للإشارة الفنية:")
    
    user_query = st.text_input("أدخل سؤالك هنا (مثلاً: ما هو تقييمك لحالة السعر الحالية؟)")
    if user_query and not df.empty:
        last_close = df['close'].iloc[-1]
        last_rsi = df['rsi'].iloc[-1]
        last_ai = df['ai_score'].iloc[-1]
        
        st.markdown("### 💡 تقرير الذكاء الاصطناعي:")
        if last_ai > 60:
            st.success(f"الوضع الحالي لـ {selected_symbol} **إيجابي (Bullish)**. مؤشر الثقة عند `{last_ai:.1f}%`، وقيمة الـ RSI تسجل `{last_rsi:.1f}`. الزخم العام يدعم الشراء مع متابعة خط الـ VWAP.")
        elif last_ai < 40:
            st.warning(f"الوضع الحالي لـ {selected_symbol} **سلبي أو هابط (Bearish)**. مؤشر الثقة منخفض عند `{last_ai:.1f}%`، والزخم يشير لضغوط بيعية.")
        else:
            st.info(f"السوق في حالة **حيادية (Neutral)** لـ {selected_symbol}. يفضل الانتظار حتى كسر المستويات العرضية.")

# --- TAB 3: EXECUTION & API ---
with tab3:
    st.header("🚀 التنفيذ الآلي والربط الحقيقي (API)")
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        api_k = st.text_input("API Key", type="password")
        api_s = st.text_input("API Secret", type="password")
        mode = st.radio("البيئة:", ["محاكاة محلية", "Binance Testnet"])
    with c_e2:
        amt = st.number_input("مبلغ الصفقة ($)", value=100.0)
        if not df.empty:
            lp = df['close'].iloc[-1]
            if st.button("🛒 تنفيذ شراء وإرسال لسجل SQLite"):
                log_trade_to_db(selected_symbol, "BUY", lp, amt, "Active")
                st.success("تم تنفيذ الصفقة وحفظها في قاعدة البيانات المحلية بنجاح!")
                send_telegram_alert(f"🛒 صفقة ناجحة على {selected_symbol} بسعر ${lp:,.2f}")

# --- TAB 4: HEATMAP & SCANNER ---
with tab4:
    st.header("🗺️ خريطة الحرارة والماسح الفوري")
    all_syms = crypto_symbols + stock_symbols
    h_data = []
    for s in all_syms:
        t_df = fetch_data(s, "1h")
        if not t_df.empty and len(t_df) > 1:
            p_n = t_df['close'].iloc[-1]
            p_p = t_df['close'].iloc[-2]
            chg = ((p_n - p_p) / p_p) * 100
            h_data.append({"Symbol": s, "Price": f"${p_n:,.2f}", "Change %": f"{chg:+.2f}%"})
    if h_data:
        st.dataframe(pd.DataFrame(h_data), use_container_width=True)

# --- TAB 5: MULTI-TIMEFRAME ---
with tab5:
    st.header("📊 تقاطع الأطر الزمنية")
    for tf in ["15m", "1h", "4h"]:
        sub = fetch_data(selected_symbol, tf)
        if not sub.empty:
            st.write(f"**الإطار الزمني {tf}:** السعر = `${sub['close'].iloc[-1]:,.2f}` | RSI = `{sub['rsi'].iloc[-1]:.1f}`")

# --- TAB 6: SQLITE TRADE JOURNAL ---
with tab6:
    st.header("📒 سجل الأداء والمحافظ (SQLite Database)")
    trades_df = get_trades_from_db()
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True)
        if st.button("🗑️ تفريغ كافة السجلات"):
            clear_trades_db()
            st.rerun()
    else:
        st.info("لا توجد سجلات صفقات في قاعدة البيانات حتى الآن.")

# --- TAB 7: RISK ENGINE ---
with tab7:
    st.header("🛡️ حاسبة المخاطر المؤسسية")
    acc = st.number_input("رأس المال ($)", value=10000.0)
    risk = st.slider("المخاطرة (%)", 0.5, 5.0, 1.0)
    if not df.empty and 'atr' in df.columns:
        cp = df['close'].iloc[-1]
        sl = cp - (df['atr'].iloc[-1] * atr_multiplier)
        tp = cp + ((cp - sl) * risk_reward_ratio)
        units = calculate_position_size(acc, risk, cp, sl)
        st.metric("الكمية الموصى بها", f"{units:.4f}")
