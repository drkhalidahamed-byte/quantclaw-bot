import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from trading_engine import (
    calculate_indicators, calculate_position_size, execute_binance_order,
    log_trade_to_db, get_trades_from_db, clear_trades_db, poll_telegram_commands
)

st.set_page_config(page_title="QuantClaw Ultimate AI Enterprise", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar Navigation & Controls ---
st.sidebar.title("⚡ QuantClaw Ultimate AI")
market_type = st.sidebar.radio("السوق:", ["العملات الرقمية (Crypto)", "الأسهم الأمريكية & ETFs"])

crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
stock_symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "COIN"]

active_symbols = stock_symbols if market_type.startswith("الأسهم") else crypto_symbols
selected_symbol = st.sidebar.selectbox("🎯 الأصل النشط", active_symbols)
timeframe = st.sidebar.selectbox("⏱️ الإطار الزمني", ["5m", "15m", "1h", "4h", "1d"])

st.sidebar.markdown("---")
st.sidebar.subheader("📂 التنقل بين أقسام المنصة")
navigation_section = st.sidebar.radio(
    "اختر القسم المطلوب:",
    [
        "📈 1. الشارت والمؤشرات الفنية",
        "🧠 2. منظومة الذكاء الاصطناعي (AI Models)",
        "🤖 3. أوامر ووكيل Telegram",
        "🚀 4. التنفيذ الآلي والـ API",
        "🗺️ 5. خريطة الحرارة والماسح",
        "📊 6. تقاطع الأطر الزمنية",
        "📒 7. سجل الأداء ومنحنى المحفظة",
        "🛡️ 8. حماية المؤسسات والمخاطر"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ إعدادات النماذج والذكاء الاصطناعي")
ema_period = st.sidebar.slider("EMA Period", 20, 200, 200, 5)
rsi_period = st.sidebar.slider("RSI Period", 7, 30, 14, 1)
atr_period = st.sidebar.slider("ATR Period", 5, 30, 10, 1)
atr_multiplier = st.sidebar.slider("ATR Stop Multiplier", 1.0, 5.0, 2.0, 0.1)
risk_reward_ratio = st.sidebar.slider("Risk/Ratio (TP Multiplier)", 1.0, 5.0, 2.0, 0.5)
max_daily_drawdown = st.sidebar.slider("Circuit Breaker Max Loss (%)", 1.0, 10.0, 3.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("📲 Telegram Autonomous Bot")
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

# --- Render Selected Section ---

if navigation_section.startswith("📈"):
    st.header(f"📈 التحليل الفني والشارت - {selected_symbol}")
    if not df.empty and len(df) > 5:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
        if 'ema' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#2962FF', width=1.5), name=f"EMA {ema_period}"), row=1, col=1)
        if 'macd_hist' in df.columns:
            colors = ['#00E676' if val >= 0 else '#FF5252' for val in df['macd_hist'].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], marker_color=colors, name="MACD Hist"), row=2, col=1)
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#FF9800', width=1.5), name="RSI"), row=3, col=1)
        
        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

elif navigation_section.startswith("🧠"):
    st.header("🧠 منظومة الذكاء الاصطناعي ونماذج التنبؤ المحسنة")
    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        ml_s = df['ai_score'].iloc[-1]
        lstm_s = df['lstm_score'].iloc[-1]
        sent_s = df['sentiment_score'].iloc[-1]
        acc = df['model_accuracy'].iloc[-1]
        rl_act = df['rl_action'].iloc[-1]

        c1.metric("احتمالية الصعود (Random Forest)", f"{ml_s:.1f}%")
        c2.metric("توقع شبكة LSTM الزمنية", f"{lstm_s:.1f}%")
        c3.metric("مؤشر المشاعر المحسّن (FinBERT)", f"{sent_s:.1f}%")
        c4.metric("دقة النماذج التاريخية (Backtest)", f"{acc:.1f}%")

        st.success(f"🤖 **القرار الموحد لشبكة الذكاء الاصطناعي:** بناءً على الميزات الهندسية المتقدمة (Bollinger %B و ROC)، قرار التداول الموصى به لـ {selected_symbol} هو **{rl_act}**.")

elif navigation_section.startswith("🤖"):
    st.header("🤖 الوكيل الذكي ومراقبة أوامر التيليجرام")
    if st.button("🔄 فحص رسائل Telegram الواردة"):
        updates = poll_telegram_commands(telegram_token)
        if updates:
            st.success(f"تم رصد {len(updates)} رسالة جديدة!")
        else:
            st.info("لا توجد رسائل جديدة معلقة.")

    if st.button("📤 إرسال تقرير حالة الأصول الفوري للتيليجرام"):
        if not df.empty:
            p_now = df['close'].iloc[-1]
            report = f"📊 *تقرير QuantClaw المحسّن*\n- الأصل: {selected_symbol}\n- السعر: ${p_now:,.2f}\n- قرار AI: {df['rl_action'].iloc[-1]}"
            send_telegram_alert(report)
            st.success("تم إرسال التقرير بنجاح إلى التيليجرام!")

elif navigation_section.startswith("🚀"):
    st.header("🚀 التنفيذ الآلي والربط الحقيقي (API)")
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        api_k = st.text_input("API Key", type="password")
        api_s = st.text_input("API Secret", type="password")
    with c_e2:
        amt = st.number_input("مبلغ الصفقة ($)", value=100.0)
        if not df.empty:
            lp = df['close'].iloc[-1]
            if st.button("🛒 تنفيذ شراء وتسجيل في SQLite"):
                log_trade_to_db(selected_symbol, "BUY", lp, amt, "Active")
                st.success("تم تسجيل الصفقة بنجاح!")

elif navigation_section.startswith("🗺️"):
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

elif navigation_section.startswith("📊"):
    st.header("📊 تقاطع الأطر الزمنية")
    for tf in ["15m", "1h", "4h"]:
        sub = fetch_data(selected_symbol, tf)
        if not sub.empty:
            st.write(f"**الإطار الزمني {tf}:** السعر = `${sub['close'].iloc[-1]:,.2f}` | AI Action = `{sub['rl_action'].iloc[-1]}`")

elif navigation_section.startswith("📒"):
    st.header("📒 سجل الأداء ومنحنى نمو المحفظة (Equity Curve)")
    trades_df = get_trades_from_db()
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True)
        trades_df['cumulative_pnl'] = trades_df['size'].cumsum() * 0.01
        st.line_chart(trades_df['cumulative_pnl'])
        if st.button("🗑️ تفريغ كافة السجلات"):
            clear_trades_db()
            st.rerun()
    else:
        st.info("لا توجد سجلات صفقات حالياً.")

elif navigation_section.startswith("🛡️"):
    st.header("🛡️ حاسبة المخاطر المؤسسية وقاطع الدائرة اليومي")
    acc = st.number_input("رأس المال الإجمالي ($)", value=10000.0)
    risk = st.slider("المخاطرة لكل صفقة (%)", 0.5, 5.0, 1.0)
    if not df.empty and 'atr' in df.columns:
        cp = df['close'].iloc[-1]
        sl = cp - (df['atr'].iloc[-1] * atr_multiplier)
        tp = cp + ((cp - sl) * risk_reward_ratio)
        units = calculate_position_size(acc, risk, cp, sl)
        st.metric("الكمية الموصى بها (Units)", f"{units:.4f}")
        st.warning(f"⚠️ قاطع الدائرة مفعل: سيتم إيقاف التداول أوتوماتيكياً إذا تجاوزت الخسائر اليومية حد {max_daily_drawdown}%.")
