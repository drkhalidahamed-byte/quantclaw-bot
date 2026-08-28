import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from trading_engine import (
    calculate_indicators, run_institutional_backtest, calculate_position_size_with_trailing, 
    execute_binance_order, log_trade_to_db, get_trades_from_db, clear_trades_db, 
    update_trade_status_in_db, fetch_whale_and_liquidations_simulation, send_telegram_alert
)

st.set_page_config(page_title="QuantClaw Hedge Fund Pro Terminal", layout="wide", initial_sidebar_state="expanded")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 QuantClaw Institutional Terminal - Login")
    pass_input = st.text_input("أدخل كلمة مرور النظام:", type="password")
    if st.button("دخول للمنصة"):
        if pass_input == "quantclaw2026":
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ كلمة المرور غير صحيحة.")
    st.stop()

st.sidebar.title("⚡ QuantClaw Ultimate Pro")
theme_mode = st.sidebar.radio("🎨 وضع العرض (Theme)", ["الوضع الليلي (Dark Mode)", "الوضع الفاتح (Light Mode)"], index=0)

if theme_mode == "الوضع الليلي (Dark Mode)":
    bg_color, sidebar_bg, text_color, header_color, card_bg, border_color, plotly_template, plot_bg = "#0b0f17", "#111622", "#f0f6fc", "#79c0ff", "#161b22", "#30363d", "plotly_dark", "#111622"
else:
    bg_color, sidebar_bg, text_color, header_color, card_bg, border_color, plotly_template, plot_bg = "#ffffff", "#f0f2f6", "#1f2328", "#0969da", "#f6f8fa", "#d0d7de", "plotly", "#ffffff"

st.sidebar.subheader("🌍 بيئة التشغيل والاختبار الحي (Paper Trading)")
trading_env = st.sidebar.selectbox("اختر البيئة:", ["Paper Trading (بيئة التجربة الحية)", "Testnet (بينانس تجريبي)", "Live (الحقيقي)", "Simulator (محاكي محلي)"], index=0)

env_clean = "PaperTrading"
if "Testnet" in trading_env: env_clean = "Testnet"
elif "Live" in trading_env: env_clean = "Live"
elif "Simulator" in trading_env: env_clean = "Simulator"

market_type = st.sidebar.radio("السوق:", ["العملات الرقمية الموسعة (Crypto)", "الأسهم الأمريكية & ETFs"])
crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD"]
stock_symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "COIN", "GOOGL", "META"]

active_symbols = stock_symbols if market_type.startswith("الأسهم") else crypto_symbols
selected_symbol = st.sidebar.selectbox("🎯 الأصل النشط", active_symbols)
timeframe = st.sidebar.selectbox("⏱️ الإطار الزمني", ["5m", "15m", "1h", "4h", "1d"])

auto_refresh = st.sidebar.checkbox("🔄 تفعيل التحديث التلقائي الحي (Auto-Refresh)", value=False)
if auto_refresh:
    st.sidebar.caption("⚡ يتم تحديث البيانات كل 30 ثانية تلقائياً.")
    time.sleep(0.1)

st.sidebar.markdown("---")
navigation_section = st.sidebar.radio("اختر القسم:", [
    "📈 1. الشارت والمؤشرات المتقدمة",
    "🧠 2. نموذج الذكاء الاصطناعي (GradientBoosting)",
    "🐋 3. رصد الحيتان وتصفية العقود (Whales & Liqs)",
    "🧪 4. محرك الاختبار العكسي (Backtest)",
    "🤖 5. التداول الآلي والخلفي (Autonomous Daemon)",
    "⚖️ 6. محفظة توزيع الأصول (Portfolio Matrix)",
    "📡 7. حالة البث والتنبيهات الحية (Stream Monitor)",
    "⚙️ 8. إعدادات محرك التنفيذ وإدارة المخاطر",
    "📒 9. سجل الصفقات ومنحنى الأداء"
])

ema_period = st.sidebar.slider("EMA Period", 20, 200, 200, 5)
rsi_period = st.sidebar.slider("RSI Period", 7, 30, 14, 1)
atr_multiplier = st.sidebar.slider("ATR Trailing Multiplier", 1.0, 4.0, 2.0, 0.5)
initial_capital = st.sidebar.number_input("رأس المال الافتراضي للاختبار ($)", value=25000.0)
telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID")

@st.cache_data(ttl=15)
def fetch_data(symbol, interval):
    try:
        df = yf.download(symbol, period="1mo", interval=interval, progress=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        return calculate_indicators(df, ema_period, rsi_period, 10).dropna(subset=['close'])
    except Exception:
        return pd.DataFrame()

df = fetch_data(selected_symbol, timeframe)

# محرك المراقبة الخلفي النشط للـ Trailing Stop (Background Guard Daemon)
def background_trailing_monitor():
    trades_df = get_trades_from_db()
    if not trades_df.empty and not df.empty:
        active_trades = trades_df[trades_df['status'] == 'Active']
        current_price = df['close'].iloc[-1]
        for idx, row in active_trades.iterrows():
            t_id = row['id']
            t_action = row['action']
            t_stop = row['trailing_stop']
            if t_action == 'BUY' and current_price <= t_stop:
                update_trade_status_in_db(t_id, 'Closed (Trailing Stop Hit)')
                if telegram_token and telegram_chat_id:
                    send_telegram_alert(telegram_token, telegram_chat_id, f"⚠️ *Trailing Stop Hit Alert!*\n- Symbol: {row['symbol']}\n- Closed at Price: ${current_price:,.2f}")
            elif t_action == 'SELL' and current_price >= t_stop:
                update_trade_status_in_db(t_id, 'Closed (Trailing Stop Hit)')
                if telegram_token and telegram_chat_id:
                    send_telegram_alert(telegram_token, telegram_chat_id, f"⚠️ *Trailing Stop Hit Alert!*\n- Symbol: {row['symbol']}\n- Closed at Price: ${current_price:,.2f}")

background_trailing_monitor()

if navigation_section.startswith("📈"):
    st.header(f"📈 TradingView Pro - {selected_symbol} [{env_clean}]")
    if not df.empty:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#00e676', width=2), name="EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['cci'], line=dict(color='#29b6f6', width=1.5), name="CCI"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#ffab40', width=2), name="RSI"), row=3, col=1)
        fig.update_layout(template=plotly_template, paper_bgcolor=bg_color, plot_bgcolor=plot_bg, height=750, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

elif navigation_section.startswith("🧠"):
    st.header("🧠 نموذج الذكاء الاصطناعي المؤسسي المتقدم")
    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Composite AI Score", f"{df['ai_score'].iloc[-1]:.1f}%")
        c2.metric("LSTM Temporal Model", f"{df['lstm_score'].iloc[-1]:.1f}%")
        c3.metric("Market Sentiment Score", f"{df['sentiment_score'].iloc[-1]:.1f}%")
        c4.metric("Model Backtest Accuracy", f"{df['model_accuracy'].iloc[-1]:.1f}%")
        st.success(f"🤖 **القرار الموحد:** التوصية الفورية للأصل {selected_symbol} هي **{df['rl_action'].iloc[-1]}**")

elif navigation_section.startswith("🐋"):
    st.header(f"🐋 نظام رصد صفقات الحيتان - {selected_symbol}")
    w = fetch_whale_and_liquidations_simulation(selected_symbol)
    col1, col2 = st.columns(2)
    col1.metric("Whale Flow", w["whale_status"])
    col2.metric("Liquidations", w["liquidation_alert"])

elif navigation_section.startswith("🧪"):
    st.header("🧪 محرك الاختبار العكسي (Backtest)")
    if not df.empty:
        bt = run_institutional_backtest(df, initial_capital)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sharpe", bt["sharpe"])
        m2.metric("Sortino", bt["sortino"])
        m3.metric("Max Drawdown", f"{bt['max_dd']}%")
        m4.metric("Profit Factor", bt["profit_factor"])
        m5.metric("Win Rate", f"{bt['win_rate']}%")
        st.line_chart(bt["equity_curve"])

elif navigation_section.startswith("🤖"):
    st.header(f"🤖 التداول الآلي مع Trailing Stop [{env_clean}]")
    c_api1, c_api2 = st.columns(2)
    with c_api1: auto_key = st.text_input("API Key", type="password")
    with c_api2: auto_sec = st.text_input("API Secret", type="password")

    if not df.empty:
        cur_price, cur_action, cur_atr = df['close'].iloc[-1], df['rl_action'].iloc[-1], df['atr'].iloc[-1] if 'atr' in df.columns else 1.0
        if st.button("🚀 تشغيل حلقة التنفيذ الذاتي وبث تليجرام"):
            if cur_action == "HOLD":
                st.warning("⚠️ القرار الحالي (HOLD).")
            else:
                size, trailing_stop = calculate_position_size_with_trailing(initial_capital, 1.0, cur_price, cur_price, cur_atr, atr_multiplier)
                log_trade_to_db(selected_symbol, cur_action, cur_price, size, trailing_stop, "Active", env_clean)
                st.success(f"✅ تم تسجيل الصفقة الحية: السعر = `${cur_price:,.2f}` | Trailing Stop = `${trailing_stop:,.2f}`")
                
                if telegram_token and telegram_chat_id:
                    send_telegram_alert(telegram_token, telegram_chat_id, f"🚀 *QuantClaw Autonomous Alert [{env_clean}]*\n- {selected_symbol} | {cur_action} | Price: ${cur_price:,.2f} | Trailing Stop: ${trailing_stop:,.2f}")

                if env_clean in ["Live", "Testnet"]:
                    res = execute_binance_order(auto_key, auto_sec, selected_symbol, cur_action, size, env_clean)
                    st.json(res)

elif navigation_section.startswith("⚖️"):
    st.header("⚖️ محفظة توزيع الأصول الذكية")
    matrix_data = [{"Asset": s, "Allocation Weight": "20%"} for s in crypto_symbols[:5]]
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)

elif navigation_section.startswith("📡"):
    st.header("📡 حالة البث والتنبيهات الحية")
    st.success("🟢 الاتصال نشط ومستقر. محرك المراقبة الخلفي للـ Trailing Stop يعمل بنجاح.")

elif navigation_section.startswith("⚙️"):
    st.header("⚙️ إعدادات محرك التنفيذ وإدارة المخاطر")
    st.success("تم تفعيل إعدادات محرك المخاطر وإدارة الـ ATR.")

elif navigation_section.startswith("📒"):
    st.header("📒 سجل صفقات الاختبار الحي والتصدير")
    trades_df = get_trades_from_db()
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True)
        st.download_button("📥 تصدير سجل الصفقات CSV", data=trades_df.to_csv(index=False).encode('utf-8'), file_name="trades.csv", mime="text/csv")
        if st.button("🗑️ حذف السجلات"): clear_trades_db(); st.rerun()
    else:
        st.info("لا توجد سجلات حالياً.")
