import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from trading_engine import (
    calculate_indicators, run_institutional_backtest, calculate_position_size, 
    execute_binance_order, log_trade_to_db, get_trades_from_db, clear_trades_db, 
    fetch_whale_and_liquidations_simulation, send_telegram_alert
)

st.set_page_config(page_title="QuantClaw Hedge Fund Pro Terminal", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("⚡ QuantClaw Ultimate Pro")

# --- زر التبديل بين الوضع الليلي والوضع العادي (Theme Switcher) ---
theme_mode = st.sidebar.radio("🎨 وضع العرض (Theme)", ["الوضع الليلي (Dark Mode)", "الوضع الفاتح (Light Mode)"], index=0)

if theme_mode == "الوضع الليلي (Dark Mode)":
    bg_color = "#0b0f17"
    sidebar_bg = "#111622"
    text_color = "#f0f6fc"
    header_color = "#79c0ff"
    card_bg = "#161b22"
    border_color = "#30363d"
    plotly_template = "plotly_dark"
    plot_bg = "#111622"
else:
    bg_color = "#ffffff"
    sidebar_bg = "#f0f2f6"
    text_color = "#1f2328"
    header_color = "#0969da"
    card_bg = "#f6f8fa"
    border_color = "#d0d7de"
    plotly_template = "plotly"
    plot_bg = "#ffffff"

# --- تطبيق الأنماط الديناميكية بناءً على اختيار المستخدم ---
st.markdown(f"""
    <style>
    .main {{ background-color: {bg_color}; color: {text_color}; }}
    .stSidebar {{ background-color: {sidebar_bg}; border-right: 1px solid {border_color}; }}
    
    h1, h2, h3, h4, h5, h6 {{ color: {header_color} !important; font-family: 'Courier New', Courier, monospace; font-weight: bold; }}
    p, span, label, .stMarkdown {{ color: {text_color} !important; font-size: 15px; }}
    
    .metric-card {{ 
        background-color: {card_bg}; 
        border: 1px solid {border_color}; 
        padding: 15px; 
        border-radius: 8px; 
        text-align: center; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stSelectbox, .stMultiSelect, .stNumberInput, .stTextInput {{ color: {text_color} !important; }}
    .stButton>button {{ 
        background-color: #238636; 
        color: white; 
        font-weight: bold; 
        border-radius: 6px; 
        border: none;
        padding: 8px 16px;
    }}
    .stButton>button:hover {{ background-color: #2ea043; }}
    </style>
""", unsafe_allow_html=True)

# --- اختيار بيئة التشغيل ---
st.sidebar.subheader("🌍 بيئة التشغيل (Execution Environment)")
trading_env = st.sidebar.selectbox(
    "اختر البيئة:",
    ["Simulator (المحاكي المحلي)", "Testnet (بيئة التجربة)", "Live (الحساب الحقيقي)"],
    index=0
)

env_clean = "Simulator"
if "Testnet" in trading_env:
    env_clean = "Testnet"
elif "Live" in trading_env:
    env_clean = "Live"

if env_clean == "Live":
    st.sidebar.error("🚨 تحذير: تعمل على الحساب الحقيقي (Live).")

market_type = st.sidebar.radio("السوق:", ["العملات الرقمية (Crypto)", "الأسهم الأمريكية & ETFs"])
crypto_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
stock_symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "COIN"]

active_symbols = stock_symbols if market_type.startswith("الأسهم") else crypto_symbols
selected_symbol = st.sidebar.selectbox("🎯 الأصل النشط", active_symbols)
timeframe = st.sidebar.selectbox("⏱️ الإطار الزمني", ["5m", "15m", "1h", "4h", "1d"])

st.sidebar.markdown("---")
st.sidebar.subheader("📂 أقسام المنصة المؤسسية")
navigation_section = st.sidebar.radio(
    "اختر القسم:",
    [
        "📈 1. الشارت والمؤشرات المتقدمة",
        "🧠 2. نموذج الذكاء الاصطناعي (GradientBoosting)",
        "🐋 3. رصد الحيتان وتصفية العقود (Whales & Liqs)",
        "🧪 4. محرك الاختبار العكسي (Backtest)",
        "🤖 5. التداول الآلي والخلفي (Autonomous Daemon)",
        "⚖️ 6. محفظة توزيع الأصول (Portfolio Matrix)",
        "📒 7. سجل الصفقات ومنحنى الأداء"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ إدارة المخاطر والتنبيهات")
ema_period = st.sidebar.slider("EMA Period", 20, 200, 200, 5)
rsi_period = st.sidebar.slider("RSI Period", 7, 30, 14, 1)
risk_reward_ratio = st.sidebar.slider("Risk/Ratio TP", 1.0, 5.0, 2.0, 0.5)
initial_capital = st.sidebar.number_input("محاكاة رأس المال ($)", value=10000.0)

telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID")

@st.cache_data(ttl=20)
def fetch_data(symbol, interval):
    try:
        period = "7d" if interval in ["1m", "5m"] else ("1mo" if interval in ["15m", "30m"] else "3mo")
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return pd.DataFrame()
        df = calculate_indicators(df, ema_period, rsi_period, 10)
        return df.dropna(subset=['close'])
    except Exception:
        return pd.DataFrame()

df = fetch_data(selected_symbol, timeframe)

# --- Render Sections ---

if navigation_section.startswith("📈"):
    st.header(f"📈 Bloomberg Terminal / TradingView Pro - {selected_symbol} [{env_clean}]")
    if not df.empty and len(df) > 5:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC"), row=1, col=1)
        if 'ema' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ema'], line=dict(color='#00e676', width=2), name=f"EMA {ema_period}"), row=1, col=1)
        if 'macd_hist' in df.columns:
            colors = ['#00e676' if val >= 0 else '#ff5252' for val in df['macd_hist'].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], marker_color=colors, name="MACD Hist"), row=2, col=1)
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#ffab40', width=2), name="RSI"), row=3, col=1)
        
        # توافق الشارت التلقائي مع الوضع المختار
        fig.update_layout(
            template=plotly_template, 
            paper_bgcolor=bg_color, 
            plot_bgcolor=plot_bg,
            height=750, 
            xaxis_rangeslider_visible=False, 
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(color=text_color, size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

elif navigation_section.startswith("🧠"):
    st.header("🧠 نموذج الذكاء الاصطناعي المؤسسي (GradientBoosting + ML)")
    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        ml_s = df['ai_score'].iloc[-1]
        lstm_s = df['lstm_score'].iloc[-1]
        sent_s = df['sentiment_score'].iloc[-1]
        acc = df['model_accuracy'].iloc[-1]

        c1.metric("GradientBoosting AI Score", f"{ml_s:.1f}%")
        c2.metric("LSTM Temporal Prediction", f"{lstm_s:.1f}%")
        c3.metric("CMF & Volume Momentum", f"{sent_s:.1f}%")
        c4.metric("Model Backtest Accuracy", f"{acc:.1f}%")

        st.success(f"🤖 **القرار الموحد للشبكة:** التوصية الفورية للأصل {selected_symbol} هي **{df['rl_action'].iloc[-1]}**.")

elif navigation_section.startswith("🐋"):
    st.header(f"🐋 نظام رصد صفقات الحيتان وتصفية العقود الآجلة - {selected_symbol}")
    whale_data = fetch_whale_and_liquidations_simulation(selected_symbol)
    
    col1, col2 = st.columns(2)
    col1.metric("حالة سيولة الحيتان (Whale Flow)", whale_data["whale_status"])
    col2.metric("حجم تصفية العقود الآجلة (Liquidations)", whale_data["liquidation_alert"])
    
    st.info("💡 يتم تحديث هذه البيانات أوتوماتيكياً لرصد التجمعات السعرية الكبرى قبل حدوث الانفجارات السعرية.")

elif navigation_section.startswith("🧪"):
    st.header(f"🧪 محرك الاختبار العكسي للاستراتيجية (Institutional Backtest)")
    if not df.empty:
        bt = run_institutional_backtest(df, initial_capital)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sharpe Ratio", bt["sharpe"])
        m2.metric("Sortino Ratio", bt["sortino"])
        m3.metric("Max Drawdown", f"{bt['max_dd']}%")
        m4.metric("Profit Factor", bt["profit_factor"])
        m5.metric("Win Rate", f"{bt['win_rate']}%")

        st.line_chart(bt["equity_curve"])

elif navigation_section.startswith("🤖"):
    st.header(f"🤖 التداول الآلي والخلفي المستمر (Autonomous Daemon) [{env_clean}]")
    st.info("💡 يقوم هذا الوضع بفحص الأصول وتنفيذ الصفقات أوتوماتيكياً وإرسال التنبيهات عبر تليجرام.")

    c_api1, c_api2 = st.columns(2)
    with c_api1:
        auto_key = st.text_input("API Key", type="password")
    with c_api2:
        auto_sec = st.text_input("API Secret", type="password")

    if not df.empty:
        cur_price = df['close'].iloc[-1]
        cur_action = df['rl_action'].iloc[-1]
        cur_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 1.0

        if st.button("🚀 تشغيل حلقة التنفيذ الذاتي الفوري"):
            if cur_action == "HOLD":
                st.warning("⚠️ القرار الحالي (HOLD). لا توجد إشارة تنفيذ.")
            else:
                sl = cur_price - (cur_atr * 2.0) if cur_action == "BUY" else cur_price + (cur_atr * 2.0)
                size = calculate_position_size(initial_capital, 1.0, cur_price, sl)
                
                log_trade_to_db(selected_symbol, cur_action, cur_price, size, "Active", env_clean)
                st.success(f"✅ تم تنفيذ الصفقة بنجاح [{env_clean}]: السعر = `${cur_price:,.2f}` | الكمية = `{size:.4f}`")
                
                if telegram_token and telegram_chat_id:
                    msg = f"🚀 *QuantClaw Autonomous Trade [{env_clean}]*\n- Symbol: {selected_symbol}\n- Action: {cur_action}\n- Price: ${cur_price:,.2f}\n- Size: {size:.4f}"
                    send_telegram_alert(telegram_token, telegram_chat_id, msg)

                if env_clean != "Simulator":
                    res = execute_binance_order(auto_key, auto_sec, selected_symbol, cur_action, size, env_clean)
                    st.json(res)

elif navigation_section.startswith("⚖️"):
    st.header("⚖️ محفظة توزيع الأصول الذكية (Multi-Asset Portfolio Matrix)")
    st.write("تحليل الارتباط وتوزيع المخاطر بين الأصول النشطة:")
    
    matrix_data = []
    for s in crypto_symbols[:4]:
        sub_df = fetch_data(s, "1h")
        if not sub_df.empty:
            last_p = sub_df['close'].iloc[-1]
            score = sub_df['ai_score'].iloc[-1] if 'ai_score' in sub_df.columns else 50.0
            matrix_data.append({"Asset": s, "Price": f"${last_p:,.2f}", "AI Score": f"{score:.1f}%", "Allocation Weight": "25%"})
            
    if matrix_data:
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)

elif navigation_section.startswith("📒"):
    st.header("📒 سجل الصفقات ومنحنى نمو المحفظة")
    trades_df = get_trades_from_db()
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True)
        if st.button("🗑️ حذف السجلات بالكامل"):
            clear_trades_db()
            st.rerun()
    else:
        st.info("لا توجد سجلات صفقات مسجلة حالياً.")
