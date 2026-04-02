import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import plotly.graph_objects as go
import shap
import numpy as np
import matplotlib.pyplot as plt
from utils import predict_future

st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

# --------- CUSTOM UI CSS ---------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0E1117;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    background-color: #1f77b4;
    color: white;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff4b4b;
}

[data-testid="stMetricValue"] {
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("📈 AI Powered Stock Market Dashboard")

# ---------------- SIDEBAR ----------------
stock = st.sidebar.selectbox(
    "📌 Select Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

period = st.sidebar.selectbox(
    "⏳ Select Period",
    ["1mo", "3mo", "6mo", "1y", "5y"]
)

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("---")

# ---------------- DATA ----------------
df = yf.download(
    stock,
    period=period,
    interval="1d",
    auto_adjust=False,
    progress=False
)

# ✅ Safety check
if df.empty:
    st.error("❌ Data load nahi ho raha")
    st.stop()

# -------- SIDEBAR LIVE INFO --------

latest_price = float(df['Close'].iloc[-1])
prev_price = float(df['Close'].iloc[-2])

st.sidebar.metric("💰 Current Price", f"₹ {round(latest_price,2)}")

if latest_price > prev_price:
    st.sidebar.success("📈 Uptrend")
else:
    st.sidebar.error("📉 Downtrend")

# Download button
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "⬇ Download Data",
    csv,
    "stock_data.csv",
    "text/csv"
)
# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Prediction", "🧠 Insights"])

# ================= TAB 1 =================
with tab1:
    st.subheader("📊 Live Stock Chart")

    # Reset index
    df = df.reset_index()

    # 🔥 FIX: flatten columns
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Debug (optional)
    st.write(df.tail())

    # Ensure numeric
    df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].astype(float)

    # ----------- CHART -----------
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    fig.update_layout(
        title="Live Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        height=600
    )

    fig.update_xaxes(rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("📌 Latest Price", f"₹ {round(df['Close'].iloc[-1],2)}")
    col2.metric("📈 High", f"₹ {round(df['High'].max(),2)}")
    col3.metric("📉 Low", f"₹ {round(df['Low'].min(),2)}")

# ================= TAB 2 =================
with tab2:
    st.subheader("🔮 Next 7 Days Prediction")

    data = df[['Close']].copy()

    for i in range(1, 6):
        data[f'lag_{i}'] = data['Close'].shift(i)

    data.dropna(inplace=True)

    last_values = data.iloc[-1][1:].values.tolist()

    if st.button("🚀 Predict Future"):
        preds = predict_future(model, last_values)

        future_df = pd.DataFrame({
            "Day": range(1, 8),
            "Predicted Price": preds
        })

        st.success("Prediction Done ✅")
        st.dataframe(future_df)
        st.line_chart(future_df.set_index("Day"))

# ================= TAB 3 (SHAP) =================
with tab3:
    st.subheader("🧠 Explainable AI (SHAP)")

    data = df[['Close']].copy()
    for i in range(1, 6):
        data[f'lag_{i}'] = data['Close'].shift(i)

    data.dropna(inplace=True)

    last_values = data.iloc[-1][1:].values.reshape(1, -1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(last_values)

    feature_names = [f"lag_{i}" for i in range(1, 6)]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values[0]
    })

    st.write("### 📊 Feature Contribution")
    st.dataframe(shap_df)

    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"], shap_df["Impact"])
    ax.set_title("SHAP Feature Impact")
    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.caption("Built with ❤️ using ML + DSA Trees")