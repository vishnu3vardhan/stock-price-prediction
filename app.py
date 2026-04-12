import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Stock Price Prediction Dashboard")
st.markdown("### Predicting NIFTY 30 Stocks using Machine Learning")

# -----------------------------
# PATH TO PREDICTIONS
# -----------------------------
PRED_DIR = "outputs/predictions"

# -----------------------------
# GET STOCK LIST
# -----------------------------
@st.cache_data
def get_stock_files():
    files = [f for f in os.listdir(PRED_DIR) if f.endswith(".csv")]
    return sorted(files)

stock_files = get_stock_files()

# -----------------------------
# SIDEBAR - STOCK SELECTOR
# -----------------------------
st.sidebar.header("📌 Select Stock")

selected_file = st.sidebar.selectbox("Choose a stock", stock_files)

stock_name = selected_file.replace(".csv", "")

file_path = os.path.join(PRED_DIR, selected_file)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data(file_path)

# -----------------------------
# METRICS
# -----------------------------
actual = df["Actual"]
predicted = df["Predicted"]

mae = abs(actual - predicted).mean()
rmse = ((actual - predicted) ** 2).mean() ** 0.5

latest_actual = actual.iloc[-1]
latest_pred = predicted.iloc[-1]

trend = "📈 Uptrend" if latest_pred > latest_actual else "📉 Downtrend"

# -----------------------------
# METRICS DISPLAY
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("Latest Trend", trend)

# -----------------------------
# PLOTLY CHART
# -----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Actual"],
    mode="lines",
    name="Actual Price",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Predicted"],
    mode="lines",
    name="Predicted Price",
    line=dict(color="orange")
))

fig.update_layout(
    title=f"{stock_name} - Actual vs Predicted Prices",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# INSIGHT BOX
# -----------------------------
st.markdown("### 🧠 Model Insight")

if mae < 10:
    st.success("Good accuracy for short-term prediction (low error).")
elif mae < 20:
    st.warning("Moderate accuracy — stock is harder to predict.")
else:
    st.error("High error — market is highly volatile for this stock.")

st.markdown(f"""
- 📌 Latest Actual Price: **{latest_actual:.2f}**
- 📌 Latest Predicted Price: **{latest_pred:.2f}**
""")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🚀 Built using Machine Learning + Streamlit + Plotly")