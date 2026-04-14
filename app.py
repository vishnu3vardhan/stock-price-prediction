import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stock Price Prediction | ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("### C V VISHNU VARDHAN CHARY")
    st.markdown("*Machine Learning Intern*")
    st.markdown("*Corizo Edtech*")
    st.markdown("[github.com/vishnu3vardhan](https://github.com/vishnu3vardhan)")
    st.markdown("---")

    # Global Info Expander
    with st.expander("About this Dashboard"):
        st.markdown("""
        **Model:** Machine Learning pipeline trained on 10 years of NSE historical data.   
        **Metrics:** MAE, RMSE, MAPE, R2.  
        **Interval:** 95% prediction interval based on historical error distribution.  
        """)

    st.markdown("---")
    st.caption("Dashboard for NSE Stock Predictions")


PRED_DIR = "outputs/predictions"


@st.cache_data
def get_stock_files():
    """Return sorted list of available stock prediction files."""
    if not os.path.exists(PRED_DIR):
        return []
    files = [f for f in os.listdir(PRED_DIR) if f.endswith(".csv")]
    return sorted(files)

@st.cache_data
def load_single_stock(file_path):
    """Load and preprocess a single stock CSV."""
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    return df

@st.cache_data
def load_all_stocks(stock_files):
    """Load all stock files and return a dictionary of dataframes."""
    all_data = {}
    for file in stock_files:
        symbol = file.replace(".csv", "")
        with st.spinner(f"Loading {symbol}..."):
            df = load_single_stock(os.path.join(PRED_DIR, file))
        all_data[symbol] = df
    return all_data

@st.cache_data
def compute_technical_indicators(df, window_sma=20, window_ema=20):
    """Add SMA and EMA to dataframe."""
    df = df.copy()
    df['SMA'] = df['Actual'].rolling(window=window_sma).mean()
    df['EMA'] = df['Actual'].ewm(span=window_ema, adjust=False).mean()
    return df


def calculate_metrics(actual, predicted):
    """Return dictionary of regression metrics."""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Accuracy': max(0, 100 - mape)
    }


st.title("NSE Stock Price Prediction Dashboard")
st.markdown("### Machine Learning Forecast Analysis")

# Load all stock data with spinner
with st.spinner("Loading all stock prediction files..."):
    stock_files = get_stock_files()
    if not stock_files:
        st.error("No prediction files found. Please ensure the 'outputs/predictions' directory contains CSV files.")
        st.stop()
    all_data = load_all_stocks(stock_files)


tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "Multi-Stock Comparison", "Model Diagnostics"])


with tab1:
    st.markdown("## Individual Stock Prediction Analysis")
    st.markdown("Select a stock and date range. Use SMA/EMA overlays for technical context.")

    # Controls row
    col_ctrl1, col_ctrl2 = st.columns([2, 2])
    with col_ctrl1:
        selected_symbol = st.selectbox(
            "Select Stock Symbol",
            options=list(all_data.keys()),
            index=0
        )
        # SMA/EMA toggles
        show_indicators = st.columns(2)
        with show_indicators[0]:
            show_sma = st.checkbox("SMA (20)", value=False, help="Simple Moving Average (20-day)")
        with show_indicators[1]:
            show_ema = st.checkbox("EMA (20)", value=False, help="Exponential Moving Average (20-day)")

    with col_ctrl2:
        df_selected = all_data[selected_symbol]
        min_date = df_selected['Date'].min().date()
        max_date = df_selected['Date'].max().date()

        # Use a slider for date range
        date_slider = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        start_date, end_date = date_slider

        # Validation: if start >= end, use default
        if start_date >= end_date:
            st.warning("Start date must be before end date. Using full range.")
            start_date, end_date = min_date, max_date

    # Apply date filter
    mask = (df_selected['Date'].dt.date >= start_date) & (df_selected['Date'].dt.date <= end_date)
    df_filtered = df_selected.loc[mask].copy()

    if df_filtered.empty:
        st.error("No data available for the selected date range. Please adjust.")
        st.stop()

    # Technical indicators
    if show_sma or show_ema:
        df_filtered = compute_technical_indicators(df_filtered)

    # Metrics
    actual = df_filtered['Actual']
    predicted = df_filtered['Predicted']
    metrics = calculate_metrics(actual, predicted)

    # KPI Cards
    st.markdown("### Key Performance Indicators")
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("MAE", f"₹{metrics['MAE']:.2f}", help=f"{metrics['MAE']/actual.mean()*100:.1f}% of avg price")
    with kpi_cols[1]:
        st.metric("RMSE", f"₹{metrics['RMSE']:.2f}")
    with kpi_cols[2]:
        st.metric("Accuracy", f"{metrics['Accuracy']:.1f}%", help="100% - MAPE")
    with kpi_cols[3]:
        latest_residual = actual.iloc[-1] - predicted.iloc[-1]
        residual_pct = (latest_residual / actual.iloc[-1]) * 100
        bias = "Over-predicted" if latest_residual < 0 else "Under-predicted"
        delta_color = "inverse" if latest_residual < 0 else "normal"
        st.metric(
            "Latest Bias", 
            bias, 
            delta=f"₹{latest_residual:+.2f} ({residual_pct:+.1f}%)",
            delta_color=delta_color
        )
        st.caption(f"Actual: ₹{actual.iloc[-1]:.2f} | Pred: ₹{predicted.iloc[-1]:.2f}")

    # Price Chart
    st.markdown("### Price Evolution")
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Actual'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#2c3e50', width=2.5),
            hovertemplate='<b>Date</b>: %{x|%d %b %Y}<br><b>Actual</b>: ₹%{y:.2f}<extra></extra>'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Predicted'],
            mode='lines',
            name='Predicted Price',
            line=dict(color='#e67e22', width=2, dash='dash'),
            hovertemplate='<b>Date</b>: %{x|%d %b %Y}<br><b>Predicted</b>: ₹%{y:.2f}<extra></extra>'
        )
    )
    if show_sma:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['Date'],
                y=df_filtered['SMA'],
                mode='lines',
                name='SMA (20)',
                line=dict(color='#3498db', width=1.5),
                hovertemplate='<b>SMA</b>: ₹%{y:.2f}<extra></extra>'
            )
        )
    if show_ema:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['Date'],
                y=df_filtered['EMA'],
                mode='lines',
                name='EMA (20)',
                line=dict(color='#9b59b6', width=1.5),
                hovertemplate='<b>EMA</b>: ₹%{y:.2f}<extra></extra>'
            )
        )

    # Prediction interval
    error_std = np.std(actual - predicted)
    upper_bound = predicted + 1.96 * error_std
    lower_bound = predicted - 1.96 * error_std
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df_filtered['Date'], df_filtered['Date'][::-1]]),
            y=pd.concat([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(230,126,34,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='95% Prediction Interval'
        )
    )

    fig.update_layout(
        title=f"{selected_symbol} - Actual vs Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_white",
        height=550,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='rgba(0,0,0,0.05)'
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: Drag the slider below the chart to zoom in on a specific time period.")

    # Error Analysis & Export
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("### Prediction Error Distribution")
        errors = actual - predicted
        fig_hist = px.histogram(
            errors, nbins=30,
            title="Residuals (Actual - Predicted)",
            labels={'value': 'Error (₹)', 'count': 'Frequency'},
            color_discrete_sequence=['#5a6c7d']
        )
        fig_hist.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.markdown("### Data Export")
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name=f"{selected_symbol}_predictions.csv",
            mime="text/csv"
        )
        st.markdown("#### Recent Records")
        st.dataframe(
            df_filtered.tail(10)[['Date', 'Actual', 'Predicted']].style.format({
                'Actual': '₹{:.2f}',
                'Predicted': '₹{:.2f}'
            }),
            use_container_width=True
        )
        # Option to show more rows
        if st.checkbox("Show more rows", key="show_more"):
            num_rows = st.number_input("Number of rows", min_value=10, max_value=100, value=20, step=10)
            st.dataframe(
                df_filtered.tail(num_rows)[['Date', 'Actual', 'Predicted']].style.format({
                    'Actual': '₹{:.2f}',
                    'Predicted': '₹{:.2f}'
                }),
                use_container_width=True
            )


with tab2:
    st.markdown("## Compare Multiple Stocks")
    st.markdown("Select up to 5 stocks to compare performance and price movements.")

    available_symbols = list(all_data.keys())
    selected_comparison = st.multiselect(
        "Choose stocks to compare",
        options=available_symbols,
        default=available_symbols[:3] if len(available_symbols) >= 3 else available_symbols,
        max_selections=5,
        help="Maximum 5 stocks for clear visualization"
    )

    if len(selected_comparison) == 5:
        st.info("Maximum 5 stocks selected. Remove one to add another.")

    if not selected_comparison:
        st.warning("Please select at least one stock for comparison.")
    else:
        st.markdown("### Normalized Price Performance (Base 100)")
        norm_data = []
        for symbol in selected_comparison:
            df = all_data[symbol].copy()
            df['Normalized_Actual'] = df['Actual'] / df['Actual'].iloc[0] * 100
            df['Symbol'] = symbol
            norm_data.append(df[['Date', 'Normalized_Actual', 'Symbol']])
        norm_df = pd.concat(norm_data)
        fig_norm = px.line(
            norm_df, x='Date', y='Normalized_Actual', color='Symbol',
            title="Relative Stock Performance (Normalized to 100 at Start)",
            template="plotly_white", height=500
        )
        fig_norm.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_norm, use_container_width=True)

        st.markdown("### Performance Metrics Comparison")
        comp_metrics = []
        for symbol in selected_comparison:
            df = all_data[symbol]
            m = calculate_metrics(df['Actual'], df['Predicted'])
            comp_metrics.append({
                'Stock': symbol,
                'MAE (₹)': f"{m['MAE']:.2f}",
                'RMSE (₹)': f"{m['RMSE']:.2f}",
                'MAPE (%)': f"{m['MAPE']:.2f}",
                'R2 Score': f"{m['R2']:.3f}",
                'Accuracy (%)': f"{m['Accuracy']:.1f}"
            })
        comp_df = pd.DataFrame(comp_metrics)
        st.dataframe(
            comp_df.style.background_gradient(subset=['MAPE (%)', 'R2 Score', 'Accuracy (%)'], cmap='RdYlGn_r'),
            use_container_width=True, hide_index=True
        )

        if len(selected_comparison) > 1:
            st.markdown("### Correlation Between Stock Movements")
            st.markdown("Higher correlation (>0.7) means stocks tend to move together.")
            pivot_data = {}
            for symbol in selected_comparison:
                df = all_data[symbol].set_index('Date')['Actual']
                pivot_data[symbol] = df
            price_df = pd.DataFrame(pivot_data).dropna()
            if not price_df.empty:
                corr_matrix = price_df.corr()
                fig_corr = px.imshow(
                    corr_matrix, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    title="Price Correlation Matrix"
                )
                fig_corr.update_layout(template="plotly_white", height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Insufficient overlapping dates for correlation analysis.")


with tab3:
    st.markdown("## Model Diagnostic Dashboard")
    st.markdown("Advanced analysis of prediction residuals and model behavior.")

    selected_diag = st.selectbox(
        "Select stock for detailed diagnostics",
        options=list(all_data.keys()),
        key='diag_select'
    )

    df_diag = all_data[selected_diag]
    actual = df_diag['Actual']
    predicted = df_diag['Predicted']
    residuals = actual - predicted

    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        fig_resid = px.scatter(
            x=predicted, y=residuals,
            title="Residuals vs Fitted Values",
            labels={'x': 'Predicted Price (₹)', 'y': 'Residual (Actual - Predicted)'},
            trendline="lowess",
            color_discrete_sequence=['#2c3e50']
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        fig_resid.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_resid, use_container_width=True)

    with col_diag2:
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        qq_df = pd.DataFrame({'Theoretical Quantiles': osm, 'Sample Quantiles': osr})
        fig_qq = px.scatter(
            qq_df, x='Theoretical Quantiles', y='Sample Quantiles',
            title="Q-Q Plot (Normality Check)",
            trendline="ols",
            color_discrete_sequence=['#2c3e50']
        )
        fig_qq.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_qq, use_container_width=True)
        st.markdown(f"**R2 of Q-Q line:** {r**2:.4f} (closer to 1 indicates normality)")

    st.markdown("### Residual Autocorrelation")
    lags = min(40, len(residuals)//2)
    autocorr = [residuals.autocorr(lag) for lag in range(1, lags+1)]
    fig_acf = px.bar(
        x=list(range(1, lags+1)), y=autocorr,
        title="Autocorrelation Function (ACF) of Residuals",
        labels={'x': 'Lag', 'y': 'Autocorrelation'}
    )
    fig_acf.add_hline(y=0, line_dash="solid", line_color="black")
    n = len(residuals)
    sig_level = 1.96 / np.sqrt(n)
    fig_acf.add_hline(y=sig_level, line_dash="dash", line_color="red", annotation_text="95% significance")
    fig_acf.add_hline(y=-sig_level, line_dash="dash", line_color="red")
    fig_acf.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_acf, use_container_width=True)


st.markdown("---")
st.markdown(
    f"**Stock Price Prediction Dashboard | Machine Learning | Streamlit & Plotly**  \n"
    f"*Last data update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*"
)


