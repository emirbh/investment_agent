import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

# Using the project's config to find the DB path
try:
    from config import DB_PATH
except ImportError:
    DB_PATH = "investment.db"

st.set_page_config(
    page_title="Dividend Agent Dashboard", 
    page_icon="📈", 
    layout="wide"
)

st.title("📈 Dividend Investment Agent")
st.markdown("Review the latest ETF predictions, historical strategy performance, and macro-economic factors.")

@st.cache_data(ttl=3600)
def query_db(query: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)

# --- Fetch Data ---
try:
    latest_run = query_db("SELECT MAX(run_date) as max_date FROM predictions").iloc[0]['max_date']
    if not latest_run:
        st.warning("No predictions found in the database. Run the pipeline first.")
        st.stop()
except Exception as e:
    st.error(f"Error accessing database at {DB_PATH}: {e}")
    st.stop()

# 1. Backtest Results
backtest_df = query_db("SELECT * FROM backtest_results ORDER BY run_date DESC")
st.header("1. Strategy Performance", divider="violet")

if not backtest_df.empty:
    latest_bt = backtest_df.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Strategy Return", value=f"{latest_bt['strategy_return']*100:.2f}%", 
                  delta=f"{(latest_bt['strategy_return'] - latest_bt['baseline_return'])*100:.2f}% vs Baseline")
    with col2:
        st.metric(label="Baseline Return", value=f"{latest_bt['baseline_return']*100:.2f}%")
    with col3:
        sharpe = latest_bt["strategy_sharpe"]
        st.metric(label="Strategy Sharpe", value=f"{sharpe:.2f}" if sharpe is not None else "—")
    with col4:
        drawdown = latest_bt["max_drawdown"]
        st.metric(
            label="Max Drawdown",
            value=f"{drawdown * 100:.2f}%" if drawdown is not None else "—",
            delta_color="inverse",
        )

    # Plot performance chart (if we had series data, but we just have table rows for backtest runs)
    # Let's plot the strategy return over different backtest run dates to show stability over time
    if len(backtest_df) > 1:
        st.subheader("Historical Backtest Runs")
        bt_melt = backtest_df.melt(id_vars=["run_date"], value_vars=["strategy_return", "baseline_return"], 
                                   var_name="Portfolio", value_name="Cumulative Return")
        fig_bt = px.line(bt_melt, x="run_date", y="Cumulative Return", color="Portfolio", 
                         title="Strategy vs Baseline Return Across Backtest Runs", markers=True)
        fig_bt.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig_bt, use_container_width=True)

# 2. Latest Predictions
st.header(f"2. ETF Weekly Predictions (Run: {latest_run})", divider="blue")

preds_df = query_db("""
    SELECT p.predicted_rank as Rank, p.ticker as Ticker, e.name as Name, 
           p.action as Action, p.predicted_ret as Expected_Return, p.confidence as Confidence,
           e.dividend_yield as Yield, e.category as Category
    FROM predictions p
    LEFT JOIN etf_universe e ON p.ticker = e.ticker
    WHERE p.run_date = ?
    ORDER BY p.predicted_rank ASC
""", (latest_run,))

# Color coding for Action
def color_action(val):
    color = 'grey'
    if val in ['BUY', 'OVERWEIGHT']: color = 'green'
    elif val in ['SELL', 'UNDERWEIGHT']: color = 'red'
    elif val == 'HOLD': color = 'orange'
    return f'color: {color}; font-weight: bold;'

st.dataframe(
    preds_df.style.map(color_action, subset=["Action"]).format(
        {"Expected_Return": "{:.2%}", "Confidence": "{:.1%}", "Yield": "{:.2f}%"},
        na_rep="—",
    ),
    use_container_width=True,
    hide_index=True,
)

# 3. Deep Dive Explorer
st.header("3. ETF Deep Dive", divider="green")

selected_tickers = st.multiselect(
    "Select ETFs to compare historic trends (normalized):", 
    preds_df['Ticker'].tolist(),
    default=[preds_df['Ticker'].iloc[0]] if not preds_df.empty else None
)

if selected_tickers:
    placeholders = ",".join("?" for _ in selected_tickers)
    query = f"SELECT ticker, date, close, dividends FROM price_history WHERE ticker IN ({placeholders}) ORDER BY date"
    prices_df = query_db(query, tuple(selected_tickers))
    
    if not prices_df.empty:
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        # Keep only last 250 days per ticker to ensure comparable 1-year windows
        prices_df = prices_df.sort_values('date').groupby('ticker').tail(250).reset_index(drop=True)
        
        # Normalize to Base 100 for comparison
        first_prices = prices_df.groupby('ticker')['close'].transform('first')
        prices_df['Normalized Price'] = (prices_df['close'] / first_prices) * 100
        
        fig_px = px.line(prices_df, x="date", y="Normalized Price", color="ticker", 
                         title="1 Year Relative Price Performance (Base 100)")
        
        divs = prices_df[prices_df['dividends'] > 0]
        if not divs.empty:
            fig_px.add_scatter(x=divs['date'], y=divs['Normalized Price'], mode='markers', 
                               marker=dict(color='yellow', size=10, symbol='star', line=dict(color='black', width=1)),
                               hovertext=divs['ticker'] + " Dividend: $" + divs['dividends'].round(3).astype(str),
                               name='Dividend Payout')
            
        st.plotly_chart(fig_px, use_container_width=True)
    else:
        st.warning("No price history available across the database for the selected tickers.")
