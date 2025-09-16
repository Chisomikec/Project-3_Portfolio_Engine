import streamlit as st
import pandas as pd
import plotly.express as px

from data_store import fetch_adjclose
from engine_core import annualise_stats, optimise_max_sharpe, simulate_rebalance, compute_metrics

# sidebar (subset + dates only) 
UNIVERSE = ['KO','JNJ','MSFT','TSLA','NVDA','XOM','SPY','SHY']

with st.sidebar:
    with st.sidebar.form("controls"):
        st.header("Universe & Window")
        subset = st.multiselect("Assets", options=UNIVERSE, default=UNIVERSE)

        if len(subset) < 5:
            st.warning(" Please select at least **5 assets** to run the optimiser.")

        start, end = st.date_input("Date range",  value=(pd.to_datetime("2020-01-01").date(), pd.to_datetime("2025-01-01").date()),)

        st.markdown("---")  # divider

        st.header("Rebalancing")
        freq = st.selectbox("Periodic", ["M", "Q", "Y", "None"], index=0)
        threshold = st.number_input("Threshold |w–w*|", 0.0, 0.5, 0.0, 0.01)
        tc_bps = st.number_input("Costs (bps)", 0.0, 100.0, 5.0, 0.5)
        init_capital = st.number_input("Initial capital", 100.0, 1_000_000.0, 1_000.0, 100.0)
        submitted = st.form_submit_button("Run optimisation")


if not submitted:
    st.title("Portfolio Optimiser")
    st.caption("Optimise asset weights (Max Sharpe) and simulate rebalancing.")

    st.info("Adjust the settings in the sidebar and click **Run optimisation** to start.")

    with st.expander("How to use", expanded=False):
        st.markdown(
            """
            1. **Pick at least 5 assets** or **run default universe** in *Universe & Window*.
            2. Choose **rebalancing** settings (frequency, threshold, costs).
            3. Click **Run optimisation** to compute weights and backtest the strategy.
            """
        )

    st.divider()
    st.subheader("Weights")
    st.write("—")
    st.subheader("Weight Distribution")
    st.write("—")
    st.divider()
    st.subheader("Equity Curve")
    st.write("—")
    st.divider()
    st.subheader("Performance Summary")
    cols = st.columns(3)
    for c, lbl in zip(cols, ["CAGR", "Volatility", "Sharpe"]):
        c.metric(lbl, "—")

    st.stop()


if len(subset) < 5:
    st.error("You must select at least **5 assets** before optimisation can run.")
    st.stop()

st.title(" Portfolio Optimiser ")
st.caption("Current optimisation method: Max Sharpe ratio")

if not subset:
    st.info("Pick at least one asset.")
    st.stop()

@st.cache_data(ttl=86400)
def fetch_adjclose_cached(tickers, start, end):
    return fetch_adjclose(tickers, start, end)

prices = fetch_adjclose_cached(subset, start, end)
if prices.empty or prices.shape[0] < 30:
    st.error("Not enough data.")
    st.stop()

rets = prices.pct_change().dropna()

# annualise and  optimise (fixed logic)
mu_ann, cov_ann = annualise_stats(rets)
weights = optimise_max_sharpe(mu_ann.loc[rets.columns], cov_ann.loc[rets.columns, rets.columns])


# display
c1, c2 = st.columns([1, 1.4])
with c1:
    st.subheader("Weights")
    st.dataframe(weights.round(4))
with c2:
    st.subheader("Weight Distribution")
    st.bar_chart(weights)
st.success("Optimisation complete!")
st.divider()


# backtest with simulator
with st.spinner("Running backtest…"):
    log_df = simulate_rebalance(
        returns=rets,
        target_weights=weights,
        init_capital=init_capital,
        freq=freq,
        threshold=(None if threshold == 0 else threshold),
        tc_bps=tc_bps/1e4,
    )
st.success("Rebalancing complete!")

st.divider()
st.subheader("Equity Curve")

fig = px.line(log_df.reset_index(), x=log_df.index.name or "index", y="pv", title="Equity Curve")
st.plotly_chart(fig, use_container_width=True)

st.divider()
metrics = compute_metrics(log_df, rf_annual=0.0)  # rf fixed at 0


st.subheader("Performance Summary")
col1, col2, col3 = st.columns(3)
col1.metric("CAGR", f"{metrics['CAGR']:.2%}")
col2.metric("Volatility", f"{metrics['ann_vol']:.2%}")
col3.metric("Sharpe", round(metrics["Sharpe"], 2))

col4, col5, col6 = st.columns(3)
col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
col5.metric("Final Value", f"{metrics['final_pv']:.0f}")
col6.metric("Total Costs", f"{metrics['total_costs']:.2f}")


