

# %%
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pypfopt import EfficientFrontier
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


def annualise_stats(ret_wide: pd.DataFrame, periods_per_year: int = 252
                    ) -> Tuple[pd.Series, pd.DataFrame]:
    """
    ret_wide: DataFrame of periodic returns (index=Date, columns=tickers)
    """
    mu_ann = ret_wide.mean() * periods_per_year
    cov_ann = ret_wide.cov() * periods_per_year
    return mu_ann, cov_ann


def optimise_max_sharpe(mu_ann: pd.Series, cov_ann: pd.DataFrame) -> pd.Series:
    tickers = list(cov_ann.index)
    ef = EfficientFrontier(mu_ann.loc[tickers], cov_ann.loc[tickers, tickers])

    # constraints: w <= 0.2 
    ef.add_constraint(lambda w: w <= 0.20)

    # constraint: SHY >= 0.05 (only if SHY is present)
    if "SHY" in tickers:
        j = tickers.index("SHY")
        ef.add_constraint(lambda w, j=j: w[j] >= 0.05)

    # objective: max_sharpe, rf=0.0 (fixed)
    ef.max_sharpe(risk_free_rate=0.0)

    w = pd.Series(ef.clean_weights()).reindex(tickers).fillna(0.0)
    return w


def simulate_rebalance(
    returns: pd.DataFrame,
    target_weights: pd.Series,
    *,
    init_capital: float = 1_000.0,
    freq: str | None = "M",          # "M", "Q", "Y" or None (disable periodic)
    threshold: float | None = None,  # e.g. 0.05 band; None disables threshold trigger
    tc_bps: float = 0.0              # transaction cost in decimal (e.g. 0.0005 = 5 bps)
) -> pd.DataFrame:
    """
    Simulate portfolio with periodic and/or threshold rebalancing.

    Parameters
    
    returns : DataFrame
        Daily (or periodic) returns, index = dates, columns = tickers.
    target_weights : Series
        Target weights, index must match returns.columns.
    init_capital : float
        Starting portfolio value.
    freq : {"M","Q","Y", None}
        Periodic rebalance cadence (month/quarter/year end). None disables periodic.
    threshold : float or None
        Absolute weight deviation band (e.g., 0.05). None disables threshold.
    tc_bps : float
        Proportional trading cost  (e.g., 0.0005 = 5 bps). Transaction cost per currency

    Returns
    
    log_df : DataFrame
        Daily log with columns: pv, rebalanced, turnover, costs, and weights per asset.
    """
    # alignment 
    returns = returns.copy().sort_index()
    returns = returns.dropna(how="any")

    target_weights = target_weights.copy().reindex(returns.columns)
    assert np.isclose(target_weights.sum(), 1.0), "target_weights must sum to 1"
    assert target_weights.index.equals(returns.columns), "target_weights must align with returns.columns"

    period_flags = pd.Series(False, index=returns.index)

    # helpers for periodic trigger 
    if freq is None:
        period_ends = pd.DatetimeIndex([])
       
    else:
        # month/quarter/year ends on calendar
        # keep only dates that exist in returns index
    
        last_idx_each_period = (
        returns.groupby(returns.index.to_period(freq)).tail(1).index
        )
        period_ends = returns.index.intersection(last_idx_each_period)
        period_flags = pd.Series(returns.index.isin(period_ends), index=returns.index)
       

    # state
    pv = float(init_capital)
    holding = pv * target_weights  #  holdings vector
    log_rows = []

    # main loop
    for day, r in returns.iterrows():
        if r.isna().any():
            log_rows.append({"date": day, "pv": pv, "rebalanced": False, "turnover": 0.0, "costs": 0.0})
            continue

        #  apply market move (drift)

        # r = r.fillna(0.0)
        holding = holding.mul(1.0 + r, fill_value=0.0)      #*= (1.0 + r.values)
        pv = float(holding.sum())
        curr_w = holding / pv

        # triggers
        
        periodic_trigger = bool(period_flags.loc[day])
        threshold_trigger = False
        if threshold is not None:

            # Option A — portfolio-level drift (L1 distance across all names):
            # Rebalance if the total absolute weight gap exceeds the threshold.
            # threshold_trigger = (curr_w.sub(target_weights).abs().sum() > total_threshold)

            # Option B — per-name drift:
            # Rebalance if ANY asset’s absolute weight gap exceeds the threshold. 
            threshold_trigger = (curr_w.sub(target_weights).abs() > threshold).any()    

        if not (periodic_trigger or threshold_trigger):
            log_rows.append({
                "date": day, "pv": pv, "rebalanced": False,
                "turnover": 0.0, "costs": 0.0, **{f"w_{c}": curr_w[c] for c in curr_w.index}
            })
            continue

        # compute trades to targets
        target_cash = pv * target_weights
        trades = target_cash - holding

        # turnover & costs
        turnover = float(np.abs(trades).sum() / pv)   # fraction of the portfolio traded
        costs = float(turnover * tc_bps * pv)

        # deduct costs from PV, then rescale targets
        pv -= costs
        target_cash = pv * target_weights

        # execute rebalance (snap to targets)
        holding = target_cash.copy()
        curr_w = holding / pv

        # log
        log_rows.append({
            "date": day, "pv": pv, "rebalanced": True,
            "turnover": turnover, "costs": costs, **{f"w_{c}": curr_w[c] for c in curr_w.index}
        })

    log_df = pd.DataFrame(log_rows).set_index("date")
    return log_df


# %%
def compute_metrics(log_df: pd.DataFrame, rf_annual: float = 0.0, period: int= 0  ):
    """
    Compute quick metrics from the simulation log.
    rf_annual: annual risk-free rate as decimal.
    Returns a dict of metrics.
    """
    pv = log_df["pv"].astype(float)
    # Daily/periodic returns of the equity curve
    r = pv.pct_change().dropna()

    # Time scaling
    periods_per_year = 252  # always daily in setup

    years = (log_df.index[-1] - log_df.index[0]).days / 365.25

    # CAGR
    cagr = np.nan
    if years > 0 and pv.iloc[0] > 0:
        cagr = (pv.iloc[-1] / pv.iloc[0]) ** (1.0 / years) - 1.0

    # Daily stats
    mean_d = r.mean()
    std_d  = r.std(ddof=1)
    rf_d   = (1 + rf_annual) ** (1/periods_per_year) - 1

    sharpe_daily = (mean_d - rf_d) / std_d if std_d > 0 else np.nan

    # Volatility & Sharpe (annualised)
    ann_ret = mean_d * periods_per_year
    ann_vol = std_d * np.sqrt(periods_per_year)
    sharpe  = (ann_ret - rf_annual) / ann_vol

    # Max drawdown
    roll_max = pv.cummax()
    dd = pv / roll_max - 1.0
    max_dd = dd.min()

    # Turnover & costs
    total_costs = float(log_df["costs"].sum())
    n_rebalances = int(log_df["rebalanced"].sum())
    avg_turnover = float(log_df.loc[log_df["rebalanced"], "turnover"].mean())

    return {
        "periods_per_year": periods_per_year,
        "years": years,
        "final_pv": float(pv.iloc[-1]),
        "CAGR": float(cagr),
        "ann_vol": float(ann_vol),
        "expected_return": float(ann_ret),
        "Sharpe": float(sharpe),
        "Sharpe_daily": float(sharpe_daily),
        "max_drawdown": float(max_dd),
        "total_costs": total_costs,
        "n_rebalances": n_rebalances,
        "avg_turnover": avg_turnover,
    }
