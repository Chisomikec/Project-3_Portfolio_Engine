import pandas as pd
import yfinance as yf

def fetch_adjclose(tickers, start, end):
    tickers = [t.upper() for t in tickers]
    df = yf.download(
        tickers,
        start=start, end=end,
        auto_adjust=False, group_by="column",
        threads=False, repair=True, progress=False
    )
    # yfinance shape handling (multiindex vs single)
    if isinstance(df.columns, pd.MultiIndex):
        adj = df["Adj Close"].copy()
    else:
        # single ticker
        col = tickers[0]
        adj = df.rename(columns={"Adj Close": col})[col].to_frame()
    adj = adj.sort_index().ffill()
    # drop any all-NaN columns
    adj = adj.dropna(axis=1, how="all")
    return adj
