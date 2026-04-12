import yfinance as yf
import pandas as pd
import os

NIFTY_30 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "ITC.NS", "HINDUNILVR.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS",
    "NESTLEIND.NS", "WIPRO.NS", "POWERGRID.NS", "NTPC.NS", "TECHM.NS",
    "HCLTECH.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS",
    "JSWSTEEL.NS", "COALINDIA.NS", "BAJAJFINSV.NS"
]

START_DATE = "2016-01-01"
END_DATE = "2026-01-31"


def fetch_stock_data(ticker, save_path="data"):
    print(f"Downloading {ticker}...")

    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE)

        if df.empty:
            print(f"❌ Skipped {ticker}")
            return

        df.reset_index(inplace=True)

        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{ticker}.csv")

        df.to_csv(file_path, index=False)

        print(f"✅ Saved: {ticker}")

    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")


def fetch_all_stocks():
    for ticker in NIFTY_30:
        fetch_stock_data(ticker)