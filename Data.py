import yfinance as yf

data = yf.download(
    "RELIANCE.NS",
    start="2015-01-01",
    end="2025-12-31",
    interval="1d"
)

print(data.head())

data.to_csv("reliance_stock_data.csv")