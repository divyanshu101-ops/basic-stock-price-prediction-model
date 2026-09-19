import yfinance as yf

df = yf.download("GOOG", start="2019-01-01", end="2026-09-19", auto_adjust=False)

df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Close", "Low", "Adj Close", "Volume"]]

df.to_csv("data/raw_data/google_stock_price_raw.csv")

print("Shape:", df.shape)
print(df.head())
print(df.tail())