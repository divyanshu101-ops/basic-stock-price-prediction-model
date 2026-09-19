import yfinance as yf

# end date exclusive hoti hai, isliye 19 Sep dene se 18 Sep tak ka data aayega
df = yf.download("GOOG", start="2019-01-01", end="2026-09-19", auto_adjust=False)

# newer yfinance MultiIndex columns deta hai, usse simple columns me badlo
df.columns = df.columns.get_level_values(0)

# tumhare purane CSV wala column order
df = df[["Open", "High", "Close", "Low", "Adj Close", "Volume"]]

df.to_csv("data/raw_data/google_stock_price_raw.csv")

print("Shape:", df.shape)
print(df.head())
print(df.tail())