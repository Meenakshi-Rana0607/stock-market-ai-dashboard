import yfinance as yf

df = yf.download("RELIANCE.NS", start="2020-01-01", end="2025-01-01")
df.to_csv("data.csv")

print("✅ Data downloaded successfully!")