import os
import time
import yfinance as yf
import pandas as pd

# Stock symbols categorized
symbols = {
    "Growth": ["PLTR", "SNOW", "SHOP", "COIN", "ROKU", "MRNA"],
    "Commodities & Crypto": ["GC=F", "SI=F", "CL=F", "BZ=F", "BTC-USD"],
    "Dividend": ["KO", "VZ", "PFE", "MMM", "T"],
    "Small-Caps": ["GME", "AMC", "PLUG", "SOFI", "RUN"]
}

# Parameters
start_date = "2015-01-01"  # Start date for data
end_date = "2025-01-01"    # End date (up to current date)
output_folder = "../data/historical_data"  # Output folder
os.makedirs(output_folder, exist_ok=True)

# Function to fetch data and save to CSV
def fetch_and_save(symbol, category):
    try:
        print(f"Fetching data for {symbol} ({category})...")

        # Download data from Yahoo Finance
        data = yf.download(symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True)

        if not data.empty:
            # Add 'Sign' column for the stock's category
            data['Sign'] = category
            # Add 'Symbol' column for the stock symbol
            data['Symbol'] = symbol
            # Add 'Timestamp' column as the index (date)
            data['Timestamp'] = data.index

            # Define the output file path
            file_path = os.path.join(output_folder, f"{symbol}.csv")

            # Save data to CSV
            data.to_csv(file_path)
            print(f"Saved data for {symbol} to {file_path}")
        else:
            print(f"No data found for {symbol}.")
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

# Fetch data for each symbol in all categories
for category, category_symbols in symbols.items():
    for symbol in category_symbols:
        fetch_and_save(symbol, category)
        time.sleep(2)  # Sleep to respect rate limiting

print("✅ All data fetched and saved.")
