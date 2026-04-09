import yfinance as yf
import numpy as np 
import pandas as pd
import time
import os
import ta

def fetch_data(tickers, start_date, end_date, delay=5):
    """Fetch stock data from Yahoo Finance"""
    ticker_data = {}
    
    for t in tickers:
        print(f"Fetching {t}...")
        ticker_data[t] = yf.Ticker(t).history(start=start_date, end=end_date)
        time.sleep(delay)
    
    return ticker_data

def add_technical_indicators(ticker_data):
    """Add technical indicators to stock data"""
    for t, df in ticker_data.items():
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], 50).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], 200).ema_indicator()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close'], window_fast=12, window_sign=9, window_slow=26).macd()
        df['BB_High'] = ta.volatility.BollingerBands(df['Close'], window=15, window_dev=2).bollinger_hband()
        df['BB_Low'] = ta.volatility.BollingerBands(df['Close'], window=15, window_dev=2).bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        ticker_data[t] = df
    
    return ticker_data

def clean_and_save_data(ticker_data, data_dir="data"):
    """Clean data and save to CSV"""
    os.makedirs(data_dir, exist_ok=True)
    
    for ticker in ticker_data:
        print(f"{ticker}: {ticker_data[ticker].shape}")
        ticker_data[ticker] = ticker_data[ticker].drop(['Dividends', 'Stock Splits'], axis=1)
        ticker_data[ticker].to_csv(f"{data_dir}/{ticker}.csv")
    
    return ticker_data

def align_and_combine_data(ticker_data, save_path=None):
    """Align data across all tickers and combine into a single array"""
    all_indices = set().union(*[ticker_data[d].index for d in ticker_data])
    
    aligned_data = []
    for ticker in ticker_data:
        aligned_data.append(ticker_data[ticker].reindex(index=all_indices))
    
    combined_data = np.stack(aligned_data, axis=1)
    filled_data = np.nan_to_num(combined_data, nan=0)
    
    if save_path:
        np.save(save_path, filled_data)
    
    return filled_data

def split_time_series_data(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split time series data maintaining temporal order
    """
    total_len = len(data)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

def main():
    """
    CORRECTED: Fetch ALL data at once, then split properly
    """
    tickers = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Fetch ALL data from 2018 to 2025 at once
    print("Fetching complete dataset (2018-2025)...")
    ticker_data = fetch_data(tickers, "2018-01-01", "2025-01-01")
    ticker_data = add_technical_indicators(ticker_data)
    ticker_data = clean_and_save_data(ticker_data, data_dir)
    
    # Combine into single array
    complete_data = align_and_combine_data(ticker_data, f"{data_dir}/complete_processed_data.npy")
    print(f"Complete data shape: {complete_data.shape}")
    
    # PROPER TEMPORAL SPLIT
    train_data, val_data, test_data = split_time_series_data(
        complete_data, 
        train_ratio=0.6,  # 60% for training (2018-2022)
        val_ratio=0.2,    # 20% for validation (2022-2024)
        test_ratio=0.2    # 20% for testing (2024-2025)
    )
    
    # Save splits
    np.save(f"{data_dir}/train_data.npy", train_data)
    np.save(f"{data_dir}/val_data.npy", val_data)
    np.save(f"{data_dir}/test_data.npy", test_data)
    
    print("DATA SPLIT SUMMARY:")
    print("=" * 50)
    print(f"Complete dataset: {complete_data.shape}")
    print(f"Training data: {train_data.shape} (60%)")
    print(f"Validation data: {val_data.shape} (20%)")
    print(f"Test data: {test_data.shape} (20%)")
    print("=" * 50)
    print("CRITICAL: Never use validation or test data during hyperparameter tuning!")

if __name__ == "__main__":
    main()