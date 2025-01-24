import pandas as pd
import polars as pl
import yfinance as yf
import os
import shutil
import kagglehub
from binance.client import Client
from datetime import datetime, timedelta

def filter_and_save_parquet(input_path: str, ticker_list:list, output_folder:str):

    # Process each ticker separately to reduce memory usage
    for ticker in ticker_list:
        # Read and filter data in streaming mode
        lazy_df = pl.scan_csv(input_path)
        filtered_df = (
            lazy_df
            .select(["Date", "Stock_symbol", "Article_title"])  # Select specific columns
            .filter(pl.col("Article_title").is_not_null())  # Filter out null values
            .filter(pl.col("Stock_symbol") == ticker)  # Filter rows matching the ticker
            .with_columns(
                pl.col("Date").str.to_datetime("%Y-%m-%d %H:%M:%S UTC")  # Convert Date column to datetime
            )
        )

        # Collect filtered data and write directly to a Parquet file
        output_path = f"{output_folder}/{ticker}_news.parquet"
        filtered_df.sink_parquet(output_path)  # Stream writing without converting to Pandas
        print(f"{ticker} is saved to: {output_path}")


def get_stock_price_data(ticker_symbol:str,start_date: str, end_date: str = None) -> pd.DataFrame:

    # Define the ticker symbol
    ticker_symbol = ticker_symbol

    # Fetch data using yfinance
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Ensure the Date column is part of the DataFrame
    stock_data.reset_index(inplace=True)

    return stock_data


def get_musk_tweets_data(destination_folder: str):
    path = kagglehub.dataset_download("gpreda/elon-musk-tweets")
    print(f"Downloaded dataset to: {path}")
    # Define the source file path
    source_folder = path
    csv_file_name = "elon_musk_tweets.csv"  # Replace with the actual CSV file name if different
    source_file_path = os.path.join(source_folder, csv_file_name)

    # Define the destination folder
    destination_folder = destination_folder
    os.makedirs(destination_folder, exist_ok=True)  # Create the destination folder if it doesn't exist

    # Define the destination file path
    destination_file_path = os.path.join(destination_folder, csv_file_name)

    # Move the file
    shutil.move(source_file_path, destination_file_path)

    print(f"File moved to: {destination_file_path}")


def get_crypto_news_data(destination_folder: str):
    path = kagglehub.dataset_download("oliviervha/crypto-news")
    print(f"Downloaded dataset to: {path}")
    # Define the source file path
    source_folder = path
    csv_file_name = "cryptonews.csv"  # Replace with the actual CSV file name if different
    source_file_path = os.path.join(source_folder, csv_file_name)

    # Define the destination folder
    destination_folder = destination_folder
    os.makedirs(destination_folder, exist_ok=True)  # Create the destination folder if it doesn't exist

    # Define the destination file path
    destination_file_path = os.path.join(destination_folder, "crypto_news.csv")

    # Move the file
    shutil.move(source_file_path, destination_file_path)

    print(f"File moved to: {destination_file_path}")


def fetch_coin_data_from_binance(
    api_key: str = None,
    api_secret: str = None,
    start_date: str = "1 Jan, 2023",
    end_date: str = None,
    symbol: str = "DOGEUSDT",
    interval: str = "1m"
) -> pd.DataFrame:
    """
    Fetch minute-level historical OHLCV data for DOGE/USDT from Binance.
    Supports large date ranges by chunking requests.
    """
    # Create Binance client
    client = Client(api_key, api_secret)
    
    # Symbol and interval
    symbol = symbol
    interval = interval  # Use the string literal for 1-minute interval
    
    # Convert start and end dates to datetime
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) if end_date else datetime.utcnow()
    
    # Initialize an empty DataFrame to store results
    all_data = []
    
    # Fetch data in chunks (max 1,000 candles per request)
    while start_datetime < end_datetime:
        # Define the end of the current chunk
        chunk_end = start_datetime + timedelta(minutes=1000)
        if chunk_end > end_datetime:
            chunk_end = end_datetime
        
        # Fetch data for the current chunk
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_datetime.strftime("%d %b, %Y %H:%M:%S"),
            end_str=chunk_end.strftime("%d %b, %Y %H:%M:%S")
        )
        
        # Append to the all_data list
        all_data.extend(klines)
        
        # Move the start_datetime forward
        start_datetime = chunk_end
    
    # Convert the collected data to a DataFrame
    df = pd.DataFrame(all_data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume", 
        "Close Time", "Ignore1", "Ignore2", "Ignore3", "Ignore4", "Ignore5"
    ])
    
    # Keep only the important columns
    df = df[["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time"]]
    
    # Convert time columns from milliseconds to datetime
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms', utc=True)
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit='ms', utc=True)
    
    # Convert numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rename for clarity
    df.rename(columns={
        "Open Time": "time_open",
        "Close Time": "time_close"
    }, inplace=True)
    
    return df