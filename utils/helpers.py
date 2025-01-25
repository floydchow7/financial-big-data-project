import numpy as np
import pandas as pd

#Helper function to read the desired file
def read_file(file_path):

    file_extension = file_path.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            # Read CSV file
            df = pd.read_csv(file_path)
            if "date" in df.columns:
                df["date"] = df["date"].apply(lambda val: pd.to_datetime(val))
            elif "Date" in df.columns:
                df["Date"] = df["Date"].apply(lambda val: pd.to_datetime(val))
        elif file_extension == 'parquet':
            # Read Parquet file
            df = pd.read_parquet(file_path)
            if "date" in df.columns:
                df["date"] = df["date"].apply(lambda val: pd.to_datetime(val))
            elif "Date" in df.columns:
                df["Date"] = df["Date"].apply(lambda val: pd.to_datetime(val))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    return df

#Helper function to write the desired file
def write_file(df, file_path):

    file_extension = file_path.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            # Write to CSV
            df.to_csv(file_path, index=False)
        elif file_extension == 'parquet':
            # Write to Parquet
            df.to_parquet(file_path, engine="pyarrow", index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        raise ValueError(f"Error writing file: {e}")



# Calculate log returns
def calculate_log_returns(data, column_name='Close'):

    data['Log_Return'] = np.log(data[column_name] / data[column_name].shift(1))
    data = data.dropna() 
    data = data.rename({"Date":"date"}, axis = 1)
    
    return data

#For Doge Coin merge with sentiment data
def process_and_merge_data_continous_price(sentiment_data,doge_price):

    # Round up 'date' to the nearest minute
    sentiment_data['date'] = sentiment_data['date'].dt.ceil('T')  # 'T' stands for minute

    sentiment_data = sentiment_data.groupby('date',as_index=False).agg({'Sentiment':'mean'})
    # Make timezones consistent
    sentiment_data['date'] = sentiment_data['date'].dt.tz_localize('UTC')

    # Merge the DataFrames on the 'date' key
    merged_data = pd.merge(
        sentiment_data,
        doge_price,
        on='date',
        how='left'  # Use 'inner' to keep only matching rows
    )

    return merged_data



def calculate_sharpe_ratio(df, return_column='Strategy_Return'):
    """
    Calculates the Sharpe ratio for the strategy returns in `return_column`.
    """
    returns = df[return_column]
    mean_return = returns.mean()
    std_dev = returns.std()
    if std_dev == 0:
        return 0
    return mean_return / std_dev

