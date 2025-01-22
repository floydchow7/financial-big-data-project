import numpy as np
import pandas as pd

#Helper function to read the desired file
def read_file(file_path):

    file_extension = file_path.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            # Read CSV file
            df = pd.read_csv(file_path)
        elif file_extension == 'parquet':
            # Read Parquet file
            df = pd.read_parquet(file_path)
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
def calculate_log_returns(data, delta = 1, column_name='Close'):
    data = data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])
    data['Log_Return'] = np.log(data[column_name] / data[column_name].shift(delta))
    data = data.dropna() 
    return data

def classify_returns(df, column_name, gamma):
    
    def classify_return(x):
        if x > gamma:
            return 1  # Positive return
        elif x < -gamma:
            return -1  # Negative return
        else:
            return 0  # Stable return
    
    df['Return_Label'] = df[column_name].apply(classify_return)
    return df

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

