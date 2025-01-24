import pandas as pd
import numpy as np
from pyinform.transferentropy import transfer_entropy 

#Helper function to filter the DataFrame to only include rows where 'Article_title' contains the name or ticker
def cleaner_df(df, ticker_lst):
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Extract the original company name and ticker from the list
    org_company_name = ticker_lst[0]
    ticker = ticker_lst[1]
    pattern = rf'\b({org_company_name}|{ticker})\b'
    # Filter the DataFrame to only include rows where 'Article_title' contains the name or ticker
    df_filtered = df[df['Article_title'].str.contains(pattern, case=False, na=False)]
    
    return df_filtered



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


def process_and_merge_data(df, sentiment_data):
    """
    Process sentiment data and merge it with returns data.
    """
    # Create a copy of sentiment_data to avoid SettingWithCopyWarning
    sentiment_data = sentiment_data.copy()

    # 1. Process sentiment data and assign news published after 4 PM to the next trading day
    # Convert 'date' column to datetime with the correct format
    sentiment_data.loc[:, 'date'] = pd.to_datetime(sentiment_data['date'], format='%Y-%m-%d %H:%M:%S')
    
    # Assign news published after 4 PM to the next trading day
    sentiment_data.loc[:, 'Trading_date'] = sentiment_data['date'].apply(
        lambda x: (x + pd.Timedelta(days=1)).date() if x.hour >= 16 else x.date()
    )

    # 2. Process returns data
    # Convert 'date' column to datetime and extract the date part
    df = df.copy()  # Create a copy of df to avoid SettingWithCopyWarning
    df.loc[:, 'date'] = pd.to_datetime(df['date']).dt.date

    # 3. Create daily sentiment averages because the return data is daily
    daily_sentiment = sentiment_data.groupby('Trading_date')['Sentiment'].mean().reset_index()

    # Ensure 'Trading_date' in daily_sentiment is of type datetime64[ns] or date
    daily_sentiment['Trading_date'] = pd.to_datetime(daily_sentiment['Trading_date']).dt.date

    # 4. Merge sentiment with returns data
    # Ensure both 'date' and 'Trading_date' are of the same type (datetime64[ns] or date)
    df['date'] = pd.to_datetime(df['date']).dt.date
    daily_sentiment['Trading_date'] = pd.to_datetime(daily_sentiment['Trading_date']).dt.date

    merged_data = pd.merge(
        daily_sentiment,
        df,
        left_on='Trading_date',
        right_on='date',
        how='left'
    )

    return merged_data




#Rediscretize the sentiment scores to calculate the TE
def discretize_sentiment_column(df, beta):
    # Apply discretization logic
    df['Sentiment_Discretized'] = np.where(
        df['Sentiment'] > beta, 1,  # Positive sentiment
        np.where(df['Sentiment'] < -beta, -1, 0)  # Negative or neutral sentiment
    )
    return df




def calculate_and_add_transfer_entropy(df, source_col, target_col, window_size, delta):
    """
    Calculate rolling Transfer Entropy (TE) between source_col and target_col.
    """
    # Remap states (function technicality regarding pyinform.transfer_entropy)
    state_mapping = {-1: 0, 0: 1, 1: 2}
    df[source_col] = df[source_col].map(state_mapping)
    df[target_col] = df[target_col].map(state_mapping)
    
    te_values = []
    
    # Rolling TE
    for i in range(len(df) - window_size - delta + 1):
        # Source window: sentiment from t to t + window_size
        source_window = df[source_col].iloc[i : i + window_size].values
        # Target window: log returns from t + delta to t + window_size + delta
        target_window = df[target_col].iloc[i + delta : i + window_size + delta].values
        
        # Calculate TE if there are no missing values
        if not pd.isnull(source_window).any() and not pd.isnull(target_window).any():
            te = transfer_entropy(source_window, target_window, k=1)  # Use smaller k because k is the lag defined in the transfer entropy function
            te_values.append(te)
        else:
            te_values.append(None)
    
    # Pad with leading None values to align with the original DataFrame
    needed_leading_nones = window_size + delta - 1
    df['Rolling_TE'] = pd.Series([None] * needed_leading_nones + te_values, index=df.index)
    
    return df




# Define the trading strategy function
def apply_trading_strategy(df, alpha, delta):
    # For the optimization pipeline to work (prevent index breaking)
    df = df.reset_index(drop=True)
    df['Strategy_Return'] = 0.0  # Initialize column for strategy returns as float
    
    for t in range(len(df) - delta):
        sentiment_score = df.loc[t, 'Sentiment']
        te = df.loc[t, 'Rolling_TE']
        
        # Check trading conditions
        if pd.notnull(te) and (sentiment_score > 0) and (te > alpha):
            df.loc[t + delta, 'Strategy_Return'] = df.loc[t + delta, 'Log_Return']
        elif pd.notnull(te) and (sentiment_score < 0) and (te > alpha):
            df.loc[t + delta, 'Strategy_Return'] = -df.loc[t + delta, 'Log_Return']
    
    # Calculate cumulative return
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    return df



#Pipeline generator function
def optimization_pipeline(df, transformations):
    for func, kwargs in transformations:
        df = func(df, **kwargs)
    return df