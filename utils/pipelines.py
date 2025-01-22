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

def process_and_merge_data(df,sentiment_data):

    # 1. Process sentiment data and assign news published after 4 PM to the next trading day
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'], dayfirst=True)
    sentiment_data['Trading_Date'] = sentiment_data['Date'].apply(
        lambda x: (x + pd.Timedelta(days=1)).date() if x.hour >= 16 else x.date()
    )

    # 2. Process returns data
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.date

    # 3. Create daily sentiment averages because the return data is daily
    daily_sentiment = sentiment_data.groupby('Trading_Date')['Sentiment'].mean().reset_index()

    # 4. Merge sentiment with returns data
    merged_data = pd.merge(
        df,
        daily_sentiment,
        left_on='Date',
        right_on='Trading_Date',
        how='inner'
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
    
    # Remap states
    state_mapping = {-1: 0, 0: 1, 1: 2}
    df[source_col] = df[source_col].map(state_mapping)
    df[target_col] = df[target_col].map(state_mapping)
    
    te_values = []
    # Rolling TE
    for i in range(len(df) - window_size + 1):
        source_window = df[source_col].iloc[i : i + window_size].values
        target_window = df[target_col].iloc[i : i + window_size].values
        
        if not pd.isnull(source_window).any() and not pd.isnull(target_window).any():
            te = transfer_entropy(source_window, target_window, k=delta)
            te_values.append(te)
        else:
            te_values.append(None)
    
    # The number of actual TE values is (len(df) - window_size + 1).
    # We need that many leading None so that total length == len(df)
    needed_leading_nones = len(df) - len(te_values)
    if needed_leading_nones < 0:
        # Safety check, means we collected more TE values than rows
        needed_leading_nones = 0
    
    df['Rolling_TE'] = pd.Series([None]*needed_leading_nones + te_values, index=df.index)
    
    return df

# Define the trading strategy function
def apply_trading_strategy(df, alpha,delta):
    # For the optimization pipeline to work (prevent from index breaking)
    df = df.reset_index(drop=True)
    df['Strategy_Return'] = 0  # Initialize column for strategy returns
    
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