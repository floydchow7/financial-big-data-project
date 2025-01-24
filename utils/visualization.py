import pandas as pd

#Function to calcualte the cumulative return of the strategy
def transform_cumulative_returns(summary_df):

    # Extract the first entry of test_cum_return
    first_cum_return = summary_df['test_cum_return'].iloc[0] + 1
    
    # Apply cumprod() to last_time_step_return starting from the second entry
    cumulative_returns = summary_df['last_time_step_return'].iloc[1:].cumprod()
    
    # Combine the first entry of test_cum_return with the cumulative returns
    transformed_returns = pd.concat([
        pd.Series([first_cum_return]),  # First entry
        cumulative_returns * first_cum_return  # Cumulative returns scaled by the first entry
    ], ignore_index=True)
    
    return transformed_returns

