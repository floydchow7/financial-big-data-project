import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def plot_transformed_cumulative_return(input_df, summary_df, name_df ,benchmark,start_index, transform_function):

    # Apply the transformation
    transformed_returns = transform_function(summary_df)

    # Add the transformed returns to the DataFrame
    summary_df['transformed_cum_return'] = transformed_returns

    start_date = input_df.iloc[start_index]["date"]
    benchmark['Date'] = pd.to_datetime(benchmark['Date'])
    benchmark = benchmark[benchmark["Date"] >= start_date]
    benchmark["Log_return"] = np.log(benchmark['Close']/benchmark['Close'].shift(1)).fillna(0)
    benchmark['benmark_cumulative_return'] = (benchmark['Log_return']+1).cumprod()
    input_df = input_df.merge(benchmark[["Date","benmark_cumulative_return"]], left_on="date", right_on="Date", how="left")
    # Plot the transformed cumulative returns
    plt.plot(
        input_df.iloc[start_index:]["date"], 
        summary_df['transformed_cum_return'], 
        label='Strategy (Tesla_news_on_tesla)'
    )

    plt.plot(
        input_df.iloc[start_index:]["date"], 
        input_df.iloc[start_index:]['benmark_cumulative_return'], 
        label='Strategy (Long Only Benchmark)'
    )



    # Add labels, title, and legend
    plt.xlabel('Date')
    plt.ylabel('Backtest Result')
    plt.title(f'Backtest Result for Strategy_{name_df}')

    # Rotate date labels for better readability
    plt.xticks(rotation=45)
    plt.legend()
    # Show the plot
    plt.show()

def plot_transformed_backtest_results(input_dfs, summary_dfs, name_dfs,benchmark_df, start_index, transform_function,title):
    """
    Plots transformed cumulative returns for multiple strategies on the same plot.

    Parameters:
    - input_dfs: List of input DataFrames containing the 'date' column.
    - summary_dfs: List of summary DataFrames where transformed returns will be added.
    - name_dfs: List of names corresponding to the strategies.
    - start_index: Starting index for slicing the data for each plot.
    - transform_function: Function used to transform cumulative returns.

    Returns:
    - None. Displays a single plot with multiple lines.
    """
    if not (len(input_dfs) == len(summary_dfs) == len(name_dfs)):
        raise ValueError("The input_dfs, summary_dfs, and name_dfs lists must have the same length.")

    plt.figure(figsize=(12, 6))  # Create a single figure for the combined plot

    for i in range(len(input_dfs)):
        # Extract current DataFrame and strategy name
        input_df = input_dfs[i]
        summary_df = summary_dfs[i]
        name_df = name_dfs[i]

        # Apply the transformation
        transformed_returns = transform_function(summary_df)

        # Add the transformed returns to the DataFrame
        summary_df['transformed_cum_return'] = transformed_returns

        # Plot the transformed cumulative returns
        plt.plot(
            input_df.iloc[start_index:]["date"], 
            summary_df['transformed_cum_return'], 
            label=f'Strategy {name_df}'
        )
    start_date = input_dfs[0].iloc[start_index]["date"]
    end_date = input_dfs[0].iloc[-1]["date"]

    benchmark_df = benchmark_df[(benchmark_df["Date"] >= start_date) & (benchmark_df["Date"] <= end_date)]
    #print(start_date)
    #print(input_dfs[0].iloc[-1]["date"])
    benchmark_df["Log_return"] = np.log(benchmark_df['Open']/benchmark_df['Open'].shift(1)).fillna(0)
    benchmark_df['benmark_cumulative_return'] = (benchmark_df['Log_return']+1).cumprod()
    input_dfs[0] = input_dfs[0].merge(benchmark_df[["Date","benmark_cumulative_return"]], left_on="date",right_on ='Date', how="left")
    plt.plot(
        input_dfs[0].iloc[start_index:]["date"], 
        input_dfs[0].iloc[start_index:]['benmark_cumulative_return'], 
        label='Strategy (Long Only Benchmark)'
    )

    # Add labels, title, and legend
    plt.xlabel('Date')
    plt.ylabel('Backtest Result')
    plt.title(title)
    plt.legend()

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    # Show the combined plot
    plt.show()

