from helpers import *
from pipelines import *
from tqdm import tqdm

def optimize_params_func(train_set, val_set, initial_params, transformations):
    """
    Example grid search to find best hyperparameters for the strategy,
    picking those that maximize the Sharpe ratio on the validation set.

    Returns:
      best_params (dict)
      best_val_sharpe (float)
    """
    # Define your search space:
    gamma_values = [0.005, 0.01]
    beta_values = [0.3, 0.5]
    delta_values = [1, 2]
    lambda_values = [5, 10]
    alpha_values = [0.1, 0.2]
    
    best_val_sharpe = -np.inf
    best_params = dict(initial_params)
    
    for gamma in gamma_values:
        for beta in beta_values:
            for delta_ in delta_values:
                for lambda_ in lambda_values:
                    for alpha_ in alpha_values:
                        # Build transformations for train/val
                        current_transformations = [
                            (classify_returns, {
                                'column_name': 'Log_Return',
                                'gamma': gamma
                            }),
                            (discretize_sentiment_column, {
                                'beta': beta
                            }),
                            (calculate_and_add_transfer_entropy, {
                                'source_col': 'Sentiment_Discretized',
                                'target_col': 'Return_Label',
                                'window_size': lambda_,
                                'delta': delta_
                            }),
                            (apply_trading_strategy, {
                                'alpha': alpha_,
                                'delta': delta_
                            })
                        ]

                        # (Optional) "Train" on train_set if there's a model to fit;
                        # if purely rule-based, we just skip or run the pipeline anyway:
                        _ = optimization_pipeline(train_set.copy(), current_transformations)

                        # Evaluate on val_set
                        val_df = optimization_pipeline(val_set.copy(), current_transformations)
                        val_sharpe = calculate_sharpe_ratio(val_df)

                        if val_sharpe > best_val_sharpe:
                            best_val_sharpe = val_sharpe
                            best_params = {
                                'gamma': gamma,
                                'beta': beta,
                                'delta': delta_,
                                'lambda': lambda_,
                                'alpha': alpha_
                            }
    
    return best_params, best_val_sharpe



def rolling_calibration_single_bar_summary(
    data,
    transformations,
    initial_params,
    window_size,
    step_size=1,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    optimize_params_func=None
):
    """
    Rolling calibration that:
      - Shows ONE progress bar for the rolling windows
      - Runs grid search inside `optimize_params_func` (no bar)
      - Stores ONE summary row per rolling window
    """

    num_points = len(data)
    
    # Indices for splitting within each window
    train_end = int(train_ratio * window_size)
    val_end   = int((train_ratio + val_ratio) * window_size)
    # Test slice is from val_end to window_size

    # 1) Determine how many rolling windows
    num_windows = 0
    start_idx = 0
    while start_idx + window_size <= num_points:
        num_windows += 1
        start_idx += step_size

    # We'll store the summary rows in a list
    summaries = []

    # Reset start_idx for the actual loop
    start_idx = 0

    with tqdm(total=num_windows, desc="Rolling Calibration (One Bar)") as pbar:
        while start_idx + window_size <= num_points:
            # Slice the rolling window
            window_data = data.iloc[start_idx : start_idx + window_size].copy()

            # Split into train/val/test
            train_set = window_data.iloc[:train_end]
            val_set   = window_data.iloc[train_end:val_end]
            test_set  = window_data.iloc[val_end:window_size]

            # 2) Grid search on (train, val) to find best_params + best_val_sharpe
            best_params, best_val_sharpe = optimize_params_func(
                train_set,
                val_set,
                initial_params,
                transformations
            )

            # 3) Apply best_params to the test set
            final_transformations = [
                (classify_returns, {
                    'column_name': 'Log_Return',
                    'gamma': best_params['gamma']
                }),
                (discretize_sentiment_column, {
                    'beta': best_params['beta']
                }),
                (calculate_and_add_transfer_entropy, {
                    'source_col': 'Sentiment_Discretized',
                    'target_col': 'Return_Label',
                    'window_size': best_params['lambda'],
                    'delta': best_params['delta']
                }),
                (apply_trading_strategy, {
                    'alpha': best_params['alpha'],
                    'delta': best_params['delta']
                })
            ]
            test_df = optimization_pipeline(test_set.copy(), final_transformations)

            # 4) Evaluate test performance
            test_sharpe = calculate_sharpe_ratio(test_df)

            # Optionally compute mean or cumulative returns
            mean_return = test_df['Strategy_Return'].mean()
            cum_return = (1 + test_df['Strategy_Return']).prod() - 1

            # 5) Create a SINGLE summary row for this window
            summary_row = {
                'start_idx': start_idx,
                'best_params': str(best_params),
                'best_val_sharpe': best_val_sharpe,
                'test_sharpe': test_sharpe,
                'test_mean_return': mean_return,
                'test_cum_return': cum_return
            }
            summaries.append(summary_row)

            # Move the rolling window
            start_idx += step_size
            pbar.update(1)  # single bar update

    # Convert summary rows to DataFrame
    summary_df = pd.DataFrame(summaries)
    return summary_df
