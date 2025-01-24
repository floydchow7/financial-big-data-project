from utils.helpers import *
from utils.pipelines import *
from tqdm import tqdm
from joblib import Parallel, delayed

def optimize_params_func(train_set, 
                        val_set,
                        initial_params,
                        gamma_values,
                        beta_values,
                        delta_values,
                        lambda_values,
                        alpha_values,
                        transformations):
    """
    Example grid search to find best hyperparameters for the strategy,
    picking those that maximize the Sharpe ratio on the validation set.

    Returns:
      best_params (dict)
      best_val_sharpe (float)
    """
    # Define your search space:
    gamma_values = gamma_values
    beta_values = beta_values
    delta_values = delta_values
    lambda_values = lambda_values
    alpha_values = alpha_values
    
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





def optimize_params_func_with_parallelization(train_set, 
                                            val_set,
                                            initial_params,
                                            gamma_values,
                                            beta_values,
                                            delta_values,
                                            lambda_values,
                                            alpha_values,
                                            transformations):
    """
    Example grid search to find best hyperparameters for the strategy,
    picking those that maximize the Sharpe ratio on the validation set.
    Uses Joblib for multithread processing.

    Returns:
      best_params (dict)
      best_val_sharpe (float)
    """
    # Define your search space:
    gamma_values = gamma_values
    beta_values = beta_values
    delta_values = delta_values
    lambda_values = lambda_values
    alpha_values = alpha_values
    
    # Function to evaluate a single combination of parameters
    def evaluate_params(gamma, beta, delta_, lambda_, alpha_):
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

        return {
            'gamma': gamma,
            'beta': beta,
            'delta': delta_,
            'lambda': lambda_,
            'alpha': alpha_,
            'val_sharpe': val_sharpe
        }

    # Use Joblib to parallelize the grid search
    results = Parallel(n_jobs=-1)(  # n_jobs=-1 uses all available cores
        delayed(evaluate_params)(gamma, beta, delta_, lambda_, alpha_)
        for gamma in gamma_values
        for beta in beta_values
        for delta_ in delta_values
        for lambda_ in lambda_values
        for alpha_ in alpha_values
    )

    # Find the best parameters based on the Sharpe ratio
    best_result = max(results, key=lambda x: x['val_sharpe'])
    best_params = {k: best_result[k] for k in ['gamma', 'beta', 'delta', 'lambda', 'alpha']}
    best_val_sharpe = best_result['val_sharpe']

    return best_params, best_val_sharpe






def rolling_calibration_single_bar_summary(
    optimize_params_func,
    step_size,
    train_ratio,
    val_ratio,
    test_ratio,
    data,
    transformations,
    initial_params,
    gamma_values,
    beta_values,
    delta_values,
    lambda_values,
    alpha_values,
    window_size
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
    val_end = int((train_ratio + val_ratio) * window_size)
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
            val_set = window_data.iloc[train_end:val_end]
            test_set = window_data.iloc[val_end:window_size]

            # 2) Grid search on (train, val) to find best_params + best_val_sharpe
            best_params, best_val_sharpe = optimize_params_func(
                train_set,
                val_set,
                initial_params,
                gamma_values,
                beta_values,
                delta_values,
                lambda_values,
                alpha_values,
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

            cumprod_lst = (1 + test_df['Strategy_Return']).cumprod()
            cum_return = cumprod_lst.iloc[-1]-1
            last_day_return = cumprod_lst.iloc[-2] / cumprod_lst.iloc[-1]


            # 5) Create a SINGLE summary row for this window
            summary_row = {
                'start_idx': start_idx,
                'best_params': str(best_params),
                'best_val_sharpe': best_val_sharpe,
                'test_sharpe': test_sharpe,
                'test_mean_return': mean_return,
                'test_cum_return': cum_return,
                'last_time_step_return':last_day_return
            }
            summaries.append(summary_row)

            # Move the rolling window
            start_idx += step_size
            pbar.update(1)  # single bar update

    # Convert summary rows to DataFrame
    summary_df = pd.DataFrame(summaries)
    return summary_df

