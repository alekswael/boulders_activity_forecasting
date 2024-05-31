# Import dependencies
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet, set_random_seed
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

def extract_yhat(forecasts, size):
    columns = forecasts.columns[3:]
    newframe = forecasts[['ds', 'yhat1']].iloc[-size:].copy()
    for col in columns:
        if 'yhat' in col:
            newframe['yhat1'] = newframe['yhat1'].fillna(forecasts[col])
    return newframe

def preprocess_data():
    # Read data
    df = pd.read_csv('data/full_dataset_15Min.csv', sep=';', index_col=0)

    # Retrieve index data and call "ds"
    df['ds'] = df.index

    # Remove index
    df.reset_index(drop=True, inplace=True)

    # Include aarhus data and ds columns
    df = df[['ds', 'Aarhus City Activity', "Middeltemperatur", "Aarhus City medlemmer", "Aarhus TOTAL medlemmer"]]

    # rename columns
    df = df.rename(columns={"Aarhus City Activity": "y"})

    # For Aarhus City, values before 22th of October are zero, so we remove them from the dataset
    # Loop through the rows, and return the index of the first row that contains a non zero value
    for i in range(len(df)):
        if df['y'].iloc[i] != 0:
            index = i
            break

    # Remove all rows before the first non zero value
    df = df.iloc[index:]

    # There is some corrupted data in the end, all values are zero. We find the spot where there are at least 300 zeros in a row, and remove all data after that.
    for i in range(len(df)):
        if sum(df['y'].iloc[i:i+300]) == 0:
            index = i
            break

    df = df.iloc[:index]

    # For some reason Neural Prophet does not like columns which are not either, 'y','ds' or a named regressor. Therefore, this fix is needed
    columnsToKeep = ['ds','y']
    regressorsList = ['Middeltemperatur'] # This is the list of regressors we want to keep.
    #regressorsList = ['Middeltemperatur', 'Aarhus City medlemmer', 'Aarhus TOTAL medlemmer'] # FULL LIST OF REGRESSORS

    # this removes all columns not 'ds','y' or any regressors
    df = df[df.columns.intersection(np.concatenate((columnsToKeep, regressorsList)))]

    return df

def historical_data_prediction(df_train, df_test):
    # Copy test df
    df_pred = df_test.copy()

    # Drop all columns except 'ds'
    df_pred = df_pred['ds']

    # Make sure 'ds' is a dataframe
    df_pred = df_pred.to_frame()

    # Make sure 'ds' is a datetime object
    df_pred['ds'] = pd.to_datetime(df_pred['ds'])

    # Offset the 'ds' column by 3 months
    df_pred['ds'] = df_pred['ds'] - pd.DateOffset(months=4)

    # Merge df_pred and df_train on 'ds', only keep the rows that are in both dataframes
    df_pred = df_pred.merge(df_train, on='ds', how='inner')

    # Rename the 'y' column to 'y_last'
    df_pred = df_pred.rename(columns={"y": "y_last"})

    # Remove the offset from 'ds'
    df_pred['ds'] = df_pred['ds'] + pd.DateOffset(months=4)

    # Now merge with df_test on 'ds', keep rows that are in test set
    df_test = df_test.merge(df_pred, on='ds', how='left')

    # Replace all Nan values in 'y_last' with 0
    df_test['y_last'] = df_test['y_last'].fillna(0)

    return df_test

def mean_of_historical_data(df_train, df_test):
    # Ensure 'date' is a datetime index
    df_train.set_index('ds', inplace=True)
    df_test.set_index('ds', inplace=True)

    # Extract the time part from the DateTimeIndex
    df_train['time_of_day'] = df_train.index.time
    df_test['time_of_day'] = df_test.index.time

    # Reset the index to make "ds" a column again
    df_train.reset_index(inplace=True)
    df_test.reset_index(inplace=True)

    # Group by the time, then calculate the mean
    average_per_15min_per_dow = df_train.groupby('time_of_day')["y"].mean()

    # Reset index for easier viewing
    average_per_15min_per_dow = average_per_15min_per_dow.reset_index(inplace=False)

    # Rename the y column to y_mean_timestep
    average_per_15min_per_dow = average_per_15min_per_dow.rename(columns={"y": "y_mean_timestep"})

    # Join with df_test on 'time_of_day', keep all rows from df_test
    df_test = df_test.merge(average_per_15min_per_dow, on='time_of_day', how='left')

    return df_test

def calculate_metrics(df_test):
    # Calculate RMSE and MAE for both the historical data and the mean of historical data
    rmse_last = root_mean_squared_error(df_test['y'], df_test['y_last'])
    mae_last = mean_absolute_error(df_test['y'], df_test['y_last'])

    rmse_mean = root_mean_squared_error(df_test['y'], df_test['y_mean_timestep'])
    mae_mean = mean_absolute_error(df_test['y'], df_test['y_mean_timestep'])

    # Create a dataframe with the metrics
    metrics_fold = pd.DataFrame([["Historical data value", rmse_last, mae_last], ["Mean timestep value", rmse_mean, mae_mean]], columns=["Model", "RMSE", "MAE"])

    return metrics_fold

def cross_validation(df, params):
    folds = NeuralProphet(**params).crossvalidation_split_df(df, freq="15Min", k=10, fold_pct=0.10, fold_overlap_pct=0.25)

    metrics_all = pd.DataFrame()
    counter = 0

    for df_train, df_test in folds:
        counter += 1

        print(len(df_train), len(df_test))

        # Make sure everything is in datetime format
        df_train['ds'] = pd.to_datetime(df_train['ds'])
        df_test['ds'] = pd.to_datetime(df_test['ds'])

        df_test = historical_data_prediction(df_train, df_test)
        df_test = mean_of_historical_data(df_train, df_test)

        metrics_fold = calculate_metrics(df_test)
        metrics_fold['fold'] = counter

        metrics_all = pd.concat([metrics_all, metrics_fold])

        print(f"INFO: Fold {counter} completed!")

    mean_rmse_mean_timestep = metrics_all[metrics_all['Model'] == "Mean timestep value"]["RMSE"].mean()
    mean_mae_mean_timestep = metrics_all[metrics_all['Model'] == "Mean timestep value"]["MAE"].mean()
    mean_rmse_historical_data = metrics_all[metrics_all['Model'] == "Historical data value"]["RMSE"].mean()
    mean_mae_historical_data = metrics_all[metrics_all['Model'] == "Historical data value"]["MAE"].mean()

    mean_metrics_mean_timestep = pd.DataFrame([["MTV_mean", mean_rmse_mean_timestep, mean_mae_mean_timestep]], columns=["Model", "RMSE", "MAE"])
    mean_metrics_historical_data = pd.DataFrame([["HDV_mean", mean_rmse_historical_data, mean_mae_historical_data]], columns=["Model", "RMSE", "MAE"])

    metrics_all = pd.concat([metrics_all, mean_metrics_mean_timestep])
    metrics_all = pd.concat([metrics_all, mean_metrics_historical_data])

    return metrics_all

def main():
    # Set random seed
    set_random_seed(0)

    # Params
    params = {
        "n_changepoints": 10,
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": 12,
        "seasonality_reg": 0.1,
        "n_lags": 4*24*7*2,
        "ar_reg": 0.1,
        "n_forecasts": 4*24*7
    }

    df = preprocess_data()
    metrics_all = cross_validation(df, params)
    metrics_all.to_csv('output/CV_baseline_metrics.csv', index=False)

if __name__ == "__main__":
    main()
