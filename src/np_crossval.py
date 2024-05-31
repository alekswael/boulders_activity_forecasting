# Import dependencies
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet, set_random_seed
import matplotlib.pyplot as plt

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

    return df, regressorsList

def run_cross_validation(df, params, regressorsList, weather_lag):
    folds = NeuralProphet(**params).crossvalidation_split_df(df, freq="15Min", k=10, fold_pct=0.10, fold_overlap_pct=0.25)
    metrics_train_all = pd.DataFrame()
    counter = 0

    for df_train, df_test in folds:
        counter += 1
        m = NeuralProphet(**params)
        m.add_lagged_regressor(names=regressorsList, n_lags=weather_lag, normalize='auto')
        m = m.add_country_holidays("DK")

        train = m.fit(df=df_train, validation_df=df_test, epochs=30, freq="15Min", early_stopping=True, progress="plot")
        metrics_train = train.iloc[-1:]

        # Add fold number
        metrics_train['fold'] = counter

        # Concat metrics
        metrics_train_all = pd.concat([metrics_train_all, metrics_train])

        # PLOT
        # Combine plots on a 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
        axs[0, 0].plot(train["epoch"], train['Loss'], label='Training Loss')
        axs[0, 0].plot(train["epoch"], train['Loss_val'], label='Validation Loss')
        axs[0, 0].set_title(f'Loss over epochs for fold {counter}')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        axs[0, 1].plot(train["epoch"], train['RMSE'], label='Training RMSE')
        axs[0, 1].plot(train["epoch"], train['RMSE_val'], label='Validation RMSE')
        axs[0, 1].set_title(f'RMSE over epochs for fold {counter}')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('RMSE')
        axs[0, 1].legend()

        axs[1, 0].plot(train["epoch"], train['MAE'], label='Training MAE')
        axs[1, 0].plot(train["epoch"], train['MAE_val'], label='Validation MAE')
        axs[1, 0].set_title(f'MAE over epochs for fold {counter}')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('MAE')
        axs[1, 0].legend()

        # Leave one plot empty
        axs[1, 1].axis('off')

        plt.savefig(f"output/combined_plots_fold_{counter}_fc1.png")

        print(f"INFO: Fold {counter} completed!")

    return metrics_train_all

def calculate_mean_metrics(metrics_train_all):
    metrics_train_all['RMSE_train_mean'] = metrics_train_all['RMSE'].mean()
    metrics_train_all['MAE_train_mean'] = metrics_train_all['MAE'].mean()
    metrics_train_all['RMSE_val_mean'] = metrics_train_all['RMSE_val'].mean()
    metrics_train_all['MAE_val_mean'] = metrics_train_all['MAE_val'].mean()

    # Print the mean RMSE and MAE for the train and test set
    print(f"Mean RMSE for 10-fold train: {metrics_train_all['RMSE'].mean()}")
    print(f"Mean MAE for 10-fold train: {metrics_train_all['MAE'].mean()}")
    print(f"Mean RMSE for 10-fold test: {metrics_train_all['RMSE_val'].mean()}")
    print(f"Mean MAE for 10-fold test: {metrics_train_all['MAE_val'].mean()}")

    # save the metrics
    metrics_train_all.to_csv(f'output/CV_metrics_all.csv', index=False)

def main():
    # Set random seed
    set_random_seed(0)

    # Data preprocessing
    df, regressorsList = preprocess_data()

    # AR-net lags
    forecast_range = 4*24*7 # 1 week forecasted
    ar_net_lags = 4*24*7*2 # "In practice, it is hard to determine accurately and is commonly set to twice the innermost periodicity or twice the forecast horizon" (Triebe at al., 2021)
    weather_lag = 4*24 # 1 day

    # Params
    params = {
        "n_changepoints": 10,
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": 12,
        "seasonality_reg": 0.1,
        "n_lags": ar_net_lags,
        "ar_reg": 0.1,
        "n_forecasts": forecast_range
    }

    # Cross validation
    metrics_train_all = run_cross_validation(df, params, regressorsList, weather_lag)

    # Calculate mean metrics
    calculate_mean_metrics(metrics_train_all)

if __name__ == "__main__":
    main()
