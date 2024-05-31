# Import dependencies
import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet, save, set_random_seed
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
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

def train_model(df, params, regressorsList, weather_lag):
    m = NeuralProphet(**params)

    m.add_lagged_regressor(names=regressorsList,
                           n_lags=weather_lag,
                           normalize='auto')

    m = m.add_country_holidays("DK")

    m.set_plotting_backend("plotly-static")

    train = m.fit(df=df,
                  freq="15min",
                  progress="plot")

    return m, train

def save_model(model, path):
    save(model, path)

def plot_training(train):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    axs[0, 0].plot(train["epoch"], train['Loss'], label='Training Loss')
    axs[0, 0].set_title(f'Loss over epochs')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(train["epoch"], train['RMSE'], label='Training RMSE')
    axs[0, 1].set_title(f'RMSE over epochs')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].legend()

    axs[1, 0].plot(train["epoch"], train['MAE'], label='Training MAE')
    axs[1, 0].set_title(f'MAE over epochs')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('MAE')
    axs[1, 0].legend()

    # Leave one plot empty
    axs[1, 1].axis('off')

    plt.savefig(f"output/combined_plots_full_model.png")

def evaluate_forecast(df_final_test, df_short, df_mean_calc, df_new):
    ################ NEURAL PROPHET ################
    # Order rows by ds column to make sure the time series is in order
    df_final_test = df_final_test.sort_values(by='ds')
    df_new = df_new.sort_values(by='ds')

    # Make sure they're datetime
    df_final_test['ds'] = pd.to_datetime(df_final_test['ds'])
    df_new['ds'] = pd.to_datetime(df_new['ds'])

    y_true = df_final_test["y"]
    y_pred = df_new["yhat1"]

    rmse_np = root_mean_squared_error(y_true, y_pred)
    mae_np = mean_absolute_error(y_true, y_pred)

    print(f"RMSE NeuralProphet: {rmse_np}")
    print(f"MAE NeuralProphet: {mae_np}")

    ################ LAST WEEKS VALUE ################
    # Calculate the RMSE and MAE for the last week
    rmse_last_week = root_mean_squared_error(df_short['y'], df_short['y_last_week'])
    mae_last_week = mean_absolute_error(df_short['y'], df_short['y_last_week'])

    print("RMSE last week: ", rmse_last_week)
    print("MAE last week: ", mae_last_week)

    ################# MEAN VALUE MODEL #################
    df_mean_calc = df_short.copy()

    # Ensure 'date' is a datetime index
    df_mean_calc.set_index('ds', inplace=True)

    # Extract the time part and day of the week from the DateTimeIndex
    df_mean_calc['time_of_day'] = df_mean_calc.index.time

    # Reset the index to make "ds" a column again
    df_mean_calc.reset_index(inplace=True)

    # Group by both the day of the week and the time, then calculate the mean
    average_per_15min = df_mean_calc.groupby('time_of_day')["y"].mean()

    # Reset index for easier viewing
    average_per_15min = average_per_15min.reset_index()

    # Replicate 7 times to get a full week
    average_per_15min = pd.concat([average_per_15min]*7, ignore_index=True)

    # Join with df_mean_calc by time_of_day column
    df_mean_calc = df_mean_calc.merge(average_per_15min, on='time_of_day', suffixes=('', '_mean'))

    # Calculate the RMSE and MAE for the mean value
    rmse_mean = root_mean_squared_error(df_mean_calc['y'], df_mean_calc['y_mean'])
    mae_mean = mean_absolute_error(df_mean_calc['y'], df_mean_calc['y_mean'])

    print("RMSE mean: ", rmse_mean)
    print("MAE mean: ", mae_mean)

    # Add to a new metrics df, use concat to append
    metrics = pd.DataFrame(columns=["Model", "RMSE", "MAE"])
    metrics = pd.concat([metrics, pd.DataFrame([["Mean value", rmse_mean, mae_mean]], columns=["Model", "RMSE", "MAE"])])
    metrics = pd.concat([metrics, pd.DataFrame([["Last weeks value", rmse_last_week, mae_last_week]], columns=["Model", "RMSE", "MAE"])])
    metrics = pd.concat([metrics, pd.DataFrame([["NeuralProphet", rmse_np, mae_np]], columns=["Model", "RMSE", "MAE"])])

    print(metrics)

    # Plot the forecast
    plt.figure(figsize=(15, 7), dpi=150)
    plt.plot(df_final_test["ds"], df_final_test["y"], label="True", color="blue")
    plt.plot(df_short["ds"], df_short["y_last_week"], label="Last weeks value", color="red")
    plt.plot(df_mean_calc["ds"], df_mean_calc["y_mean"], label="Mean value", color="green")
    plt.plot(df_new["ds"], df_new["yhat1"], label="NeuralProphet", color="orange")
    plt.title("True and forecasted values for last week of data (672 timesteps)")
    plt.xlabel("Timestep")
    plt.ylabel("Activity")
    plt.legend()
    plt.savefig('output/all_models_forecast.png')

def evaluate_forecast_without_opening_hours(df_final_test_index):
    # Calculate RMSE and MAE for the last week, mean value and NeuralProphet
    rmse_last_week = root_mean_squared_error(df_final_test_index['y'], df_final_test_index['y_last_week'])
    mae_last_week = mean_absolute_error(df_final_test_index['y'], df_final_test_index['y_last_week'])

    rmse_mean = root_mean_squared_error(df_final_test_index['y'], df_final_test_index['y_mean'])
    mae_mean = mean_absolute_error(df_final_test_index['y'], df_final_test_index['y_mean'])

    rmse_np = root_mean_squared_error(df_final_test_index['y'], df_final_test_index['yhat1'])
    mae_np = mean_absolute_error(df_final_test_index['y'], df_final_test_index['yhat1'])

    # Print the metrics
    print("RMSE last week (08:00-23:00): ", rmse_last_week)
    print("MAE last week (08:00-23:00): ", mae_last_week)
    print("RMSE mean (08:00-23:00): ", rmse_mean)
    print("MAE mean (08:00-23:00): ", mae_mean)
    print("RMSE NeuralProphet (08:00-23:00): ", rmse_np)
    print("MAE NeuralProphet (08:00-23:00): ", mae_np)

    # Add to the metrics dataframe
    metrics = pd.concat([metrics, pd.DataFrame([["Mean value (08:00-23:00)", rmse_mean, mae_mean]], columns=["Model", "RMSE", "MAE"])])
    metrics = pd.concat([metrics, pd.DataFrame([["Last weeks value (08:00-23:00)", rmse_last_week, mae_last_week]], columns=["Model", "RMSE", "MAE"])])
    metrics = pd.concat([metrics, pd.DataFrame([["NeuralProphet (08:00-23:00)", rmse_np, mae_np]], columns=["Model", "RMSE", "MAE"])])

    # Save the metrics to a csv
    metrics.to_csv("output/all_models_metrics.csv", index=False)

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
        "n_changepoints": 10, # default=10, changepoints in the linear model. I don't really now what I should set this to based on my knowledge of the data - perhaps lower it?
        "yearly_seasonality": True, # Defualt value for True is 6
        "weekly_seasonality": True, # Default value for True is 3
        "daily_seasonality": 12, # Number of Fourier terms per day, default is 6. SET TO 12, because we have 15 minute data
        "seasonality_reg": 0.1,
        "n_lags": ar_net_lags, # Auto-Regressive terms, find a way to calculate this
        "ar_reg": 0.1, # Regularization parameter for AR-net, "Alternatively, a conservatively large order can be chosen when used in combination with regularization to obtain a sparse AR-model."
        "n_forecasts": forecast_range # Number of forecasts to make
    }

    # Split to get test set
    df_final_test = df.iloc[-forecast_range:]
    df_short = df.copy()

    # Last week's value
    df_short['y_last_week'] = df['y'].shift(forecast_range)
    df = df.iloc[:-forecast_range]

    # Final test
    m, train = train_model(df, params, regressorsList, weather_lag)
    save_model(m, "output/model_1.np")
    plot_training(train)

    # Forecast
    df_future = m.make_future_dataframe(df, n_historic_predictions=True, periods=forecast_range)
    forecast = m.predict(df_future)
    df_new = extract_yhat(forecast, forecast_range)

    # Evaluate forecast
    evaluate_forecast(df_final_test, df_short, df_new)

    # Evaluate forecast without opening hours
    df_final_test_index = df_final_test.between_time('08:00', '23:00')
    evaluate_forecast_without_opening_hours(df_final_test_index)

if __name__ == "__main__":
    main()
