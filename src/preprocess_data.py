# Import dependencies
import os
import pandas as pd

def load_data():
    # Load the data
    df = pd.read_csv('data/boulders/AttendanceHistorys.csv', sep=";")
    return df

def preprocess_data(df):
    # Make timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=False)

    # Sort by timestamp, and reset index
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    # Drop rows
    df = df[df["Activity"] != "Aarhus Ud låge"]

    # Replace rows with "Aarhus Ind låge" with "Aarhus"
    df["Activity"] = df["Activity"].replace("Aarhus C Indgang", "Aarhus City Indgang")
    df["Activity"] = df["Activity"].replace("Aarhus Indgang", "Aarhus Nord Indgang")
    df["Activity"] = df["Activity"].replace("Hvidovre", "Hvidovre Indgang")

    # Drop first 6 row, as these seem odd and perhaps are corrupt
    df = df[6:]

    # Make a column for each unique value of "Activity"
    df = pd.concat([df, pd.get_dummies(df['Activity'], dtype=int)], axis=1)

    # Drop the "Activity" column
    df = df.drop(columns="Activity")

    # Drop the columns
    df.drop(columns=["CustomerActivityLinkID", "Type", "Status", "CustomerContractLinkID", "Comment", "AttendanceLocationName", "CustomerID", "AttendanceLocationType", "Id"], inplace=True)

    return df

def save_data(df):
    # Save the data
    df.to_csv('data/boulders/AttendanceHistorys_clean.csv', index=False, sep=";")

def resample_data(df):
    # Sort by timestamp
    df.sort_values("Timestamp", inplace=True)

    # Resample by the hour
    df.set_index("Timestamp", inplace=True)

    # Count by hour
    df_hourly = df.resample("15Min").sum()

    return df_hourly

def add_empty_rows(df_hourly):
    # Start time
    start_time = pd.Timestamp("2021-05-06 00:00:00")

    # Add empty rows for missing hours
    idx = pd.date_range(start=start_time, end=df_hourly.index[-1], freq="15Min")

    # Add rows according to the idx list, fill with 0
    df_hourly = df_hourly.reindex(idx, fill_value=0)

    return df_hourly

def nullify_hours(df, column, start_time, end_time):
    # All values observed between closing/opening are set to 0.
    # Define a function to apply the changes
    count = [0]
    df_temp = df.between_time(start_time, end_time)
    df_temp[column] = df_temp[column].apply(lambda x: (count.append(1), 0)[1] if x > 0 else x)
    df.loc[df_temp.index, column] = df_temp[column]
    print(f'Changed {len(count) - 1} non-zero values in {column}')

def rolling_activity(df_hourly, window_size):
    # Window size
    # Calculate the rolling activity in each gym, setting the window to 2 hours. Make all columns into integers
    df_hourly["Aarhus City Activity"] = df_hourly["Aarhus City Indgang"].rolling(window=window_size).sum()
    df_hourly["Aarhus Nord Activity"] = df_hourly["Aarhus Nord Indgang"].rolling(window=window_size).sum()
    df_hourly["Hvidovre Activity"] = df_hourly["Hvidovre Indgang"].rolling(window=window_size).sum()
    df_hourly["København Activity"] = df_hourly["København Indgang"].rolling(window=window_size).sum()
    df_hourly["Odense Activity"] = df_hourly["Odense Indgang"].rolling(window=window_size).sum()
    df_hourly["Valby Activity"] = df_hourly["Valby Indgang"].rolling(window=window_size).sum()

    # Fill nans with 0
    df_hourly.fillna(0, inplace=True)

    df_hourly = df_hourly.astype(int)

    return df_hourly

def join_dmi_data():
    # Get full file path of contents of data/dmi folder
    dmi_files = os.listdir('data/dmi/monthly')

    wd = os.getcwd()

    # Load each csv, and add to each other
    df = pd.DataFrame()

    for csv in dmi_files:
        current_df = pd.read_csv(f"{wd}/data/dmi/monthly/{csv}", sep = ";")
        df = pd.concat([df, current_df], ignore_index=True)

    # Change column names
    df = df.rename(columns={"DateTime": "Timestamp"})

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=False)

    df.sort_values(by="Timestamp", inplace=True)

    # Save to csv
    df.to_csv("data/dmi/dmi.csv", index=False, sep=";")

def join_weather_data(df_hourly):
    df_dmi = pd.read_csv('data/dmi/dmi.csv', sep=";")

    df_dmi["Timestamp"] = pd.to_datetime(df_dmi["Timestamp"])

    df_dmi.set_index("Timestamp", inplace=True)

    # Since dmi data is hourly, we will resample this to 15 minutes
    # Drop duplicates from the index
    df_dmi = df_dmi.loc[~df_dmi.index.duplicated(keep='first')]

    # Now you can resample
    df_dmi = df_dmi.resample("15Min").ffill()

    # Now join the dataframes by timestamp. 
    df_final = df_hourly.join(df_dmi, how="inner")

    df_final = df_final.drop(columns=["Maksimumtemperatur", "Minimumtemperatur"])

    return df_final

def join_medlemsdata(df_final):
    df_medlem = pd.read_csv('data/boulders/Medlemsdata_clean.csv', sep=";")

    df_medlem["Timestamp"] = pd.to_datetime(df_medlem["Timestamp"])
    df_medlem.set_index("Timestamp", inplace=True)

    # Since the data is daily, we will resample this to 15 minutes
    df_medlem = df_medlem.resample("15Min").ffill()

    # Join the dataframes by timestamp
    df_final = df_final.join(df_medlem, how="inner")

    return df_final

def save_final_dataset(df_final):
    # Save the final dataframe
    df_final.to_csv('data/full_dataset_15Min.csv', sep=";")

def main():
    df = load_data()
    df = preprocess_data(df)
    save_data(df)
    df_hourly = resample_data(df)
    df_hourly = add_empty_rows(df_hourly)
    nullify_hours(df_hourly, "Aarhus City Indgang", '23:00:00', '07:45:00')
    nullify_hours(df_hourly, "Aarhus Nord Indgang", '22:00:00', '09:45:00')
    nullify_hours(df_hourly, "Hvidovre Indgang", '22:00:00', '09:45:00')
    nullify_hours(df_hourly, "København Indgang", '23:00:00', '07:45:00')
    nullify_hours(df_hourly, "Odense Indgang", '22:00:00', '09:45:00')
    nullify_hours(df_hourly, "Valby Indgang", '22:00:00', '09:45:00')
    df_hourly = rolling_activity(df_hourly, window_size=8)
    nullify_hours(df_hourly, "Aarhus City Activity", '23:00:00', '07:45:00')
    nullify_hours(df_hourly, "Aarhus Nord Activity", '22:00:00', '09:45:00')
    nullify_hours(df_hourly, "Hvidovre Activity", '22:00:00', '09:45:00')
    nullify_hours(df_hourly, "København Activity", '23:00:00', '07:45:00')
    nullify_hours(df_hourly, "Odense Activity", '22:00:00', '09:45:00')
    nullify_hours(df_hourly, "Valby Activity", '22:00:00', '09:45:00')
    join_dmi_data()
    df_final = join_weather_data(df_hourly)
    df_final = join_medlemsdata(df_final)
    save_final_dataset(df_final)

if __name__ == "__main__":
    main()
