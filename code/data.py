import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class covidData():

    def __init__(self, temporal_lag, current_date, forecast_horizon):
        self.temporal_lag = temporal_lag
        self. current_date = current_date
        self.forecast_horizon = forecast_horizon
        self.all_ts_data = None
        self.all_targets = None


    def load_dataframes(self, dir="../data/", print_info=False):
        # load time series data
        timeseries_df = pd.read_csv(dir+"temporal_data_"+\
                                    str(self.current_date)+".csv")

        # preprocess ts data
        timeseries_df = timeseries_df.replace([np.inf, -np.inf], np.NaN)
        na_cols = timeseries_df.columns[timeseries_df.isna().any()].tolist()
        for col in na_cols:
            timeseries_df[col] = timeseries_df.groupby(\
                                                      ["current_date","GEOID"])\
                                               [col].transform(\
                                                  lambda x: x.fillna(x.mean()))
        target_cols = [0,3,4,5]
        self.all_targets = timeseries_df.iloc[:,target_cols]
        if print_info:
            print("All TS shape", timeseries_df.shape)

        cols = [0,1,2,3,4]
        timeseries_df = timeseries_df.iloc[:,cols]

        self.all_ts_data = timeseries_df


    def create_train_test_datasets(self, print_info=False):
        temporal_lag = self.temporal_lag
        current_date = self.current_date
        test_horizon = self.forecast_horizon

        # create train/test sets
        geoid_list = np.unique(self.all_ts_data.GEOID)
        dates_recorded = np.unique(self.all_ts_data.current_date)
        no_timestamps = len(dates_recorded)
        if print_info:
            print("No timestamps: ", no_timestamps)

        X_train_ts, X_train_se, y_train = [], [], []
        X_test_ts, X_test_se, y_test = [], [], []
        test_info = []

        for i,id in enumerate(geoid_list):
            county_ts = self.all_ts_data[self.all_ts_data["GEOID"]==id].\
                                                         reset_index(drop=True)
            county_ts["current_date"] = pd.to_datetime(county_ts["current_date"],
                                                       format="%Y-%m-%d")
            county_ts.sort_values(by="current_date", inplace=True, axis=0)
            county_ts.reset_index(inplace=True)
            current_index = county_ts.index\
                                [county_ts["current_date"]==current_date][0]

            if print_info:
                if i==0:
                    print("Current index: ", current_index)
            county_ts = county_ts.values[:,5]
            county_ts = county_ts.reshape((-1,1))

            county_targets = self.all_targets[self.all_targets["GEOID"]==id]\
                                              .values[:,2]

            county_test_info = self.all_targets[self.all_targets["GEOID"]==id]\
                                                   .values[:,0]

            if print_info:
                if i==0:
                    print("County TS data shape:", county_ts.shape)
                    print("County target shape:", county_targets.shape)
                    print("County Test Info data shape:",
                          county_test_info.shape)

            y_index = 0

            while y_index+temporal_lag+test_horizon <= current_index:
                ts_instance = county_ts[y_index:y_index+temporal_lag+1,:]
                X_train_ts.append(ts_instance)
                y_train.append(county_ts[y_index+temporal_lag+test_horizon])
                if print_info:
                    if i==0:
                        print("Train X:", ts_instance)
                        print("Train Target: ", county_ts[\
                                            y_index+temporal_lag+test_horizon])

                y_index += 1

            X_test_ts.append(county_ts[
                            (current_index-temporal_lag):(current_index+1),:])
            test_info.append(county_test_info[current_index])
            if print_info:
                if i==0:
                    print("Test X:", county_ts[\
                             (current_index-temporal_lag):(current_index+1),:])

        data_dict = {
            "X_train_ts": np.array(X_train_ts, dtype="float32"),
            "y_train": np.array(y_train, dtype="float32"),
            "X_test_ts": np.array(X_test_ts, dtype="float32"),
            "test_info": np.array(test_info)
        }

        return data_dict
