import os
import csv
import sys
import numpy as np
import pandas as pd
import geopandas as gp
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Concatenate
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from datetime import date, timedelta


class covidLSTMQuantiles():

    def __init__(self, temporal_lag, no_temporal_ft, opt="adam",
                 loss_fn="mse"):
        self.temporal_lag = temporal_lag
        self.no_temporal_ft = no_temporal_ft
        self.outputs = []
        self.quantiles = [.025, .100, .250, .500, .750, .900, .975]

        # define model architecure
        ts_input = Input(shape=(temporal_lag+1, no_temporal_ft))
        lstm_1_out = LSTM(64, activation="relu", return_sequences=True)(ts_input)
        lstm_2_out = LSTM(64, activation="relu")(lstm_1_out)
        dense_1_out = Dense(64)(lstm_2_out)

        # create an output for each quantile
        for i,quantile in enumerate(self.quantiles):
            output = Dense(1, name="q_"+str(i))(dense_1_out)
            self.outputs.append(output)

        # loss
        q_loss = MultiQuantileLoss(self.quantiles)

        # intialize & compile
        model = Model(inputs=ts_input, outputs=self.outputs)
        model.compile(optimizer=opt, loss=q_loss.call)

        self.model = model


    def train(self, X_ts, y, no_epochs):
        tf.random.set_seed(18)
        self.no_epochs = no_epochs
        self.model.fit(x=X_ts, y=y, epochs=no_epochs, verbose=1)


    def test(self, X_ts, y, test_info_df):
        predictions_test = self.model.predict(x=X_ts)
        predictions_test = np.transpose(np.array(predictions_test))[0]
        assert predictions_test.shape[0]==test_info_df.shape[0]

        results = np.hstack((test_info_df,
                             y.reshape((-1,1)),
                             predictions_test))

        col_names=["GEOID", "target_date", "raw_cases","correct_value"]
        for quantile in self.quantiles:
            col_names.append("q_"+str(int(quantile*1000))+"_pred")

        results_df = pd.DataFrame(data=results, columns=col_names)
        self.results = results_df
        return results_df


    def predict(self, X_ts, test_info_df):
        predictions_test = self.model.predict(x=X_ts)
        predictions_test = np.transpose(np.array(predictions_test))[0]
        assert predictions_test.shape[0]==test_info_df.shape[0]

        results = np.hstack((test_info_df.reshape((-1,1)),
                             predictions_test))
        print(results)

        col_names=["GEOID"]
        for quantile in self.quantiles:
            col_names.append("q_"+str(int(quantile*1000))+"_pred")

        results_df = pd.DataFrame(data=results, columns=col_names)
        results_df = results_df.astype({"GEOID": np.int})
        self.results = results_df

        return results_df


    def save_results(self, current_date, target_date, test_horizon,
                     print_info=False, run=1):
        results_df = self.results
        temporal_lag = self.temporal_lag

        results_folder = "../results/quantile_predictions/"
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)

        results_file = "../results/quantile_predictions/predictions_"+\
                       current_date+"_horizon_"+str(test_horizon)+\
                       "_lag_"+str(temporal_lag)+"_run_"+str(run)+".csv"

        if not os.path.isfile(results_file):
            counties_df = gp.read_file("../data/Conterminous_US_counties"+\
                                                                    ".geojson")
            cols = ["FIPS", "POPULATION"]
            counties_df = counties_df[cols]

            results_df = results_df.merge(counties_df, right_on="FIPS",
                                          left_on="GEOID", how="left")

            for quantile in self.quantiles:
                quantile = str(int(quantile*1000))
                results_df["q_"+quantile+"_pred"] =\
                      results_df["q_"+quantile+"_pred"].astype(float)
                results_df["q_"+quantile+"_pred_transform"] =\
                          np.exp(results_df["q_"+quantile+"_pred"])-1
                results_df["q_"+quantile+"_pred_cases"] =\
                       results_df["q_"+quantile+"_pred_transform"]*\
                                                 results_df["POPULATION"]/10000

            results_df.drop(cols, axis=1, inplace=True)
            results_df.to_csv(results_file, index=False)
            print("predictions saved for model "+str(run))

class MultiQuantileLoss(tf.keras.losses.Loss):

    def __init__(self, quantiles:list):
        super(MultiQuantileLoss, self).__init__()
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        # get quantile value
        q_id = int(y_pred.name.split("/")[1][2:])
        q = self.quantiles[q_id]

        # minimize quantile error
        q_error = tf.subtract(y_true, y_pred)
        q_loss = tf.reduce_mean(tf.maximum(q*q_error, (q-1)*q_error), axis=-1)
        return q_loss


def ensemble_results(current_date, test_horizon, lag, ensemble_sz=10):
    quantile_list = [.025, .100, .250, .500, .750, .900, .975]
    for i, quantile in enumerate(quantile_list):
        index = 9 + i*2

        for run in range(1,ensemble_sz+1):
            results_file = "../results/quantile_predictions/predictions_"+\
                           current_date+"_horizon_"+str(test_horizon)+"_lag_"+\
                           str(lag)+"_run_"+str(run)+".csv"
            if not os.path.isfile(results_file):
                print("Filename: ", results_file)
                print("Not a correct filename!!! SKIPPING")
                continue
            single_run_preds = pd.read_csv(results_file)

            if run==1:
                cols = [0,index]
                all_preds = single_run_preds.iloc[:,cols]
                print(all_preds)
                if i==0:
                    ens_preds = all_preds.iloc[:,0]
            else:
                all_preds = pd.concat([all_preds, single_run_preds.iloc[:,index]],
                                       axis=1)
                print(ens_preds)

        col_name = "pred_q_"+str(int(quantile*1000))
        median = pd.DataFrame(np.median(all_preds.iloc[:,2:], axis=1),
                              columns = [col_name])
        ens_preds = pd.concat([ens_preds, median], axis=1)

    ens_results_file = "../results/quantile_predictions/ensemble_predictions_"+\
                       current_date+"_horizon_"+str(test_horizon)+"_lag_"+\
                       str(lag)+".csv"

    if not os.path.isfile(ens_results_file):
        ens_preds.to_csv(ens_results_file, index=False)

    for run in range(1, ensemble_sz+1):
        results_file = "../results/quantile_predictions/predictions_"+\
                       current_date+"_horizon_"+str(test_horizon)+"_lag_"+\
                       str(lag)+"_run_"+str(run)+".csv"
        if os.path.isfile(results_file):
            os.remove(results_file)


def organize_csv_for_forecast_hub(current_date, test_horizon, save=True):
    quantile_list = [25, 100, 250, 500, 750, 900, 975]

    all_results_file = "../results/quantile_predictions/ensemble_predictions_"+\
                       current_date+"_horizon_"+str(test_horizon)+"_lag_9.csv"
    all_results_df = pd.read_csv(all_results_file)

    submission_file = "../results/"+current_date+"-CUBoulder-COVIDLSTM.csv"

    header = ["forecast_date", "target", "target_end_date", "location", "type",
              "quantile", "value"]
    print(header)
    if not os.path.isfile(submission_file):
        with open(submission_file, mode="w") as file:
            file_writer = csv.writer(file, delimiter=",")
            file_writer.writerow(header)

    date_list = current_date.split("-")
    current_date_obj = date(int(date_list[0]),
                            int(date_list[1]),
                            int(date_list[2]))
    target_date = str(current_date_obj + timedelta(days=(test_horizon*7-1)))
    counties_list = np.unique(all_results_df.GEOID.values)

    for county in counties_list:
        county_results = all_results_df[all_results_df.GEOID==county]
        fips = str(county)
        while len(fips) < 5:
            fips = "0"+str(fips)
        for i, quantile in enumerate(quantile_list):
            if len(str(quantile))==2:
                quantile = "0"+str(quantile)
            pred_value = county_results.iloc[0, 1+i]
            if pred_value < 0:
                pred_value = 0
            row = [current_date,
                   str(test_horizon)+" wk ahead inc case",
                   target_date,
                   fips,
                   "quantile",
                   "0."+str(quantile),
                   pred_value]
            print(row)
            if save:
                with open(submission_file, mode="a") as file:
                    file_writer = csv.writer(file, delimiter=",")
                    file_writer.writerow(row)
            if quantile==500:
                row = [current_date,
                       str(test_horizon)+" wk ahead inc case",
                       target_date,
                       fips,
                       "point",
                       "NA",
                       pred_value]
                print(row)
                if save:
                    with open(submission_file, mode="a") as file:
                        file_writer = csv.writer(file, delimiter=",")
                        file_writer.writerow(row)
