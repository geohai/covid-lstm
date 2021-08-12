# covid-lstm

Python code for predicting COVID-19 cases at the county-level in the US using a stacked LSTM.

To run:

(1) Open the 'code' folder on the comand line

(2) run: python3 run_forecast.py -d YYYY-MM-DD, where YYYY-MM-DD is the forecast_date. For example:

    python3 run_forecast.py -d 2020-12-27

(3) Wait approx 30-40 minutes.

(4) Find the resulting csv file in the 'results' folder. It contains county-level case predictions for 1-, 2-, 3-, and 4-wk horizons in the format requested by the COVID-19 ForecastHub submission guidelines.

Please note that the forecast date must be a Sunday in accordance with the COVID-19 ForecastHub submission guidelines.
