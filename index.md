# Geospatial Human-Centered Artificial Intelligence Lab. CuBoulder

## COVID-LSTM: Predicting COVID-19 cases at the county-level in the US using a stacked LSTM.


COVID-LSTM is a stacked LSTM model that uses ...

### Team Members:

1. **Benjamin Lucas** (Postdoctoral Research Associate)
2. **Behzad Vahedi** (PhD Student)
3. **Mortreza Karimzadeh** (Assisstant Professor)

The Python code will be available [here](https://github.com/geohai/covid-lstm) soon!






To run:

(1) Open the 'code' folder on the comand line

(2) run: `python3 run_forecast.py -d YYYY-MM-DD`, where YYYY-MM-DD is the forecast_date. For example:

```python
python3 run_forecast.py -d 2020-12-27
```

(3) Wait approx 30-40 minutes.

(4) Find the resulting csv file in the 'results' folder. It contains county-level case predictions for 1-, 2-, 3-, and 4-wk horizons in the format requested by the COVID-19 ForecastHub submission guidelines.

Please note that the forecast date must be a Sunday in accordance with the COVID-19 ForecastHub submission guidelines.



### Support or Contact

Having trouble with Pages? Check out our [metadata](https://github.com/geohai/covid-lstm/blob/main/metadata.yml) or [contact us](mailto:benjamin.Lucas@colorado.edu) and we’ll help you sort it out.
