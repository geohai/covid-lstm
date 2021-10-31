# covid-lstm

Python code for predicting COVID-19 cases at the county-level in the US using a stacked LSTM.

Package requirements:
```
    tensorflow (tf-gpu)
    pandas
    geopandas
    scikit-learn 
    keras
```
To cite this work, please use

@article{lucas2021spatiotemporal,
  title={A spatiotemporal machine learning approach to forecasting COVID-19 incidence at the county level in the United States},
  author={Lucas, Benjamin and Vahedi, Behzad and Karimzadeh, Morteza},
  journal={arXiv preprint arXiv:2109.12094},
  year={2021}
}

And for more background, please refer to:

@article{vahedi2021predicting,
  title={Predicting County-Level COVID-19 Cases using Spatiotemporal Machine Learning: Modeling Human Interactions using Social Media and Cell-Phone Data},
  author={Vahedi, Behzad and Karimzadeh, Morteza and Zoraghein, Hamidreza},
  year={2021}
}

To run:

(1) Open the 'code' folder on the comand line

(2) run: python3 run_forecast.py -d YYYY-MM-DD, where YYYY-MM-DD is the forecast_date. For example:

    python3 run_forecast.py -d 2020-12-27

(3) Wait approx 30-40 minutes.

(4) Find the resulting csv file in the 'results' folder. It contains county-level case predictions for 1-, 2-, 3-, and 4-wk horizons in the format requested by the COVID-19 ForecastHub submission guidelines.

Please note that the forecast date must be a Sunday in accordance with the COVID-19 ForecastHub submission guidelines.

Acknowledgement: 
“This work was supported by the Population Council, and the University of Colorado Population Center (CUPC) funded by Eunice Kennedy Shriver National Institute of Child Health & Human Development of the National Institutes of Health (P2CHD066613). The content is solely the responsibility of the authors, and does not reflect the views of the Population Council, or official views of the NIH, CUPC, or the University of Colorado.”
