import os
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta, date
import requests, zipfile, io
from zipfile import ZipFile
from scipy.stats import linregress
import time

def get_us_counties(contiguous=True):

    url="https://drive.google.com/file/d/10c4S0THYgJxKjaUNUiLhcVFecte4xE44/view?usp=sharing"
    url2='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = gpd.read_file(url2)
    df['FIPS'] = df['GEOID'].astype(int)

    return df


def get_JH_covid_data(target, smooth):

    assert isinstance(smooth, bool), "Smooth must be a boolean variable!"

    if target == 'case':
        jh_data_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
        cols_to_drp = ['UID', 'iso2', 'iso3', 'code3','Country_Region', 'Lat', 'Long_']

    elif target=='death':
        jh_data_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
        cols_to_drp = ['UID', 'iso2', 'iso3', 'code3','Country_Region', 'Lat', 'Long_','Population']
    else:
        print("invalid argument for target. Acceptable values are: 'case' or 'death'")
        return None

    jh_covid_df = pd.read_csv(jh_data_url)

    # preprocessing JH COVID data
    jh_covid_df.dropna(axis=0, how='any', inplace=True)
    jh_covid_df['FIPS'] = jh_covid_df['FIPS'].astype('int64')
    jh_covid_df.drop(columns=cols_to_drp, inplace=True)

    #Important: check to see the column index is adherent to the imported df
    first_date = datetime.strptime(jh_covid_df.columns[4], '%m/%d/%y').date()
    last_date = datetime.strptime(jh_covid_df.columns[-1], '%m/%d/%y').date()

    current_date = last_date
    previous_date = last_date - timedelta (days=1)

    while current_date > first_date:

        #For unix, replace # with - in the time format
        current_col = current_date.strftime('%-m/%-d/%y') #replace # with - in Mac or Linux
        previous_col = previous_date.strftime('%-m/%-d/%y')
        jh_covid_df[previous_col] = np.where(jh_covid_df[previous_col] > jh_covid_df[current_col], jh_covid_df[current_col], jh_covid_df[previous_col])
        current_date = current_date - timedelta(days=1)
        previous_date = previous_date - timedelta(days=1)

    if smooth:
        jh_covid_df.iloc[:,4:] = jh_covid_df.iloc[:,4:].rolling(7,min_periods=1,axis=1).mean()

    return jh_covid_df


def combine_data(forecast_date, contiguous_counties, covid_df, covid_df_non_smooth):

    assert forecast_date.weekday()==6 ## forecast date should be a Sunday

    T_end = forecast_date - timedelta(days=1) # Saturday
    T_start = T_end - timedelta(weeks=1) # Saturday

    dates = [T_end, T_start]
    dates_case_str = [item.strftime('%-m/%-d/%y') for item in dates]

    jh_df = covid_df[['FIPS', *dates_case_str]]

    jh_df_non_smooth  = covid_df_non_smooth[['FIPS', *dates_case_str]]

    temp = contiguous_counties
    temp['current_date'] = forecast_date.strftime('%Y-%m-%d')

    # add covid-related columns
    temp = temp.merge(jh_df, on='FIPS', how='left')

    temp['DELTA_INC_RATE'] = (temp[dates_case_str[0]] - temp[dates_case_str[1]]) / temp['POPULATION'] * 10000
    temp['LOG_DELTA_INC_RATE'] = np.log(temp['DELTA_INC_RATE'] + 1)

    # drop unnecessary columns
    temp.drop(columns='DELTA_INC_RATE', inplace=True)
    temp.drop(columns=dates_case_str, inplace=True)

    output = temp.merge(jh_df_non_smooth, on='FIPS', how='left')
    output['DELTA_CASE_JH'] = output[dates_case_str[0]] - output[dates_case_str[1]]

    output.drop(columns=dates_case_str, inplace=True)

    return output


def download_data(current_date):

    counties_df = get_us_counties(contiguous=True)

    covid_df = get_JH_covid_data(target="case", smooth=True)
    covid_df_contiguous = covid_df[covid_df['FIPS'].isin(counties_df['FIPS'])].copy()

    covid_df_non_smooth = get_JH_covid_data('case', smooth=False)
    covid_df_non_smooth_contiguous = covid_df_non_smooth[covid_df_non_smooth['FIPS'].isin(counties_df['FIPS'])].copy()

    df_lagged_list = []

    forecast_date = date(int(current_date.split("-")[0]),
                         int(current_date.split("-")[1]),
                         int(current_date.split("-")[2]))

    while forecast_date >= date(2020, 4, 5):
        df_week = combine_data(forecast_date,
                               counties_df,
                               covid_df_contiguous,
                               covid_df_non_smooth_contiguous)
        df_lagged_list.append(df_week)

        forecast_date -= timedelta(days=7)

    df_lagged = pd.concat(df_lagged_list, axis=0)

    cols_to_save = ['GEOID', 'NAME', 'STATE', 'current_date',
                    'LOG_DELTA_INC_RATE', 'DELTA_CASE_JH']

    df_lagged = df_lagged[cols_to_save]
    df_lagged.sort_values(by='GEOID', inplace=True)

    file_name = 'temporal_data_{}.csv'.format(current_date)
    df_lagged.to_csv('../data/' + file_name, index=False)

if __name__ == '__main__':
    download_data("2020-12-27")
